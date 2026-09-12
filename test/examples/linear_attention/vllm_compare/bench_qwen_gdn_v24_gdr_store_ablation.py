#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import qwen_gdn_kkt_avelang_v24_bt16_mfma_layout
from qwen_gdn_chunked_avelang_v24_gdr_store_ablation import qwen_gdn_chunk_gdr_avelang_v24_store_ablation


VARIANTS = ["baseline_copy", "no_vn_store", "no_h_store", "no_h_no_vn_store"]


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int):
    torch.manual_seed(seed)
    q = l2norm(torch.randn(1, t, 4, 128, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(1, t, 4, 128, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(1, t, 8, 128, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(1, t, 8, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(1, t, 8, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = (torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def prepare_gdr_inputs(t: int):
    _q, k, v, g, beta, initial_state = make_inputs(t, seed=24000 + t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=16)
    a = qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(k, g_cumsum, beta, chunk_size=16)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=16)
    w, u = qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=16)
    gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=16)
    sync()
    return k, w, u, gdr_decay, gdr_g_last_exp, initial_state


def time_fn(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    sync()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def run_t(t: int, warmup: int, repeat: int):
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = prepare_gdr_inputs(t)
    rows = {}
    for variant in VARIANTS:
        fn = lambda variant=variant: qwen_gdn_chunk_gdr_avelang_v24_store_ablation(
            k,
            w,
            u,
            gdr_decay,
            gdr_g_last_exp,
            initial_state,
            variant=variant,
            chunk_size=16,
        )
        rows[variant] = time_fn(fn, warmup, repeat)
    base = rows["baseline_copy"]
    print(
        f"T={t} "
        + " ".join(f"{name}={rows[name]:.6f}ms speedup={base / rows[name]:.4f}x" for name in VARIANTS)
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    for t in args.T:
        run_t(t, args.warmup, args.repeat)


if __name__ == "__main__":
    main()

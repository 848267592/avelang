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
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 1234):
    torch.manual_seed(seed)
    b, hk, hv, kdim = 1, 4, 8, 128
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    return k, g, beta


def make_a(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)


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


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def run_t(t: int, warmup: int, repeat: int) -> None:
    k, g, beta = make_inputs(t, seed=18000 + t)
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,dtype=BF16,layout=[B,T,H,D]")

    a16 = make_a(k, g, beta, 16)
    a32 = make_a(k, g, beta, 32)
    a64 = make_a(k, g, beta, 64)

    solve_v6_16 = lambda: qwen_gdn_solve_avelang_v6_standalone(a16, chunk_size=16)
    solve_v6_32 = lambda: qwen_gdn_solve_avelang_v6_standalone(a32, chunk_size=32)
    solve_v18_32 = lambda: qwen_gdn_solve_avelang_v18_layout(a32, chunk_size=32)
    solve_v6_64 = lambda: qwen_gdn_solve_avelang_v6_standalone(a64, chunk_size=64)
    solve_v18_64 = lambda: qwen_gdn_solve_avelang_v18_layout(a64, chunk_size=64)

    ref32 = solve_v6_32()
    new32 = solve_v18_32()
    ref64 = solve_v6_64()
    new64 = solve_v18_64()
    sync()

    v6_16_ms = time_fn(solve_v6_16, warmup, repeat)
    v6_32_ms = time_fn(solve_v6_32, warmup, repeat)
    v18_32_ms = time_fn(solve_v18_32, warmup, repeat)
    v6_64_ms = time_fn(solve_v6_64, warmup, repeat)
    v18_64_ms = time_fn(solve_v18_64, warmup, repeat)

    print(
        "solve_latency_ms,"
        f"T={t},"
        f"v6_bt16={v6_16_ms:.6f},"
        f"v6_bt32={v6_32_ms:.6f},"
        f"v18_bt32={v18_32_ms:.6f},"
        f"v6_bt64={v6_64_ms:.6f},"
        f"v18_bt64={v18_64_ms:.6f}"
    )
    print(
        "solve_speedup,"
        f"T={t},"
        f"bt32_vs_v6={v6_32_ms / v18_32_ms:.4f},"
        f"bt64_vs_v6={v6_64_ms / v18_64_ms:.4f},"
        f"bt32_usable_under_0p15ms={v18_32_ms < 0.15},"
        f"bt64_under_0p5ms={v18_64_ms < 0.5}"
    )
    print(
        "solve_accuracy,"
        f"T={t},"
        f"bt32_max_abs={max_err(new32, ref32):.9g},"
        f"bt64_max_abs={max_err(new64, ref64):.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")
    for t in args.T:
        run_t(t, args.warmup, args.repeat)


if __name__ == "__main__":
    main()

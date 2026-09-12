#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics

import torch

from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import qwen_gdn_gdr_decay_avelang_v17
from qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout,
)
from qwen_gdn_chunked_avelang_v25_gdr_bt64_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v25_bt64_bv32_mfma_layout,
    qwen_gdn_chunk_gdr_torch_ref_bt64,
)


def sync() -> None:
    torch.cuda.synchronize()


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 2525):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    k = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    w = torch.randn(b, t, hv, kdim, device="cuda", dtype=torch.float32).contiguous() * 0.05
    u = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.float32).contiguous() * 0.05
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, w, u, g, initial_state


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


def run_t(t: int, warmup: int, repeat: int, check_ref: bool):
    k, w, u, g, initial_state = make_inputs(t, seed=252500 + t)
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16-k,chunk_gdr_only")

    gdr_decay16, gdr_g_last_exp16 = qwen_gdn_gdr_decay_avelang_v17(g, chunk_size=16)
    f_v23 = lambda: qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout(
        k,
        w,
        u,
        g,
        gdr_decay16,
        gdr_g_last_exp16,
        initial_state=initial_state,
        chunk_size=16,
    )
    f_v25 = lambda: qwen_gdn_chunk_gdr_avelang_v25_bt64_bv32_mfma_layout(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        chunk_size=64,
    )

    h25, vn25, fs25 = f_v25()
    sync()
    err_h = err_vn = err_fs = float("nan")
    if check_ref:
        h_ref, vn_ref, fs_ref = qwen_gdn_chunk_gdr_torch_ref_bt64(k, w, u, g, initial_state=initial_state, chunk_size=64)
        sync()
        err_h = max_err(h25, h_ref)
        err_vn = max_err(vn25, vn_ref)
        err_fs = max_err(fs25, fs_ref)

    v23_ms = time_fn(f_v23, warmup, repeat)
    v25_ms = time_fn(f_v25, warmup, repeat)
    row = {
        "T": t,
        "v23_bt16_chunk_gdr_ms": v23_ms,
        "v25_bt64_bv32_chunk_gdr_ms": v25_ms,
        "speedup_v25_vs_v23": v23_ms / v25_ms,
        "h_err": err_h,
        "vn_err": err_vn,
        "final_state_err": err_fs,
    }
    print(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--no-ref", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    rows = [run_t(t, args.warmup, args.repeat, not args.no_ref) for t in args.T]
    print("summary")
    for r in rows:
        print(
            f"T={r['T']} v23_bt16={r['v23_bt16_chunk_gdr_ms']:.4f}ms "
            f"v25_bt64_bv32={r['v25_bt64_bv32_chunk_gdr_ms']:.4f}ms "
            f"speedup={r['speedup_v25_vs_v23']:.4f} "
            f"h_err={r['h_err']:.6g} vn_err={r['vn_err']:.6g} fs_err={r['final_state_err']:.6g}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal correctness repro for qwen_gdn_chunked_avelang_v11_mfma_layout_fixed.py.

This intentionally tests only the narrow v11 MFMA chunk_gdr path:
    B=1,T=16,Hk=4,Hv=8,K=128,V=128,BF16,chunk=16,block_v=16,block_k=64

It compares h/vn/final_state against the v10 scalar chunk_gdr oracle.  Do not
use this script as a performance benchmark.
"""

from __future__ import annotations

import argparse
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v10_vllm_layout_fixed import qwen_gdn_chunk_gdr_avelang_v10_vllm_layout
from qwen_gdn_chunked_avelang_v11_mfma_layout_fixed import qwen_gdn_chunk_gdr_avelang_v11_mfma_layout


def max_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def max_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-6)
    return (diff / denom).max().item()


def print_err(name: str, actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    torch.cuda.synchronize()
    abs_err = max_abs_err(actual, expected)
    rel_err = max_rel_err(actual, expected)
    print(f"{name}_max_abs={abs_err:.9g}")
    print(f"{name}_max_rel={rel_err:.9g}")
    return abs_err, rel_err


def run_case(*, with_initial_state: bool, seed: int, atol: float) -> bool:
    torch.manual_seed(seed)
    device = "cuda"
    B, T, Hk, Hv, K, V = 1, 16, 4, 8, 128, 128
    chunk = 16

    k = torch.randn(B, T, Hk, K, device=device, dtype=torch.bfloat16).contiguous()
    v = torch.randn(B, T, Hv, V, device=device, dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device=device)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(B, T, Hv, device=device)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(B, Hv, V, K, device=device, dtype=torch.float32) * 0.01).contiguous()

    print("target_shape,B=1,T=16,Hk=4,Hv=8,K=128,V=128,dtype=BF16,chunk=16")
    print(f"with_initial_state={with_initial_state}")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k, v, g_cumsum, beta, a_solved, chunk_size=chunk, prefer_optimized=True
    )

    print("running_v10_oracle")
    h10, vn10, final10 = qwen_gdn_chunk_gdr_avelang_v10_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk,
        use_parallel_chunk_gdr=True,
        parallel_mode="chunk_vk",
        block_v=4,
        block_k=64,
    )
    torch.cuda.synchronize()

    print("running_v11_mfma")
    h11, vn11, final11 = qwen_gdn_chunk_gdr_avelang_v11_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk,
        use_mfma_chunk_gdr=True,
        prefer_optimized=True,
        block_v=16,
        block_k=64,
    )
    torch.cuda.synchronize()

    h_abs, _ = print_err("h", h11, h10)
    print("vn_vs_u_max_abs", (vn11.float() - u.float()).abs().max().item())
    vn_abs, _ = print_err("vn", vn11, vn10)
    final_abs, _ = print_err("final_state", final11, final10)

    ok = h_abs <= atol and vn_abs <= atol and final_abs <= atol
    print(f"case_status={'ok' if ok else 'failed'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--both", action="store_true", help="Run both with and without initial_state.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This repro requires a CUDA/HIP GPU.")

    if args.both:
        ok0 = run_case(with_initial_state=False, seed=args.seed, atol=args.atol)
        ok1 = run_case(with_initial_state=True, seed=args.seed + 1, atol=args.atol)
        if not (ok0 and ok1):
            raise SystemExit(1)
        return

    ok = run_case(with_initial_state=not args.no_initial_state, seed=args.seed, atol=args.atol)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

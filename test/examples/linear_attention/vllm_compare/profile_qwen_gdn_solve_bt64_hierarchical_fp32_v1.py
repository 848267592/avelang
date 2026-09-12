#!/usr/bin/env python3
"""Minimal dispatch runner for rocprof of the opt-in Stage 5B S0 solve."""

from __future__ import annotations

import argparse

import torch

from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.T <= 0 or args.T % 64 != 0:
        raise ValueError("T must be positive and divisible by 64")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    torch.manual_seed(20260716)
    a = torch.randn((1, args.T, 8, 64), device="cuda", dtype=torch.float32) * 0.02
    lower = torch.tril(torch.ones((64, 64), device="cuda", dtype=torch.float32), diagonal=-1)
    a = (a * lower.repeat(args.T // 64, 1).view(1, args.T, 1, 64)).contiguous()
    for _ in range(args.warmup + args.repeat):
        qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

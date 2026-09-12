#!/usr/bin/env python3
"""Preallocated BV64/BV32 block-dot arms for rocprofv3."""

from __future__ import annotations

import argparse
import json

import torch

import repro_qwen_gdn_direct_k64_block_dot_ab as bv64
import repro_qwen_gdn_direct_k64_block_dot_bv32_coop as bv32
import repro_qwen_gdn_direct_k64_update_current_abi as base


def make_launch(implementation: str, tokens: int, seed: int):
    k, v_new, g, initial = base._make_inputs(tokens, seed)
    h = torch.empty((1, tokens // 64, 8, 128, 128), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, 8, 128, 128), device=k.device, dtype=torch.float32)
    if implementation == "bv64":
        bv64.set_block_dot_lowering("specialized")
        kernel = bv64._qwen_gdn_direct_k64_block_dot_ab_kernel
        grid = (8 * 2, 1, 1)
    elif implementation == "bv32":
        bv32.set_block_dot_lowering()
        kernel = bv32._qwen_gdn_direct_k64_block_dot_bv32_coop_kernel
        grid = (8 * 4, 1, 1)
    else:
        raise ValueError(f"unsupported implementation: {implementation}")

    def launch() -> None:
        kernel[lambda: (grid, (128, 1, 1))](
            k, v_new, g, initial, h, final_state, tokens, tokens // 64, True, num_warps=2
        )

    return launch, h, final_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["bv64", "bv32"], required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if args.T <= 0 or args.T % 64:
        raise ValueError("T must be a positive multiple of 64")
    launch, h, final_state = make_launch(args.implementation, args.T, args.seed)
    launch()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(args.repeat):
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "T": args.T,
                "median_ms": sorted(samples)[len(samples) // 2],
                "h_finite": bool(torch.isfinite(h.float()).all().item()),
                "final_state_finite": bool(torch.isfinite(final_state).all().item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

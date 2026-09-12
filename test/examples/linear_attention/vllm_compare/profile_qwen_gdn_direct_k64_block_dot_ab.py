#!/usr/bin/env python3
"""One preallocated block-dot lowering arm for rocprofv3."""

from __future__ import annotations

import argparse
import json

import torch

import repro_qwen_gdn_direct_k64_block_dot_ab as block_dot
import repro_qwen_gdn_direct_k64_update_current_abi as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowering", choices=["generic", "specialized"], required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if args.T <= 0 or args.T % 64:
        raise ValueError("T must be a positive multiple of 64")
    block_dot.set_block_dot_lowering(args.lowering)
    k, v_new, g, initial = base._make_inputs(args.T, args.seed)
    h = torch.empty((1, args.T // 64, 8, 128, 128), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, 8, 128, 128), device=k.device, dtype=torch.float32)
    kernel = block_dot._qwen_gdn_direct_k64_block_dot_ab_kernel

    def launch() -> None:
        kernel[lambda: ((16, 1, 1), (128, 1, 1))](
            k, v_new, g, initial, h, final_state, args.T, args.T // 64, True, num_warps=2
        )

    launch()  # Compile and module-load outside the measured loop.
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
    print(json.dumps({
        "lowering": args.lowering,
        "T": args.T,
        "median_ms": sorted(samples)[len(samples) // 2],
        "h_finite": bool(torch.isfinite(h.float()).all().item()),
        "final_state_finite": bool(torch.isfinite(final_state).all().item()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preallocated rocprof target for the BV32 typed K/V operand staging A/B."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

import repro_qwen_gdn_direct_k64_block_dot_bv32_coop as bv32
import repro_qwen_gdn_direct_k64_update_current_abi as base


def make_launch(operand: str, tokens: int, seed: int):
    os.environ["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = operand
    bv32.set_block_dot_lowering()
    k, v_new, g, initial = base._make_inputs(tokens, seed)
    h = torch.empty((1, tokens // 64, 8, 128, 128), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, 8, 128, 128), device=k.device, dtype=torch.float32)

    def launch() -> None:
        bv32._qwen_gdn_direct_k64_block_dot_bv32_coop_kernel[lambda: ((32, 1, 1), (128, 1, 1))](
            k,
            v_new,
            g,
            initial,
            h,
            final_state,
            tokens,
            tokens // 64,
            True,
            num_warps=2,
        )

    return launch, h, final_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operand", choices=["scalar", "typed_vector"], required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    launch, h, final_state = make_launch(args.operand, args.T, args.seed)
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
                "T": args.T,
                "operand": args.operand,
                "median_ms": statistics.median(samples),
                "h_finite": bool(torch.isfinite(h.float()).all().item()),
                "final_state_finite": bool(torch.isfinite(final_state).all().item()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preallocated rocprof target for C0.5 typed/swizzled LDS lowering."""

from __future__ import annotations

import argparse
import json
import os

import torch

import repro_qwen_gdn_direct_k64_block_dot_bv32_coop as repro
import repro_qwen_gdn_direct_k64_update_current_abi as base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operand",
        choices=["persistent_typed_block", "persistent_typed_lds_layout"],
        required=True,
    )
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"
    os.environ["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = args.operand
    k, v_new, g, initial = base._make_inputs(args.T, args.seed)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return repro.qwen_gdn_direct_k64_block_dot_bv32_coop(k, v_new, g, initial)

    h, final = launch()
    for _ in range(args.warmup):
        h, final = launch()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        h, final = launch()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "operand": args.operand,
                "T": args.T,
                "finite": bool(torch.isfinite(h.float()).all() and torch.isfinite(final).all()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

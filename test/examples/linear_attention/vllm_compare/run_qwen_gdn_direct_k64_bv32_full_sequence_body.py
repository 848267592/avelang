#!/usr/bin/env python3
"""Launch one preallocated B0/B2 recurrence body for rocprof collection."""

from __future__ import annotations

import argparse

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead as b2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("b0", "b2"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    b2.p1._set_c0_lowering()
    k, w, u, g, initial_state = p2._make_long_case(args.T, args.seed)
    runner = b0.run_body if args.variant == "b0" else b2.run_body
    launch, _, _, _ = runner(k, w, u, g, initial_state)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        launch()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Production-body entry point for the R4-tail U/V-new LDS bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue_u_vnew_lds_bridge as bridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    k, w, u, g, initial_state = p2._make_long_case(args.T, args.seed + args.T)
    launch, h, v_new, final_state = bridge.run_body(
        k, w, u, g, initial_state, dump_hsaco_dir=args.dump_hsaco_dir)
    for _ in range(args.repeat - 1):
        launch()
    torch.cuda.synchronize()
    print(json.dumps({
        "plan": bridge.PLAN,
        "T": args.T,
        "repeat": args.repeat,
        "grid": [bridge.r4.GRID, 1, 1],
        "workgroup": [bridge.r4.WORKGROUP, 1, 1],
        "emit_audit": False,
        "finite": all(bool(value.isfinite().all()) for value in (h, v_new, final_state)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

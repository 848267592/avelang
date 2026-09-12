#!/usr/bin/env python3
"""Production-shaped compile entry point for the State-KV pred M/N-swap probe.

This is intentionally not a performance or correctness runner.  It invokes the
same public-shaped body construction as state_kv_dual_dot, with
``emit_audit=False`` and only the pred producer constexpr enabled, so the
resulting MLIR/MIR/ISA can answer the ownership question without mixing in an
I/O or update change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv as candidate


PLAN = "gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_pred_mn_swap"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dump-hsaco-dir", type=Path, required=True)
    args = parser.parse_args()

    k, w, u, g, initial_state = p2._make_long_case(args.T, args.seed + args.T)
    launch, h, v_new, final_state = candidate.run_pred_mn_swap_body(
        k, w, u, g, initial_state, dump_hsaco_dir=args.dump_hsaco_dir)
    for _ in range(args.repeat - 1):
        launch()
    torch.cuda.synchronize()
    print(json.dumps({
        "plan": PLAN,
        "T": args.T,
        "repeat": args.repeat,
        "grid": [candidate.GRID, 1, 1],
        "workgroup": [candidate.WORKGROUP, 1, 1],
        "emit_audit": False,
        "pred_mn_axis_swap": True,
        "finite": all(bool(x.isfinite().all()) for x in (h, v_new, final_state)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

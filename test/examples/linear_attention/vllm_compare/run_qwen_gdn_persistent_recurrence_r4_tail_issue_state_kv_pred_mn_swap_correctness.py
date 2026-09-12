#!/usr/bin/env python3
"""P2 correctness runner for State-KV pred M/N swap, with optional V4 I/O."""

from __future__ import annotations

import argparse
import json

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0
import repro_qwen_gdn_persistent_recurrence_r0 as r0
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue_state_kv as candidate


def run_one(t: int, seed: int, *, v4_io: bool) -> dict[str, object]:
    candidate._set_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    outputs = r0._allocate(t, initial_state, u)
    launch = candidate._make_launch(
        k, w, u, g, initial_state, outputs, emit_audit=True,
        pred_mn_axis_swap=True, pred_mn_v4_io=v4_io)
    launch()
    torch.cuda.synchronize()
    actual = dict(zip(
        ("h", "pred_f32", "pred_bf16", "v_new", "v_decay",
         "state_after", "final_state"), outputs))
    controls = {
        "p2_host_microscope": b0._run_p2_host_microscope(k, w, u, g, initial_state),
        "device_contract": b0._reference(k, w, u, g, initial_state),
    }
    torch.cuda.synchronize()
    comparisons = {name: b0._compare(actual, reference, label=name)
                   for name, reference in controls.items()}
    return {
        "T": t,
        "pred_mn_v4_io": v4_io,
        "comparisons": comparisons,
        "pred_f32_pass": all(row["pred_f32_max_abs"] <= 5e-5
                             for row in comparisons.values()),
        "full_pass": all(row["pass"] for row in comparisons.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--v4-io", action="store_true")
    args = parser.parse_args()
    rows = [run_one(t, args.seed + t, v4_io=args.v4_io) for t in args.T]
    print(json.dumps(rows, indent=2))
    if not all(row["full_pass"] for row in rows):
        raise SystemExit("P2 correctness gate failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict R4 tail-issue causality control.

This dispatches the exact R4 source kernel.  The only selected compiler
change is ``gfx942_bt64_bv32_joint_v4_tail_issue``: it moves the four next
W/K global-to-LDS paths to the completed-current-core tail, without invoking
the modulo scheduler or a packet ring.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0
import repro_qwen_gdn_persistent_recurrence_r0 as r0
import repro_qwen_gdn_persistent_recurrence_r1 as r1
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4


BT = r4.BT
LOWERING = "gfx942_bt64_bv32_joint_v4_tail_issue"
_HSACO_DUMP_DIR: Path | None = None


def _set_tail_issue_lowering() -> None:
    r4.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = LOWERING
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"


def _make_launch(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, outputs: tuple[torch.Tensor, ...], *, emit_audit: bool,
) -> Callable[[], None]:
    _set_tail_issue_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        r4._qwen_gdn_persistent_recurrence_r4_joint_v4_kernel[
            lambda: ((r4.GRID, 1, 1), (r4.WORKGROUP, 1, 1))
        ](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_tokens // BT, emit_audit,
            True, False, num_warps=2,
        )

    return launch


def _run_kernel(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=True)
    previous_dump_dir = r4._HSACO_DUMP_DIR
    r4._HSACO_DUMP_DIR = _HSACO_DUMP_DIR
    try:
        r4._maybe_dump_hsaco(launch)
    finally:
        r4._HSACO_DUMP_DIR = previous_dump_dir
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    return {
        "h": h, "pred_f32": pred_f32, "pred_bf16": pred_bf16,
        "v_new": v_new, "v_decay": v_decay, "state_after": state_after,
        "final_state": final_state, "_launch": launch,
    }


def run_body(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=False)
    launch()
    return launch, outputs[0], outputs[3], outputs[6]


def run_correctness_length(t: int, *, seed: int) -> dict[str, Any]:
    _set_tail_issue_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    # References are compiled only after the candidate has completed.  This
    # avoids the experimental mode matching their separate recurrence region.
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "tail_issue_vs_p2_host_microscope": b0._run_p2_host_microscope(
            k, w, u, g, initial_state),
        "tail_issue_vs_device_contract": b0._reference(
            k, w, u, g, initial_state),
    }
    torch.cuda.synchronize()
    comparisons = {
        label: b0._compare(actual, control, label=label)
        for label, control in controls.items()
    }
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in (
        "h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state",
    ))
    return {
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "comparisons": comparisons,
        "per_chunk": b0._per_chunk_rows(
            actual, controls["tail_issue_vs_device_contract"]),
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rocprof_outputs/qwen_persistent_recurrence_r4_tail_issue",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tail_issue_correctness.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({k: v for k, v in result.items() if k != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("R4 tail-issue correctness gate failed")


if __name__ == "__main__":
    main()

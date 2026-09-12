#!/usr/bin/env python3
"""Native single-core microtile plans for the R4 tail-issue recurrence."""

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
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4


BT = r4.BT
MODE_PREFIX = "gfx942_bt64_bv32_microtile_experimental_"
CORE_LASTUSE_MODE_PREFIX = "gfx942_bt64_bv32_core_lastuse_experimental_"
_HSACO_DUMP_DIR: Path | None = None


def plan_name(w_packets: int, k_packets: int, placement: str, distance: int) -> str:
    if w_packets not in (1, 2, 4) or k_packets not in (1, 2, 4):
        raise ValueError("W/K packets per group must be 1, 2, or 4")
    if placement not in ("tail", "lastuse") or distance not in (0, 1):
        raise ValueError("placement must be tail/lastuse and distance must be 0/1")
    return f"{MODE_PREFIX}w{w_packets}_k{k_packets}_{placement}_d{distance}"


def core_lastuse_plan_name(w_packets: int, k_packets: int, placement: str) -> str:
    """Name a real core-MFMA last-use issue plan (commit remains at tail)."""
    if w_packets not in (1, 2, 4) or k_packets not in (1, 2, 4):
        raise ValueError("W/K packets per group must be 1, 2, or 4")
    if placement not in ("immediate", "delay1"):
        raise ValueError("placement must be immediate or delay1")
    return f"{CORE_LASTUSE_MODE_PREFIX}w{w_packets}_k{k_packets}_{placement}"


def _set_microtile_lowering(plan: str) -> None:
    if not (plan.startswith(MODE_PREFIX) or
            plan.startswith(CORE_LASTUSE_MODE_PREFIX)):
        raise ValueError(f"not a microtile/core-lastuse plan: {plan}")
    r4.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = plan
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"


def _make_launch(
    plan: str, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, outputs: tuple[torch.Tensor, ...], *, emit_audit: bool,
) -> Callable[[], None]:
    _set_microtile_lowering(plan)
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} is not divisible by BT={BT}")
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
    plan: str, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(plan, k, w, u, g, initial_state, outputs, emit_audit=True)
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
    plan: str, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(plan, k, w, u, g, initial_state, outputs, emit_audit=False)
    launch()
    return launch, outputs[0], outputs[3], outputs[6]


def run_correctness_length(plan: str, t: int, *, seed: int) -> dict[str, Any]:
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(plan, k, w, u, g, initial_state)
    torch.cuda.synchronize()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "microtile_vs_p2_host_microscope": b0._run_p2_host_microscope(
            k, w, u, g, initial_state),
        "microtile_vs_device_contract": b0._reference(k, w, u, g, initial_state),
    }
    comparisons = {
        label: b0._compare(actual, control, label=label)
        for label, control in controls.items()
    }
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in (
        "h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state",
    ))
    return {
        "plan": plan,
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "comparisons": comparisons,
        "per_chunk": b0._per_chunk_rows(
            actual, controls["microtile_vs_device_contract"]),
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--out-dir", type=Path, default=Path("microtile_correctness"))
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(args.plan, t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "microtile_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2) if args.json else "\n".join(
        json.dumps({key: value for key, value in row.items() if key != "per_chunk"}, sort_keys=True)
        for row in results
    ))
    if not all(bool(row["pass"]) for row in results):
        raise SystemExit("microtile recurrence correctness gate failed")


if __name__ == "__main__":
    main()

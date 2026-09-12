#!/usr/bin/env python3
"""R2 joint_v2 full recurrence using the shared R1 semantic kernel.

R2 is a compile-time schedule choice, not a second pred/update implementation:
the kernel body, BF16 boundary and Direct-K64 block-dot remain shared with R1.
Only the planner/lowering mode and the next-K issue position differ.
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
import repro_qwen_gdn_persistent_recurrence_r2_joint_v2 as joint_v2


BT = r1.BT
LOWERING = "gfx942_bt64_bv32_joint_v2"


def _set_r2_lowering() -> None:
    b0.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = LOWERING
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"


def _make_launch(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    *,
    emit_audit: bool,
) -> Callable[[], None]:
    _set_r2_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        joint_v2._qwen_gdn_persistent_recurrence_r2_joint_v2_kernel[
            lambda: ((r1.GRID, 1, 1), (r1.WORKGROUP, 1, 1))
        ](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new,
            v_decay, state_after, final_state, num_tokens, num_chunks,
            emit_audit, True, num_warps=2,
        )

    return launch


def _run_kernel(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=True)
    launch()
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    return {
        "h": h,
        "pred_f32": pred_f32,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
        "v_decay": v_decay,
        "state_after": state_after,
        "final_state": final_state,
        "_launch": launch,
    }


def run_body(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=False)
    launch()
    return launch, outputs[0], outputs[3], outputs[6]


def run_correctness_length(t: int, *, seed: int) -> dict[str, Any]:
    _set_r2_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    r1_actual = r1._run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    microscope = b0._run_p2_host_microscope(k, w, u, g, initial_state)
    reference = b0._reference(k, w, u, g, initial_state)
    comparisons = {
        "r2_vs_r1": b0._compare(actual, r1_actual, label="r2_vs_r1"),
        "r2_vs_p2": b0._compare(actual, microscope, label="r2_vs_p2_host_microscope"),
        "r2_vs_device_contract": b0._compare(actual, reference, label="r2_vs_device_contract"),
    }
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in (
        "h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state",
    ))
    return {
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "comparisons": comparisons,
        "per_chunk": b0._per_chunk_rows(actual, reference),
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_persistent_recurrence_r2_joint_v2",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [run_correctness_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "r2_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({key: value for key, value in result.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("R2 correctness gate failed; do not run resource or performance collection.")


if __name__ == "__main__":
    main()

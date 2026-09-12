#!/usr/bin/env python3
"""Distance-one modulo software-pipeline candidate over the validated R4 body.

The source kernel is deliberately the R4 full nonzero-W recurrence: mathematics,
BV32 ownership, typed packet producer and LDS-mediated K retile are unchanged.
``LOWERING`` selects the compiler-owned prologue/steady/epilogue scheduler.
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
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as joint_v4


BT = r1.BT
LOWERING = "gfx942_bt64_bv32_software_pipeline"
_HSACO_DUMP_DIR: Path | None = None


def _set_software_pipeline_lowering() -> None:
    b0.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = LOWERING
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"


def _make_launch(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, outputs: tuple[torch.Tensor, ...], *, emit_audit: bool,
) -> Callable[[], None]:
    _set_software_pipeline_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        joint_v4._qwen_gdn_persistent_recurrence_r4_joint_v4_kernel[
            lambda: ((r1.GRID, 1, 1), (r1.WORKGROUP, 1, 1))
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
    # Reuse the R4 diagnostic wrapper only to capture the compiled code object.
    # It does not supply a schedule: the selected mode is consumed by the
    # compiler-owned modulo pass above.
    previous_dump_dir = joint_v4._HSACO_DUMP_DIR
    joint_v4._HSACO_DUMP_DIR = _HSACO_DUMP_DIR
    try:
        joint_v4._maybe_dump_hsaco(launch)
    finally:
        joint_v4._HSACO_DUMP_DIR = previous_dump_dir
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
    _set_software_pipeline_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    # The AveLang in-process JIT cache keys the source specialization rather
    # than the lowering-mode environment.  Cross-mode kernel comparisons are
    # therefore run in fresh processes by the benchmark/audit harness.  These
    # two independent references are mode-free and give this candidate a real
    # correctness gate in a single process.
    # Controls intentionally compile different source wrappers.  Keep their
    # mode at the validated R4 contract rather than letting the candidate's
    # scheduler attempt to match an unrelated reference region in the same
    # process.  This does not affect `actual`, which has already dispatched
    # and synchronized above.
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "software_pipeline_vs_p2_host_microscope": b0._run_p2_host_microscope(
            k, w, u, g, initial_state),
        "software_pipeline_vs_device_contract": b0._reference(
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
            actual, controls["software_pipeline_vs_device_contract"]),
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
        / "rocprof_outputs/qwen_persistent_recurrence_software_pipeline",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "software_pipeline_correctness.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({k: v for k, v in result.items() if k != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("software-pipeline correctness gate failed")


if __name__ == "__main__":
    main()

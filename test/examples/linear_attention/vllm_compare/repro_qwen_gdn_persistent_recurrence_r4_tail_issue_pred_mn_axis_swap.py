#!/usr/bin/env python3
"""Mechanism-only R4-tail pred-producer M/N-axis exchange.

This is intentionally not a correctness or performance candidate.  It keeps
the R4-tail recurrence, U/V-new boundary, update block-dot, state ownership,
and tail issue unchanged, and changes exactly one pred-MFMA property:

    historical physical C: M=V, N=T  (state, W)
    candidate  physical C: M=T, N=V  (W, state)

The candidate does not insert an LDS bridge, packet gather, or a restoration
to a VxT view after pred.  It exists solely to capture MLIR/LLVM/MIR/ISA and
to validate the resulting accumulator ownership before any U/V-new work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r0 as r0
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as tail_issue


PLAN = "gfx942_bt64_bv32_joint_v4_tail_issue_pred_mn_axis_swap"
_HSACO_DUMP_DIR: Path | None = None


def _make_launch(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, outputs: tuple[torch.Tensor, ...],
) -> Callable[[], None]:
    tail_issue._set_tail_issue_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % r4.BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={r4.BT}")
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        r4._qwen_gdn_persistent_recurrence_r4_joint_v4_kernel[
            lambda: ((r4.GRID, 1, 1), (r4.WORKGROUP, 1, 1))
        ](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_tokens // r4.BT,
            False, True, False, True, num_warps=2,
        )

    return launch


def run_body(t: int, *, seed: int, dump_hsaco_dir: Path | None) -> dict[str, object]:
    """Compile and launch the production-shaped (emit_audit=False) mechanism."""
    global _HSACO_DUMP_DIR
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    outputs = r0._allocate(t, initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs)
    previous_dump_dir = r4._HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = dump_hsaco_dir
    r4._HSACO_DUMP_DIR = dump_hsaco_dir
    try:
        r4._maybe_dump_hsaco(launch)
    finally:
        r4._HSACO_DUMP_DIR = previous_dump_dir
        _HSACO_DUMP_DIR = None
    torch.cuda.synchronize()
    return {
        "plan": PLAN,
        "T": t,
        "emit_audit": False,
        "logical_grid": r4.GRID,
        "workgroup": r4.WORKGROUP,
        "pred_mn_axis_swap": True,
        "note": "mechanism-only capture; correctness is intentionally not run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--dump-hsaco-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    result = run_body(args.T, seed=args.seed, dump_hsaco_dir=args.dump_hsaco_dir)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

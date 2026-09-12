#!/usr/bin/env python3
"""R0 persistent-recurrence semantic container with legacy-B0 lowering.

This file intentionally owns no pred/update schedule.  It launches the exact
B0 kernel body with a compile-time semantic delimiter enabled; the compiler
forms that body into ``ave.gpu.amdgpu_qwen_persistent_recurrence`` and the
``legacy_b0`` lowering inlines the region before the existing block-dot pass.
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


BT = b0.BT
BV = b0.BV
H_V = b0.H_V
KDIM = b0.KDIM
GRID = b0.GRID
WORKGROUP = b0.WORKGROUP
_HSACO_DUMP_DIR: Path | None = None


def _set_r0_lowering() -> None:
    """Use the existing C0 block-dot path and only R0's structural lowering."""
    b0.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "legacy_b0"


def _allocate(num_tokens: int, initial_state: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    chunks = num_tokens // BT
    h = torch.empty((1, chunks, H_V, KDIM, KDIM), device=u.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=u.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(u)
    v_new = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, chunks, H_V, KDIM, KDIM), device=u.device, dtype=torch.float32)
    final_state = torch.empty_like(initial_state)
    return h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state


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
    _set_r0_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        b0._qwen_gdn_direct_k64_bv32_full_sequence_b0_kernel[
            lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))
        ](
            k,
            w,
            u,
            g,
            initial_state,
            h,
            pred_f32,
            pred_bf16,
            v_new,
            v_decay,
            state_after,
            final_state,
            num_tokens,
            num_chunks,
            emit_audit,
            True,
            num_warps=2,
        )

    return launch


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    # B0's JIT function is deliberately shared.  Its existing dump hook is
    # therefore the faithful code-object capture point for the R0 constexpr
    # specialization as well.
    previous = b0._HSACO_DUMP_DIR
    b0._HSACO_DUMP_DIR = _HSACO_DUMP_DIR
    try:
        b0._maybe_dump_hsaco(launch)
    finally:
        b0._HSACO_DUMP_DIR = previous


def _run_kernel(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = _allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=True)
    _maybe_dump_hsaco(launch)
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
    """Return a preallocated, audit-elided R0 body launch for diagnostics."""
    outputs = _allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=False)
    launch()
    return launch, outputs[0], outputs[3], outputs[6]


def _comparison(actual: dict[str, Any], expected: dict[str, Any], label: str) -> dict[str, Any]:
    return b0._compare(actual, expected, label=label)


def run_correctness_length(t: int, *, seed: int) -> dict[str, Any]:
    _set_r0_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    r0 = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    b0_actual = b0._run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    microscope = b0._run_p2_host_microscope(k, w, u, g, initial_state)
    reference = b0._reference(k, w, u, g, initial_state)
    comparisons = {
        "r0_vs_b0": _comparison(r0, b0_actual, "r0_vs_b0"),
        "r0_vs_p2": _comparison(r0, microscope, "r0_vs_p2_host_microscope"),
        "r0_vs_device_contract": _comparison(r0, reference, "r0_vs_device_contract"),
    }
    per_chunk = b0._per_chunk_rows(r0, reference)
    finite = all(bool(torch.isfinite(r0[name]).all()) for name in (
        "h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state",
    ))
    return {
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "comparisons": comparisons,
        "per_chunk": per_chunk,
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_persistent_recurrence_r0",
    )
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "r0_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({key: value for key, value in result.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("R0 correctness gate failed; do not run resource or performance collection.")


if __name__ == "__main__":
    main()

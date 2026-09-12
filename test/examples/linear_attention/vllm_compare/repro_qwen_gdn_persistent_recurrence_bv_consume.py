#!/usr/bin/env python3
"""Full R4-tail persistent recurrence with logical BV-consume tiles.

This is intentionally a thin launch/correctness harness around the R4 source
kernel's BV-consume form.  It does not contain an alternate recurrence or an
isolated update experiment: each launch keeps one first-class persistent loop,
the R4 typed W/K producer, one LDS W/K bank, and tail issue/commit.
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
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4


BT = r4.BT
BV_CHOICES = (16, 32, 64)
_HSACO_DUMP_DIR: Path | None = None


def lowering_for(bv_consume: int) -> str:
    if bv_consume not in BV_CHOICES:
        raise ValueError(f"BV-consume must be one of {BV_CHOICES}, got {bv_consume}")
    return f"gfx942_bt64_bv{bv_consume}_joint_v4_tail_issue"


def _set_lowering(bv_consume: int) -> None:
    r4.p1._set_c0_lowering()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = lowering_for(bv_consume)
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = "distributed"


def _maybe_dump_hsaco(launch: Callable[[], None], bv_consume: int) -> None:
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    target = _HSACO_DUMP_DIR / f"qwen_persistent_recurrence_bv{bv_consume}.hsaco"
    if target.exists():
        launch()
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        # BV16/BV32 share the physically-V32 source function, while BV64 uses
        # the four-wave function.  Both names carry this stable prefix.
        if not dumped and "persistent_recurrence_bv" in src.fn.fn.__name__:
            target.write_bytes(binary)
            dumped = True
            print(f"dumped_hsaco: {target}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError("BV-consume HSACO dump requested but no kernel was compiled")


def _make_launch(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
    *,
    bv_consume: int,
    emit_audit: bool,
) -> Callable[[], None]:
    _set_lowering(bv_consume)
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    if 128 % bv_consume:
        raise ValueError(f"BV-consume must divide 128, got {bv_consume}")
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    if bv_consume == 64:
        kernel = r4._qwen_gdn_persistent_recurrence_bv_consume_kernel
    else:
        kernel = r4._qwen_gdn_persistent_recurrence_bv16_bv32_kernel
    workgroup = 256 if bv_consume == 64 else r4.WORKGROUP

    def launch() -> None:
        kernel[
            lambda: ((r4.H_V * (r4.KDIM // bv_consume), 1, 1), (workgroup, 1, 1))
        ](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_tokens // BT, emit_audit,
            True, bv_consume, num_warps=4 if bv_consume == 64 else 2,
        )

    return launch


def _run_kernel(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    bv_consume: int,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs,
                          bv_consume=bv_consume, emit_audit=True)
    _maybe_dump_hsaco(launch, bv_consume)
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    return {
        "h": h, "pred_f32": pred_f32, "pred_bf16": pred_bf16,
        "v_new": v_new, "v_decay": v_decay, "state_after": state_after,
        "final_state": final_state, "_launch": launch,
    }


def run_body(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    bv_consume: int,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs,
                          bv_consume=bv_consume, emit_audit=False)
    launch()
    return launch, outputs[0], outputs[3], outputs[6]


def run_correctness_length(t: int, *, seed: int, bv_consume: int) -> dict[str, Any]:
    _set_lowering(bv_consume)
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state, bv_consume=bv_consume)
    torch.cuda.synchronize()
    # The references are separate recurrence compilation units; compile them
    # under baseline R4 only after the candidate has completed.
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "bv_consume_vs_p2_host_microscope": b0._run_p2_host_microscope(k, w, u, g, initial_state),
        "bv_consume_vs_device_contract": b0._reference(k, w, u, g, initial_state),
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
        "bv_consume": bv_consume,
        "lowering": lowering_for(bv_consume),
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "comparisons": comparisons,
        "per_chunk": b0._per_chunk_rows(actual, controls["bv_consume_vs_device_contract"]),
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bv", type=int, choices=BV_CHOICES, required=True)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_persistent_recurrence_bv_consume",
    )
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [run_correctness_length(t, seed=args.seed + t, bv_consume=args.bv) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"bv{args.bv}_correctness.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(json.dumps({key: value for key, value in row.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(row["pass"]) for row in rows):
        raise SystemExit("BV-consume correctness gate failed")


if __name__ == "__main__":
    main()

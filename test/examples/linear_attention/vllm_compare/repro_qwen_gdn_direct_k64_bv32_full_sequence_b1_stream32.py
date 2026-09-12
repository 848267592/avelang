#!/usr/bin/env python3
"""B1 compiler-owned stream32 Direct-K64 BV32 full recurrence.

This is an experimental-only replacement of B0's *inside-one-chunk* schedule.
The device-side chunk loop, BF16 ABI, BV32 ownership, MFMA32 geometry, output
contract and P0/P1/P2 correctness reference are intentionally inherited from
B0.  The only source-level scheduling change is the opt-in compiler operation
``qwen_gdn_recurrence_step_bf16_f32``.  Its late lowering owns:

  pred T32 -> BF16 V-new/V-decay T32 -> Direct-K64 update T32.

No production selector or external HSACO is changed here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

import torch

import avelang
import avelang.language as al

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1 as p1
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0


BT = b0.BT
BV = b0.BV
WORKGROUP = b0.WORKGROUP
GRID = b0.GRID
H_V = b0.H_V
KDIM = b0.KDIM
STATE_ATOL = b0.STATE_ATOL
_HSACO_DUMP_DIR: Path | None = None


def _set_b1_lowering() -> None:
    """Keep B1 opt-in and retain the frozen C0 specialized configuration."""
    p1._set_c0_lowering()
    os.environ["AVELANG_QWEN_GDN_RECURRENCE_STEP_LOWERING"] = "stream_t32"


@avelang.jit
def _qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    pred_f32_ptr: al.Pointer(al.f32),
    pred_bf16_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
    v_decay_ptr: al.Pointer(al.bf16),
    state_after_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    emit_audit: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, KDIM), (num_tokens * 4 * KDIM, 4 * KDIM, KDIM, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, KDIM, KDIM),
            (num_chunks * H_V * KDIM * KDIM, H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1),
        ),
    )
    pred_f32 = al.make_tensor(
        pred_f32_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    pred_bf16 = al.make_tensor(
        pred_bf16_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    v_new = al.make_tensor(
        v_new_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    v_decay = al.make_tensor(
        v_decay_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    state_after = al.make_tensor(
        state_after_ptr,
        al.f32,
        al.make_layout(
            (1, num_chunks, H_V, KDIM, KDIM),
            (num_chunks * H_V * KDIM * KDIM, H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_row = lane & 31
    mfma_lane_group = lane >> 5
    program_id = al.block_id(0)
    value_head_idx = program_id >> 2
    value_base = (program_id & 3) * BV
    key_head_idx = value_head_idx >> 1

    # This persistent state mapping is verbatim B0. The new compiler op only
    # changes one BT64 recurrence step after these vectors are established.
    h_lo = al.full((16,), 0.0, al.f32)
    h_hi = al.full((16,), 0.0, al.f32)
    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_row
        if wave_id == 0:
            h_lo[acc_i] = initial_state[0, value_head_idx, global_v, local_k]
            h_hi[acc_i] = initial_state[0, value_head_idx, global_v, 32 + local_k]
        else:
            h_lo[acc_i] = initial_state[0, value_head_idx, global_v, 64 + local_k]
            h_hi[acc_i] = initial_state[0, value_head_idx, global_v, 96 + local_k]

    # Physical B1 LDS map: 8 KiB state, 8 KiB W/K phase reuse, 8 KiB pred
    # partial planes, and 2 KiB current V-decay. No BT64 V-decay or separate
    # full W/K allocation exists in the source graph.
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    phase_bf16 = al.make_shared((64, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    vdecay_stage = al.make_shared((1, BV, 32), al.bf16)

    for chunk_idx in al.range(num_chunks):
        next_state = al.amdgpu.qwen_gdn_recurrence_step_bf16_f32(
            state_bf16,
            phase_bf16,
            pred_partial,
            vdecay_stage,
            k,
            w,
            u,
            g,
            h,
            pred_f32,
            pred_bf16,
            v_new,
            v_decay,
            state_after,
            tid,
            chunk_idx,
            value_head_idx,
            key_head_idx,
            value_base,
            h_lo,
            h_hi,
            emit_audit,
        )
        for acc_i in al.range(16):
            h_lo[acc_i] = next_state[acc_i]
            h_hi[acc_i] = next_state[16 + acc_i]
        al.syncthreads()

    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_row
        if wave_id == 0:
            final_state[0, value_head_idx, global_v, local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 32 + local_k] = h_hi[acc_i]
        else:
            final_state[0, value_head_idx, global_v, 64 + local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 96 + local_k] = h_hi[acc_i]


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    target = _HSACO_DUMP_DIR / f"{_qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32_kernel.fn.__name__}.hsaco"
    if target.exists():
        launch()
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and "full_sequence_b1_stream32_kernel" in src.fn.fn.__name__:
            target.write_bytes(binary)
            dumped = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError("B1 HSACO dump requested, but no matching kernel compiled")


def _allocate_outputs(num_tokens: int, initial_state: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, ...]:
    num_chunks = num_tokens // BT
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=u.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=u.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(u)
    v_new = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=u.device, dtype=torch.float32)
    final_state = torch.empty_like(initial_state)
    return h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state


def _run_kernel(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    _set_b1_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = _allocate_outputs(
        num_tokens, initial_state, u
    )

    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32_kernel[
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
            True,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch)
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
    """Return B1 body launch with only non-ABI audit writes compile-time-elided."""
    _set_b1_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = _allocate_outputs(
        num_tokens, initial_state, u
    )

    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_full_sequence_b1_stream32_kernel[
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
            False,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch)
    return launch, h, v_new, final_state


def _stats(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float | bool]:
    maximum, mean = b0._max_mean(lhs, rhs)
    return {"max_abs": maximum, "mean_abs": mean, "byte_equal": bool(torch.equal(lhs, rhs))}


def run_correctness_length(t: int, *, seed: int, include_p2: bool = True) -> dict[str, Any]:
    _set_b1_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    reference = b0._reference(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    b0_actual = b0._run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    p2_actual: dict[str, torch.Tensor] | None = None
    if include_p2:
        p2_actual = b0._run_p2_host_microscope(k, w, u, g, initial_state)
        torch.cuda.synchronize()
    actual_tensors = {name: value for name, value in actual.items() if name != "_launch"}
    b0_tensors = {name: value for name, value in b0_actual.items() if name != "_launch"}
    device = b0._compare(actual_tensors, reference, label="device_contract_reference")
    b0_compare = b0._compare(actual_tensors, b0_tensors, label="b1_vs_b0")
    p2_compare = (
        b0._compare(actual_tensors, p2_actual, label="p2_host_microscope") if p2_actual is not None else None
    )
    per_chunk = b0._per_chunk_rows(actual_tensors, reference)
    finite = all(
        bool(torch.isfinite(actual_tensors[name]).all())
        for name in ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state")
    )
    return {
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "device_contract": device,
        "b1_vs_b0": b0_compare,
        "p2_host_microscope": p2_compare,
        "per_chunk": per_chunk,
        "pass": finite
        and bool(device["pass"])
        and bool(b0_compare["pass"])
        and (p2_compare is None or bool(p2_compare["pass"])),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b1_stream32",
    )
    parser.add_argument("--skip-p2", action="store_true", help="Only for compile debugging; not a B1 correctness gate.")
    parser.add_argument(
        "--b1-only-compile",
        action="store_true",
        help="Compile and execute only B1 once to preserve B1-only IR/LTO snapshots.",
    )
    parser.add_argument(
        "--b1-only-body-compile",
        action="store_true",
        help="Compile and execute only the audit-elided B1 body once for resource collection.",
    )
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    if args.b1_only_compile and args.b1_only_body_compile:
        parser.error("choose only one B1-only compile mode")
    if args.b1_only_body_compile:
        if len(args.T) != 1:
            parser.error("--b1-only-body-compile requires exactly one --T value")
        k, w, u, g, initial_state = p2._make_long_case(args.T[0], args.seed + args.T[0])
        run_body(k, w, u, g, initial_state)
        torch.cuda.synchronize()
        result = {"T": args.T[0], "b1_only_body_compile": True, "status": "compiled_and_executed"}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, sort_keys=True))
        return
    if args.b1_only_compile:
        if len(args.T) != 1:
            parser.error("--b1-only-compile requires exactly one --T value")
        k, w, u, g, initial_state = p2._make_long_case(args.T[0], args.seed + args.T[0])
        _run_kernel(k, w, u, g, initial_state)
        torch.cuda.synchronize()
        result = {"T": args.T[0], "b1_only_compile": True, "status": "compiled_and_executed"}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, sort_keys=True))
        return
    results = [
        run_correctness_length(t, seed=args.seed + t, include_p2=not args.skip_p2)
        for t in args.T
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b1_stream32_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({key: value for key, value in result.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("B1 stream32 correctness gate failed; do not run resource or performance collection.")


if __name__ == "__main__":
    main()

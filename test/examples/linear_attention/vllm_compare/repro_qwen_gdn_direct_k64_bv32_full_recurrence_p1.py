#!/usr/bin/env python3
"""P1: one-chunk nonzero-W pred plus C0 direct-K64 composition audit.

P1 is enabled only after P0's BF16 MFMA32 pred mapping succeeds.  It is a
single BT64 recurrence chunk with the existing C0 persistent typed block-dot
consumer.  No timing code exists in this file: each output is an intermediate
correctness checkpoint for the pred -> V-new -> V-decay -> update handoff.
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

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0 as p0


BT = p0.BT
BV = p0.BV
KDIM = p0.KDIM
H_K = p0.H_K
H_V = p0.H_V
WORKGROUP = p0.WORKGROUP
GRID = p0.GRID
STATE_ATOL = 0.02
_HSACO_DUMP_DIR: Path | None = None


def _set_c0_lowering() -> None:
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"
    os.environ["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = "persistent_typed_block"


@avelang.jit
def _qwen_gdn_direct_k64_bv32_full_p1_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    raw_acc_ptr: al.Pointer(al.f32),
    pred_partial_ptr: al.Pointer(al.f32),
    pred_f32_ptr: al.Pointer(al.f32),
    pred_bf16_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
    v_decay_ptr: al.Pointer(al.bf16),
    delta_ptr: al.Pointer(al.f32),
    state_after_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
):
    """One BT64 recurrence chunk, with fixed P0 pred mapping and C0 update."""

    k = al.make_tensor(
        k_ptr, al.bf16,
        al.make_layout((1, BT, H_K, KDIM), (BT * H_K * KDIM, H_K * KDIM, KDIM, 1)),
    )
    w = al.make_tensor(
        w_ptr, al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    u = al.make_tensor(
        u_ptr, al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, BT, H_V), (BT * H_V, H_V, 1)))
    initial_state = al.make_tensor(
        initial_state_ptr, al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    h = al.make_tensor(
        h_ptr, al.bf16,
        al.make_layout((1, 1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    raw_acc = al.make_tensor(
        raw_acc_ptr, al.f32,
        al.make_layout((H_V, 4, 2, 2, 64, 16), (4 * 2 * 2 * 64 * 16, 2 * 2 * 64 * 16, 2 * 64 * 16, 64 * 16, 16, 1)),
    )
    pred_partial_out = al.make_tensor(
        pred_partial_ptr, al.f32,
        al.make_layout((H_V, 4, 2, 2, 32, 32), (4 * 2 * 2 * 32 * 32, 2 * 2 * 32 * 32, 2 * 32 * 32, 32 * 32, 32, 1)),
    )
    pred_f32 = al.make_tensor(
        pred_f32_ptr, al.f32,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    pred_bf16 = al.make_tensor(
        pred_bf16_ptr, al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    v_new = al.make_tensor(
        v_new_ptr, al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    v_decay = al.make_tensor(
        v_decay_ptr, al.bf16,
        al.make_layout((1, BT, H_V, KDIM), (BT * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    delta = al.make_tensor(
        delta_ptr, al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    state_after = al.make_tensor(
        state_after_ptr, al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    final_state = al.make_tensor(
        final_state_ptr, al.f32,
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

    # C0 owns V32 rows cooperatively: wave 0 carries K[0:64], wave 1 K[64:128].
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

    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    vdecay_stage = al.make_shared((1, BV, BT), al.bf16)
    k_stage = al.make_shared((BT, 64), al.bf16)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))
    g_last = g[0, BT - 1, value_head_idx]
    g_last_exp = al.exp(g_last)

    for rep_state in al.range(32):
        linear = tid + rep_state * WORKGROUP
        k_half = linear // (BV * 64)
        rem = linear - k_half * (BV * 64)
        local_v = rem // 64
        local_k = rem - local_v * 64
        state_bf16[k_half, local_v, local_k] = al.convert(
            initial_state[0, value_head_idx, value_base + local_v, k_half * 64 + local_k], al.bf16
        )
    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_row
        if wave_id == 0:
            h[0, 0, value_head_idx, global_v, local_k] = al.convert(h_lo[acc_i], al.bf16)
            h[0, 0, value_head_idx, global_v, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
        else:
            h[0, 0, value_head_idx, global_v, 64 + local_k] = al.convert(h_lo[acc_i], al.bf16)
            h[0, 0, value_head_idx, global_v, 96 + local_k] = al.convert(h_hi[acc_i], al.bf16)
    al.syncthreads()

    # This pred body is intentionally identical in ownership and operand
    # indexing to the P0-passing kernel.  It is not an alternative schedule.
    for token_tile in al.range(2):
        token_base = token_tile * 32
        for rep_w in al.range(32):
            linear = tid + rep_w * WORKGROUP
            k_half = linear // (32 * 64)
            rem = linear - k_half * (32 * 64)
            token_off = rem // 64
            local_k = rem - token_off * 64
            w_bf16[k_half, token_off, local_k] = w[
                0, token_base + token_off, value_head_idx, k_half * 64 + local_k
            ]
        al.syncthreads()

        pred_acc = al.full((16,), 0.0, al.f32)
        for kpack in al.range(4):
            k_vec = kpack * 2 + mfma_lane_group
            w_words = w_vec[wave_id, lane_row, k_vec]
            state_words = state_vec[wave_id, lane_row, k_vec]
            w_frag = al.view(w_words, al.Tensor((2, 4, 1), al.bf16))
            state_frag = al.view(state_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[0], w_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[1], w_frag[1], pred_acc)

        for acc_i in al.range(16):
            raw_acc[value_head_idx, program_id & 3, token_tile, wave_id, lane, acc_i] = pred_acc[acc_i]
            out_row = lane_row
            out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
        al.syncthreads()

        for rep_dump in al.range(16):
            linear = tid + rep_dump * WORKGROUP
            k_half = linear // (32 * BV)
            rem = linear - k_half * (32 * BV)
            token_off = rem // BV
            local_v = rem - token_off * BV
            pred_partial_out[value_head_idx, program_id & 3, token_tile, k_half, token_off, local_v] = pred_partial[
                k_half, token_off, local_v
            ]

        for rep_out in al.range(8):
            linear = tid + rep_out * WORKGROUP
            token_off = linear // BV
            local_v = linear - token_off * BV
            token_idx = token_base + token_off
            global_v = value_base + local_v
            pred = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
            corrected = al.convert(u[0, token_idx, value_head_idx, global_v], al.f32) - pred
            # The current recurrence ABI materializes V-new as BF16.  Update
            # must consume that exact boundary value, rather than an unrounded
            # FP32 corrected temporary that only happens to be nearby.
            corrected_bf16 = al.convert(corrected, al.bf16)
            corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
            decay = al.exp(g_last - g[0, token_idx, value_head_idx])
            pred_f32[0, token_idx, value_head_idx, global_v] = pred
            pred_bf16[0, token_idx, value_head_idx, global_v] = al.convert(pred, al.bf16)
            v_new[0, token_idx, value_head_idx, global_v] = corrected_bf16
            vdecay_stage[0, local_v, token_idx] = al.convert(corrected_from_bf16 * decay, al.bf16)
            v_decay[0, token_idx, value_head_idx, global_v] = al.convert(corrected_from_bf16 * decay, al.bf16)
        al.syncthreads()

    # C0 persistent typed block-dot consumes the CTA-local V-decay directly.
    # It still receives the ABI V-new pointer but staged lowering must not
    # reload it.  Each owning wave commits exactly its own K64 pair.
    for k_half in al.range(2):
        update_pair = al.amdgpu.block_dot_bf16_f32_staged_vdecay(
            vdecay_stage, k_stage, k, v_new, g, tid, 0, value_head_idx,
            key_head_idx, value_base, k_half, g_last, h_lo, h_hi,
        )
        for acc_i in al.range(16):
            local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            global_v = value_base + lane_row
            old_lo = h_lo[acc_i]
            old_hi = h_hi[acc_i]
            if k_half == 0:
                if wave_id == 0:
                    h_lo[acc_i] = update_pair[acc_i]
                    h_hi[acc_i] = update_pair[16 + acc_i]
                    delta[0, value_head_idx, global_v, local_k] = update_pair[acc_i] - old_lo * g_last_exp
                    delta[0, value_head_idx, global_v, 32 + local_k] = update_pair[16 + acc_i] - old_hi * g_last_exp
            else:
                if wave_id == 1:
                    h_lo[acc_i] = update_pair[acc_i]
                    h_hi[acc_i] = update_pair[16 + acc_i]
                    delta[0, value_head_idx, global_v, 64 + local_k] = update_pair[acc_i] - old_lo * g_last_exp
                    delta[0, value_head_idx, global_v, 96 + local_k] = update_pair[16 + acc_i] - old_hi * g_last_exp
    al.syncthreads()

    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_row
        if wave_id == 0:
            state_after[0, value_head_idx, global_v, local_k] = h_lo[acc_i]
            state_after[0, value_head_idx, global_v, 32 + local_k] = h_hi[acc_i]
            final_state[0, value_head_idx, global_v, local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 32 + local_k] = h_hi[acc_i]
        else:
            state_after[0, value_head_idx, global_v, 64 + local_k] = h_lo[acc_i]
            state_after[0, value_head_idx, global_v, 96 + local_k] = h_hi[acc_i]
            final_state[0, value_head_idx, global_v, 64 + local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 96 + local_k] = h_hi[acc_i]


def _case_g(case: p0.Case, seed: int) -> torch.Tensor:
    if case.name == "random_low_amplitude_nonzero_w":
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return (torch.randn((1, BT, H_V), device="cuda", dtype=torch.float32, generator=generator) * 0.02).contiguous()
    return torch.zeros((1, BT, H_V), device="cuda", dtype=torch.float32)


def _reference(case: p0.Case, k: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_ref = p0._reference(case)
    g_last = g[:, BT - 1 : BT]
    decay = torch.exp(g_last - g)
    v_decay = (pred_ref["v_new"].float() * decay[..., None]).to(torch.bfloat16)
    delta = torch.empty((1, H_V, KDIM, KDIM), device="cuda", dtype=torch.float32)
    state_after = torch.empty_like(delta)
    h = case.initial_state.to(torch.bfloat16).unsqueeze(1)
    for head in range(H_V):
        key_head = head // 2
        delta[0, head] = v_decay[0, :, head].float().t() @ k[0, :, key_head].float()
        state_after[0, head] = case.initial_state[0, head] * torch.exp(g[0, BT - 1, head]) + delta[0, head]
    return {**pred_ref, "v_decay": v_decay, "delta": delta, "h": h, "state_after": state_after, "final_state": state_after}


def _make_k(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((1, BT, H_K, KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.02).to(torch.bfloat16).contiguous()


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    """Capture the exact correctness kernel once for metadata/ISA auditing."""
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    target = _HSACO_DUMP_DIR / f"{_qwen_gdn_direct_k64_bv32_full_p1_kernel.fn.__name__}.hsaco"
    if target.exists():
        launch()
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and "full_p1_kernel" in src.fn.fn.__name__:
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
        raise RuntimeError("P1 HSACO dump requested, but no matching kernel compiled")


def _run_kernel(case: p0.Case, k: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
    h = torch.empty((1, 1, H_V, KDIM, KDIM), device="cuda", dtype=torch.bfloat16)
    raw_acc = torch.empty((H_V, 4, 2, 2, 64, 16), device="cuda", dtype=torch.float32)
    pred_partial = torch.empty((H_V, 4, 2, 2, 32, 32), device="cuda", dtype=torch.float32)
    pred_f32 = torch.empty((1, BT, H_V, KDIM), device="cuda", dtype=torch.float32)
    pred_bf16 = torch.empty_like(case.w)
    v_new = torch.empty_like(case.u)
    v_decay = torch.empty_like(case.u)
    delta = torch.empty((1, H_V, KDIM, KDIM), device="cuda", dtype=torch.float32)
    state_after = torch.empty_like(delta)
    final_state = torch.empty_like(delta)
    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_full_p1_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
            k, case.w, case.u, g, case.initial_state, h, raw_acc, pred_partial,
            pred_f32, pred_bf16, v_new, v_decay, delta, state_after, final_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch)
    return {
        "h": h,
        "raw_acc": raw_acc,
        "pred_partial": pred_partial,
        "pred_f32": pred_f32,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
        "v_decay": v_decay,
        "delta": delta,
        "state_after": state_after,
        "final_state": final_state,
    }


def _first_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any] | None:
    for name in ("h", "raw_acc", "pred_partial", "pred_f32", "pred_bf16", "v_new", "v_decay", "delta", "state_after", "final_state"):
        diff = (actual[name].float() - expected[name].float()).abs()
        limit = p0.P0_BF16_ATOL if name in {"h", "pred_bf16", "v_new", "v_decay"} else p0.P0_FP32_ATOL
        if name in {"delta", "state_after", "final_state"}:
            limit = STATE_ATOL
        maximum = float(diff.max().item())
        if maximum > limit:
            flat = int(diff.flatten().argmax().item())
            index = tuple(int(item.item()) for item in torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
            return {"stage": name, "index": index, "expected": float(expected[name][index].float().item()), "actual": float(actual[name][index].float().item()), "abs_error": maximum, "limit": limit}
    return None


def _summary(case: p0.Case, actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {"case": case.name}
    for name in ("h", "raw_acc", "pred_partial", "pred_f32", "pred_bf16", "v_new", "v_decay", "delta", "state_after", "final_state"):
        error = (actual[name].float() - expected[name].float()).abs()
        result[f"{name}_max_abs"] = float(error.max().item())
        result[f"{name}_mean_abs"] = float(error.mean().item())
    result["finite"] = all(bool(torch.isfinite(value).all()) for value in actual.values())
    result["first_error"] = _first_error(actual, expected)
    result["pass"] = result["finite"] and result["first_error"] is None
    return result


def run_p1(*, seed: int, out_dir: Path) -> list[dict[str, Any]]:
    _set_c0_lowering()
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(p0._machine_cases(seed)):
        k = _make_k(seed + 1000 + index)
        g = _case_g(case, seed + 2000 + index)
        actual = _run_kernel(case, k, g)
        torch.cuda.synchronize()
        expected = _reference(case, k, g)
        torch.cuda.synchronize()
        rows.append(_summary(case, actual, expected))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p1_single_chunk_composition_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = run_p1(seed=args.seed, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    if not all(bool(row["pass"]) for row in rows):
        raise SystemExit("P1 one-chunk composition gate failed; feedback ladder and all benchmarking are blocked.")


if __name__ == "__main__":
    main()

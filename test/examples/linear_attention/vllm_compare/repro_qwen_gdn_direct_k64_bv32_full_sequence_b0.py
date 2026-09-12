#!/usr/bin/env python3
"""B0: correctness-locked native Direct-K64 BV32 full recurrence baseline.

This experimental kernel moves the P1-passing nonzero-W pred plus C0
Direct-K64 update body into one device-side BT64 chunk loop.  It deliberately
does not add a new pipeline, ownership scheme, LDS layout, or lifetime trick.
The only difference from P2 is that the FP32 state feedback stays inside the
CTA rather than returning to the host between P1 launches.
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
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1 as p1
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2


BT = p0.BT
BV = p0.BV
KDIM = p0.KDIM
H_K = p0.H_K
H_V = p0.H_V
WORKGROUP = p0.WORKGROUP
GRID = p0.GRID
STATE_ATOL = p1.STATE_ATOL
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _qwen_gdn_direct_k64_bv32_full_sequence_b0_kernel(
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
    persistent_semantic: al.constexpr,
):
    """One CTA keeps one V32 x K128 FP32 state tile across all BT64 chunks."""

    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, KDIM), (num_tokens * H_K * KDIM, H_K * KDIM, KDIM, 1)),
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

    # R0 reuses this exact B0 source body. These compile-time delimiters are
    # absent from historical B0 launches. For R0, the formation pass captures
    # the complete device-side loop as one semantic region, then legacy_b0
    # lowering inlines it unchanged before block-dot lowering.
    if persistent_semantic:
        al.amdgpu.qwen_persistent_recurrence_begin(
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new,
            v_decay, state_after, final_state, num_chunks, emit_audit,
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

    # C0 BV32 ownership is unchanged: wave 0 owns K[0:64], wave 1 K[64:128]
    # for the same V32 rows. h_lo/h_hi are the only recurrence feedback carrier.
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

    # These allocations and the pred mapping are the P1 body.  C0's staged
    # block-dot consumes vdecay_stage directly and must not reload global V-new.
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    vdecay_stage = al.make_shared((1, BV, BT), al.bf16)
    k_stage = al.make_shared((BT, 64), al.bf16)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last = g[0, chunk_start + BT - 1, value_head_idx]
        g_last_exp = al.exp(g_last)

        # H is a BF16 pre-update snapshot.  state_bf16 is separately produced
        # from FP32 h_lo/h_hi for pred; neither H nor state_bf16 carries feedback.
        for acc_i in al.range(16):
            local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            global_v = value_base + lane_row
            if wave_id == 0:
                h[0, chunk_idx, value_head_idx, global_v, local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
                state_bf16[0, lane_row, local_k] = al.convert(h_lo[acc_i], al.bf16)
                state_bf16[0, lane_row, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
            else:
                h[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = al.convert(h_hi[acc_i], al.bf16)
                state_bf16[1, lane_row, local_k] = al.convert(h_lo[acc_i], al.bf16)
                state_bf16[1, lane_row, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
        al.syncthreads()

        # This is P0's corrected MFMA32 pred mapping, reused verbatim.  Each
        # loop writes every element of the corresponding shared buffer.
        for token_tile in al.range(2):
            token_base = token_tile * 32
            for rep_w in al.range(32):
                linear = tid + rep_w * WORKGROUP
                k_half = linear // (32 * 64)
                rem = linear - k_half * (32 * 64)
                token_off = rem // 64
                local_k = rem - token_off * 64
                w_bf16[k_half, token_off, local_k] = w[
                    0, chunk_start + token_base + token_off, value_head_idx, k_half * 64 + local_k
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
                out_row = lane_row
                out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
            al.syncthreads()

            for rep_out in al.range(8):
                linear = tid + rep_out * WORKGROUP
                token_off = linear // BV
                local_v = linear - token_off * BV
                token_idx = chunk_start + token_base + token_off
                global_v = value_base + local_v
                pred = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
                corrected = al.convert(u[0, token_idx, value_head_idx, global_v], al.f32) - pred
                # Current-vLLM ABI boundary: update consumes FP32(BF16(V-new)),
                # never the nearby unrounded corrected FP32 temporary.
                corrected_bf16 = al.convert(corrected, al.bf16)
                corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
                decay = al.exp(g_last - g[0, token_idx, value_head_idx])
                if emit_audit:
                    pred_f32[0, token_idx, value_head_idx, global_v] = pred
                    pred_bf16[0, token_idx, value_head_idx, global_v] = al.convert(pred, al.bf16)
                v_new[0, token_idx, value_head_idx, global_v] = corrected_bf16
                vdecay_stage[0, local_v, token_base + token_off] = al.convert(corrected_from_bf16 * decay, al.bf16)
                if emit_audit:
                    v_decay[0, token_idx, value_head_idx, global_v] = al.convert(corrected_from_bf16 * decay, al.bf16)
            al.syncthreads()

        # The staged C0 op uses exactly the P1 K32 accumulation order.  Only
        # its chunk_start is dynamic now, because B0 owns the full sequence.
        for k_half in al.range(2):
            update_pair = al.amdgpu.block_dot_bf16_f32_staged_vdecay(
                vdecay_stage, k_stage, k, v_new, g, tid, chunk_start, value_head_idx,
                key_head_idx, value_base, k_half, g_last, h_lo, h_hi,
            )
            for acc_i in al.range(16):
                if k_half == 0:
                    if wave_id == 0:
                        h_lo[acc_i] = update_pair[acc_i]
                        h_hi[acc_i] = update_pair[16 + acc_i]
                else:
                    if wave_id == 1:
                        h_lo[acc_i] = update_pair[acc_i]
                        h_hi[acc_i] = update_pair[16 + acc_i]
        al.syncthreads()

        for acc_i in al.range(16):
            local_k = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            global_v = value_base + lane_row
            if emit_audit:
                if wave_id == 0:
                    state_after[0, chunk_idx, value_head_idx, global_v, local_k] = h_lo[acc_i]
                    state_after[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = h_hi[acc_i]
                else:
                    state_after[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = h_lo[acc_i]
                    state_after[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = h_hi[acc_i]
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

    if persistent_semantic:
        al.amdgpu.qwen_persistent_recurrence_end()


def _reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Device-contract reference: FP32 feedback, BF16 pred/V-new/V-decay ABI."""

    num_tokens = int(k.shape[1])
    num_chunks = num_tokens // BT
    state = initial_state.clone()
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=k.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(w)
    v_new = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BT
        stop = start + BT
        h[:, chunk_idx] = state.to(torch.bfloat16)
        state_operand = state.to(torch.bfloat16).float()
        for head in range(H_V):
            pred_f32[0, start:stop, head] = w[0, start:stop, head].float() @ state_operand[0, head].t()
        pred_bf16[:, start:stop] = pred_f32[:, start:stop].to(torch.bfloat16)
        v_new[:, start:stop] = (u[:, start:stop].float() - pred_f32[:, start:stop]).to(torch.bfloat16)
        g_last = g[:, stop - 1 : stop]
        decay = torch.exp(g_last - g[:, start:stop])
        v_decay[:, start:stop] = (v_new[:, start:stop].float() * decay[..., None]).to(torch.bfloat16)
        next_state = torch.empty_like(state)
        for head in range(H_V):
            key_head = head // 2
            delta = v_decay[0, start:stop, head].float().t() @ k[0, start:stop, key_head].float()
            next_state[0, head] = state[0, head] * torch.exp(g[0, stop - 1, head]) + delta
        state = next_state
        state_after[:, chunk_idx] = state
    return {
        "h": h,
        "pred_f32": pred_f32,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
        "v_decay": v_decay,
        "state_after": state_after,
        "final_state": state,
    }


def _run_p2_host_microscope(
    k_all: torch.Tensor,
    w_all: torch.Tensor,
    u_all: torch.Tensor,
    g_all: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replay the already-correct P1 body per chunk for byte-level B0 checks."""

    num_tokens = int(k_all.shape[1])
    num_chunks = num_tokens // BT
    actual_state = initial_state.clone()
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k_all.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=k_all.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(w_all)
    v_new = torch.empty_like(u_all)
    v_decay = torch.empty_like(u_all)
    state_after = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k_all.device, dtype=torch.float32)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BT
        stop = start + BT
        case = p0.Case(
            f"b0_p2_T{num_tokens}_chunk{chunk_idx}",
            w_all[:, start:stop].contiguous(),
            u_all[:, start:stop].contiguous(),
            actual_state,
        )
        actual = p1._run_kernel(case, k_all[:, start:stop].contiguous(), g_all[:, start:stop].contiguous())
        h[:, chunk_idx] = actual["h"][:, 0]
        pred_f32[:, start:stop] = actual["pred_f32"]
        pred_bf16[:, start:stop] = actual["pred_bf16"]
        v_new[:, start:stop] = actual["v_new"]
        v_decay[:, start:stop] = actual["v_decay"]
        state_after[:, chunk_idx] = actual["state_after"]
        actual_state = actual["state_after"].detach()
    return {
        "h": h,
        "pred_f32": pred_f32,
        "pred_bf16": pred_bf16,
        "v_new": v_new,
        "v_decay": v_decay,
        "state_after": state_after,
        "final_state": actual_state,
    }


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    target = _HSACO_DUMP_DIR / f"{_qwen_gdn_direct_k64_bv32_full_sequence_b0_kernel.fn.__name__}.hsaco"
    if target.exists():
        launch()
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and "full_sequence_b0_kernel" in src.fn.fn.__name__:
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
        raise RuntimeError("B0 HSACO dump requested, but no matching kernel compiled")


def _run_kernel(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    p1._set_c0_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=k.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(w)
    v_new = torch.empty_like(u)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    final_state = torch.empty_like(initial_state)

    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_full_sequence_b0_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_chunks, True, False, num_warps=2,
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
    """Return the production-shape B0 body launch with audit stores compiled out.

    This uses the same P1 pred, BF16 V-new boundary, C0 update and FP32
    feedback as the correctness arm. Only the non-ABI diagnostic outputs are
    compile-time-elided; H, V-new and final state remain materialized.
    """

    p1._set_c0_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    num_chunks = num_tokens // BT
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    v_new = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)
    # These argument buffers remain allocated to preserve the shared source
    # signature, but emit_audit=False removes all accesses to them from ISA.
    pred_f32 = torch.empty((1, num_tokens, H_V, KDIM), device=k.device, dtype=torch.float32)
    pred_bf16 = torch.empty_like(w)
    v_decay = torch.empty_like(u)
    state_after = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)

    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_full_sequence_b0_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_chunks, False, False, num_warps=2,
        )

    launch()
    return launch, h, v_new, final_state


def _max_mean(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _first_error(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict[str, Any] | None:
    for name in ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state"):
        maximum, _ = _max_mean(actual[name], expected[name])
        limit = p0.P0_BF16_ATOL if name in {"h", "pred_bf16", "v_new", "v_decay"} else p0.P0_FP32_ATOL
        if name in {"state_after", "final_state"}:
            limit = STATE_ATOL
        if maximum > limit:
            diff = (actual[name].float() - expected[name].float()).abs()
            flat = int(diff.flatten().argmax().item())
            index = tuple(int(value.item()) for value in torch.unravel_index(torch.tensor(flat, device=diff.device), diff.shape))
            return {
                "stage": name,
                "index": index,
                "expected": float(expected[name][index].float().item()),
                "actual": float(actual[name][index].float().item()),
                "abs_error": maximum,
                "limit": limit,
            }
    return None


def _compare(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {"comparison": label}
    for name in ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state"):
        maximum, mean = _max_mean(actual[name], expected[name])
        row[f"{name}_max_abs"] = maximum
        row[f"{name}_mean_abs"] = mean
        row[f"{name}_byte_equal"] = bool(torch.equal(actual[name], expected[name]))
    row["first_error"] = _first_error(actual, expected)
    row["pass"] = row["first_error"] is None
    return row


def _per_chunk_rows(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_idx in range(int(actual["h"].shape[1])):
        row: dict[str, Any] = {"chunk": chunk_idx}
        start = chunk_idx * BT
        stop = start + BT
        for name, lhs, rhs in (
            ("pred_f32", actual["pred_f32"][:, start:stop], expected["pred_f32"][:, start:stop]),
            ("v_new", actual["v_new"][:, start:stop], expected["v_new"][:, start:stop]),
            ("v_decay", actual["v_decay"][:, start:stop], expected["v_decay"][:, start:stop]),
            ("state_after", actual["state_after"][:, chunk_idx], expected["state_after"][:, chunk_idx]),
        ):
            maximum, mean = _max_mean(lhs, rhs)
            row[f"{name}_max_abs"] = maximum
            row[f"{name}_mean_abs"] = mean
        # Validate the two recurrence carriers directly: H is snapshot only,
        # whereas next pred reads the still-FP32 persistent h vectors.
        snapshot_max, _ = _max_mean(actual["h"][:, chunk_idx], expected["h"][:, chunk_idx])
        row["h_snapshot_max_abs"] = snapshot_max
        row["finite"] = all(bool(torch.isfinite(actual[name]).all()) for name in ("h", "pred_f32", "v_new", "v_decay", "state_after"))
        rows.append(row)
    return rows


def run_correctness_length(t: int, *, seed: int, include_p2: bool = True) -> dict[str, Any]:
    p1._set_c0_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    device_reference = _reference(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    device_check = _compare(actual, device_reference, label="device_contract_reference")
    p2_check: dict[str, Any] | None = None
    if include_p2:
        microscope = _run_p2_host_microscope(k, w, u, g, initial_state)
        torch.cuda.synchronize()
        p2_check = _compare(actual, microscope, label="p2_host_microscope")
    per_chunk = _per_chunk_rows(actual, device_reference)
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state"))
    return {
        "T": t,
        "chunks": t // BT,
        "finite": finite,
        "device_contract": device_check,
        "p2_host_microscope": p2_check,
        "per_chunk": per_chunk,
        "pass": finite and bool(device_check["pass"]) and (p2_check is None or bool(p2_check["pass"])),
    }


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b0",
    )
    parser.add_argument("--skip-p2", action="store_true", help="Only for fast compile/debug; not a B0 correctness gate.")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(t, seed=args.seed + t, include_p2=not args.skip_p2) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "b0_full_sequence_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({key: value for key, value in result.items() if key != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("B0 correctness gate failed; do not run resource or performance collection.")


if __name__ == "__main__":
    main()

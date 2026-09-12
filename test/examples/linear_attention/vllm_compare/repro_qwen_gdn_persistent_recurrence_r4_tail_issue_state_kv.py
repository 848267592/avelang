#!/usr/bin/env python3
"""R4-tail KxV persistent-state / dual-dot orientation candidate.

The public H and final-state ABI remains [V, K].  Inside the single R4-tail
recurrence loop, however, each wave holds FP32 feedback as K32xV32 fragments.
The prediction consumes a KxV LDS state directly as W[T,K] @ state[K,V], and
the existing R4 specialized block-dot is selected in its state-KV mode for
K^T[K,T] @ V-decay[T,V].  No U/V-new packet bridge is used.
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

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0
import repro_qwen_gdn_persistent_recurrence_r0 as r0
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as tail_issue


BT = r4.BT
BV = r4.BV
KDIM = r4.KDIM
H_K = r4.H_K
H_V = r4.H_V
WORKGROUP = r4.WORKGROUP
GRID = r4.GRID
PLAN = "gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_dual_dot"
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _state_kv_kernel(
    k_ptr: al.Pointer(al.bf16), w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16), g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32), h_ptr: al.Pointer(al.bf16),
    pred_f32_ptr: al.Pointer(al.f32), pred_bf16_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16), v_decay_ptr: al.Pointer(al.bf16),
    state_after_ptr: al.Pointer(al.f32), final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr, num_chunks: al.constexpr,
    emit_audit: al.constexpr, persistent_semantic: al.constexpr,
    pred_mn_axis_swap: al.constexpr = False,
    pred_mn_v4_io: al.constexpr = False,
):
    """One CTA: FP32 feedback fragments are KxV; external ABI stays VxK.

    ``pred_mn_axis_swap`` is deliberately a constexpr-only producer experiment.
    Its true branch removes the transient ``state_pred[V,K]`` operand view and
    feeds the existing KxV feedback directly as MFMA B, i.e. W[T,K] @ S[K,V].
    ``pred_mn_v4_io`` is a second, independently-specialized mechanism probe:
    it preserves the resulting T1xV4 MFMA owner through the two-wave partial
    reduction and uses one b64 U load and one b64 V-new store per contiguous
    V4 packet.  It deliberately leaves H and the existing V-decay/update path
    untouched.
    """
    k = al.make_tensor(k_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_K, KDIM),
                       (num_tokens * H_K * KDIM, H_K * KDIM, KDIM, 1)))
    w = al.make_tensor(w_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    u = al.make_tensor(u_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    g = al.make_tensor(g_ptr, al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    initial_state = al.make_tensor(initial_state_ptr, al.f32,
        al.make_layout((1, H_V, KDIM, KDIM),
                       (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)))
    h = al.make_tensor(h_ptr, al.bf16,
        al.make_layout((1, num_chunks, H_V, KDIM, KDIM),
                       (num_chunks * H_V * KDIM * KDIM, H_V * KDIM * KDIM,
                        KDIM * KDIM, KDIM, 1)))
    pred_f32 = al.make_tensor(pred_f32_ptr, al.f32,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    pred_bf16 = al.make_tensor(pred_bf16_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    v_new = al.make_tensor(v_new_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    v_decay = al.make_tensor(v_decay_ptr, al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM),
                       (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)))
    state_after = al.make_tensor(state_after_ptr, al.f32,
        al.make_layout((1, num_chunks, H_V, KDIM, KDIM),
                       (num_chunks * H_V * KDIM * KDIM, H_V * KDIM * KDIM,
                        KDIM * KDIM, KDIM, 1)))
    final_state = al.make_tensor(final_state_ptr, al.f32,
        al.make_layout((1, H_V, KDIM, KDIM),
                       (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)))

    # Raw descriptors are resolved before constexpr branches are specialized.
    # They are therefore declared unconditionally and eliminated from control
    # arms that do not select the T1xV4 I/O candidate.
    io_bf16_bytes = num_tokens * H_V * KDIM * 2
    u_rsrc = al.amdgpu.make_rsrc(u, io_bf16_bytes)
    v_new_rsrc = al.amdgpu.make_rsrc(v_new, io_bf16_bytes)
    raw_zero = al.convert(0, al.i32)

    if persistent_semantic:
        al.amdgpu.qwen_persistent_recurrence_begin(
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new,
            v_decay, state_after, final_state, num_chunks, emit_audit)

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_row = lane & 31
    mfma_lane_group = lane >> 5
    program_id = al.block_id(0)
    value_head_idx = program_id >> 2
    value_base = (program_id & 3) * BV
    key_head_idx = value_head_idx >> 1

    # In this candidate a lane identifies a K row, and its sixteen
    # accumulators identify the V32 columns.  h_k_lo/h_k_hi therefore cover
    # K0:32/K32:64 of the lane's K64 half rather than V/K columns of R4.
    h_k_lo = al.full((16,), 0.0, al.f32)
    h_k_hi = al.full((16,), 0.0, al.f32)
    for acc_i in al.range(16):
        local_v = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + local_v
        if wave_id == 0:
            h_k_lo[acc_i] = initial_state[0, value_head_idx, global_v, lane_row]
            h_k_hi[acc_i] = initial_state[0, value_head_idx, global_v, 32 + lane_row]
        else:
            h_k_lo[acc_i] = initial_state[0, value_head_idx, global_v, 64 + lane_row]
            h_k_hi[acc_i] = initial_state[0, value_head_idx, global_v, 96 + lane_row]

    # The false branch is the historical state-KV control: it materializes a
    # VxK dot operand.  The true branch instead owns a KxV BF16 snapshot that
    # is read directly by pred as B[K,V].  AveLang resolves constexpr control
    # flow after its AST-to-MLIR construction, so both branch-local memrefs
    # must be declared here.  The production audit below explicitly checks
    # that the unused VxK allocation/path is eliminated after specialization.
    state_pred_kv = al.make_shared((2, 64, BV), al.bf16)
    state_pred_vk = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, BT, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    vdecay_stage = al.make_shared((1, BV, BT), al.bf16)
    k_stage = al.make_shared((2, BT, 64), al.bf16)
    w_vec = al.view(w_bf16, al.i32,
        al.make_layout((2, 2, 32, 8, 4),
                       (2 * 32 * 8 * 4, 32 * 8 * 4, 8 * 4, 4, 1)))
    state_vec = al.view(state_pred_vk, al.i32,
        al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))

    current_w0 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, 0, value_head_idx, 0)
    current_w1 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, 0, value_head_idx, 1)
    current_k0 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, 0, key_head_idx, 0)
    current_k1 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, 0, key_head_idx, 1)
    al.amdgpu.qwen_bt64_pipeline_stage_commit(current_w0, w_bf16)
    al.amdgpu.qwen_bt64_pipeline_stage_commit(current_w1, w_bf16)
    al.amdgpu.qwen_bt64_pipeline_stage_commit(current_k0, k_stage)
    al.amdgpu.qwen_bt64_pipeline_stage_commit(current_k1, k_stage)
    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last = g[0, chunk_start + BT - 1, value_head_idx]

        # H's public ABI is deliberately left VxK.  The direct pred branch
        # writes a KxV LDS snapshot instead; it never creates a VxK LDS view.
        for acc_i in al.range(16):
            local_v = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
            global_v = value_base + local_v
            if wave_id == 0:
                h[0, chunk_idx, value_head_idx, global_v, lane_row] = al.convert(h_k_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 32 + lane_row] = al.convert(h_k_hi[acc_i], al.bf16)
                if pred_mn_axis_swap:
                    state_pred_kv[0, lane_row, local_v] = al.convert(h_k_lo[acc_i], al.bf16)
                    state_pred_kv[0, 32 + lane_row, local_v] = al.convert(h_k_hi[acc_i], al.bf16)
                else:
                    state_pred_vk[0, local_v, lane_row] = al.convert(h_k_lo[acc_i], al.bf16)
                    state_pred_vk[0, local_v, 32 + lane_row] = al.convert(h_k_hi[acc_i], al.bf16)
            else:
                h[0, chunk_idx, value_head_idx, global_v, 64 + lane_row] = al.convert(h_k_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 96 + lane_row] = al.convert(h_k_hi[acc_i], al.bf16)
                if pred_mn_axis_swap:
                    state_pred_kv[1, lane_row, local_v] = al.convert(h_k_lo[acc_i], al.bf16)
                    state_pred_kv[1, 32 + lane_row, local_v] = al.convert(h_k_hi[acc_i], al.bf16)
                else:
                    state_pred_vk[1, local_v, lane_row] = al.convert(h_k_lo[acc_i], al.bf16)
                    state_pred_vk[1, local_v, 32 + lane_row] = al.convert(h_k_hi[acc_i], al.bf16)
        al.syncthreads()

        # Pred is W[T,K] @ state[K,V].  The direct branch constructs the
        # MFMA B input from state_pred_kv[wave, K, lane_row] (K-strided at a
        # fixed V), then emits mfma(W, state).  Thus the physical MFMA output
        # axes are M=T and N=V; it does not restore the old VxK operand view.
        for token_tile in al.range(2):
            token_base = token_tile * 32
            pred_acc = al.full((16,), 0.0, al.f32)
            if pred_mn_axis_swap:
                # The MFMA32 microscope proves that this wrapper's argument
                # order is physical B,A (not A,B): arg0[lane,slot] owns
                # B[K,N] and arg1 owns A[M,K].  The typed W producer's
                # established K traversal is ``2*kpack + lane_group`` with
                # two fixed BF16x4 halves.  Keep that exact traversal; the
                # microscope-derived correction is the B,A argument order.
                for kpack in al.range(4):
                    k_vec = kpack * 2 + mfma_lane_group
                    w_words = w_vec[wave_id, token_tile, lane_row, k_vec]
                    w_frag = al.view(w_words, al.Tensor((2, 4, 1), al.bf16))
                    state_lo = al.amdgpu.qwen_pred_state_kv_frag_load_bf16x4(
                        state_pred_kv, wave_id, k_vec, lane_row, 0)
                    state_hi = al.amdgpu.qwen_pred_state_kv_frag_load_bf16x4(
                        state_pred_kv, wave_id, k_vec, lane_row, 1)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                        state_lo, w_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                        state_hi, w_frag[1], pred_acc)
            else:
                for kpack in al.range(4):
                    k_vec = kpack * 2 + mfma_lane_group
                    w_words = w_vec[wave_id, token_tile, lane_row, k_vec]
                    w_frag = al.view(w_words, al.Tensor((2, 4, 1), al.bf16))
                    state_words = state_vec[wave_id, lane_row, k_vec]
                    state_frag = al.view(state_words, al.Tensor((2, 4, 1), al.bf16))
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[0], w_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[1], w_frag[1], pred_acc)
            for acc_i in al.range(16):
                out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
                pred_partial[wave_id, lane_row, out_col] = pred_acc[acc_i]
            al.syncthreads()

            if pred_mn_v4_io:
                # Direct M=T/N=V ownership is T1xV4.  On each wave-0 lane
                # l, r=l mod 32 and g=floor(l/32), packet q owns
                #   (token_base+r, value_base+8q+4g+[0,4)).
                # The 64 lanes and q=[0,4) cover this T32xV32 tile exactly
                # once.  Wave 1 has already contributed the other K64 partial
                # in pred_partial; it does not participate in I/O ownership.
                # Thus this is four statically addressed b64 packets/lane, not
                # a 128-iteration exec-masked packet loop or an LDS bridge.
                if wave_id == 0:
                    token_idx = chunk_start + token_base + lane_row
                    decay = al.exp(g_last - g[0, token_idx, value_head_idx])
                    for packet_q in al.range(4):
                        local_v = packet_q * 8 + mfma_lane_group * 4
                        global_v = value_base + local_v
                        io_offset = (
                            ((token_idx * H_V + value_head_idx) * KDIM + global_v) * 2)
                        # The byte address is the per-lane vindex.  Keeping
                        # soffset at zero is essential: passing this varying
                        # value as soffset makes AMDGPU scalarize it through
                        # v_readfirstlane/saveexec, which is precisely the
                        # rejected exec-masked packet loop.
                        u_words = al.amdgpu.raw_buffer_load_x2(
                            u_rsrc, al.convert(io_offset, al.i32), raw_zero, 0)
                        u_packet = al.view(u_words, al.Tensor((4,), al.bf16))
                        pred_packet = al.full((4,), 0.0, al.f32)
                        corrected_packet = al.full((4,), 0.0, al.f32)
                        v_new_bits = al.full((4,), 0, al.u16)
                        v_decay_packet = al.full((4,), 0.0, al.bf16)
                        for packet_i in al.range(4):
                            local_v_i = local_v + packet_i
                            global_v_i = global_v + packet_i
                            pred_packet[packet_i] = (
                                pred_partial[0, lane_row, local_v_i]
                                + pred_partial[1, lane_row, local_v_i])
                            corrected_packet[packet_i] = (
                                al.convert(u_packet[packet_i], al.f32)
                                - pred_packet[packet_i])
                            corrected_bf16 = al.convert(
                                corrected_packet[packet_i], al.bf16)
                            corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
                            v_new_bits[packet_i] = al.bitcast(corrected_bf16, al.u16)
                            v_decay_packet[packet_i] = al.convert(
                                corrected_from_bf16 * decay, al.bf16)
                            # Audit sinks intentionally remain scalar and are
                            # compile-time absent from the production body.
                            if emit_audit:
                                pred_f32[0, token_idx, value_head_idx, global_v_i] = pred_packet[packet_i]
                                pred_bf16[0, token_idx, value_head_idx, global_v_i] = al.convert(pred_packet[packet_i], al.bf16)
                                v_decay[0, token_idx, value_head_idx, global_v_i] = v_decay_packet[packet_i]
                            # The typed V-decay layout consumed by the current
                            # specialized update block-dot is unchanged.
                            vdecay_stage[0, local_v_i, token_base + lane_row] = v_decay_packet[packet_i]
                        v_new_words = al.full((2,), 0, al.u32)
                        for word_i in al.range(2):
                            bit_base = word_i * 2
                            lo = al.convert(v_new_bits[bit_base], al.u32)
                            hi = al.convert(v_new_bits[bit_base + 1], al.u32)
                            v_new_words[word_i] = lo | (hi << 16)
                        al.amdgpu.raw_buffer_store_x2(
                            v_new_words, v_new_rsrc,
                            al.convert(io_offset, al.i32), raw_zero, 0)
            else:
                # The normal public T x V boundary remains the control path.
                for rep_out in al.range(8):
                    linear = tid + rep_out * WORKGROUP
                    token_off = linear // BV
                    local_v = linear - token_off * BV
                    token_idx = chunk_start + token_base + token_off
                    global_v = value_base + local_v
                    pred = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
                    corrected = al.convert(u[0, token_idx, value_head_idx, global_v], al.f32) - pred
                    corrected_bf16 = al.convert(corrected, al.bf16)
                    corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
                    if emit_audit:
                        pred_f32[0, token_idx, value_head_idx, global_v] = pred
                        pred_bf16[0, token_idx, value_head_idx, global_v] = al.convert(pred, al.bf16)
                    v_new[0, token_idx, value_head_idx, global_v] = corrected_bf16
                    vdecay_stage[0, local_v, token_base + token_off] = al.convert(
                        corrected_from_bf16 * al.exp(g_last - g[0, token_idx, value_head_idx]), al.bf16)
                    if emit_audit:
                        v_decay[0, token_idx, value_head_idx, global_v] = vdecay_stage[0, local_v, token_base + token_off]
            al.syncthreads()

        # R4's exact preloaded-K specialized block-dot now emits the dual
        # operand orientation.  Its returned low/high vectors are K32xV32
        # fragments and feed the same loop-carried FP32 feedback directly.
        for k_half in al.range(2):
            update_pair = al.amdgpu.block_dot_bf16_f32_staged_vdecay_preloaded_k_state_kv(
                vdecay_stage, k_stage, k, v_new, g, tid, chunk_start,
                value_head_idx, key_head_idx, value_base, k_half, g_last,
                h_k_lo, h_k_hi)
            for acc_i in al.range(16):
                if k_half == 0:
                    if wave_id == 0:
                        h_k_lo[acc_i] = update_pair[acc_i]
                        h_k_hi[acc_i] = update_pair[16 + acc_i]
                else:
                    if wave_id == 1:
                        h_k_lo[acc_i] = update_pair[acc_i]
                        h_k_hi[acc_i] = update_pair[16 + acc_i]

        # Strict R4-tail issue/commit boundary: all recurrence consumers have
        # retired before this same-bank next W/K producer executes.
        al.syncthreads()
        next_start = al.min(chunk_start + BT, num_tokens - BT)
        next_w0 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, next_start, value_head_idx, 0)
        next_w1 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, next_start, value_head_idx, 1)
        next_k0 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, next_start, key_head_idx, 0)
        next_k1 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, next_start, key_head_idx, 1)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_w0, w_bf16)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_w1, w_bf16)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_k0, k_stage)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_k1, k_stage)
        al.syncthreads()

        if emit_audit:
            for acc_i in al.range(16):
                local_v = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
                global_v = value_base + local_v
                if wave_id == 0:
                    state_after[0, chunk_idx, value_head_idx, global_v, lane_row] = h_k_lo[acc_i]
                    state_after[0, chunk_idx, value_head_idx, global_v, 32 + lane_row] = h_k_hi[acc_i]
                else:
                    state_after[0, chunk_idx, value_head_idx, global_v, 64 + lane_row] = h_k_lo[acc_i]
                    state_after[0, chunk_idx, value_head_idx, global_v, 96 + lane_row] = h_k_hi[acc_i]
        al.syncthreads()

    for acc_i in al.range(16):
        local_v = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
        global_v = value_base + local_v
        if wave_id == 0:
            final_state[0, value_head_idx, global_v, lane_row] = h_k_lo[acc_i]
            final_state[0, value_head_idx, global_v, 32 + lane_row] = h_k_hi[acc_i]
        else:
            final_state[0, value_head_idx, global_v, 64 + lane_row] = h_k_lo[acc_i]
            final_state[0, value_head_idx, global_v, 96 + lane_row] = h_k_hi[acc_i]
    if persistent_semantic:
        al.amdgpu.qwen_persistent_recurrence_end()


def _set_lowering() -> None:
    tail_issue._set_tail_issue_lowering()


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    """Dump this candidate's production/audit HSACO, never R4's wrapper."""
    if _HSACO_DUMP_DIR is None:
        launch()
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    target = _HSACO_DUMP_DIR / f"{_state_kv_kernel.fn.__name__}.hsaco"
    if target.exists():
        launch()
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and _state_kv_kernel.fn.__name__ in src.fn.fn.__name__:
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
        raise RuntimeError("state-KV HSACO dump requested, but no matching kernel compiled")


def _make_launch(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, outputs: tuple[torch.Tensor, ...], *, emit_audit: bool,
    pred_mn_axis_swap: bool = False,
    pred_mn_v4_io: bool = False,
) -> Callable[[], None]:
    _set_lowering()
    num_tokens = int(k.shape[1])
    if num_tokens % BT:
        raise ValueError(f"T={num_tokens} must be divisible by BT={BT}")
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs

    def launch() -> None:
        _state_kv_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
            k, w, u, g, initial_state, h, pred_f32, pred_bf16, v_new, v_decay,
            state_after, final_state, num_tokens, num_tokens // BT, emit_audit,
            True, pred_mn_axis_swap, pred_mn_v4_io, num_warps=2)

    return launch


def _run_kernel(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor | Callable[[], None]]:
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=True)
    _maybe_dump_hsaco(launch)
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    return {"h": h, "pred_f32": pred_f32, "pred_bf16": pred_bf16,
            "v_new": v_new, "v_decay": v_decay, "state_after": state_after,
            "final_state": final_state, "_launch": launch}


def run_body(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, *, dump_hsaco_dir: Path | None = None,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    global _HSACO_DUMP_DIR
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(k, w, u, g, initial_state, outputs, emit_audit=False)
    if dump_hsaco_dir is None:
        launch()
    else:
        previous_dump_dir = _HSACO_DUMP_DIR
        _HSACO_DUMP_DIR = dump_hsaco_dir
        try:
            _maybe_dump_hsaco(launch)
        finally:
            _HSACO_DUMP_DIR = previous_dump_dir
    return launch, outputs[0], outputs[3], outputs[6]


def run_pred_mn_swap_body(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, *, dump_hsaco_dir: Path | None = None,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile/launch the direct KxV pred M/N-swap mechanism candidate."""
    global _HSACO_DUMP_DIR
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(
        k, w, u, g, initial_state, outputs, emit_audit=False,
        pred_mn_axis_swap=True)
    if dump_hsaco_dir is None:
        launch()
    else:
        previous_dump_dir = _HSACO_DUMP_DIR
        _HSACO_DUMP_DIR = dump_hsaco_dir
        try:
            _maybe_dump_hsaco(launch)
        finally:
            _HSACO_DUMP_DIR = previous_dump_dir
    return launch, outputs[0], outputs[3], outputs[6]


def run_pred_mn_swap_v4_io_body(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor, *, dump_hsaco_dir: Path | None = None,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production body for direct T1xV4 b64 U/V-new ownership."""
    global _HSACO_DUMP_DIR
    outputs = r0._allocate(int(k.shape[1]), initial_state, u)
    launch = _make_launch(
        k, w, u, g, initial_state, outputs, emit_audit=False,
        pred_mn_axis_swap=True, pred_mn_v4_io=True)
    if dump_hsaco_dir is None:
        launch()
    else:
        previous_dump_dir = _HSACO_DUMP_DIR
        _HSACO_DUMP_DIR = dump_hsaco_dir
        try:
            _maybe_dump_hsaco(launch)
        finally:
            _HSACO_DUMP_DIR = previous_dump_dir
    return launch, outputs[0], outputs[3], outputs[6]


def run_pred_mn_swap_v4_io_correctness_length(t: int, *, seed: int) -> dict[str, Any]:
    """Full P2 correctness gate for the direct pred-M/N + b64 I/O candidate."""
    _set_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    outputs = r0._allocate(t, initial_state, u)
    launch = _make_launch(
        k, w, u, g, initial_state, outputs, emit_audit=True,
        pred_mn_axis_swap=True, pred_mn_v4_io=True)
    _maybe_dump_hsaco(launch)
    h, pred_f32, pred_bf16, v_new, v_decay, state_after, final_state = outputs
    actual: dict[str, torch.Tensor | Callable[[], None]] = {
        "h": h, "pred_f32": pred_f32, "pred_bf16": pred_bf16,
        "v_new": v_new, "v_decay": v_decay, "state_after": state_after,
        "final_state": final_state, "_launch": launch,
    }
    torch.cuda.synchronize()
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "state_kv_pred_mn_v4_io_vs_p2_host_microscope": b0._run_p2_host_microscope(k, w, u, g, initial_state),
        "state_kv_pred_mn_v4_io_vs_device_contract": b0._reference(k, w, u, g, initial_state),
    }
    torch.cuda.synchronize()
    comparisons = {name: b0._compare(actual, control, label=name)
                   for name, control in controls.items()}
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in
                 ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state"))
    return {
        "plan": "gfx942_bt64_bv32_joint_v4_tail_issue_state_kv_pred_mn_swap_v4_io",
        "T": t, "chunks": t // BT, "finite": finite,
        "comparisons": comparisons,
        "per_chunk": b0._per_chunk_rows(actual, controls["state_kv_pred_mn_v4_io_vs_device_contract"]),
        "pass": finite and all(bool(row["pass"]) for row in comparisons.values()),
    }


def run_correctness_length(t: int, *, seed: int) -> dict[str, Any]:
    _set_lowering()
    k, w, u, g, initial_state = p2._make_long_case(t, seed)
    actual = _run_kernel(k, w, u, g, initial_state)
    torch.cuda.synchronize()
    # References must compile after the state-KV module so its exact producer
    # and specialized block-dot selection cannot bleed into the control.
    os.environ["AVELANG_PERSISTENT_RECURRENCE_LOWERING"] = "gfx942_bt64_bv32_joint_v4"
    controls = {
        "state_kv_vs_p2_host_microscope": b0._run_p2_host_microscope(k, w, u, g, initial_state),
        "state_kv_vs_device_contract": b0._reference(k, w, u, g, initial_state),
    }
    torch.cuda.synchronize()
    comparisons = {name: b0._compare(actual, control, label=name)
                   for name, control in controls.items()}
    finite = all(bool(torch.isfinite(actual[name]).all()) for name in
                 ("h", "pred_f32", "pred_bf16", "v_new", "v_decay", "state_after", "final_state"))
    return {"plan": PLAN, "T": t, "chunks": t // BT, "finite": finite,
            "comparisons": comparisons,
            "per_chunk": b0._per_chunk_rows(actual, controls["state_kv_vs_device_contract"]),
            "pass": finite and all(bool(row["pass"]) for row in comparisons.values())}


def main() -> None:
    global _HSACO_DUMP_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--out-dir", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "rocprof_outputs/qwen_persistent_recurrence_r4_tail_issue_state_kv")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    results = [run_correctness_length(t, seed=args.seed + t) for t in args.T]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "state_kv_correctness.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(json.dumps({k: v for k, v in result.items() if k != "per_chunk"}, sort_keys=True))
    if not all(bool(result["pass"]) for result in results):
        raise SystemExit("R4-tail state-KV correctness gate failed")


if __name__ == "__main__":
    main()

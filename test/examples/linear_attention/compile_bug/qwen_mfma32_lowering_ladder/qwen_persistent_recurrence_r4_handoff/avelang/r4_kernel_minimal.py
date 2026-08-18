#!/usr/bin/env python3
"""Minimal R4 joint_v4 kernel body for source-level teaching.

This is the byte-preserved primary R4 kernel body extracted from the full
experiment file. Correctness/benchmark wrappers stay outside this file.
"""

from __future__ import annotations

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 128
H_K = 4
H_V = 8
WORKGROUP = 128
GRID = H_V * (KDIM // BV)

@avelang.jit
def _qwen_gdn_persistent_recurrence_r4_joint_v4_kernel(
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
    io_packet_ownership: al.constexpr,
    pred_mn_axis_swap: al.constexpr = False,
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

    # The production ABI is row-major BF16 for U/V-new and H.  Packet mode 1
    # gives one lane a contiguous V8 (U/V-new) or K4 (H) segment.  Packet mode
    # 2 is deliberately narrower: it keeps R4's H/state ownership, turns the
    # dead wave-1 pred-partial plane into a token-major LDS bridge, and gives
    # only U/V-new a contiguous V8 owner.
    # These descriptors intentionally live outside the constexpr branch:
    # AveLang resolves raw-buffer operands during AST generation, before it
    # prunes constexpr control flow.  They are dead in the scalar control arm.
    io_bf16_bytes = num_tokens * H_V * KDIM * 2
    h_bf16_bytes = num_chunks * H_V * KDIM * KDIM * 2
    u_rsrc = al.amdgpu.make_rsrc(u, io_bf16_bytes)
    h_rsrc = al.amdgpu.make_rsrc(h, h_bf16_bytes)
    v_new_rsrc = al.amdgpu.make_rsrc(v_new, io_bf16_bytes)
    raw_zero = al.convert(0, al.i32)

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

    # joint_v1 owns one full current W/K bank, never a second LDS bank. The
    # next tile remains an opaque compiler-owned token until this bank dies.
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, BT, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    vdecay_stage = al.make_shared((1, BV, BT), al.bf16)
    k_stage = al.make_shared((2, BT, 64), al.bf16)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 2, 32, 8, 4), (2 * 32 * 8 * 4, 32 * 8 * 4, 8 * 4, 4, 1)))
    # In bridge mode, plane zero becomes a [token, V8-packet, dword] view
    # after the two partial planes have been reduced.  This is a view only:
    # it reuses the existing pred_partial LDS allocation and never invokes a
    # cross-lane register transpose.
    pred_partial_words = al.view(
        pred_partial, al.u32,
        al.make_layout((2, 32, 4, 8), (32 * 4 * 8, 4 * 8, 8, 1)),
    )

    # Prologue: four opaque packet stages populate the only W/K bank. The
    # compiler chooses lane packets and load placement after the joint plan.
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
        g_last_exp = al.exp(g_last)

        # H is a BF16 pre-update snapshot.  state_bf16 is separately produced
        # from FP32 h_lo/h_hi for pred; neither H nor state_bf16 carries feedback.
        if io_packet_ownership == 1:
            # Four adjacent accumulator values form one 64-bit H packet.
            # Moving the wave selection outside this four-element group also
            # turns sixteen scalar-output exec-mask regions into four.
            for acc_group in al.range(4):
                acc_base = acc_group * 4
                local_k_base = acc_group * 8 + mfma_lane_group * 4
                for packet_i in al.range(4):
                    h_lo_value = al.convert(h_lo[acc_base + packet_i], al.bf16)
                    h_hi_value = al.convert(h_hi[acc_base + packet_i], al.bf16)
                    if wave_id == 0:
                        state_bf16[0, lane_row, local_k_base + packet_i] = h_lo_value
                        state_bf16[0, lane_row, 32 + local_k_base + packet_i] = h_hi_value
                    else:
                        state_bf16[1, lane_row, local_k_base + packet_i] = h_lo_value
                        state_bf16[1, lane_row, 32 + local_k_base + packet_i] = h_hi_value
                # A private bf16 vector currently leaves an unrealized cast in
                # the persistent-region lowering. Form the same bit-exact
                # BF16 pairs directly in i32 instead.
                h_lo_words = al.full((2,), 0, al.u32)
                h_hi_words = al.full((2,), 0, al.u32)
                for word_i in al.range(2):
                    pair_base = acc_base + word_i * 2
                    h_lo_bits0 = al.convert(al.bitcast(al.convert(h_lo[pair_base], al.bf16), al.u16), al.u32)
                    h_lo_bits1 = al.convert(al.bitcast(al.convert(h_lo[pair_base + 1], al.bf16), al.u16), al.u32)
                    h_hi_bits0 = al.convert(al.bitcast(al.convert(h_hi[pair_base], al.bf16), al.u16), al.u32)
                    h_hi_bits1 = al.convert(al.bitcast(al.convert(h_hi[pair_base + 1], al.bf16), al.u16), al.u32)
                    h_lo_words[word_i] = h_lo_bits0 | (h_lo_bits1 << 16)
                    h_hi_words[word_i] = h_hi_bits0 | (h_hi_bits1 << 16)
                global_v = value_base + lane_row
                if wave_id == 0:
                    h_lo_offset = (
                        (((chunk_idx * H_V + value_head_idx) * KDIM + global_v) * KDIM + local_k_base) * 2
                    )
                    h_hi_offset = h_lo_offset + 32 * 2
                    al.amdgpu.raw_buffer_store_x2(
                        h_lo_words, h_rsrc, raw_zero, al.convert(h_lo_offset, al.i32), 0)
                    al.amdgpu.raw_buffer_store_x2(
                        h_hi_words, h_rsrc, raw_zero, al.convert(h_hi_offset, al.i32), 0)
                else:
                    h_lo_offset = (
                        (((chunk_idx * H_V + value_head_idx) * KDIM + global_v) * KDIM + 64 + local_k_base) * 2
                    )
                    h_hi_offset = h_lo_offset + 32 * 2
                    al.amdgpu.raw_buffer_store_x2(
                        h_lo_words, h_rsrc, raw_zero, al.convert(h_lo_offset, al.i32), 0)
                    al.amdgpu.raw_buffer_store_x2(
                        h_hi_words, h_rsrc, raw_zero, al.convert(h_hi_offset, al.i32), 0)
        else:
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

        # R4 preserves R2's interleaved next-chunk issue. The joint_v4 plan
        # changes the W/K packet-to-LDS producer chain, not this verified
        # recurrence arithmetic or its single-bank tail-commit boundary.
        # Keep stage-token SSA values unconditional. AveLang's current JIT does
        # not form a token phi across two separate dynamic if regions. On the
        # epilogue this stages chunk N - 1 again; its tail commit is dead after
        # the final state store, while all steady-state iterations stage i + 1.
        next_start = al.min(chunk_start + BT, num_tokens - BT)
        next_w0 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, next_start, value_head_idx, 0)
        next_w1 = al.amdgpu.qwen_bt64_pipeline_stage_load(w, tid, next_start, value_head_idx, 1)

        # This is P0's corrected MFMA32 pred mapping. W now comes from the
        # prologue/tail LDS bank.  ``pred_mn_axis_swap`` is deliberately a
        # mechanism-only constexpr: it exchanges the two dot operands so
        # MFMA's physical output axes are M=T, N=V.  The existing output
        # address map is then already [token, value], without a post-pred LDS
        # bridge, gather, or VxT restoration.  All regular R4 callers keep
        # the historical false value.
        for token_tile in al.range(2):
            token_base = token_tile * 32
            pred_acc = al.full((16,), 0.0, al.f32)
            for kpack in al.range(4):
                k_vec = kpack * 2 + mfma_lane_group
                w_words = w_vec[wave_id, token_tile, lane_row, k_vec]
                state_words = state_vec[wave_id, lane_row, k_vec]
                w_frag = al.view(w_words, al.Tensor((2, 4, 1), al.bf16))
                state_frag = al.view(state_words, al.Tensor((2, 4, 1), al.bf16))
                if pred_mn_axis_swap:
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(w_frag[0], state_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(w_frag[1], state_frag[1], pred_acc)
                else:
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[0], w_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(state_frag[1], w_frag[1], pred_acc)

            for acc_i in al.range(16):
                out_row = lane_row
                out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
            al.syncthreads()

            if io_packet_ownership == 1:
                # 128 lanes own the 128 contiguous V8 packets of this T32xV32
                # tile. The local V/new and V-decay consumers remain unchanged.
                packet = tid
                token_off = packet // 4
                local_v = (packet - token_off * 4) * 8
                token_idx = chunk_start + token_base + token_off
                global_v = value_base + local_v
                io_offset = (((token_idx * H_V + value_head_idx) * KDIM + global_v) * 2)
                u_words = al.amdgpu.raw_buffer_load_x4(
                    u_rsrc, raw_zero, al.convert(io_offset, al.i32), 0)
                u_packet = al.view(u_words, al.Tensor((8,), al.bf16))
                v_new_bits = al.full((8,), 0, al.u16)
                # g is one scalar per (token, value-head), not per V. Packet
                # ownership therefore removes seven reloads without issuing a
                # pointless cross-head wide transaction.
                decay = al.exp(g_last - g[0, token_idx, value_head_idx])
                for packet_i in al.range(8):
                    pred = (
                        pred_partial[0, token_off, local_v + packet_i]
                        + pred_partial[1, token_off, local_v + packet_i]
                    )
                    corrected = al.convert(u_packet[packet_i], al.f32) - pred
                    corrected_bf16 = al.convert(corrected, al.bf16)
                    corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
                    v_new_bits[packet_i] = al.bitcast(corrected_bf16, al.u16)
                    if emit_audit:
                        pred_f32[0, token_idx, value_head_idx, global_v + packet_i] = pred
                        pred_bf16[0, token_idx, value_head_idx, global_v + packet_i] = al.convert(pred, al.bf16)
                    vdecay_value = al.convert(corrected_from_bf16 * decay, al.bf16)
                    vdecay_stage[0, local_v + packet_i, token_base + token_off] = vdecay_value
                    if emit_audit:
                        v_decay[0, token_idx, value_head_idx, global_v + packet_i] = vdecay_value
                v_new_words = al.full((4,), 0, al.u32)
                for word_i in al.range(4):
                    bit_base = word_i * 2
                    bits0 = al.convert(v_new_bits[bit_base], al.u32)
                    bits1 = al.convert(v_new_bits[bit_base + 1], al.u32)
                    v_new_words[word_i] = bits0 | (bits1 << 16)
                al.amdgpu.raw_buffer_store_x4(
                    v_new_words, v_new_rsrc, raw_zero, al.convert(io_offset, al.i32), 0)
            elif io_packet_ownership == 2:
                # LDS-mediated U/V-new bridge.  The R4 producer mapping is
                # V1xT8, so first reduce its two wave partials into plane 0
                # in the existing pred_partial allocation.  Each work-item
                # owns a unique (token, V) element; plane 1 is dead after
                # this write.  The following barrier turns plane 0 into the
                # token-major [T32, V32] bridge consumed by T1xV8 packets.
                for rep_out in al.range(8):
                    linear = tid + rep_out * WORKGROUP
                    bridge_token = linear // BV
                    bridge_v = linear - bridge_token * BV
                    pred_partial[0, bridge_token, bridge_v] = (
                        pred_partial[0, bridge_token, bridge_v]
                        + pred_partial[1, bridge_token, bridge_v]
                    )
                al.syncthreads()

                # One static lane packet owns T1xV8.  The source is a typed
                # LDS dword vector view, not ds_bpermute/gather or a dynamic
                # packet loop.  U and V-new use the matching b128 raw-buffer
                # packet; V-decay returns to exactly the old staged mapping
                # consumed by the specialized update block-dot.
                packet = tid
                token_off = packet // 4
                packet_v = packet - token_off * 4
                local_v = packet_v * 8
                token_idx = chunk_start + token_base + token_off
                global_v = value_base + local_v
                io_offset = (((token_idx * H_V + value_head_idx) * KDIM + global_v) * 2)
                pred_words = pred_partial_words[0, token_off, packet_v]
                pred_packet = al.view(pred_words, al.Tensor((8,), al.f32))
                u_words = al.amdgpu.raw_buffer_load_x4(
                    u_rsrc, raw_zero, al.convert(io_offset, al.i32), 0)
                u_packet = al.view(u_words, al.Tensor((8,), al.bf16))
                v_new_bits = al.full((8,), 0, al.u16)
                decay = al.exp(g_last - g[0, token_idx, value_head_idx])
                for packet_i in al.range(8):
                    pred = pred_packet[packet_i]
                    corrected = al.convert(u_packet[packet_i], al.f32) - pred
                    corrected_bf16 = al.convert(corrected, al.bf16)
                    corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
                    v_new_bits[packet_i] = al.bitcast(corrected_bf16, al.u16)
                    if emit_audit:
                        pred_f32[0, token_idx, value_head_idx, global_v + packet_i] = pred
                        pred_bf16[0, token_idx, value_head_idx, global_v + packet_i] = al.convert(pred, al.bf16)
                    vdecay_value = al.convert(corrected_from_bf16 * decay, al.bf16)
                    vdecay_stage[0, local_v + packet_i, token_base + token_off] = vdecay_value
                    if emit_audit:
                        v_decay[0, token_idx, value_head_idx, global_v + packet_i] = vdecay_value
                v_new_words = al.full((4,), 0, al.u32)
                for word_i in al.range(4):
                    bit_base = word_i * 2
                    bits0 = al.convert(v_new_bits[bit_base], al.u32)
                    bits1 = al.convert(v_new_bits[bit_base + 1], al.u32)
                    v_new_words[word_i] = bits0 | (bits1 << 16)
                al.amdgpu.raw_buffer_store_x4(
                    v_new_words, v_new_rsrc, raw_zero, al.convert(io_offset, al.i32), 0)
            else:
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

        next_k0 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, next_start, key_head_idx, 0)
        next_k1 = al.amdgpu.qwen_bt64_pipeline_stage_load(k, tid, next_start, key_head_idx, 1)

        # The staged C0 op consumes the current K bank directly. It must not
        # issue a second global K producer after the joint plan owns staging.
        for k_half in al.range(2):
            update_pair = al.amdgpu.block_dot_bf16_f32_staged_vdecay_preloaded_k(
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

        # Current W/K consumers have finished. Only now may the late compiler
        # commit the already-issued next packets to this same LDS bank.
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_w0, w_bf16)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_w1, w_bf16)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_k0, k_stage)
        al.amdgpu.qwen_bt64_pipeline_stage_commit(next_k1, k_stage)
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

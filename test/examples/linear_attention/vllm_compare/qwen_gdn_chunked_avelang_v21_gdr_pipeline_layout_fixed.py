"""Qwen GDN v21 BT16 chunk_gdr software-pipeline prototypes.

v21 keeps the v17 predecay BT16 path as the baseline shape and only swaps the
chunk_gdr stage.  The new kernels keep the v17 four-wave K-quarter split and
try conservative software pipelining for chunk-local global inputs:

* ``w``: double-buffer W staging.
* ``wk``: double-buffer W and transposed K staging.

The recurrence state is intentionally not double-buffered or reordered because
chunk ``i + 1`` must see chunk ``i``'s updated state.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    _require_fp32_cuda_contiguous,
    _validate_bf16_chunk_gdr_stage,
    _validate_bf16_qkvgb,
    _validate_chunk_size,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
    _require_v14_target_shape,
    _validate_initial_state_v14,
)

BT = 16
BV = 16


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_w_pipeline_mfma(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    gdr_decay = al.make_tensor(
        gdr_decay_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8, 16), (num_chunks * 8 * 16, 8 * 16, 16, 1)),
    )
    gdr_g_last_exp = al.make_tensor(
        gdr_g_last_exp_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8), (num_chunks * 8, 8, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((BV, 128), al.bf16)
    w_bf16_buf = al.make_shared((2, BT, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    pred_partial = al.make_shared((4, BT, BV), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_bf16_buf, al.i32, al.make_layout((2, BT, 16, 4), (1024, 64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64
        vv = idx // 32
        kk_local = idx - vv * 32
        kk = kk_local + wave_id * 32
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)

    # Preload chunk 0 W into buffer 0.
    for rep_w0 in al.range(8):
        idx_w0 = lane + rep_w0 * 64
        row_w0 = idx_w0 // 32
        kk_local_w0 = idx_w0 - row_w0 * 32
        col_w0 = kk_local_w0 + wave_id * 32
        w_bf16_buf[0, row_w0, col_w0] = al.convert(w[0, row_w0, value_head_idx, col_w0], al.bf16)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        cur = chunk_idx % 2
        nxt = 1 - cur
        g_last_exp = gdr_g_last_exp[0, chunk_idx, value_head_idx]

        for rep_h in al.range(8):
            idx_h = lane + rep_h * 64
            vv_h = idx_h // 32
            kk_local_h = idx_h - vv_h * 32
            kk_h = kk_local_h + wave_id * 32
            global_v_h = value_base + vv_h
            h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]

        for rep_all in al.range(8):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 32
            kk_local_all = idx_all - row_all * 32
            col_all = kk_local_all + wave_id * 32
            token_idx = chunk_start + row_all
            state_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            k_all_t[col_all, row_all] = k[0, token_idx, key_head_idx, col_all]

        al.syncthreads()

        pred_acc = al.full((4,), 0.0, al.f32)
        k_vec32 = lane_group + wave_id * 4
        a_words = w_vec[cur, lane_col, k_vec32]
        b_words = state_vec[lane_col, k_vec32]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

        for r_pred in al.range(4):
            token_offset_pred = lane_group * 4 + r_pred
            value_offset_pred = lane_col
            pred_partial[wave_id, token_offset_pred, value_offset_pred] = pred_acc[r_pred]

        al.syncthreads()

        if wave_id == 0:
            for r in al.range(4):
                token_offset = lane_group * 4 + r
                value_offset = lane_col
                token_idx = chunk_start + token_offset
                global_v = value_base + value_offset
                pred_value = (
                    pred_partial[0, token_offset, value_offset]
                    + pred_partial[1, token_offset, value_offset]
                    + pred_partial[2, token_offset, value_offset]
                    + pred_partial[3, token_offset, value_offset]
                )
                v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
                vn[0, token_idx, value_head_idx, global_v] = v_new
                decay = gdr_decay[0, chunk_idx, value_head_idx, token_offset]
                v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        # Bring the next chunk W into the alternate buffer before current update.
        if chunk_idx < num_chunks - 1:
            next_start = chunk_start + 16
            for rep_wn in al.range(8):
                idx_wn = lane + rep_wn * 64
                row_wn = idx_wn // 32
                kk_local_wn = idx_wn - row_wn * 32
                col_wn = kk_local_wn + wave_id * 32
                token_wn = next_start + row_wn
                w_bf16_buf[nxt, row_wn, col_wn] = al.convert(w[0, token_wn, value_head_idx, col_wn], al.bf16)

        al.syncthreads()

        for local_tile in al.range(2):
            global_tile = wave_id * 2 + local_tile
            base_k = global_tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 1:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
            if lane_group == 2:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 3:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc[r_up]

        al.syncthreads()

    for rep_final in al.range(8):
        idx_final = lane + rep_final * 64
        vv_final = idx_final // 32
        kk_local_final = idx_final - vv_final * 32
        kk_final = kk_local_final + wave_id * 32
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_wk_pipeline_mfma(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    gdr_decay = al.make_tensor(
        gdr_decay_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8, 16), (num_chunks * 8 * 16, 8 * 16, 16, 1)),
    )
    gdr_g_last_exp = al.make_tensor(
        gdr_g_last_exp_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8), (num_chunks * 8, 8, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((BV, 128), al.bf16)
    w_bf16_buf = al.make_shared((2, BT, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t_buf = al.make_shared((2, 128, BT), al.bf16)
    pred_partial = al.make_shared((4, BT, BV), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_bf16_buf, al.i32, al.make_layout((2, BT, 16, 4), (1024, 64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t_buf, al.i32, al.make_layout((2, 128, 2, 4), (1024, 8, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64
        vv = idx // 32
        kk_local = idx - vv * 32
        kk = kk_local + wave_id * 32
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)

    # Preload chunk 0 W/K into buffer 0.
    for rep_0 in al.range(8):
        idx_0 = lane + rep_0 * 64
        row_0 = idx_0 // 32
        kk_local_0 = idx_0 - row_0 * 32
        col_0 = kk_local_0 + wave_id * 32
        w_bf16_buf[0, row_0, col_0] = al.convert(w[0, row_0, value_head_idx, col_0], al.bf16)
        k_all_t_buf[0, col_0, row_0] = k[0, row_0, key_head_idx, col_0]

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        cur = chunk_idx % 2
        nxt = 1 - cur
        g_last_exp = gdr_g_last_exp[0, chunk_idx, value_head_idx]

        for rep_h in al.range(8):
            idx_h = lane + rep_h * 64
            vv_h = idx_h // 32
            kk_local_h = idx_h - vv_h * 32
            kk_h = kk_local_h + wave_id * 32
            global_v_h = value_base + vv_h
            h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]

        for rep_state in al.range(8):
            idx_state = lane + rep_state * 64
            row_state = idx_state // 32
            kk_local_state = idx_state - row_state * 32
            col_state = kk_local_state + wave_id * 32
            state_bf16[row_state, col_state] = al.convert(state[row_state, col_state], al.bf16)

        al.syncthreads()

        pred_acc = al.full((4,), 0.0, al.f32)
        k_vec32 = lane_group + wave_id * 4
        a_words = w_vec[cur, lane_col, k_vec32]
        b_words = state_vec[lane_col, k_vec32]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

        for r_pred in al.range(4):
            token_offset_pred = lane_group * 4 + r_pred
            value_offset_pred = lane_col
            pred_partial[wave_id, token_offset_pred, value_offset_pred] = pred_acc[r_pred]

        al.syncthreads()

        if wave_id == 0:
            for r in al.range(4):
                token_offset = lane_group * 4 + r
                value_offset = lane_col
                token_idx = chunk_start + token_offset
                global_v = value_base + value_offset
                pred_value = (
                    pred_partial[0, token_offset, value_offset]
                    + pred_partial[1, token_offset, value_offset]
                    + pred_partial[2, token_offset, value_offset]
                    + pred_partial[3, token_offset, value_offset]
                )
                v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
                vn[0, token_idx, value_head_idx, global_v] = v_new
                decay = gdr_decay[0, chunk_idx, value_head_idx, token_offset]
                v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        # Bring the next chunk W/K into the alternate buffer before current update.
        if chunk_idx < num_chunks - 1:
            next_start = chunk_start + 16
            for rep_n in al.range(8):
                idx_n = lane + rep_n * 64
                row_n = idx_n // 32
                kk_local_n = idx_n - row_n * 32
                col_n = kk_local_n + wave_id * 32
                token_n = next_start + row_n
                w_bf16_buf[nxt, row_n, col_n] = al.convert(w[0, token_n, value_head_idx, col_n], al.bf16)
                k_all_t_buf[nxt, col_n, row_n] = k[0, token_n, key_head_idx, col_n]

        al.syncthreads()

        for local_tile in al.range(2):
            global_tile = wave_id * 2 + local_tile
            base_k = global_tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[cur, base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 1:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[cur, base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
            if lane_group == 2:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[cur, base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 3:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[cur, base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc[r_up]

        al.syncthreads()

    for rep_final in al.range(8):
        idx_final = lane + rep_final * 64
        vv_final = idx_final // 32
        kk_local_final = idx_final - vv_final * 32
        kk_final = kk_local_final + wave_id * 32
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


def _validate_gdr_decay_v21(
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    *,
    num_chunks: int,
    device: torch.device,
) -> None:
    _require_fp32_cuda_contiguous("gdr_decay", gdr_decay)
    _require_fp32_cuda_contiguous("gdr_g_last_exp", gdr_g_last_exp)
    if tuple(gdr_decay.shape) != (1, num_chunks, 8, 16):
        raise ValueError(f"gdr_decay must have shape [1,{num_chunks},8,16], got {tuple(gdr_decay.shape)}.")
    if tuple(gdr_g_last_exp.shape) != (1, num_chunks, 8):
        raise ValueError(f"gdr_g_last_exp must have shape [1,{num_chunks},8], got {tuple(gdr_g_last_exp.shape)}.")
    if gdr_decay.device != device or gdr_g_last_exp.device != device:
        raise ValueError("gdr_decay and gdr_g_last_exp must be on the same device as k.")


def qwen_gdn_chunk_gdr_avelang_v21_4wave_pipeline_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
    variant: str = "wk",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("v21 pipeline chunk_gdr only supports chunk_size=16.")
    if variant not in {"w", "wk"}:
        raise ValueError("v21 pipeline variant must be 'w' or 'wk'.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(k, w, u, g, chunk_size)
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v21 pipeline chunk_gdr only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v21 pipeline chunk_gdr requires num_tokens divisible by 16.")
    num_chunks = _num_chunks(num_tokens, chunk_size)
    _validate_gdr_decay_v21(gdr_decay, gdr_g_last_exp, num_chunks=num_chunks, device=k.device)
    initial_state_arg, has_initial_state = _validate_initial_state_v14(initial_state, device=k.device)

    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    kernel = (
        _qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_w_pipeline_mfma
        if variant == "w"
        else _qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_wk_pipeline_mfma
    )
    kernel[lambda: ((64, 1, 1), (256, 1, 1))](
        k,
        w,
        u,
        gdr_decay,
        gdr_g_last_exp,
        initial_state_arg,
        h,
        vn,
        final_state,
        num_tokens,
        num_chunks,
        has_initial_state,
    )
    return h, vn, final_state


def qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
    variant: str = "wk",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_chunk_size(chunk_size)
    if chunk_size != BT:
        raise ValueError("v21 pipeline full path only supports chunk_size=16.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v14_target_shape(q, k, v, g, beta)
    _validate_initial_state_v14(initial_state, device=q.device)
    if q.shape[1] % BT != 0:
        raise ValueError("v21 pipeline full path requires num_tokens divisible by 16.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size)
    gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk_size)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v21_4wave_pipeline_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        gdr_decay,
        gdr_g_last_exp,
        initial_state=initial_state,
        chunk_size=chunk_size,
        variant=variant,
    )
    output = qwen_gdn_chunk_o_avelang_v14_mfma_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
    )
    return g_cumsum, output, a_solved, h, final_state, gdr_decay, gdr_g_last_exp


def qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
    variant: str = "wk",
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state, _, _ = qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        variant=variant,
    )
    return output, final_state


__all__ = [
    "_qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_w_pipeline_mfma",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v21_4wave_wk_pipeline_mfma",
    "qwen_gdn_chunk_gdr_avelang_v21_4wave_pipeline_mfma_layout",
    "qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout_full",
    "qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout",
]

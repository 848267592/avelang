"""Qwen GDN v11 narrow MFMA tiled chunk_gdr prototype.

This file keeps v9/v10 scalar paths as fallbacks and adds a deliberately narrow
MFMA chunk_gdr path for the vLLM Qwen3Next TP4 per-rank target:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16, chunk_size=BT=16, BV=16, BK=64

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
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v9_vllm_layout_fixed import qwen_gdn_chunk_o_avelang_v9_vllm_layout

BT = 16
BV = 16
BK = 64


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_10_scalar_update(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
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
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
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

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    h0_bf16 = al.make_shared((BV, BK), al.bf16)
    h1_bf16 = al.make_shared((BV, BK), al.bf16)
    w0_bf16 = al.make_shared((BT, BK), al.bf16)
    w1_bf16 = al.make_shared((BT, BK), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    h_all_bf16 = al.make_shared((BV, 128), al.bf16)
    w_all_bf16 = al.make_shared((BT, 128), al.bf16)

    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))
    h_all_vec = al.view(h_all_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_all_vec = al.view(w_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            state[vv, kk + 64] = initial_state[0, value_head_idx, global_v, kk + 64]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)
            state[vv, kk + 64] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        last_token = chunk_start + 15
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            global_v = value_base + vv
            h[0, chunk_idx, value_head_idx, global_v, kk] = state[vv, kk]
            h[0, chunk_idx, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]
            h0_bf16[vv, kk] = al.convert(state[vv, kk], al.bf16)
            h1_bf16[vv, kk] = al.convert(state[vv, kk + 64], al.bf16)
            token_offset = idx // BK
            k_offset = idx - token_offset * BK
            token_idx = chunk_start + token_offset
            w0_bf16[token_offset, k_offset] = al.convert(w[0, token_idx, value_head_idx, k_offset], al.bf16)
            w1_bf16[token_offset, k_offset] = al.convert(w[0, token_idx, value_head_idx, k_offset + BK], al.bf16)

        for rep_all in al.range(32):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 128
            col_all = idx_all - row_all * 128
            h_all_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            w_all_bf16[row_all, col_all] = al.convert(w[0, chunk_start + row_all, value_head_idx, col_all], al.bf16)

        al.syncthreads()

        # DEBUG isolate path: MFMA pred with the same contiguous [16,128]
        # shared/view layout as the passing standalone pred prototype.
        pred_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            k_vec128 = lane_group + batch128 * 4
            a_words128 = w_all_vec[lane_col, k_vec128]
            b_words128 = h_all_vec[lane_col, k_vec128]
            a_frag128 = al.view(a_words128, al.Tensor((2, 4, 1), al.bf16))
            b_frag128 = al.view(b_words128, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[0], b_frag128[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[1], b_frag128[1], pred_acc)




        for r in al.range(4):
            token_offset = lane_group * 4 + r
            value_offset = lane_col
            token_idx = chunk_start + token_offset
            global_v = value_base + value_offset
            v_new = u[0, token_idx, value_head_idx, global_v] - pred_acc[r]
            vn[0, token_idx, value_head_idx, global_v] = v_new
            decay = al.exp(g_last - g[0, token_idx, value_head_idx])
            v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            state[vv, kk] = state[vv, kk] * g_last_exp
            state[vv, kk + 64] = state[vv, kk + 64] * g_last_exp

        al.syncthreads()

        # Correctness workaround: scalar BF16-operand state update.  The MFMA
        # update path is isolated as the current failure point when it follows
        # pred MFMA in the same kernel.
        for rep_up in al.range(16):
            idx_up = lane + rep_up * 64
            vv_up = idx_up // BK
            kk_up = idx_up - vv_up * BK
            acc0 = al.convert(0.0, al.f32)
            acc1 = al.convert(0.0, al.f32)
            for tok_up in al.range(16):
                token_up = chunk_start + tok_up
                v_re = vn[0, token_up, value_head_idx, value_base + vv_up]
                d_re = al.exp(g_last - g[0, token_up, value_head_idx])
                vd_b = al.convert(v_re * d_re, al.bf16)
                vd_f = al.convert(vd_b, al.f32)
                k0_f = al.convert(k[0, token_up, key_head_idx, kk_up], al.f32)
                k1_f = al.convert(k[0, token_up, key_head_idx, kk_up + 64], al.f32)
                acc0 = acc0 + vd_f * k0_f
                acc1 = acc1 + vd_f * k1_f
            state[vv_up, kk_up] = state[vv_up, kk_up] + acc0
            state[vv_up, kk_up + 64] = state[vv_up, kk_up + 64] + acc1

        al.syncthreads()

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        final_state[0, value_head_idx, global_v, kk] = state[vv, kk]
        final_state[0, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]



@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_12_clean_scalar_update(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
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
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
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

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    h_all_bf16 = al.make_shared((BV, 128), al.bf16)
    w_all_bf16 = al.make_shared((BT, 128), al.bf16)

    h_all_vec = al.view(h_all_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_all_vec = al.view(w_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            state[vv, kk + 64] = initial_state[0, value_head_idx, global_v, kk + 64]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)
            state[vv, kk + 64] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        last_token = chunk_start + 15
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            global_v = value_base + vv
            h[0, chunk_idx, value_head_idx, global_v, kk] = state[vv, kk]
            h[0, chunk_idx, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]

        for rep_all in al.range(32):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 128
            col_all = idx_all - row_all * 128
            h_all_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            w_all_bf16[row_all, col_all] = al.convert(w[0, chunk_start + row_all, value_head_idx, col_all], al.bf16)

        al.syncthreads()

        # DEBUG isolate path: MFMA pred with the same contiguous [16,128]
        # shared/view layout as the passing standalone pred prototype.
        pred_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            k_vec128 = lane_group + batch128 * 4
            a_words128 = w_all_vec[lane_col, k_vec128]
            b_words128 = h_all_vec[lane_col, k_vec128]
            a_frag128 = al.view(a_words128, al.Tensor((2, 4, 1), al.bf16))
            b_frag128 = al.view(b_words128, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[0], b_frag128[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[1], b_frag128[1], pred_acc)




        for r in al.range(4):
            token_offset = lane_group * 4 + r
            value_offset = lane_col
            token_idx = chunk_start + token_offset
            global_v = value_base + value_offset
            v_new = u[0, token_idx, value_head_idx, global_v] - pred_acc[r]
            vn[0, token_idx, value_head_idx, global_v] = v_new

        al.syncthreads()

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            state[vv, kk] = state[vv, kk] * g_last_exp
            state[vv, kk + 64] = state[vv, kk + 64] * g_last_exp

        al.syncthreads()

        # Correctness-safe scalar BF16-operand state update.  Integrated
        # pred-MFMA -> update-MFMA is intentionally avoided; see mfma_bug.md.
        for rep_up in al.range(16):
            idx_up = lane + rep_up * 64
            vv_up = idx_up // BK
            kk_up = idx_up - vv_up * BK
            acc0 = al.convert(0.0, al.f32)
            acc1 = al.convert(0.0, al.f32)
            for tok_up in al.range(16):
                token_up = chunk_start + tok_up
                v_re = vn[0, token_up, value_head_idx, value_base + vv_up]
                d_re = al.exp(g_last - g[0, token_up, value_head_idx])
                vd_b = al.convert(v_re * d_re, al.bf16)
                vd_f = al.convert(vd_b, al.f32)
                k0_f = al.convert(k[0, token_up, key_head_idx, kk_up], al.f32)
                k1_f = al.convert(k[0, token_up, key_head_idx, kk_up + 64], al.f32)
                acc0 = acc0 + vd_f * k0_f
                acc1 = acc1 + vd_f * k1_f
            state[vv_up, kk_up] = state[vv_up, kk_up] + acc0
            state[vv_up, kk_up + 64] = state[vv_up, kk_up + 64] + acc1

        al.syncthreads()

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        final_state[0, value_head_idx, global_v, kk] = state[vv, kk]
        final_state[0, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]



@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
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
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
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

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    h_all_bf16 = al.make_shared((BV, 128), al.bf16)
    w_all_bf16 = al.make_shared((BT, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    h_all_vec = al.view(h_all_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_all_vec = al.view(w_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            state[vv, kk + 64] = initial_state[0, value_head_idx, global_v, kk + 64]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)
            state[vv, kk + 64] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        last_token = chunk_start + 15
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            global_v = value_base + vv
            h[0, chunk_idx, value_head_idx, global_v, kk] = state[vv, kk]
            h[0, chunk_idx, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]

        for rep_all in al.range(32):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 128
            col_all = idx_all - row_all * 128
            h_all_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            w_all_bf16[row_all, col_all] = al.convert(w[0, chunk_start + row_all, value_head_idx, col_all], al.bf16)
            k_all_t[col_all, row_all] = k[0, chunk_start + row_all, key_head_idx, col_all]

        al.syncthreads()

        # DEBUG isolate path: MFMA pred with the same contiguous [16,128]
        # shared/view layout as the passing standalone pred prototype.
        pred_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            k_vec128 = lane_group + batch128 * 4
            a_words128 = w_all_vec[lane_col, k_vec128]
            b_words128 = h_all_vec[lane_col, k_vec128]
            a_frag128 = al.view(a_words128, al.Tensor((2, 4, 1), al.bf16))
            b_frag128 = al.view(b_words128, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[0], b_frag128[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[1], b_frag128[1], pred_acc)




        for r in al.range(4):
            token_offset = lane_group * 4 + r
            value_offset = lane_col
            token_idx = chunk_start + token_offset
            global_v = value_base + value_offset
            v_new = u[0, token_idx, value_head_idx, global_v] - pred_acc[r]
            vn[0, token_idx, value_head_idx, global_v] = v_new
            decay = al.exp(g_last - g[0, token_idx, value_head_idx])
            v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            state[vv, kk] = state[vv, kk] * g_last_exp
            state[vv, kk + 64] = state[vv, kk + 64] * g_last_exp

        al.syncthreads()

        # Integrated update MFMA: delta_H[BV,128] = v_decay[BT,BV]^T @ k_chunk[BT,128].
        # This follows the passing standalone staged-delta prototype exactly:
        # lane_group 0/1/2/3 select the four packed token fragments.
        for tile in al.range(8):
            base_k = tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words = vdecay_vec[lane_col, 0]
                b_words = kall_vec[base_k + lane_col, 0]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
            if lane_group == 1:
                a_words = vdecay_vec[lane_col, 0]
                b_words = kall_vec[base_k + lane_col, 0]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
            if lane_group == 2:
                a_words = vdecay_vec[lane_col, 1]
                b_words = kall_vec[base_k + lane_col, 1]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
            if lane_group == 3:
                a_words = vdecay_vec[lane_col, 1]
                b_words = kall_vec[base_k + lane_col, 1]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] + acc[r_up]

            al.syncthreads()

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        final_state[0, value_head_idx, global_v, kk] = state[vv, kk]
        final_state[0, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]



@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_11_rtloop_scalar_update(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    num_chunks_rt: al.u32,
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
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
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

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    h0_bf16 = al.make_shared((BV, BK), al.bf16)
    h1_bf16 = al.make_shared((BV, BK), al.bf16)
    w0_bf16 = al.make_shared((BT, BK), al.bf16)
    w1_bf16 = al.make_shared((BT, BK), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    h_all_bf16 = al.make_shared((BV, 128), al.bf16)
    w_all_bf16 = al.make_shared((BT, 128), al.bf16)

    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))
    h_all_vec = al.view(h_all_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_all_vec = al.view(w_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            state[vv, kk + 64] = initial_state[0, value_head_idx, global_v, kk + 64]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)
            state[vv, kk + 64] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks_rt):
        chunk_start = chunk_idx * 16
        last_token = chunk_start + 15
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            global_v = value_base + vv
            h[0, chunk_idx, value_head_idx, global_v, kk] = state[vv, kk]
            h[0, chunk_idx, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]
            h0_bf16[vv, kk] = al.convert(state[vv, kk], al.bf16)
            h1_bf16[vv, kk] = al.convert(state[vv, kk + 64], al.bf16)
            token_offset = idx // BK
            k_offset = idx - token_offset * BK
            token_idx = chunk_start + token_offset
            w0_bf16[token_offset, k_offset] = al.convert(w[0, token_idx, value_head_idx, k_offset], al.bf16)
            w1_bf16[token_offset, k_offset] = al.convert(w[0, token_idx, value_head_idx, k_offset + BK], al.bf16)

        for rep_all in al.range(32):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 128
            col_all = idx_all - row_all * 128
            h_all_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            w_all_bf16[row_all, col_all] = al.convert(w[0, chunk_start + row_all, value_head_idx, col_all], al.bf16)

        al.syncthreads()

        # DEBUG isolate path: MFMA pred with the same contiguous [16,128]
        # shared/view layout as the passing standalone pred prototype.
        pred_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            k_vec128 = lane_group + batch128 * 4
            a_words128 = w_all_vec[lane_col, k_vec128]
            b_words128 = h_all_vec[lane_col, k_vec128]
            a_frag128 = al.view(a_words128, al.Tensor((2, 4, 1), al.bf16))
            b_frag128 = al.view(b_words128, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[0], b_frag128[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[1], b_frag128[1], pred_acc)




        for r in al.range(4):
            token_offset = lane_group * 4 + r
            value_offset = lane_col
            token_idx = chunk_start + token_offset
            global_v = value_base + value_offset
            v_new = u[0, token_idx, value_head_idx, global_v] - pred_acc[r]
            vn[0, token_idx, value_head_idx, global_v] = v_new
            decay = al.exp(g_last - g[0, token_idx, value_head_idx])
            v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            state[vv, kk] = state[vv, kk] * g_last_exp
            state[vv, kk + 64] = state[vv, kk + 64] * g_last_exp

        al.syncthreads()

        # Correctness workaround: scalar BF16-operand state update.  The MFMA
        # update path is isolated as the current failure point when it follows
        # pred MFMA in the same kernel.
        for rep_up in al.range(16):
            idx_up = lane + rep_up * 64
            vv_up = idx_up // BK
            kk_up = idx_up - vv_up * BK
            acc0 = al.convert(0.0, al.f32)
            acc1 = al.convert(0.0, al.f32)
            for tok_up in al.range(16):
                token_up = chunk_start + tok_up
                v_re = vn[0, token_up, value_head_idx, value_base + vv_up]
                d_re = al.exp(g_last - g[0, token_up, value_head_idx])
                vd_b = al.convert(v_re * d_re, al.bf16)
                vd_f = al.convert(vd_b, al.f32)
                k0_f = al.convert(k[0, token_up, key_head_idx, kk_up], al.f32)
                k1_f = al.convert(k[0, token_up, key_head_idx, kk_up + 64], al.f32)
                acc0 = acc0 + vd_f * k0_f
                acc1 = acc1 + vd_f * k1_f
            state[vv_up, kk_up] = state[vv_up, kk_up] + acc0
            state[vv_up, kk_up + 64] = state[vv_up, kk_up + 64] + acc1

        al.syncthreads()

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        final_state[0, value_head_idx, global_v, kk] = state[vv, kk]
        final_state[0, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]


def _is_v11_supported_chunk_gdr(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, chunk_size: int, block_v: int, block_k: int | None) -> bool:
    if k.dtype != torch.bfloat16:
        return False
    if chunk_size != BT or block_v != BV or (block_k is not None and block_k != BK):
        return False
    if tuple(k.shape) != (1, k.shape[1], 4, 128):
        return False
    if tuple(u.shape) != (1, k.shape[1], 8, 128):
        return False
    if tuple(w.shape) != (1, k.shape[1], 8, 128):
        return False
    if tuple(g.shape) != (1, k.shape[1], 8):
        return False
    if k.shape[1] % BT != 0:
        return False
    return True


def _validate_initial_state_v11(initial_state: torch.Tensor | None, *, device: torch.device) -> tuple[torch.Tensor | None, bool]:
    has_initial_state = initial_state is not None
    if initial_state is None:
        return None, has_initial_state
    _require_fp32_cuda_contiguous("initial_state", initial_state)
    if tuple(initial_state.shape) != (1, 8, 128, 128):
        raise ValueError(f"initial_state must have shape (1, 8, 128, 128), got {tuple(initial_state.shape)}.")
    if initial_state.device != device:
        raise ValueError(f"initial_state must be on device {device}, got {initial_state.device}.")
    return initial_state, has_initial_state


def qwen_gdn_chunk_gdr_avelang_v11_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
    use_mfma_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    block_v: int = BV,
    block_k: int | None = BK,
    fallback: str = "v10",
    use_clean_kernel: bool = True,
    use_runtime_chunk_loop: bool = False,
    use_update_mfma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not use_mfma_chunk_gdr or not prefer_optimized or not _is_v11_supported_chunk_gdr(k, w, u, g, chunk_size, block_v, block_k):
        return qwen_gdn_chunk_gdr_avelang_v10_vllm_layout(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=chunk_size,
            use_parallel_chunk_gdr=True,
            prefer_optimized=True,
            parallel_mode="chunk_vk" if fallback == "v10" else "vk",
            block_v=4,
            block_k=64,
        )

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(k, w, u, g, chunk_size)
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v11 MFMA path only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    initial_state_arg, has_initial_state = _validate_initial_state_v11(initial_state, device=k.device)

    num_chunks = _num_chunks(num_tokens, chunk_size)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    grid_size = 8 * 8
    if use_runtime_chunk_loop:
        raise RuntimeError("v11 runtime chunk loop experiment failed correctness and is disabled; see v11_high_perf_status_report.md.")
    if use_update_mfma:
        kernel = _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update
    else:
        kernel = (
            _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_12_clean_scalar_update
            if use_clean_kernel
            else _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_10_scalar_update
        )
    kernel[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k,
        w,
        u,
        g,
        initial_state_arg,
        h,
        vn,
        final_state,
        num_tokens,
        num_chunks,
        has_initial_state,
    )
    return h, vn, final_state


def qwen_gdn_chunked_avelang_v11_mfma_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
    use_mfma_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    block_v: int = BV,
    block_k: int | None = BK,
    use_parallel_chunk_o: bool = True,
    chunk_o_block_v: int = 4,
    chunk_o_block_k: int | None = 16,
    use_clean_kernel: bool = True,
    use_runtime_chunk_loop: bool = False,
    use_update_mfma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_chunk_size(chunk_size)
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return qwen_gdn_chunked_avelang_v10_vllm_layout(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
        )
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _validate_initial_state_v11(initial_state, device=q.device)

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size, prefer_optimized=True)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v11_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_mfma_chunk_gdr=use_mfma_chunk_gdr,
        prefer_optimized=prefer_optimized,
        block_v=block_v,
        block_k=block_k,
        use_clean_kernel=use_clean_kernel,
        use_runtime_chunk_loop=use_runtime_chunk_loop,
        use_update_mfma=use_update_mfma,
    )
    output = qwen_gdn_chunk_o_avelang_v9_vllm_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_o=use_parallel_chunk_o,
        chunk_o_parallel_mode="vk",
        chunk_o_block_v=chunk_o_block_v,
        chunk_o_block_k=chunk_o_block_k,
    )
    return g_cumsum, output, a_solved, h, final_state


def qwen_gdn_chunked_avelang_v11_mfma_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
    use_mfma_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    block_v: int = BV,
    block_k: int | None = BK,
    use_parallel_chunk_o: bool = True,
    chunk_o_block_v: int = 4,
    chunk_o_block_k: int | None = 16,
    use_clean_kernel: bool = True,
    use_runtime_chunk_loop: bool = False,
    use_update_mfma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v11_mfma_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        use_mfma_chunk_gdr=use_mfma_chunk_gdr,
        prefer_optimized=prefer_optimized,
        block_v=block_v,
        block_k=block_k,
        use_parallel_chunk_o=use_parallel_chunk_o,
        chunk_o_block_v=chunk_o_block_v,
        chunk_o_block_k=chunk_o_block_k,
        use_clean_kernel=use_clean_kernel,
        use_runtime_chunk_loop=use_runtime_chunk_loop,
        use_update_mfma=use_update_mfma,
    )
    return output, final_state


__all__ = [
    "_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_10_scalar_update",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_12_clean_scalar_update",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update",
    "qwen_gdn_chunk_gdr_avelang_v11_mfma_layout",
    "qwen_gdn_chunked_avelang_v11_mfma_layout_full",
    "qwen_gdn_chunked_avelang_v11_mfma_layout",
]

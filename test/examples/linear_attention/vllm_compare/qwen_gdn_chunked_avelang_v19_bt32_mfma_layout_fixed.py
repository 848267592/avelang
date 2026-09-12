"""Qwen GDN v19 BT32 MFMA full-path prototype.

v19 opens the larger chunk path after v18 made BT32 solve cheap.  It keeps the
fixed Qwen3Next TP4 per-rank target:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16 q/k/v, FP32 g/beta/output/final_state
    chunk_size=BT=32, BV=16

Unsupported shapes raise ValueError.  There is no fallback to v6 for the
performance path.
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
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import qwen_gdn_chunk_o_avelang_v13_mfma_layout
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout

BT = 32
BT_SUB = 16
BV = 16


@avelang.jit
def _qwen_gdn_w_bf16_kernel_v19_bt32_mfma(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    row_block = program_id % 2
    tile_idx = (program_id // 2) % 8
    value_head_idx = (program_id // 16) % 8
    chunk_idx = program_id // 128

    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT
    row_base = row_block * BT_SUB
    col_base = tile_idx * 16

    a0_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a1_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b0_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b1_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a0_vec = al.view(a0_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    a1_vec = al.view(a1_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b0_vec = al.view(b0_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b1_vec = al.view(b1_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    for rep in al.range(4):
        idx = lane + rep * 64
        row = idx // BT_SUB
        col = idx - row * BT_SUB
        token_idx = chunk_start + row_base + row
        source_idx0 = chunk_start + col
        source_idx1 = chunk_start + BT_SUB + col
        a_value0 = a[0, token_idx, value_head_idx, col]
        a_value1 = a[0, token_idx, value_head_idx, BT_SUB + col]
        beta_value0 = beta[0, source_idx0, value_head_idx]
        beta_value1 = beta[0, source_idx1, value_head_idx]
        g_value0 = g[0, source_idx0, value_head_idx]
        g_value1 = g[0, source_idx1, value_head_idx]
        a0_bf16[row, col] = al.convert(a_value0 * beta_value0 * al.exp(g_value0), al.bf16)
        a1_bf16[row, col] = al.convert(a_value1 * beta_value1 * al.exp(g_value1), al.bf16)
        b0_bf16[row, col] = k[0, source_idx0, key_head_idx, col_base + row]
        b1_bf16[row, col] = k[0, source_idx1, key_head_idx, col_base + row]

    al.syncthreads()

    out_acc = al.full((4,), 0.0, al.f32)
    if lane_group == 0:
        a_words0 = a0_vec[lane_col, 0]
        b_words0 = b0_vec[lane_col, 0]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[0], b_frag0[0], out_acc)
    if lane_group == 1:
        a_words0 = a0_vec[lane_col, 0]
        b_words0 = b0_vec[lane_col, 0]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[1], b_frag0[1], out_acc)
    if lane_group == 2:
        a_words0 = a0_vec[lane_col, 1]
        b_words0 = b0_vec[lane_col, 1]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[0], b_frag0[0], out_acc)
    if lane_group == 3:
        a_words0 = a0_vec[lane_col, 1]
        b_words0 = b0_vec[lane_col, 1]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[1], b_frag0[1], out_acc)
    if lane_group == 0:
        a_words1 = a1_vec[lane_col, 0]
        b_words1 = b1_vec[lane_col, 0]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[0], b_frag1[0], out_acc)
    if lane_group == 1:
        a_words1 = a1_vec[lane_col, 0]
        b_words1 = b1_vec[lane_col, 0]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[1], b_frag1[1], out_acc)
    if lane_group == 2:
        a_words1 = a1_vec[lane_col, 1]
        b_words1 = b1_vec[lane_col, 1]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[0], b_frag1[0], out_acc)
    if lane_group == 3:
        a_words1 = a1_vec[lane_col, 1]
        b_words1 = b1_vec[lane_col, 1]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[1], b_frag1[1], out_acc)

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        token_idx_out = chunk_start + row_base + token_offset
        out_col = col_base + lane_col
        correction = al.convert(0.0, al.f32)
        for s0 in al.range(BT_SUB):
            source_idx_corr = chunk_start + s0
            a_fp32 = a[0, token_idx_out, value_head_idx, s0] * beta[0, source_idx_corr, value_head_idx] * al.exp(
                g[0, source_idx_corr, value_head_idx]
            )
            a_staged = al.convert(a0_bf16[token_offset, s0], al.f32)
            b_value = al.convert(k[0, source_idx_corr, key_head_idx, out_col], al.f32)
            correction = correction + (a_fp32 - a_staged) * b_value
        for s1 in al.range(BT_SUB):
            source_idx_corr1 = chunk_start + BT_SUB + s1
            a_fp32_1 = a[0, token_idx_out, value_head_idx, BT_SUB + s1] * beta[
                0,
                source_idx_corr1,
                value_head_idx,
            ] * al.exp(g[0, source_idx_corr1, value_head_idx])
            a_staged_1 = al.convert(a1_bf16[token_offset, s1], al.f32)
            b_value_1 = al.convert(k[0, source_idx_corr1, key_head_idx, out_col], al.f32)
            correction = correction + (a_fp32_1 - a_staged_1) * b_value_1
        w[0, token_idx_out, value_head_idx, out_col] = out_acc[r] + correction


@avelang.jit
def _qwen_gdn_u_bf16_kernel_v19_bt32_mfma(
    v_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    v = al.make_tensor(
        v_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    row_block = program_id % 2
    tile_idx = (program_id // 2) % 8
    value_head_idx = (program_id // 16) % 8
    chunk_idx = program_id // 128

    chunk_start = chunk_idx * BT
    row_base = row_block * BT_SUB
    col_base = tile_idx * 16

    a0_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a1_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b0_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b1_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a0_vec = al.view(a0_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    a1_vec = al.view(a1_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b0_vec = al.view(b0_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b1_vec = al.view(b1_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    for rep in al.range(4):
        idx = lane + rep * 64
        row = idx // BT_SUB
        col = idx - row * BT_SUB
        token_idx = chunk_start + row_base + row
        source_idx0 = chunk_start + col
        source_idx1 = chunk_start + BT_SUB + col
        a_value0 = a[0, token_idx, value_head_idx, col]
        a_value1 = a[0, token_idx, value_head_idx, BT_SUB + col]
        beta_value0 = beta[0, source_idx0, value_head_idx]
        beta_value1 = beta[0, source_idx1, value_head_idx]
        a0_bf16[row, col] = al.convert(a_value0 * beta_value0, al.bf16)
        a1_bf16[row, col] = al.convert(a_value1 * beta_value1, al.bf16)
        b0_bf16[row, col] = v[0, source_idx0, value_head_idx, col_base + row]
        b1_bf16[row, col] = v[0, source_idx1, value_head_idx, col_base + row]

    al.syncthreads()

    out_acc = al.full((4,), 0.0, al.f32)
    if lane_group == 0:
        a_words0 = a0_vec[lane_col, 0]
        b_words0 = b0_vec[lane_col, 0]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[0], b_frag0[0], out_acc)
    if lane_group == 1:
        a_words0 = a0_vec[lane_col, 0]
        b_words0 = b0_vec[lane_col, 0]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[1], b_frag0[1], out_acc)
    if lane_group == 2:
        a_words0 = a0_vec[lane_col, 1]
        b_words0 = b0_vec[lane_col, 1]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[0], b_frag0[0], out_acc)
    if lane_group == 3:
        a_words0 = a0_vec[lane_col, 1]
        b_words0 = b0_vec[lane_col, 1]
        a_frag0 = al.view(a_words0, al.Tensor((2, 4, 1), al.bf16))
        b_frag0 = al.view(b_words0, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag0[1], b_frag0[1], out_acc)
    if lane_group == 0:
        a_words1 = a1_vec[lane_col, 0]
        b_words1 = b1_vec[lane_col, 0]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[0], b_frag1[0], out_acc)
    if lane_group == 1:
        a_words1 = a1_vec[lane_col, 0]
        b_words1 = b1_vec[lane_col, 0]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[1], b_frag1[1], out_acc)
    if lane_group == 2:
        a_words1 = a1_vec[lane_col, 1]
        b_words1 = b1_vec[lane_col, 1]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[0], b_frag1[0], out_acc)
    if lane_group == 3:
        a_words1 = a1_vec[lane_col, 1]
        b_words1 = b1_vec[lane_col, 1]
        a_frag1 = al.view(a_words1, al.Tensor((2, 4, 1), al.bf16))
        b_frag1 = al.view(b_words1, al.Tensor((2, 4, 1), al.bf16))
        out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag1[1], b_frag1[1], out_acc)

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        token_idx_out = chunk_start + row_base + token_offset
        out_col = col_base + lane_col
        correction = al.convert(0.0, al.f32)
        for s0 in al.range(BT_SUB):
            source_idx_corr = chunk_start + s0
            a_fp32 = a[0, token_idx_out, value_head_idx, s0] * beta[0, source_idx_corr, value_head_idx]
            a_staged = al.convert(a0_bf16[token_offset, s0], al.f32)
            b_value = al.convert(v[0, source_idx_corr, value_head_idx, out_col], al.f32)
            correction = correction + (a_fp32 - a_staged) * b_value
        for s1 in al.range(BT_SUB):
            source_idx_corr1 = chunk_start + BT_SUB + s1
            a_fp32_1 = a[0, token_idx_out, value_head_idx, BT_SUB + s1] * beta[
                0,
                source_idx_corr1,
                value_head_idx,
            ]
            a_staged_1 = al.convert(a1_bf16[token_offset, s1], al.f32)
            b_value_1 = al.convert(v[0, source_idx_corr1, value_head_idx, out_col], al.f32)
            correction = correction + (a_fp32_1 - a_staged_1) * b_value_1
        u[0, token_idx_out, value_head_idx, out_col] = out_acc[r] + correction


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_8wave_mfma(
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

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    token_half_id = wave_id // 4
    k_quarter_id = wave_id - token_half_id * 4

    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((BV, 128), al.bf16)
    w_bf16 = al.make_shared((BT, 128), al.bf16)
    v_decay0_t = al.make_shared((BV, BT_SUB), al.bf16)
    v_decay1_t = al.make_shared((BV, BT_SUB), al.bf16)
    k0_t = al.make_shared((128, BT_SUB), al.bf16)
    k1_t = al.make_shared((128, BT_SUB), al.bf16)
    pred_partial = al.make_shared((2, 4, BT_SUB, BV), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    vdecay0_vec = al.view(v_decay0_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    vdecay1_vec = al.view(v_decay1_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    k0_vec = al.view(k0_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))
    k1_vec = al.view(k1_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    if token_half_id == 0:
        for rep in al.range(8):
            idx = lane + rep * 64
            vv = idx // 32
            kk_local = idx - vv * 32
            kk = kk_local + k_quarter_id * 32
            global_v = value_base + vv
            if has_initial_state:
                state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            else:
                state[vv, kk] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        last_token = chunk_start + 31
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        if token_half_id == 0:
            for rep_h in al.range(8):
                idx_h = lane + rep_h * 64
                vv_h = idx_h // 32
                kk_local_h = idx_h - vv_h * 32
                kk_h = kk_local_h + k_quarter_id * 32
                global_v_h = value_base + vv_h
                h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]

        for rep_all in al.range(8):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 32
            kk_local_all = idx_all - row_all * 32
            col_all = kk_local_all + k_quarter_id * 32
            token_offset = token_half_id * BT_SUB + row_all
            token_idx = chunk_start + token_offset
            w_bf16[token_offset, col_all] = al.convert(w[0, token_idx, value_head_idx, col_all], al.bf16)
            if token_half_id == 0:
                state_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
                k0_t[col_all, row_all] = k[0, token_idx, key_head_idx, col_all]
            if token_half_id == 1:
                k1_t[col_all, row_all] = k[0, token_idx, key_head_idx, col_all]

        al.syncthreads()

        pred_acc = al.full((4,), 0.0, al.f32)
        k_vec32 = lane_group + k_quarter_id * 4
        a_words = w_vec[lane_col + token_half_id * BT_SUB, k_vec32]
        b_words = state_vec[lane_col, k_vec32]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

        for r_pred in al.range(4):
            token_offset_pred = lane_group * 4 + r_pred
            value_offset_pred = lane_col
            pred_partial[token_half_id, k_quarter_id, token_offset_pred, value_offset_pred] = pred_acc[r_pred]

        al.syncthreads()

        if k_quarter_id == 0:
            for r in al.range(4):
                token_offset_local = lane_group * 4 + r
                token_offset_abs = token_half_id * BT_SUB + token_offset_local
                token_idx_v = chunk_start + token_offset_abs
                value_offset = lane_col
                global_v = value_base + value_offset
                pred_value = (
                    pred_partial[token_half_id, 0, token_offset_local, value_offset]
                    + pred_partial[token_half_id, 1, token_offset_local, value_offset]
                    + pred_partial[token_half_id, 2, token_offset_local, value_offset]
                    + pred_partial[token_half_id, 3, token_offset_local, value_offset]
                )
                v_new = u[0, token_idx_v, value_head_idx, global_v] - pred_value
                vn[0, token_idx_v, value_head_idx, global_v] = v_new
                decay = al.exp(g_last - g[0, token_idx_v, value_head_idx])
                if token_half_id == 0:
                    v_decay0_t[value_offset, token_offset_local] = al.convert(v_new * decay, al.bf16)
                if token_half_id == 1:
                    v_decay1_t[value_offset, token_offset_local] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for local_tile in al.range(2):
            global_tile = k_quarter_id * 2 + local_tile
            base_k = global_tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if token_half_id == 0:
                if lane_group == 0:
                    a_words_u = vdecay0_vec[lane_col, 0]
                    b_words_u = k0_vec[base_k + lane_col, 0]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                if lane_group == 1:
                    a_words_u = vdecay0_vec[lane_col, 0]
                    b_words_u = k0_vec[base_k + lane_col, 0]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
                if lane_group == 2:
                    a_words_u = vdecay0_vec[lane_col, 1]
                    b_words_u = k0_vec[base_k + lane_col, 1]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                if lane_group == 3:
                    a_words_u = vdecay0_vec[lane_col, 1]
                    b_words_u = k0_vec[base_k + lane_col, 1]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
            if token_half_id == 1:
                if lane_group == 0:
                    a_words_u = vdecay1_vec[lane_col, 0]
                    b_words_u = k1_vec[base_k + lane_col, 0]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                if lane_group == 1:
                    a_words_u = vdecay1_vec[lane_col, 0]
                    b_words_u = k1_vec[base_k + lane_col, 0]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
                if lane_group == 2:
                    a_words_u = vdecay1_vec[lane_col, 1]
                    b_words_u = k1_vec[base_k + lane_col, 1]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                if lane_group == 3:
                    a_words_u = vdecay1_vec[lane_col, 1]
                    b_words_u = k1_vec[base_k + lane_col, 1]
                    a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                    acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

            out_col = base_k + lane_col
            if token_half_id == 0:
                for r_up in al.range(4):
                    vv_up = lane_group * 4 + r_up
                    state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc[r_up]

            al.syncthreads()

            if token_half_id == 1:
                for r_up_1 in al.range(4):
                    vv_up_1 = lane_group * 4 + r_up_1
                    state[vv_up_1, out_col] = state[vv_up_1, out_col] + acc[r_up_1]

            al.syncthreads()

    if token_half_id == 0:
        for rep_final in al.range(8):
            idx_final = lane + rep_final * 64
            vv_final = idx_final // 32
            kk_local_final = idx_final - vv_final * 32
            kk_final = kk_local_final + k_quarter_id * 32
            global_v_final = value_base + vv_final
            final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma(
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
    w_sub_bf16 = al.make_shared((BT_SUB, 128), al.bf16)
    v_decay0_t = al.make_shared((BV, BT_SUB), al.bf16)
    v_decay1_t = al.make_shared((BV, BT_SUB), al.bf16)
    k0_t = al.make_shared((128, BT_SUB), al.bf16)
    k1_t = al.make_shared((128, BT_SUB), al.bf16)
    pred_partial = al.make_shared((4, BT_SUB, BV), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_sub_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    vdecay0_vec = al.view(v_decay0_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    vdecay1_vec = al.view(v_decay1_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    k0_vec = al.view(k0_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))
    k1_vec = al.view(k1_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

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

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        last_token = chunk_start + 31
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

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
            w_sub_bf16[row_all, col_all] = al.convert(w[0, token_idx, value_head_idx, col_all], al.bf16)
            k0_t[col_all, row_all] = k[0, token_idx, key_head_idx, col_all]

        al.syncthreads()

        pred_acc = al.full((4,), 0.0, al.f32)
        k_vec32 = lane_group + wave_id * 4
        a_words = w_vec[lane_col, k_vec32]
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
                token_idx_v = chunk_start + token_offset
                global_v = value_base + value_offset
                pred_value = (
                    pred_partial[0, token_offset, value_offset]
                    + pred_partial[1, token_offset, value_offset]
                    + pred_partial[2, token_offset, value_offset]
                    + pred_partial[3, token_offset, value_offset]
                )
                v_new = u[0, token_idx_v, value_head_idx, global_v] - pred_value
                vn[0, token_idx_v, value_head_idx, global_v] = v_new
                decay = al.exp(g_last - g[0, token_idx_v, value_head_idx])
                v_decay0_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for rep_all_1 in al.range(8):
            idx_all_1 = lane + rep_all_1 * 64
            row_all_1 = idx_all_1 // 32
            kk_local_all_1 = idx_all_1 - row_all_1 * 32
            col_all_1 = kk_local_all_1 + wave_id * 32
            token_idx_1 = chunk_start + BT_SUB + row_all_1
            w_sub_bf16[row_all_1, col_all_1] = al.convert(w[0, token_idx_1, value_head_idx, col_all_1], al.bf16)
            k1_t[col_all_1, row_all_1] = k[0, token_idx_1, key_head_idx, col_all_1]

        al.syncthreads()

        pred_acc_1 = al.full((4,), 0.0, al.f32)
        a_words_1 = w_vec[lane_col, k_vec32]
        b_words_1 = state_vec[lane_col, k_vec32]
        a_frag_1 = al.view(a_words_1, al.Tensor((2, 4, 1), al.bf16))
        b_frag_1 = al.view(b_words_1, al.Tensor((2, 4, 1), al.bf16))
        pred_acc_1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_1[0], b_frag_1[0], pred_acc_1)
        pred_acc_1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_1[1], b_frag_1[1], pred_acc_1)

        for r_pred_1 in al.range(4):
            token_offset_pred_1 = lane_group * 4 + r_pred_1
            value_offset_pred_1 = lane_col
            pred_partial[wave_id, token_offset_pred_1, value_offset_pred_1] = pred_acc_1[r_pred_1]

        al.syncthreads()

        if wave_id == 0:
            for r1 in al.range(4):
                token_offset_1 = lane_group * 4 + r1
                value_offset_1 = lane_col
                token_idx_v_1 = chunk_start + BT_SUB + token_offset_1
                global_v_1 = value_base + value_offset_1
                pred_value_1 = (
                    pred_partial[0, token_offset_1, value_offset_1]
                    + pred_partial[1, token_offset_1, value_offset_1]
                    + pred_partial[2, token_offset_1, value_offset_1]
                    + pred_partial[3, token_offset_1, value_offset_1]
                )
                v_new_1 = u[0, token_idx_v_1, value_head_idx, global_v_1] - pred_value_1
                vn[0, token_idx_v_1, value_head_idx, global_v_1] = v_new_1
                decay_1 = al.exp(g_last - g[0, token_idx_v_1, value_head_idx])
                v_decay1_t[value_offset_1, token_offset_1] = al.convert(v_new_1 * decay_1, al.bf16)

        al.syncthreads()

        for local_tile in al.range(2):
            global_tile = wave_id * 2 + local_tile
            base_k = global_tile * 16
            acc0 = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u0 = vdecay0_vec[lane_col, 0]
                b_words_u0 = k0_vec[base_k + lane_col, 0]
                a_frag_u0 = al.view(a_words_u0, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u0[0], b_frag_u0[0], acc0)
            if lane_group == 1:
                a_words_u0 = vdecay0_vec[lane_col, 0]
                b_words_u0 = k0_vec[base_k + lane_col, 0]
                a_frag_u0 = al.view(a_words_u0, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u0[1], b_frag_u0[1], acc0)
            if lane_group == 2:
                a_words_u0 = vdecay0_vec[lane_col, 1]
                b_words_u0 = k0_vec[base_k + lane_col, 1]
                a_frag_u0 = al.view(a_words_u0, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u0[0], b_frag_u0[0], acc0)
            if lane_group == 3:
                a_words_u0 = vdecay0_vec[lane_col, 1]
                b_words_u0 = k0_vec[base_k + lane_col, 1]
                a_frag_u0 = al.view(a_words_u0, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u0[1], b_frag_u0[1], acc0)

            acc1 = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u1 = vdecay1_vec[lane_col, 0]
                b_words_u1 = k1_vec[base_k + lane_col, 0]
                a_frag_u1 = al.view(a_words_u1, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u1[0], b_frag_u1[0], acc1)
            if lane_group == 1:
                a_words_u1 = vdecay1_vec[lane_col, 0]
                b_words_u1 = k1_vec[base_k + lane_col, 0]
                a_frag_u1 = al.view(a_words_u1, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u1[1], b_frag_u1[1], acc1)
            if lane_group == 2:
                a_words_u1 = vdecay1_vec[lane_col, 1]
                b_words_u1 = k1_vec[base_k + lane_col, 1]
                a_frag_u1 = al.view(a_words_u1, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u1[0], b_frag_u1[0], acc1)
            if lane_group == 3:
                a_words_u1 = vdecay1_vec[lane_col, 1]
                b_words_u1 = k1_vec[base_k + lane_col, 1]
                a_frag_u1 = al.view(a_words_u1, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u1[1], b_frag_u1[1], acc1)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc0[r_up] + acc1[r_up]

        al.syncthreads()

    for rep_final in al.range(8):
        idx_final = lane + rep_final * 64
        vv_final = idx_final // 32
        kk_local_final = idx_final - vv_final * 32
        kk_final = kk_local_final + wave_id * 32
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


def _require_v19_target_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
) -> None:
    if tuple(q.shape) != (1, q.shape[1], 4, 128):
        raise ValueError("q must have shape [1,T,4,128].")
    num_tokens = q.shape[1]
    if tuple(k.shape) != (1, num_tokens, 4, 128):
        raise ValueError("k must have shape [1,T,4,128].")
    if tuple(v.shape) != (1, num_tokens, 8, 128):
        raise ValueError("v must have shape [1,T,8,128].")
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("g must have shape [1,T,8].")
    if beta is not None and tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("beta must have shape [1,T,8].")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("v19 BT32 requires q/k/v dtype torch.bfloat16.")
    if g.dtype != torch.float32:
        raise ValueError("v19 BT32 requires g dtype torch.float32.")
    if beta is not None and beta.dtype != torch.float32:
        raise ValueError("v19 BT32 requires beta dtype torch.float32.")


def _validate_initial_state_v19(initial_state: torch.Tensor | None, *, device: torch.device) -> tuple[torch.Tensor | None, bool]:
    has_initial_state = initial_state is not None
    if initial_state is None:
        return None, has_initial_state
    _require_fp32_cuda_contiguous("initial_state", initial_state)
    if tuple(initial_state.shape) != (1, 8, 128, 128):
        raise ValueError(f"initial_state must have shape (1, 8, 128, 128), got {tuple(initial_state.shape)}.")
    if initial_state.device != device:
        raise ValueError(f"initial_state must be on device {device}, got {initial_state.device}.")
    return initial_state, has_initial_state


def qwen_gdn_w_u_avelang_v19_bt32_mfma_layout(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("v19 BT32 w_u MFMA only supports chunk_size=32.")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("v19 BT32 w_u requires k/v dtype torch.bfloat16.")
    if g.dtype != torch.float32 or beta.dtype != torch.float32 or a_solved.dtype != torch.float32:
        raise ValueError("v19 BT32 w_u requires g/beta/a_solved dtype torch.float32.")
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta), ("a_solved", a_solved)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a CUDA/HIP device.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
    if tuple(k.shape) != (1, k.shape[1], 4, 128):
        raise ValueError("k must have shape [1,T,4,128].")
    num_tokens = k.shape[1]
    if tuple(v.shape) != (1, num_tokens, 8, 128):
        raise ValueError("v must have shape [1,T,8,128].")
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("g must have shape [1,T,8].")
    if tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("beta must have shape [1,T,8].")
    if tuple(a_solved.shape) != (1, num_tokens, 8, BT):
        raise ValueError("a_solved must have shape [1,T,8,32].")
    if num_tokens % BT != 0:
        raise ValueError("v19 BT32 w_u requires num_tokens divisible by 32.")

    num_chunks = _num_chunks(num_tokens, chunk_size)
    w = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
    u = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
    grid_size = num_chunks * 8 * 8 * 2
    _qwen_gdn_w_bf16_kernel_v19_bt32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k,
        g,
        beta,
        a_solved,
        w,
        num_tokens,
        num_chunks,
    )
    _qwen_gdn_u_bf16_kernel_v19_bt32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        v,
        beta,
        a_solved,
        u,
        num_tokens,
        num_chunks,
    )
    return w, u


def qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("v19 BT32 chunk_gdr MFMA only supports chunk_size=32.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(k, w, u, g, chunk_size)
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v19 BT32 chunk_gdr only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v19 BT32 chunk_gdr requires num_tokens divisible by 32.")
    initial_state_arg, has_initial_state = _validate_initial_state_v19(initial_state, device=k.device)

    num_chunks = _num_chunks(num_tokens, chunk_size)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    _qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma[lambda: ((64, 1, 1), (256, 1, 1))](
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


def qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_chunk_size(chunk_size)
    if chunk_size != BT:
        raise ValueError("v19 BT32 full MFMA only supports chunk_size=32.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v19_target_shape(q, k, v, g, beta)
    _validate_initial_state_v19(initial_state, device=q.device)
    if q.shape[1] % BT != 0:
        raise ValueError("v19 BT32 full MFMA requires num_tokens divisible by 32.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v19_bt32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output = qwen_gdn_chunk_o_avelang_v13_mfma_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
    )
    return g_cumsum, output, a_solved, h, final_state


def qwen_gdn_chunked_avelang_v19_bt32_mfma_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    return output, final_state


__all__ = [
    "_qwen_gdn_w_bf16_kernel_v19_bt32_mfma",
    "_qwen_gdn_u_bf16_kernel_v19_bt32_mfma",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_8wave_mfma",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v19_bt32_4wave_mfma",
    "qwen_gdn_w_u_avelang_v19_bt32_mfma_layout",
    "qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout",
    "qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_full",
    "qwen_gdn_chunked_avelang_v19_bt32_mfma_layout",
]

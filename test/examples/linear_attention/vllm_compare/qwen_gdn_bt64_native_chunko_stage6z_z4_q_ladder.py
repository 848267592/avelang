"""Stage 6Z Z4 Q-provenance ladder.

This file contains three independent arms forked from fixed Z2.  They are
diagnostic chunk-o bodies only and are not wired into X2 or production:

* Z4A vector-Q-only: keep all three Q producer passes, but use a legal
  ``raw_buffer_load_x4`` packet for each contiguous BF16x8 Q slice.
* Z4B partial-Q-residency: keep scalar Q loads, retain a 4 KiB Q tile, and
  let only score half 0 reuse it.  Score half 1 keeps the Z2 global reload.
* Z4C full-K32-Q-fusion: keep scalar Q loads and use one outer K32 loop so
  one Q slice feeds inter, score half 0, and score half 1 before phase reuse.

The arms deliberately do not change K/H/V-new/g/output, the WG256 mapping,
the MFMA32 instruction, or the BF16 output contract.  Z4C intentionally keeps
three FP32 accumulator vectors live; its resource result is part of the
experiment rather than something hidden by a fallback.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    _validate_stage6w_chunko_inputs,
)


BT = 64
BV = 64
BK = 32
WORKGROUP = 256
Z4_WORKGROUP_CONTRACT = 256


def _q_layout(num_tokens: int):
    return al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1))


def _vn_layout(num_tokens: int):
    return al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1))


def _h_layout(num_chunks: int):
    return al.make_layout(
        (1, num_chunks, H_V, V_DIM, K_DIM),
        (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1),
    )


def _g_layout(num_tokens: int):
    return al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1))


def _out_layout(num_tokens: int):
    return al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1))


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Z4A: vectorize only the Q global producer; keep three passes."""
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, V_DIM, K_DIM),
            (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )

    tid = al.thread_id(0)
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5
    wave_id = tid >> 6
    row_half = wave_id >> 1
    value_half = wave_id & 1
    program_id = al.block_id(0)
    v_block_idx = program_id % 2
    value_head_idx = (program_id // 2) % H_V
    chunk_idx = program_id // (2 * H_V)
    value_base = v_block_idx * BV
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx >> 1

    phase = al.make_shared((256, 32), al.bf16)
    phase_vec = al.view(phase, al.Tensor((256, 4, 4), al.i32))
    q_rsrc = al.amdgpu.make_rsrc(q, num_tokens * H_K * K_DIM * 2)
    zero = al.convert(0, al.i32)

    inter_acc = al.full((16,), 0.0, al.f32)

    # Phase A remains one complete Q pass.  Each of the 256 lanes loads eight
    # contiguous BF16 values, then applies the same FP32 scale and BF16 round
    # trip as fixed Z2 before writing phase.
    for k_stage in al.range(4):
        row = tid // 4
        col_base = (tid - row * 4) * 8
        q_offset = ((chunk_start + row) * 512 + key_head_idx * 128 + k_stage * BK + col_base) * 2
        packed_q = al.amdgpu.raw_buffer_load_x4(q_rsrc, zero, al.convert(q_offset, al.i32), 0)
        q_values = al.view(packed_q, al.Tensor((8,), al.bf16))
        for element in al.range(8):
            col = col_base + element
            phase[row, col] = al.convert(al.convert(q_values[element], al.f32) * scale, al.bf16)
            phase[64 + row, col] = h[0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            h_words = phase_vec[64 + value_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
        al.syncthreads()

    # Phase B keeps the two source-half Q passes.  Only the Q producer is
    # vectorized; K, score, g, barriers and all MFMA consumers are unchanged.
    for source_half in al.range(2):
        score_stage_base = source_half * 128
        score_acc = al.full((16,), 0.0, al.f32)
        for k_stage in al.range(4):
            row = tid // 4
            col_base = (tid - row * 4) * 8
            q_offset = ((chunk_start + row) * 512 + key_head_idx * 128 + k_stage * BK + col_base) * 2
            packed_q = al.amdgpu.raw_buffer_load_x4(q_rsrc, zero, al.convert(q_offset, al.i32), 0)
            q_values = al.view(packed_q, al.Tensor((8,), al.bf16))
            for element in al.range(8):
                col = col_base + element
                phase[score_stage_base + row, col] = al.convert(
                    al.convert(q_values[element], al.f32) * scale, al.bf16
                )
            for rep in al.range(4):
                idx = tid + rep * WORKGROUP
                k_row = idx // BK
                k_col = idx - k_row * BK
                phase[score_stage_base + 64 + k_row, k_col] = k[
                    0, chunk_start + source_half * 32 + k_row, key_head_idx, k_stage * BK + k_col
                ]
            al.syncthreads()

            for kt in al.range(2):
                word = kt * 2 + lane_group
                q_words = phase_vec[score_stage_base + row_half * 32 + lane_col, word]
                k_words = phase_vec[score_stage_base + 64 + lane_col, word]
                if value_half == 0:
                    q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                    k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc)
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc)
            al.syncthreads()

        if value_half == 0:
            for r in al.range(16):
                out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
                token_offset = row_half * 32 + lane_col
                source_offset = source_half * 32 + out_col
                score = al.convert(0.0, al.f32)
                if source_offset <= token_offset:
                    score = score_acc[r] * al.exp(
                        g[0, chunk_start + token_offset, value_head_idx]
                        - g[0, chunk_start + source_offset, value_head_idx]
                    )
                phase[token_offset * 2 + source_half, out_col] = al.convert(score, al.bf16)
        al.syncthreads()

    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[128 + value_offset * 2 + token_offset // 32, token_offset - (token_offset // 32) * 32] = vn[
            0, chunk_start + token_offset, value_head_idx, value_base + value_offset
        ]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[(row_half * 32 + lane_col) * 2 + source_half, word]
            v_words = phase_vec[128 + (value_half * 32 + lane_col) * 2 + source_half, word]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_offset = row_half * 32 + lane_col
        result = inter_acc[r] * al.exp(g[0, chunk_start + token_offset, value_head_idx]) + intra_acc[r]
        out[0, chunk_start + token_offset, value_head_idx, value_base + value_half * 32 + out_col] = al.convert(
            result, al.bf16
        )


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Z4B: scalar Q loads, with only score half 0 reusing a 4 KiB Q tile."""
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, V_DIM, K_DIM),
            (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )

    tid = al.thread_id(0)
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5
    wave_id = tid >> 6
    row_half = wave_id >> 1
    value_half = wave_id & 1
    program_id = al.block_id(0)
    v_block_idx = program_id % 2
    value_head_idx = (program_id // 2) % H_V
    chunk_idx = program_id // (2 * H_V)
    value_base = v_block_idx * BV
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx >> 1

    phase = al.make_shared((256, 32), al.bf16)
    phase_vec = al.view(phase, al.Tensor((256, 4, 4), al.i32))
    inter_acc = al.full((16,), 0.0, al.f32)
    score_acc0 = al.full((16,), 0.0, al.f32)
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            q_scaled = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                al.bf16,
            )
            phase[row, col] = q_scaled
            phase[64 + row, col] = h[0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            h_words = phase_vec[64 + value_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
        al.syncthreads()

        # The just-consumed Q K32 slice is still live in phase.  Replace only
        # the H rows with K half 0 and feed the same Q rows to the first score
        # consumer before advancing to the next K32 slice.  This removes one
        # global Q producer pass without allocating another full Q tile.
        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[64 + row, col] = k[0, chunk_start + row, key_head_idx, k_stage * BK + col]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            k_words = phase_vec[64 + lane_col, word]
            if value_half == 0:
                q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                score_acc0 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc0)
                score_acc0 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc0)
        al.syncthreads()

    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score_acc0[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[token_offset * 2, out_col] = al.convert(score, al.bf16)
    al.syncthreads()

    # Score half 1 retains the fixed-Z2 scalar Q producer and its separate
    # accumulator.  Its phase rows are independent of the half-0 result.
    score_acc1 = al.full((16,), 0.0, al.f32)
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[128 + row, col] = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                al.bf16,
            )
        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[128 + 64 + row, col] = k[
                0, chunk_start + 32 + row, key_head_idx, k_stage * BK + col
            ]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[128 + row_half * 32 + lane_col, word]
            k_words = phase_vec[128 + 64 + lane_col, word]
            if value_half == 0:
                q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                score_acc1 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc1)
                score_acc1 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc1)
        al.syncthreads()

    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = 32 + out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score_acc1[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[token_offset * 2 + 1, out_col] = al.convert(score, al.bf16)
    al.syncthreads()

    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[128 + value_offset * 2 + token_offset // 32, token_offset - (token_offset // 32) * 32] = vn[
            0, chunk_start + token_offset, value_head_idx, value_base + value_offset
        ]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[(row_half * 32 + lane_col) * 2 + source_half, word]
            v_words = phase_vec[128 + (value_half * 32 + lane_col) * 2 + source_half, word]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_offset = row_half * 32 + lane_col
        result = inter_acc[r] * al.exp(g[0, chunk_start + token_offset, value_head_idx]) + intra_acc[r]
        out[0, chunk_start + token_offset, value_head_idx, value_base + value_half * 32 + out_col] = al.convert(
            result, al.bf16
        )


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Z4C: one scalar Q pass feeds all three K32 consumers."""
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, V_DIM, K_DIM),
            (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )

    tid = al.thread_id(0)
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5
    wave_id = tid >> 6
    row_half = wave_id >> 1
    value_half = wave_id & 1
    program_id = al.block_id(0)
    v_block_idx = program_id % 2
    value_head_idx = (program_id // 2) % H_V
    chunk_idx = program_id // (2 * H_V)
    value_base = v_block_idx * BV
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx >> 1

    phase = al.make_shared((256, 32), al.bf16)
    phase_vec = al.view(phase, al.Tensor((256, 4, 4), al.i32))

    # The three accumulators are intentionally live together.  If the arm
    # creates an AGPR/VGPR cliff, that is the requested lifetime result.
    inter_acc = al.full((16,), 0.0, al.f32)
    score_acc0 = al.full((16,), 0.0, al.f32)
    score_acc1 = al.full((16,), 0.0, al.f32)

    # One K32 outer loop.  The Q slice is loaded once into rows [0,64), used
    # by inter, score half 0, and score half 1, then allowed to be overwritten.
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[row, col] = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                al.bf16,
            )
            phase[64 + row, col] = h[0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            h_words = phase_vec[64 + value_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
        # H rows are dead after inter.  Reuse them for K half 0.
        al.syncthreads()

        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[64 + row, col] = k[0, chunk_start + row, key_head_idx, k_stage * BK + col]
        al.syncthreads()
        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            k_words = phase_vec[64 + lane_col, word]
            if value_half == 0:
                q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                score_acc0 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc0)
                score_acc0 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc0)
        al.syncthreads()

        # Reuse the same K rows for source half 1.  The Q rows remain live.
        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[64 + row, col] = k[0, chunk_start + 32 + row, key_head_idx, k_stage * BK + col]
        al.syncthreads()
        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = phase_vec[row_half * 32 + lane_col, word]
            k_words = phase_vec[64 + lane_col, word]
            if value_half == 0:
                q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                score_acc1 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc1)
                score_acc1 = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc1)
        al.syncthreads()

    # Serialize both score halves only after all K32 reductions.  The causal
    # math and BF16 score materialization match fixed Z2.
    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source0 = out_col
            source1 = 32 + out_col
            score0 = al.convert(0.0, al.f32)
            score1 = al.convert(0.0, al.f32)
            if source0 <= token_offset:
                score0 = score_acc0[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source0, value_head_idx]
                )
            if source1 <= token_offset:
                score1 = score_acc1[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source1, value_head_idx]
                )
            phase[token_offset * 2, out_col] = al.convert(score0, al.bf16)
            phase[token_offset * 2 + 1, out_col] = al.convert(score1, al.bf16)
    al.syncthreads()

    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[128 + value_offset * 2 + token_offset // 32, token_offset - (token_offset // 32) * 32] = vn[
            0, chunk_start + token_offset, value_head_idx, value_base + value_offset
        ]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[(row_half * 32 + lane_col) * 2 + source_half, word]
            v_words = phase_vec[128 + (value_half * 32 + lane_col) * 2 + source_half, word]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_offset = row_half * 32 + lane_col
        result = inter_acc[r] * al.exp(g[0, chunk_start + token_offset, value_head_idx]) + intra_acc[r]
        out[0, chunk_start + token_offset, value_head_idx, value_base + value_half * 32 + out_col] = al.convert(
            result, al.bf16
        )


def _launch(
    kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    output: torch.Tensor,
    *,
    scale: float | None,
    chunk_size: int,
) -> None:
    if WORKGROUP != Z4_WORKGROUP_CONTRACT:
        raise RuntimeError("Stage 6Z Z4 shape contract violation: WG must remain 256")
    t, num_chunks = _validate_stage6w_chunko_inputs(q, k, vn, h, g, chunk_size=chunk_size)
    if (
        output.dtype != torch.bfloat16
        or tuple(output.shape) != tuple(vn.shape)
        or not output.is_cuda
        or not output.is_contiguous()
        or output.device != q.device
    ):
        raise ValueError("Stage 6Z Z4 output must be contiguous BF16 [1,T,8,128] on the input device.")
    if scale is None:
        scale = K_DIM ** -0.5
    kernel[lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))](
        q, k, vn, h, g, output, float(scale), t, num_chunks
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into(
    q, k, vn, h, g, output, *, scale: float | None = None, chunk_size: int = BT
) -> None:
    _launch(
        _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q,
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into(
    q, k, vn, h, g, output, *, scale: float | None = None, chunk_size: int = BT
) -> None:
    _launch(
        _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency,
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into(
    q, k, vn, h, g, output, *, scale: float | None = None, chunk_size: int = BT
) -> None:
    _launch(
        _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion,
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q(q, k, vn, h, g, *, scale=None, chunk_size=BT):
    output = torch.empty_like(vn)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into(
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency(q, k, vn, h, g, *, scale=None, chunk_size=BT):
    output = torch.empty_like(vn)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into(
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion(q, k, vn, h, g, *, scale=None, chunk_size=BT):
    output = torch.empty_like(vn)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into(
        q, k, vn, h, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q",
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency",
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into",
]

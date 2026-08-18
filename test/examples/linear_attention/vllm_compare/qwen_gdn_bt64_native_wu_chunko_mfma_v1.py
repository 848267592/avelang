"""Opt-in BT64 MFMA W/U and chunk-o stages for the gfx942 asm-v0 experiment.

The kernels deliberately reuse the v24/v14 64-lane 16x16x16 BF16 MFMA
microkernel.  BT64 is represented as four physical token-16 tiles: W/U sums
all four source tiles, while chunk-o visits the complete 4x4 lower-triangular
token-tile matrix.  The frozen asm recurrence and production v24 path are
not modified.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import _require_target


BT = 64
BT_SUB = 16
BV = 16
H_K = 4
H_V = 8
K_DIM = 128
V_DIM = 128


@avelang.jit
def _qwen_gdn_w_bf16_kernel_bt64_from_v24_mfma_v1(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 512, 512, 128, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 1024, 1024, 128, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    col_tile = program_id % 8
    row_tile = (program_id // 8) % 4
    value_head_idx = (program_id // 32) % 8
    chunk_idx = program_id // 256
    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT
    row_base = row_tile * BT_SUB
    col_base = col_tile * BT_SUB

    a_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a_vec = al.view(a_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b_vec = al.view(b_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    out_acc = al.full((4,), 0.0, al.f32)

    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(4):
            idx = lane + rep * 64
            row = idx // BT_SUB
            col = idx - row * BT_SUB
            token_idx = chunk_start + row_base + row
            source_idx = chunk_start + source_base + col
            a_value = a[0, token_idx, value_head_idx, source_base + col]
            coeff = a_value * beta[0, source_idx, value_head_idx] * al.exp(g[0, source_idx, value_head_idx])
            a_bf16[row, col] = al.convert(coeff, al.bf16)
            b_bf16[row, col] = k[0, source_idx, key_head_idx, col_base + row]
        al.syncthreads()

        if lane_group == 0:
            a_frag = al.view(a_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 1:
            a_frag = al.view(a_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        if lane_group == 2:
            a_frag = al.view(a_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 3:
            a_frag = al.view(a_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        al.syncthreads()

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        token_idx = chunk_start + row_base + token_offset
        out_col = col_base + lane_col
        correction = al.convert(0.0, al.f32)
        for source_offset in al.range(BT):
            source_idx = chunk_start + source_offset
            coeff = a[0, token_idx, value_head_idx, source_offset] * beta[0, source_idx, value_head_idx] * al.exp(
                g[0, source_idx, value_head_idx]
            )
            # The four MFMA contributions each used the BF16-rounded coefficient.
            # Reconstruct that same rounding here instead of retaining all four
            # shared tiles through the output correction.
            staged = al.convert(al.convert(coeff, al.bf16), al.f32)
            correction = correction + (coeff - staged) * al.convert(k[0, source_idx, key_head_idx, out_col], al.f32)
        w[0, token_idx, value_head_idx, out_col] = out_acc[r] + correction


@avelang.jit
def _qwen_gdn_u_bf16_kernel_bt64_from_v24_mfma_v1(
    v_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 1024, 1024, 128, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)))
    u = al.make_tensor(u_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 1024, 1024, 128, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    col_tile = program_id % 8
    row_tile = (program_id // 8) % 4
    value_head_idx = (program_id // 32) % 8
    chunk_idx = program_id // 256
    chunk_start = chunk_idx * BT
    row_base = row_tile * BT_SUB
    col_base = col_tile * BT_SUB

    a_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    b_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a_vec = al.view(a_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    b_vec = al.view(b_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    out_acc = al.full((4,), 0.0, al.f32)

    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(4):
            idx = lane + rep * 64
            row = idx // BT_SUB
            col = idx - row * BT_SUB
            token_idx = chunk_start + row_base + row
            source_idx = chunk_start + source_base + col
            coeff = a[0, token_idx, value_head_idx, source_base + col] * beta[0, source_idx, value_head_idx]
            a_bf16[row, col] = al.convert(coeff, al.bf16)
            b_bf16[row, col] = v[0, source_idx, value_head_idx, col_base + row]
        al.syncthreads()

        if lane_group == 0:
            a_frag = al.view(a_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 1:
            a_frag = al.view(a_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        if lane_group == 2:
            a_frag = al.view(a_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 3:
            a_frag = al.view(a_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        al.syncthreads()

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        token_idx = chunk_start + row_base + token_offset
        out_col = col_base + lane_col
        correction = al.convert(0.0, al.f32)
        for source_offset in al.range(BT):
            source_idx = chunk_start + source_offset
            coeff = a[0, token_idx, value_head_idx, source_offset] * beta[0, source_idx, value_head_idx]
            staged = al.convert(coeff, al.bf16)
            correction = correction + (coeff - al.convert(staged, al.f32)) * al.convert(v[0, source_idx, value_head_idx, out_col], al.f32)
        u[0, token_idx, value_head_idx, out_col] = out_acc[r] + correction


@avelang.jit
def _qwen_gdn_chunk_o_bf16_kernel_bt64_from_v24_mfma_v1(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    q = al.make_tensor(q_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 512, 512, 128, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 512, 512, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 1024, 1024, 128, 1)))
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout((1, num_chunks, 8, 128, 128), (num_chunks * 8 * 16384, 8 * 16384, 16384, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 1024, 1024, 128, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    out_tile = program_id % 4
    v_block_idx = (program_id // 4) % 8
    value_head_idx = (program_id // 32) % 8
    chunk_idx = program_id // 256
    out_base = out_tile * BT_SUB
    value_base = v_block_idx * BV
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2

    q_scaled_bf16 = al.make_shared((BT_SUB, 128), al.bf16)
    k_bf16 = al.make_shared((BT_SUB, 128), al.bf16)
    h_bf16 = al.make_shared((BV, 128), al.bf16)
    score_decay_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    vn_t_bf16 = al.make_shared((BV, BT_SUB), al.bf16)
    q_vec = al.view(q_scaled_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    h_vec = al.view(h_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    score_vec = al.view(score_decay_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    vn_vec = al.view(vn_t_bf16, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        token_idx = chunk_start + out_base + row
        q_scaled_bf16[row, col] = al.convert(al.convert(q[0, token_idx, key_head_idx, col], al.f32) * scale, al.bf16)
        h_bf16[row, col] = h[0, chunk_idx, value_head_idx, value_base + row, col]
    al.syncthreads()

    inter_acc = al.full((4,), 0.0, al.f32)
    for batch128 in al.range(4):
        vec_idx = lane_group + batch128 * 4
        a_frag = al.view(q_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(h_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], inter_acc)
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], inter_acc)

    intra_acc = al.full((4,), 0.0, al.f32)
    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(32):
            idx = lane + rep * 64
            row = idx // 128
            col = idx - row * 128
            k_bf16[row, col] = k[0, chunk_start + source_base + row, key_head_idx, col]
        for rep in al.range(4):
            idx = lane + rep * 64
            value_offset = idx // BT_SUB
            source_offset = idx - value_offset * BT_SUB
            vn_t_bf16[value_offset, source_offset] = al.convert(
                vn[0, chunk_start + source_base + source_offset, value_head_idx, value_base + value_offset], al.bf16
            )
        al.syncthreads()

        score_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            vec_idx = lane_group + batch128 * 4
            a_frag = al.view(q_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(k_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
            score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], score_acc)
            score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], score_acc)

        for r in al.range(4):
            token_offset = lane_group * 4 + r
            source_offset = lane_col
            output_abs = out_base + token_offset
            source_abs = source_base + source_offset
            score_value = al.convert(0.0, al.f32)
            if source_abs <= output_abs:
                score_value = score_acc[r] * al.exp(
                    g[0, chunk_start + output_abs, value_head_idx]
                    - g[0, chunk_start + source_abs, value_head_idx]
                )
            score_decay_bf16[token_offset, source_offset] = al.convert(score_value, al.bf16)
        al.syncthreads()

        if lane_group == 0:
            a_frag = al.view(score_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], intra_acc)
        if lane_group == 1:
            a_frag = al.view(score_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], intra_acc)
        if lane_group == 2:
            a_frag = al.view(score_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], intra_acc)
        if lane_group == 3:
            a_frag = al.view(score_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], intra_acc)
        al.syncthreads()

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        token_idx = chunk_start + out_base + token_offset
        out[0, token_idx, value_head_idx, value_base + lane_col] = (
            inter_acc[r] * al.exp(g[0, token_idx, value_head_idx]) + intra_acc[r]
        )


def _require_wu_bt64_inputs(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, a_solved: torch.Tensor
) -> tuple[int, int]:
    tensors = (("k", k, torch.bfloat16), ("v", v, torch.bfloat16), ("g", g, torch.float32), ("beta", beta, torch.float32), ("a_solved", a_solved, torch.float32))
    for name, tensor, dtype in tensors:
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA/HIP {dtype}.")
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on {k.device}.")
    num_tokens = k.shape[1]
    if tuple(k.shape) != (1, num_tokens, H_K, K_DIM) or tuple(v.shape) != (1, num_tokens, H_V, V_DIM):
        raise ValueError("BT64 W/U requires k=[1,T,4,128] and v=[1,T,8,128].")
    if tuple(g.shape) != (1, num_tokens, H_V) or tuple(beta.shape) != (1, num_tokens, H_V):
        raise ValueError("BT64 W/U requires g/beta=[1,T,8].")
    if tuple(a_solved.shape) != (1, num_tokens, H_V, BT) or num_tokens % BT:
        raise ValueError("BT64 W/U requires a_solved=[1,T,8,64] and T divisible by 64.")
    return num_tokens, _num_chunks(num_tokens, BT)


def qwen_gdn_w_u_bt64_from_v24_mfma_v1(
    k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, a_solved: torch.Tensor, *, chunk_size: int = BT
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native 64-thread BF16 MFMA W/U for the fixed BT64 asm ABI."""
    if chunk_size != BT:
        raise ValueError("native BT64 MFMA W/U only supports chunk_size=64.")
    num_tokens, num_chunks = _require_wu_bt64_inputs(k, v, g, beta, a_solved)
    w = torch.empty((1, num_tokens, H_V, K_DIM), dtype=torch.float32, device=k.device)
    u = torch.empty_like(w)
    grid_size = num_chunks * H_V * 4 * 8
    _qwen_gdn_w_bf16_kernel_bt64_from_v24_mfma_v1[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k, g, beta, a_solved, w, num_tokens, num_chunks
    )
    _qwen_gdn_u_bf16_kernel_bt64_from_v24_mfma_v1[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        v, beta, a_solved, u, num_tokens, num_chunks
    )
    return w, u


def _require_chunko_bt64_inputs(
    q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor, h_bf16: torch.Tensor, g: torch.Tensor
) -> tuple[int, int]:
    tensors = (("q", q, torch.bfloat16), ("k", k, torch.bfloat16), ("v_new", v_new, torch.float32), ("h_bf16", h_bf16, torch.bfloat16), ("g", g, torch.float32))
    for name, tensor, dtype in tensors:
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA/HIP {dtype}.")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}.")
    num_tokens = q.shape[1]
    num_chunks = _num_chunks(num_tokens, BT)
    if tuple(q.shape) != (1, num_tokens, H_K, K_DIM) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("native BT64 chunk_o requires q/k=[1,T,4,128].")
    if tuple(v_new.shape) != (1, num_tokens, H_V, V_DIM) or tuple(g.shape) != (1, num_tokens, H_V):
        raise ValueError("native BT64 chunk_o requires v_new=[1,T,8,128] and g=[1,T,8].")
    if tuple(h_bf16.shape) != (1, num_chunks, H_V, V_DIM, K_DIM) or num_tokens % BT:
        raise ValueError("native BT64 chunk_o requires BF16 h=[1,T/64,8,128,128] and T divisible by 64.")
    return num_tokens, num_chunks


def qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Native BT64 MFMA chunk-o consuming asm-v0 BF16 h directly."""
    if chunk_size != BT:
        raise ValueError("native BT64 MFMA chunk_o only supports chunk_size=64.")
    num_tokens, num_chunks = _require_chunko_bt64_inputs(q, k, v_new, h_bf16, g)
    if scale is None:
        scale = K_DIM ** -0.5
    out = torch.empty_like(v_new)
    grid_size = num_chunks * H_V * 8 * 4
    _qwen_gdn_chunk_o_bf16_kernel_bt64_from_v24_mfma_v1[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        q, k, v_new, h_bf16, g, out, float(scale), num_tokens, num_chunks
    )
    return out


def qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Opt-in Stage 3 graph; all non-W/U/chunk-o stages remain Stage 2 code."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=BT, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum, "a": a, "a_solved": a_solved, "w": w, "u": u,
        "h_bf16": h_bf16, "v_new": v_new, "final_state": final_state,
        "output_fp32": output_fp32, "output": output_fp32.to(q.dtype), "initial_state": h0,
    }


def qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    fallback: Callable[[], tuple[torch.Tensor, torch.Tensor | None]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    try:
        stages = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(
            q, k, v, g, beta, initial_state=initial_state, scale=scale
        )
    except ValueError:
        if fallback is not None:
            return fallback()
        raise
    return stages["output"], stages["final_state"] if output_final_state else None


__all__ = [
    "qwen_gdn_w_u_bt64_from_v24_mfma_v1",
    "qwen_gdn_chunk_o_bt64_from_v24_mfma_v1",
    "qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1",
    "qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages",
]

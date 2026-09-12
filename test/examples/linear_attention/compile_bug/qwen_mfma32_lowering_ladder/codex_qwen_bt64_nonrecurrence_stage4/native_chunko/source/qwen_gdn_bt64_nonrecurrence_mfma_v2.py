"""Opt-in Stage 4 native BT64 non-recurrence kernels for gfx942.

This module leaves the frozen asm-v0 recurrence and all production paths
untouched.  The first Stage 4 gate is KKT-S0: a direct 4x4 decomposition of
the BT64 token matrix into native 16x16 BF16 MFMA tiles.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0
from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
)
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import _require_target


BT = 64
BT_SUB = 16
H_K = 4
H_V = 8
K_DIM = 128
V_DIM = 128


@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * H_K * K_DIM, H_K * K_DIM, K_DIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    tile_id = program_id % 16
    value_head_idx = (program_id // 16) % H_V
    chunk_idx = program_id // (16 * H_V)
    row_tile = tile_id // 4
    col_tile = tile_id - row_tile * 4
    row_base = row_tile * BT_SUB
    col_base = col_tile * BT_SUB
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2

    row_k_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    col_k_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    row_k_vec = al.view(row_k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    col_k_vec = al.view(col_k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    acc = al.full((4,), 0.0, al.f32)

    # Upper-triangular programs only zero their output tile.  Lower and
    # diagonal programs execute the same proven MFMA16 dot pattern as v24.
    if row_tile >= col_tile:
        for rep in al.range(32):
            idx = lane + rep * 64
            row = idx // K_DIM
            col = idx - row * K_DIM
            row_k_bf16[row, col] = k[0, chunk_start + row_base + row, key_head_idx, col]
            col_k_bf16[row, col] = k[0, chunk_start + col_base + row, key_head_idx, col]
        al.syncthreads()

        for batch128 in al.range(4):
            vec_idx = lane_group + batch128 * 4
            a_words = row_k_vec[lane_col, vec_idx]
            b_words = col_k_vec[lane_col, vec_idx]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

    for r in al.range(4):
        token_offset = row_base + lane_group * 4 + r
        source_offset = col_base + lane_col
        token_idx = chunk_start + token_offset
        source_idx = chunk_start + source_offset
        value = al.convert(0.0, al.f32)
        if source_offset < token_offset:
            decay = al.exp(g[0, token_idx, value_head_idx] - g[0, source_idx, value_head_idx])
            value = beta[0, token_idx, value_head_idx] * acc[r] * decay
        out[0, token_idx, value_head_idx, source_offset] = value


def _require_kkt_bt64_inputs(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("Stage 4 BT64 KKT only supports chunk_size=64.")
    for name, tensor, dtype in (("k", k, torch.bfloat16), ("g", g, torch.float32), ("beta", beta, torch.float32)):
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA/HIP {dtype}.")
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on {k.device}.")
    num_tokens = k.shape[1]
    if tuple(k.shape) != (1, num_tokens, H_K, K_DIM):
        raise ValueError("Stage 4 BT64 KKT requires k=[1,T,4,128].")
    if tuple(g.shape) != (1, num_tokens, H_V) or tuple(beta.shape) != tuple(g.shape):
        raise ValueError("Stage 4 BT64 KKT requires g/beta=[1,T,8].")
    if num_tokens % BT:
        raise ValueError("Stage 4 BT64 KKT requires T divisible by 64.")
    return num_tokens, _num_chunks(num_tokens, BT)


def qwen_gdn_kkt_bt64_mfma_v2_s0(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Compute strict-lower BT64 KKT using native 16x16 MFMA tiles."""
    num_tokens, num_chunks = _require_kkt_bt64_inputs(k, g, beta, chunk_size)
    out = torch.empty((1, num_tokens, H_V, BT), dtype=torch.float32, device=k.device)
    grid_size = num_chunks * H_V * 16
    _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k, g, beta, out, num_tokens, num_chunks
    )
    return out


@avelang.jit
def _qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    use_residual_correction: al.constexpr,
    use_mfma_residual: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, num_tokens, H_V, K_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    col_tile = program_id % 8
    value_head_idx = (program_id // 8) % H_V
    chunk_idx = program_id // 64
    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT
    row_base = wave_id * BT_SUB
    col_base = col_tile * BT_SUB

    a_bf16 = al.make_shared((BT, BT_SUB), al.bf16)
    b_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a_vec = al.view(a_bf16, al.i32, al.make_layout((BT, 2, 4), (8, 4, 1)))
    b_vec = al.view(b_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    out_acc = al.full((4,), 0.0, al.f32)

    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(4):
            idx = tid + rep * 256
            row = idx // BT_SUB
            col = idx - row * BT_SUB
            token_idx = chunk_start + row
            source_idx = chunk_start + source_base + col
            coeff = a[0, token_idx, value_head_idx, source_base + col] * beta[0, source_idx, value_head_idx]
            coeff = coeff * al.exp(g[0, source_idx, value_head_idx])
            a_bf16[row, col] = al.convert(coeff, al.bf16)

        b_row = tid // BT_SUB
        b_col = tid - b_row * BT_SUB
        b_bf16[b_row, b_col] = k[0, chunk_start + source_base + b_col, key_head_idx, col_base + b_row]
        al.syncthreads()

        if lane_group == 0:
            a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 1:
            a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        if lane_group == 2:
            a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 3:
            a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        al.syncthreads()

        if use_mfma_residual:
            for rep_residual in al.range(4):
                idx_residual = tid + rep_residual * 256
                row_residual = idx_residual // BT_SUB
                col_residual = idx_residual - row_residual * BT_SUB
                token_idx_residual = chunk_start + row_residual
                source_idx_residual = chunk_start + source_base + col_residual
                coeff_residual = (
                    a[0, token_idx_residual, value_head_idx, source_base + col_residual]
                    * beta[0, source_idx_residual, value_head_idx]
                )
                coeff_residual = coeff_residual * al.exp(g[0, source_idx_residual, value_head_idx])
                coeff_staged = al.convert(al.convert(coeff_residual, al.bf16), al.f32)
                a_bf16[row_residual, col_residual] = al.convert(coeff_residual - coeff_staged, al.bf16)
            al.syncthreads()

            if lane_group == 0:
                a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
            if lane_group == 1:
                a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
            if lane_group == 2:
                a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
            if lane_group == 3:
                a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
            al.syncthreads()

    for r in al.range(4):
        token_offset = row_base + lane_group * 4 + r
        token_idx = chunk_start + token_offset
        out_col = col_base + lane_col
        result = out_acc[r]
        if use_residual_correction:
            correction = al.convert(0.0, al.f32)
            for source_offset in al.range(BT):
                source_idx = chunk_start + source_offset
                coeff = a[0, token_idx, value_head_idx, source_offset] * beta[0, source_idx, value_head_idx]
                coeff = coeff * al.exp(g[0, source_idx, value_head_idx])
                staged = al.convert(al.convert(coeff, al.bf16), al.f32)
                correction = correction + (coeff - staged) * al.convert(k[0, source_idx, key_head_idx, out_col], al.f32)
            result = result + correction
        w[0, token_idx, value_head_idx, out_col] = result


@avelang.jit
def _qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0(
    v_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    use_residual_correction: al.constexpr,
    use_mfma_residual: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)))
    u = al.make_tensor(u_ptr, al.f32, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    col_tile = program_id % 8
    value_head_idx = (program_id // 8) % H_V
    chunk_idx = program_id // 64
    chunk_start = chunk_idx * BT
    row_base = wave_id * BT_SUB
    col_base = col_tile * BT_SUB

    a_bf16 = al.make_shared((BT, BT_SUB), al.bf16)
    b_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    a_vec = al.view(a_bf16, al.i32, al.make_layout((BT, 2, 4), (8, 4, 1)))
    b_vec = al.view(b_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    out_acc = al.full((4,), 0.0, al.f32)

    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(4):
            idx = tid + rep * 256
            row = idx // BT_SUB
            col = idx - row * BT_SUB
            token_idx = chunk_start + row
            source_idx = chunk_start + source_base + col
            coeff = a[0, token_idx, value_head_idx, source_base + col] * beta[0, source_idx, value_head_idx]
            a_bf16[row, col] = al.convert(coeff, al.bf16)

        b_row = tid // BT_SUB
        b_col = tid - b_row * BT_SUB
        b_bf16[b_row, b_col] = v[0, chunk_start + source_base + b_col, value_head_idx, col_base + b_row]
        al.syncthreads()

        if lane_group == 0:
            a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 1:
            a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        if lane_group == 2:
            a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
        if lane_group == 3:
            a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
        al.syncthreads()

        if use_mfma_residual:
            for rep_residual in al.range(4):
                idx_residual = tid + rep_residual * 256
                row_residual = idx_residual // BT_SUB
                col_residual = idx_residual - row_residual * BT_SUB
                token_idx_residual = chunk_start + row_residual
                source_idx_residual = chunk_start + source_base + col_residual
                coeff_residual = (
                    a[0, token_idx_residual, value_head_idx, source_base + col_residual]
                    * beta[0, source_idx_residual, value_head_idx]
                )
                coeff_staged = al.convert(al.convert(coeff_residual, al.bf16), al.f32)
                a_bf16[row_residual, col_residual] = al.convert(coeff_residual - coeff_staged, al.bf16)
            al.syncthreads()

            if lane_group == 0:
                a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
            if lane_group == 1:
                a_frag = al.view(a_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
            if lane_group == 2:
                a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], out_acc)
            if lane_group == 3:
                a_frag = al.view(a_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                out_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], out_acc)
            al.syncthreads()

    for r in al.range(4):
        token_offset = row_base + lane_group * 4 + r
        token_idx = chunk_start + token_offset
        out_col = col_base + lane_col
        result = out_acc[r]
        if use_residual_correction:
            correction = al.convert(0.0, al.f32)
            for source_offset in al.range(BT):
                source_idx = chunk_start + source_offset
                coeff = a[0, token_idx, value_head_idx, source_offset] * beta[0, source_idx, value_head_idx]
                staged = al.convert(al.convert(coeff, al.bf16), al.f32)
                correction = correction + (coeff - staged) * al.convert(v[0, source_idx, value_head_idx, out_col], al.f32)
            result = result + correction
        u[0, token_idx, value_head_idx, out_col] = result


def qwen_gdn_w_u_bt64_mfma_v2_s0(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
    use_residual_correction: bool = True,
    use_mfma_residual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("Stage 4 W/U-S0 only supports chunk_size=64.")
    num_tokens, num_chunks = _require_kkt_bt64_inputs(k, g, beta, chunk_size)
    if v.dtype != torch.bfloat16 or not v.is_cuda or not v.is_contiguous() or v.device != k.device:
        raise ValueError("v must be contiguous CUDA/HIP BF16 on the same device as k.")
    if tuple(v.shape) != (1, num_tokens, H_V, V_DIM):
        raise ValueError("Stage 4 W/U-S0 requires v=[1,T,8,128].")
    if a_solved.dtype != torch.float32 or not a_solved.is_cuda or not a_solved.is_contiguous():
        raise ValueError("a_solved must be contiguous CUDA/HIP FP32.")
    if a_solved.device != k.device or tuple(a_solved.shape) != (1, num_tokens, H_V, BT):
        raise ValueError("Stage 4 W/U-S0 requires a_solved=[1,T,8,64] on the same device.")
    w = torch.empty((1, num_tokens, H_V, K_DIM), dtype=torch.float32, device=k.device)
    u = torch.empty_like(w)
    grid_size = num_chunks * H_V * 8
    _qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid_size, 1, 1), (256, 1, 1))](
        k, g, beta, a_solved, w, num_tokens, num_chunks, use_residual_correction, use_mfma_residual
    )
    _qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid_size, 1, 1), (256, 1, 1))](
        v, beta, a_solved, u, num_tokens, num_chunks, use_residual_correction, use_mfma_residual
    )
    return w, u


def qwen_gdn_w_u_bt64_mfma_v2_s1(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WU-S1: replace the scalar residual pass with a second BF16 residual MFMA."""
    return qwen_gdn_w_u_bt64_mfma_v2_s0(
        k,
        v,
        g,
        beta,
        a_solved,
        chunk_size=chunk_size,
        use_residual_correction=False,
        use_mfma_residual=True,
    )


def qwen_gdn_w_u_bt64_mfma_v2_no_correction_failed(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Historical failed WU-S1 candidate retained for diagnostic reproduction."""
    return qwen_gdn_w_u_bt64_mfma_v2_s0(
        k,
        v,
        g,
        beta,
        a_solved,
        chunk_size=chunk_size,
        use_residual_correction=False,
        use_mfma_residual=False,
    )


@avelang.jit
def _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0(
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
    q = al.make_tensor(q_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.f32, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout((1, num_chunks, H_V, V_DIM, K_DIM), (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % H_V
    chunk_idx = program_id // 64
    row_base = wave_id * BT_SUB
    value_base = v_block_idx * BT_SUB
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2

    q_scaled_bf16 = al.make_shared((BT, K_DIM), al.bf16)
    h_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    k_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    score_decay_bf16 = al.make_shared((4, BT_SUB, BT_SUB), al.bf16)
    vn_t_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    q_vec = al.view(q_scaled_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    h_vec = al.view(h_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    score_vec = al.view(score_decay_bf16, al.i32, al.make_layout((4, BT_SUB, 2, 4), (128, 8, 4, 1)))
    vn_vec = al.view(vn_t_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    for rep in al.range(32):
        idx = tid + rep * 256
        row = idx // K_DIM
        col = idx - row * K_DIM
        q_scaled_bf16[row, col] = al.convert(
            al.convert(q[0, chunk_start + row, key_head_idx, col], al.f32) * scale,
            al.bf16,
        )
    for rep in al.range(8):
        idx = tid + rep * 256
        row = idx // K_DIM
        col = idx - row * K_DIM
        h_bf16[row, col] = h[0, chunk_idx, value_head_idx, value_base + row, col]
    al.syncthreads()

    inter_acc = al.full((4,), 0.0, al.f32)
    for batch128 in al.range(4):
        vec_idx = lane_group + batch128 * 4
        a_frag = al.view(q_vec[row_base + lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(h_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], inter_acc)
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], inter_acc)

    intra_acc = al.full((4,), 0.0, al.f32)

    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(8):
            idx = tid + rep * 256
            row = idx // K_DIM
            col = idx - row * K_DIM
            k_bf16[row, col] = k[0, chunk_start + source_base + row, key_head_idx, col]
        value_offset = tid // BT_SUB
        source_offset = tid - value_offset * BT_SUB
        vn_t_bf16[value_offset, source_offset] = al.convert(
            vn[0, chunk_start + source_base + source_offset, value_head_idx, value_base + value_offset],
            al.bf16,
        )
        al.syncthreads()

        if source_tile < 4:
            score_acc = al.full((4,), 0.0, al.f32)
            for batch128_score in al.range(4):
                vec_idx_score = lane_group + batch128_score * 4
                a_frag_score = al.view(
                    q_vec[row_base + lane_col, vec_idx_score], al.Tensor((2, 4, 1), al.bf16)
                )
                b_frag_score = al.view(k_vec[lane_col, vec_idx_score], al.Tensor((2, 4, 1), al.bf16))
                score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                    a_frag_score[0], b_frag_score[0], score_acc
                )
                score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                    a_frag_score[1], b_frag_score[1], score_acc
                )

            for r_score in al.range(4):
                token_offset = lane_group * 4 + r_score
                output_abs = row_base + token_offset
                source_abs = source_base + lane_col
                score_value = al.convert(0.0, al.f32)
                if source_abs <= output_abs:
                    score_value = score_acc[r_score] * al.exp(
                        g[0, chunk_start + output_abs, value_head_idx]
                        - g[0, chunk_start + source_abs, value_head_idx]
                    )
                score_decay_bf16[wave_id, token_offset, lane_col] = al.convert(score_value, al.bf16)
        al.syncthreads()

        if source_tile < 4:
            if lane_group == 0:
                a_frag_intra = al.view(score_vec[wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
            if lane_group == 1:
                a_frag_intra = al.view(score_vec[wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
            if lane_group == 2:
                a_frag_intra = al.view(score_vec[wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
            if lane_group == 3:
                a_frag_intra = al.view(score_vec[wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
        al.syncthreads()

    for r_out in al.range(4):
        token_offset_out = row_base + lane_group * 4 + r_out
        token_idx_out = chunk_start + token_offset_out
        out[0, token_idx_out, value_head_idx, value_base + lane_col] = (
            inter_acc[r_out] * al.exp(g[0, token_idx_out, value_head_idx]) + intra_acc[r_out]
        )


def qwen_gdn_chunk_o_bt64_mfma_v2_s0(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    if chunk_size != BT:
        raise ValueError("Stage 4 chunk-o-S0 only supports chunk_size=64.")
    num_tokens = q.shape[1]
    num_chunks = _num_chunks(num_tokens, BT)
    for name, tensor, dtype in (
        ("q", q, torch.bfloat16),
        ("k", k, torch.bfloat16),
        ("v_new", v_new, torch.float32),
        ("h_bf16", h_bf16, torch.bfloat16),
        ("g", g, torch.float32),
    ):
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != q.device:
            raise ValueError(f"{name} must be contiguous CUDA/HIP {dtype} on {q.device}.")
    if tuple(q.shape) != (1, num_tokens, H_K, K_DIM) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Stage 4 chunk-o-S0 requires q/k=[1,T,4,128].")
    if tuple(v_new.shape) != (1, num_tokens, H_V, V_DIM) or tuple(g.shape) != (1, num_tokens, H_V):
        raise ValueError("Stage 4 chunk-o-S0 requires v_new=[1,T,8,128] and g=[1,T,8].")
    if tuple(h_bf16.shape) != (1, num_chunks, H_V, V_DIM, K_DIM) or num_tokens % BT:
        raise ValueError("Stage 4 chunk-o-S0 requires h_bf16=[1,T/64,8,128,128] and T divisible by 64.")
    if scale is None:
        scale = K_DIM ** -0.5
    output = torch.empty_like(v_new)
    grid_size = num_chunks * H_V * 8
    _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0[lambda: ((grid_size, 1, 1), (256, 1, 1))](
        q, k, v_new, h_bf16, g, output, float(scale), num_tokens, num_chunks
    )
    return output


def qwen_gdn_full_bt64_stage4_kkt_s0_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Stage 3 graph with only KKT replaced by Stage 4 KKT-S0."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Stage 4 graph after incremental KKT-S0 plus W/U-S0 integration."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s0(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Stage 4 graph with KKT-S0 and residual-MFMA WU-S1."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage4_all_s0_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Stage 4 graph with KKT-S0, WU-S1, and chunk-o-S0."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage4_all_s0(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    stages = qwen_gdn_full_bt64_stage4_all_s0_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale
    )
    return stages["output"], stages["final_state"] if output_final_state else None


def qwen_gdn_full_bt64_stage4_kkt_s0(
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
        stages = qwen_gdn_full_bt64_stage4_kkt_s0_stages(
            q, k, v, g, beta, initial_state=initial_state, scale=scale
        )
    except ValueError:
        if fallback is not None:
            return fallback()
        raise
    return stages["output"], stages["final_state"] if output_final_state else None


__all__ = [
    "_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0",
    "_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0",
    "_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0",
    "_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0",
    "qwen_gdn_kkt_bt64_mfma_v2_s0",
    "qwen_gdn_w_u_bt64_mfma_v2_s0",
    "qwen_gdn_w_u_bt64_mfma_v2_s1",
    "qwen_gdn_w_u_bt64_mfma_v2_no_correction_failed",
    "qwen_gdn_chunk_o_bt64_mfma_v2_s0",
    "qwen_gdn_full_bt64_stage4_kkt_s0_stages",
    "qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages",
    "qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages",
    "qwen_gdn_full_bt64_stage4_all_s0_stages",
    "qwen_gdn_full_bt64_stage4_all_s0",
    "qwen_gdn_full_bt64_stage4_kkt_s0",
]

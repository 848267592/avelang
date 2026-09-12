"""Qwen GDN v20 BT32 native 32x32 MFMA prototype.

v20 keeps the BT32 full path from v19, but replaces the BT32 KKT and w_u
stages with native 32x32x8 BF16 MFMA kernels.  The target is fixed to the
Qwen3Next TP4 per-rank shape:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16 q/k/v, FP32 g/beta/intermediates
    chunk_size=BT=32, BV=16

Unsupported shapes raise ValueError.  There is no fallback in the v20 wrappers.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    _validate_bf16_qkvgb,
    _validate_chunk_size,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import qwen_gdn_chunk_o_avelang_v13_mfma_layout
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout
from qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout,
)

BT = 32
BV = 16


@avelang.jit
def _qwen_gdn_mfma_32x32_smoke_kernel_v20(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    c_ptr: al.Pointer(al.f32),
):
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    c = al.make_tensor(c_ptr, al.f32, al.make_layout((32, 32), (32, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5

    a_vec = al.view(a, al.Tensor((32, 4, 4), al.i32))
    b_vec = al.view(b, al.Tensor((32, 4, 4), al.i32))
    a_smem = al.make_shared((64, 4), al.i32)
    b_smem = al.make_shared((64, 4), al.i32)
    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(2):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = a_vec[lane_col, k_vec]
        b_smem[lane] = b_vec[lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        c[lane_col, out_col] = acc[r]


@avelang.jit
def _qwen_gdn_w_bf16_kernel_v20_bt32_32x32_mfma(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5
    program_id = al.block_id(0)
    tile_idx = program_id % 4
    value_head_idx = (program_id // 4) % 8
    chunk_idx = program_id // 32

    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT
    col_base = tile_idx * 32

    a_bf16 = al.make_shared((32, 32), al.bf16)
    b_bf16 = al.make_shared((32, 32), al.bf16)
    a_vec = al.view(a_bf16, al.Tensor((32, 4, 4), al.i32))
    b_vec = al.view(b_bf16, al.Tensor((32, 4, 4), al.i32))
    a_smem = al.make_shared((64, 4), al.i32)
    b_smem = al.make_shared((64, 4), al.i32)

    for rep in al.range(16):
        idx = lane + rep * 64
        row = idx // 32
        col = idx - row * 32
        token_idx = chunk_start + row
        source_idx = chunk_start + col
        a_value = a[0, token_idx, value_head_idx, col]
        beta_value = beta[0, source_idx, value_head_idx]
        g_value = g[0, source_idx, value_head_idx]
        a_bf16[row, col] = al.convert(a_value * beta_value * al.exp(g_value), al.bf16)
        b_bf16[row, col] = k[0, source_idx, key_head_idx, col_base + row]

    al.syncthreads()

    acc = al.full((16,), 0.0, al.f32)
    for kt in al.range(2):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = a_vec[lane_col, k_vec]
        b_smem[lane] = b_vec[lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        token_offset = lane_col
        out_col_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_idx_out = chunk_start + token_offset
        out_col = col_base + out_col_offset
        correction = al.convert(0.0, al.f32)
        for s in al.range(BT):
            source_idx_corr = chunk_start + s
            a_fp32 = a[0, token_idx_out, value_head_idx, s] * beta[0, source_idx_corr, value_head_idx] * al.exp(
                g[0, source_idx_corr, value_head_idx]
            )
            a_staged = al.convert(a_bf16[token_offset, s], al.f32)
            b_value = al.convert(k[0, source_idx_corr, key_head_idx, out_col], al.f32)
            correction = correction + (a_fp32 - a_staged) * b_value
        w[0, token_idx_out, value_head_idx, out_col] = acc[r] + correction


@avelang.jit
def _qwen_gdn_u_bf16_kernel_v20_bt32_32x32_mfma(
    v_ptr: al.Pointer(al.bf16),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)))
    u = al.make_tensor(u_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5
    program_id = al.block_id(0)
    tile_idx = program_id % 4
    value_head_idx = (program_id // 4) % 8
    chunk_idx = program_id // 32

    chunk_start = chunk_idx * BT
    col_base = tile_idx * 32

    a_bf16 = al.make_shared((32, 32), al.bf16)
    b_bf16 = al.make_shared((32, 32), al.bf16)
    a_vec = al.view(a_bf16, al.Tensor((32, 4, 4), al.i32))
    b_vec = al.view(b_bf16, al.Tensor((32, 4, 4), al.i32))
    a_smem = al.make_shared((64, 4), al.i32)
    b_smem = al.make_shared((64, 4), al.i32)

    for rep in al.range(16):
        idx = lane + rep * 64
        row = idx // 32
        col = idx - row * 32
        token_idx = chunk_start + row
        source_idx = chunk_start + col
        a_value = a[0, token_idx, value_head_idx, col]
        beta_value = beta[0, source_idx, value_head_idx]
        a_bf16[row, col] = al.convert(a_value * beta_value, al.bf16)
        b_bf16[row, col] = v[0, source_idx, value_head_idx, col_base + row]

    al.syncthreads()

    acc = al.full((16,), 0.0, al.f32)
    for kt in al.range(2):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = a_vec[lane_col, k_vec]
        b_smem[lane] = b_vec[lane_col, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        token_offset = lane_col
        out_col_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_idx_out = chunk_start + token_offset
        out_col = col_base + out_col_offset
        correction = al.convert(0.0, al.f32)
        for s in al.range(BT):
            source_idx_corr = chunk_start + s
            a_fp32 = a[0, token_idx_out, value_head_idx, s] * beta[0, source_idx_corr, value_head_idx]
            a_staged = al.convert(a_bf16[token_offset, s], al.f32)
            b_value = al.convert(v[0, source_idx_corr, value_head_idx, out_col], al.f32)
            correction = correction + (a_fp32 - a_staged) * b_value
        u[0, token_idx_out, value_head_idx, out_col] = acc[r] + correction


@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_v20_bt32_32x32_mfma(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 31
    lane_group = lane >> 5
    program_id = al.block_id(0)
    value_head_idx = program_id % 8
    chunk_idx = program_id // 8
    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT

    a_vec = al.view(k, al.Tensor((1, num_tokens, 4, 16, 4), al.i32))
    a_smem = al.make_shared((64, 4), al.i32)
    b_smem = al.make_shared((64, 4), al.i32)
    acc = al.full((16,), 0.0, al.f32)

    for kt in al.range(8):
        k_vec = kt * 2 + lane_group
        a_smem[lane] = a_vec[0, chunk_start + lane_col, key_head_idx, k_vec]
        b_smem[lane] = a_vec[0, chunk_start + lane_col, key_head_idx, k_vec]

        al.syncthreads()

        a_words = a_smem[lane]
        b_words = b_smem[lane]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

        al.syncthreads()

    for r in al.range(16):
        token_offset = lane_col
        source_offset = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_idx = chunk_start + token_offset
        source_idx = chunk_start + source_offset
        value = al.convert(0.0, al.f32)
        if source_offset < token_offset:
            decay = al.exp(g[0, token_idx, value_head_idx] - g[0, source_idx, value_head_idx])
            value = beta[0, token_idx, value_head_idx] * acc[r] * decay
        out[0, token_idx, value_head_idx, source_offset] = value


def _require_v20_target_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor | None = None) -> None:
    if tuple(q.shape) != (1, q.shape[1], 4, 128):
        raise ValueError("v20 supports q shape [1,T,4,128] only.")
    num_tokens = q.shape[1]
    if tuple(k.shape) != (1, num_tokens, 4, 128):
        raise ValueError("v20 supports k shape [1,T,4,128] only.")
    if tuple(v.shape) != (1, num_tokens, 8, 128):
        raise ValueError("v20 supports v shape [1,T,8,128] only.")
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("v20 supports g shape [1,T,8] only.")
    if beta is not None and tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("v20 supports beta shape [1,T,8] only.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("v20 requires q/k/v dtype torch.bfloat16.")
    if g.dtype != torch.float32:
        raise ValueError("v20 requires g dtype torch.float32.")
    if beta is not None and beta.dtype != torch.float32:
        raise ValueError("v20 requires beta dtype torch.float32.")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a CUDA/HIP device.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")
    if beta is not None:
        if not beta.is_cuda or not beta.is_contiguous() or beta.device != q.device:
            raise ValueError("beta must be CUDA/HIP, contiguous, and on the same device as q.")


def _require_v20_kkt_inputs(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> int:
    if chunk_size != BT:
        raise ValueError("v20 KKT only supports chunk_size=32.")
    if k.dtype != torch.bfloat16 or g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("v20 KKT requires BF16 k and FP32 g/beta.")
    if not k.is_cuda or not g.is_cuda or not beta.is_cuda:
        raise ValueError("v20 KKT requires CUDA/HIP tensors.")
    if not k.is_contiguous() or not g.is_contiguous() or not beta.is_contiguous():
        raise ValueError("v20 KKT requires contiguous tensors.")
    if tuple(k.shape) != (1, k.shape[1], 4, 128):
        raise ValueError("k must have shape [1,T,4,128].")
    num_tokens = k.shape[1]
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("g must have shape [1,T,8].")
    if tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("beta must have shape [1,T,8].")
    if num_tokens % BT != 0:
        raise ValueError("v20 KKT requires num_tokens divisible by 32.")
    return num_tokens


def qwen_gdn_mfma_32x32_smoke_v20(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("smoke requires BF16 a/b.")
    if tuple(a.shape) != (32, 32) or tuple(b.shape) != (32, 32):
        raise ValueError("smoke requires a/b shape [32,32].")
    if not a.is_cuda or not b.is_cuda or not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("smoke requires CUDA/HIP contiguous tensors.")
    c = torch.empty((32, 32), dtype=torch.float32, device=a.device)
    _qwen_gdn_mfma_32x32_smoke_kernel_v20[lambda: ((1, 1, 1), (64, 1, 1))](a, b, c)
    return c


def qwen_gdn_kkt_avelang_v20_bt32_mfma_layout(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    num_tokens = _require_v20_kkt_inputs(k, g, beta, chunk_size)
    num_chunks = _num_chunks(num_tokens, chunk_size)
    out = torch.empty((1, num_tokens, 8, BT), dtype=torch.float32, device=k.device)
    grid_size = num_chunks * 8
    _qwen_gdn_kkt_bf16_kernel_v20_bt32_32x32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k,
        g,
        beta,
        out,
        num_tokens,
        num_chunks,
    )
    return out


def qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("v20 w_u only supports chunk_size=32.")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("v20 w_u requires k/v dtype torch.bfloat16.")
    if g.dtype != torch.float32 or beta.dtype != torch.float32 or a_solved.dtype != torch.float32:
        raise ValueError("v20 w_u requires g/beta/a_solved dtype torch.float32.")
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
        raise ValueError("v20 w_u requires num_tokens divisible by 32.")

    num_chunks = _num_chunks(num_tokens, chunk_size)
    w = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
    u = torch.empty((1, num_tokens, 8, 128), dtype=torch.float32, device=k.device)
    grid_size = num_chunks * 8 * 4
    _qwen_gdn_w_bf16_kernel_v20_bt32_32x32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k,
        g,
        beta,
        a_solved,
        w,
        num_tokens,
        num_chunks,
    )
    _qwen_gdn_u_bf16_kernel_v20_bt32_32x32_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        v,
        beta,
        a_solved,
        u,
        num_tokens,
        num_chunks,
    )
    return w, u


def qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_full(
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
        raise ValueError("v20 full only supports chunk_size=32.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v20_target_shape(q, k, v, g, beta)
    if initial_state is not None:
        if initial_state.dtype != torch.float32 or not initial_state.is_cuda or not initial_state.is_contiguous():
            raise ValueError("initial_state must be FP32 CUDA/HIP contiguous.")
        if tuple(initial_state.shape) != (1, 8, 128, 128):
            raise ValueError("initial_state must have shape [1,8,128,128].")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q.")
    if q.shape[1] % BT != 0:
        raise ValueError("v20 full requires num_tokens divisible by 32.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v20_bt32_mfma_layout(k, g_cumsum, beta, chunk_size=chunk_size)
    a_solved = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output = qwen_gdn_chunk_o_avelang_v13_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk_size)
    return g_cumsum, output, a_solved, h, final_state


def qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout(
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
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_full(
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
    "_qwen_gdn_mfma_32x32_smoke_kernel_v20",
    "_qwen_gdn_w_bf16_kernel_v20_bt32_32x32_mfma",
    "_qwen_gdn_u_bf16_kernel_v20_bt32_32x32_mfma",
    "_qwen_gdn_kkt_bf16_kernel_v20_bt32_32x32_mfma",
    "qwen_gdn_mfma_32x32_smoke_v20",
    "qwen_gdn_kkt_avelang_v20_bt32_mfma_layout",
    "qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout",
    "qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_full",
    "qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout",
]

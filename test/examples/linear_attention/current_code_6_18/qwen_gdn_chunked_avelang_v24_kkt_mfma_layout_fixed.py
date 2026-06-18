"""Qwen GDN v24 BT16 native KKT MFMA path.

v24 keeps the current v23 production BT16 path, but replaces the old v6
single-thread KKT stage with a native 16x16x16 BF16 MFMA kernel.  The target is
fixed to the Qwen3Next TP4 per-rank benchmark shape:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16 q/k/v, FP32 g/beta/intermediates
    chunk_size=BT=16, BV=16

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
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    _require_v14_target_shape,
    _validate_initial_state_v14,
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout,
)

BT = 16
BV = 16


@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_v24_bt16_mfma(
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
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, BT), (num_tokens * 8 * BT, 8 * BT, BT, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    value_head_idx = program_id % 8
    chunk_idx = program_id // 8
    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT

    k_bf16 = al.make_shared((BT, 128), al.bf16)
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        k_bf16[row, col] = k[0, chunk_start + row, key_head_idx, col]

    al.syncthreads()

    acc = al.full((4,), 0.0, al.f32)
    for batch128 in al.range(4):
        k_vec128 = lane_group + batch128 * 4
        a_words = k_vec[lane_col, k_vec128]
        b_words = k_vec[lane_col, k_vec128]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        source_offset = lane_col
        token_idx = chunk_start + token_offset
        source_idx = chunk_start + source_offset
        value = al.convert(0.0, al.f32)
        if source_offset < token_offset:
            decay = al.exp(g[0, token_idx, value_head_idx] - g[0, source_idx, value_head_idx])
            value = beta[0, token_idx, value_head_idx] * acc[r] * decay
        out[0, token_idx, value_head_idx, source_offset] = value


def _require_v24_kkt_inputs(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> int:
    if chunk_size != BT:
        raise ValueError("v24 KKT MFMA only supports chunk_size=16.")
    if k.dtype != torch.bfloat16 or g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("v24 KKT MFMA requires BF16 k and FP32 g/beta.")
    for name, tensor in (("k", k), ("g", g), ("beta", beta)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a CUDA/HIP device.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
    if tuple(k.shape) != (1, k.shape[1], 4, 128):
        raise ValueError("k must have shape [1,T,4,128].")
    num_tokens = k.shape[1]
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("g must have shape [1,T,8].")
    if tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("beta must have shape [1,T,8].")
    if num_tokens % BT != 0:
        raise ValueError("v24 KKT MFMA requires num_tokens divisible by 16.")
    return num_tokens


def qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    num_tokens = _require_v24_kkt_inputs(k, g, beta, chunk_size)
    num_chunks = _num_chunks(num_tokens, chunk_size)
    out = torch.empty((1, num_tokens, 8, BT), dtype=torch.float32, device=k.device)
    grid_size = num_chunks * 8
    _qwen_gdn_kkt_bf16_kernel_v24_bt16_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        k,
        g,
        beta,
        out,
        num_tokens,
        num_chunks,
    )
    return out


def qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_chunk_size(chunk_size)
    if chunk_size != BT:
        raise ValueError("v24 full path only supports chunk_size=16.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v14_target_shape(q, k, v, g, beta)
    _validate_initial_state_v14(initial_state, device=q.device)
    if q.shape[1] % BT != 0:
        raise ValueError("v24 full path requires num_tokens divisible by 16.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(k, g_cumsum, beta, chunk_size=chunk_size)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size)
    gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk_size)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        gdr_decay,
        gdr_g_last_exp,
        initial_state=initial_state,
        chunk_size=chunk_size,
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


def qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(
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
    _, output, _, _, final_state, _, _ = qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_full(
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


def qwen_gdn_kkt_avelang_v24_reference_v6_layout(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Explicit reference hook for tests/benchmarks; not used by v24 full path."""
    if chunk_size != BT:
        raise ValueError("v24 v6 KKT reference hook only supports chunk_size=16.")
    return qwen_gdn_kkt_avelang_v6_standalone(k, g, beta, chunk_size=chunk_size, prefer_optimized=True)


__all__ = [
    "_qwen_gdn_kkt_bf16_kernel_v24_bt16_mfma",
    "qwen_gdn_kkt_avelang_v24_bt16_mfma_layout",
    "qwen_gdn_kkt_avelang_v24_reference_v6_layout",
    "qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_full",
    "qwen_gdn_chunked_avelang_v24_kkt_mfma_layout",
]

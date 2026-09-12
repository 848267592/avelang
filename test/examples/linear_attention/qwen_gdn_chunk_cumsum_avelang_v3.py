"""Qwen GDN v3：实现 chunk 内局部 g cumsum。

当前版本只实现 `qwen_gdn_ref.torch_cumsum(g, chunk_size=...)`
在 `cu_seqlens=None` 时的行为：每个 chunk 内累加，chunk 边界处重新从 0 开始。
相比 v0/v1/v2 的完整 naive recurrent forward，这一版开始拆 Qwen-style
chunked pipeline 的第一个小步骤。
当前版本仍然不支持完整 GDN forward、cu_seqlens、kkt/solve/w/u、chunk_gdr、
chunk_o、backward、raw_buffer、shared memory、MFMA、向量化和性能优化。
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al


@avelang.jit
def _qwen_gdn_chunk_cumsum_kernel_v3(
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_heads: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
):
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_heads), (num_tokens * num_heads, num_heads, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_heads), (num_tokens * num_heads, num_heads, 1)),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_heads * num_chunks

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            chunk_idx = program_id % num_chunks
            head_idx = (program_id // num_chunks) % num_heads
            batch_idx = program_id // (num_heads * num_chunks)

            # 每个 program 只负责一个 batch/head/chunk，acc 在 chunk 边界重置。
            acc = al.convert(0.0, al.f32)
            for offset in al.range(chunk_size):
                token_idx = chunk_idx * chunk_size + offset
                if token_idx < num_tokens:
                    acc = acc + g[batch_idx, token_idx, head_idx]
                    out[batch_idx, token_idx, head_idx] = acc


def _require_fp32_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def qwen_gdn_chunk_cumsum_avelang_v3(
    g: torch.Tensor,
    *,
    chunk_size: int = 64,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """运行 correctness-first 的 chunk-local g cumsum kernel。"""
    _require_fp32_cuda_contiguous("g", g)
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, H], got {tuple(g.shape)}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

    batch_size, num_tokens, num_heads = g.shape
    if out is None:
        out = torch.empty_like(g)
    else:
        _require_fp32_cuda_contiguous("out", out)
        if out.shape != g.shape:
            raise ValueError(f"out must have shape {tuple(g.shape)}, got {tuple(out.shape)}.")
        if out.device != g.device:
            raise ValueError(f"out must be on device {g.device}, got {out.device}.")

    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    grid_size = batch_size * num_heads * num_chunks
    _qwen_gdn_chunk_cumsum_kernel_v3[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        g,
        out,
        batch_size,
        num_tokens,
        num_heads,
        chunk_size,
        num_chunks,
    )
    return out

"""Qwen GDN v18 larger-chunk solve prototype.

This file intentionally leaves the v17 baseline untouched.  v18 starts by
fixing the solve stage for larger chunks so BT=32/64 can be evaluated without
the v6 single-thread triangular recurrence dominating runtime.

Supported target for this prototype:

    B=1, Hv=8, FP32 a/a_solved, chunk_size in {32, 64}

Unsupported shapes raise ValueError instead of falling back to v6.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    _require_fp32_cuda_contiguous,
    _validate_bf16_qkvgb,
    _validate_chunk_size,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v13_mfma_layout,
    qwen_gdn_chunk_o_avelang_v13_mfma_layout,
)


@avelang.jit
def _qwen_gdn_solve_kernel_v18_parallel(
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
):
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, chunk_size), (num_tokens * 8 * chunk_size, 8 * chunk_size, chunk_size, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, chunk_size), (num_tokens * 8 * chunk_size, 8 * chunk_size, chunk_size, 1)),
    )

    tid = al.thread_id(0)
    program_id = al.block_id(0)
    head_idx = program_id % 8
    chunk_idx = program_id // 8
    chunk_start = chunk_idx * chunk_size

    groups_per_col = 4
    if chunk_size == 64:
        groups_per_col = 2

    group_idx = tid // chunk_size
    col_idx_thread = tid - group_idx * chunk_size

    mat = al.make_shared((chunk_size, chunk_size), al.f32)
    row_buf = al.make_shared((chunk_size,), al.f32)
    partial = al.make_shared((4, 64), al.f32)

    for rep_load in al.range(32):
        flat_idx = tid + rep_load * 128
        if flat_idx < chunk_size * chunk_size:
            row_idx = flat_idx // chunk_size
            col_idx = flat_idx - row_idx * chunk_size
            token_idx = chunk_start + row_idx
            value = al.convert(0.0, al.f32)
            if token_idx < num_tokens:
                value = al.convert(0.0, al.f32) - a[0, token_idx, head_idx, col_idx]
            mat[row_idx, col_idx] = value

    al.syncthreads()

    for row_idx_solve in al.range(1, chunk_size):
        if tid < chunk_size:
            row_buf[tid] = mat[row_idx_solve, tid]
        al.syncthreads()

        if group_idx < groups_per_col:
            part = al.convert(0.0, al.f32)
            if col_idx_thread < row_idx_solve:
                for inner_rep in al.range(32):
                    inner_idx = group_idx + inner_rep * groups_per_col
                    if inner_idx < row_idx_solve:
                        part = part + row_buf[inner_idx] * mat[inner_idx, col_idx_thread]
            partial[group_idx, col_idx_thread] = part

        al.syncthreads()

        if group_idx == 0:
            if col_idx_thread < row_idx_solve:
                acc = row_buf[col_idx_thread]
                for reduce_group in al.range(4):
                    if reduce_group < groups_per_col:
                        acc = acc + partial[reduce_group, col_idx_thread]
                mat[row_idx_solve, col_idx_thread] = acc

        al.syncthreads()

    for rep_store in al.range(32):
        flat_idx_out = tid + rep_store * 128
        if flat_idx_out < chunk_size * chunk_size:
            row_idx_out = flat_idx_out // chunk_size
            col_idx_out = flat_idx_out - row_idx_out * chunk_size
            token_idx_out = chunk_start + row_idx_out
            if token_idx_out < num_tokens:
                out_value = mat[row_idx_out, col_idx_out]
                if col_idx_out == row_idx_out:
                    out_value = out_value + al.convert(1.0, al.f32)
                out[0, token_idx_out, head_idx, col_idx_out] = out_value


def _validate_v18_solve_input(a: torch.Tensor, chunk_size: int) -> tuple[int, int, int]:
    if chunk_size not in (32, 64):
        raise ValueError("v18 solve only supports chunk_size=32 or chunk_size=64.")
    _require_fp32_cuda_contiguous("a", a)
    if a.ndim != 4:
        raise ValueError(f"a must have shape [1,T,8,{chunk_size}], got {tuple(a.shape)}.")
    batch_size, num_tokens, num_heads, a_chunk_size = a.shape
    if (batch_size, num_heads, a_chunk_size) != (1, 8, chunk_size):
        raise ValueError(
            "v18 solve only supports target shape [1,T,8,chunk_size] "
            f"with chunk_size={chunk_size}, got {tuple(a.shape)}."
        )
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    return batch_size, num_tokens, num_heads


def qwen_gdn_solve_avelang_v18_layout(a: torch.Tensor, *, chunk_size: int) -> torch.Tensor:
    """Parallel chunk-local solve for BT=32/64.

    Matches qwen_gdn_solve_avelang_v6_standalone numerically for the fixed
    Qwen GDN target shape, while parallelizing each row update across a
    128-thread workgroup.
    """

    _, num_tokens, _ = _validate_v18_solve_input(a, chunk_size)
    num_chunks = _num_chunks(num_tokens, chunk_size)
    out = torch.empty_like(a)
    grid_size = num_chunks * 8
    _qwen_gdn_solve_kernel_v18_parallel[lambda: ((grid_size, 1, 1), (128, 1, 1))](
        a,
        out,
        num_tokens,
        chunk_size,
        num_chunks,
    )
    return out


def qwen_gdn_solve_avelang_v18_bt32_layout(a: torch.Tensor, *, chunk_size: int = 32) -> torch.Tensor:
    if chunk_size != 32:
        raise ValueError("qwen_gdn_solve_avelang_v18_bt32_layout requires chunk_size=32.")
    return qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)


def qwen_gdn_solve_avelang_v18_bt64_layout(a: torch.Tensor, *, chunk_size: int = 64) -> torch.Tensor:
    if chunk_size != 64:
        raise ValueError("qwen_gdn_solve_avelang_v18_bt64_layout requires chunk_size=64.")
    return qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)


def _require_v18_bt32_target_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
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
    if tuple(beta.shape) != (1, num_tokens, 8):
        raise ValueError("beta must have shape [1,T,8].")


def _validate_initial_state_v18(initial_state: torch.Tensor | None, *, device: torch.device) -> None:
    if initial_state is None:
        return
    _require_fp32_cuda_contiguous("initial_state", initial_state)
    if tuple(initial_state.shape) != (1, 8, 128, 128):
        raise ValueError(f"initial_state must have shape (1, 8, 128, 128), got {tuple(initial_state.shape)}.")
    if initial_state.device != device:
        raise ValueError(f"initial_state must be on device {device}, got {initial_state.device}.")


def qwen_gdn_chunked_avelang_v18_bt32_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimal BT32 smoke path.

    This path exists to prove that v18 solve can drop into the old BT32 flow.
    It still uses v6 w_u and v13 chunk_gdr/chunk_o, so full-path performance is
    intentionally not treated as a valid v18 optimization result.
    """

    _validate_chunk_size(chunk_size)
    if chunk_size != 32:
        raise ValueError("v18 BT32 smoke path only supports chunk_size=32.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v18_bt32_target_shape(q, k, v, g, beta)
    _validate_initial_state_v18(initial_state, device=q.device)
    if q.shape[1] % 32 != 0:
        raise ValueError("v18 BT32 smoke path requires num_tokens divisible by 32.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size, prefer_optimized=True)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v13_mfma_layout(
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


def qwen_gdn_chunked_avelang_v18_bt32_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v18_bt32_layout_full(
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
    "_qwen_gdn_solve_kernel_v18_parallel",
    "qwen_gdn_solve_avelang_v18_layout",
    "qwen_gdn_solve_avelang_v18_bt32_layout",
    "qwen_gdn_solve_avelang_v18_bt64_layout",
    "qwen_gdn_chunked_avelang_v18_bt32_layout_full",
    "qwen_gdn_chunked_avelang_v18_bt32_layout",
]

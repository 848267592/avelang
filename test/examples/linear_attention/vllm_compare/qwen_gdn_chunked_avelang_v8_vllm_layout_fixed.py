"""Qwen GDN v8 vLLM-layout parallel chunk_gdr.

This version targets the real v6/v7 bottleneck: chunk_gdr's workgroup size of 1.
Only chunk_gdr is replaced.  The rest of the forward pipeline reuses the v6
vLLM-layout implementation:

    cumsum -> KKT -> solve -> w/u -> v8 chunk_gdr -> chunk_o

The first v8 optimization maps one workgroup to a block of value dimensions:

    block_id  -> (B, Hv, V block)
    thread_id -> value_idx inside that V block

Each thread still owns one value dimension and keeps its own state[K], so the
math order for each value_idx matches v6.  Unlike the failed v7 value_tile
experiment, this uses actual workgroup lanes instead of expanding one scalar
thread's local state.

The second v8 optimization keeps the V-block mapping but splits each value lane
across K lanes:

    block_id     -> (B, Hv, V block)
    thread_id.x  -> value_idx inside the V block
    thread_id.y  -> K lane for the per-token dot/recurrent state update

Each K lane owns a strided slice of state[K].  The expensive
pred = sum_k w[k] * state[k] is reduced through shared memory, so the largest
remaining scalar loop in chunk_gdr is no longer carried by one lane.
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
    _validate_qkvgb,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_gdr_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_chunked_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)


V8_VLLM_LAYOUT_OPTIMIZATION_SUMMARY = {
    "target": "chunk_gdr",
    "mapping": "vblock or vk: block_id -> (B, Hv, V block), thread_id -> value_idx[/K lane]",
    "default_parallel_mode": "vk",
    "default_block_v": 4,
    "default_block_k": "auto, 64 when head_dim_k >= 64",
    "state_layout": "vLLM value-major: state [B, Hv, V, K], h [B, C, Hv, V, K]",
    "fallback": "use_parallel_chunk_gdr=False, non-BF16 q/k/v, or parallel_mode='vblock' keeps the first v8 kernel",
}


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v8_vblock(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
    block_v: al.constexpr,
    num_v_blocks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_k),
            (num_tokens * num_v_heads * head_dim_k, num_v_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_v, head_dim_k),
            (num_v_heads * head_dim_v * head_dim_k, head_dim_v * head_dim_k, head_dim_k, 1),
        ),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_chunks, num_v_heads, head_dim_v, head_dim_k),
            (
                num_chunks * num_v_heads * head_dim_v * head_dim_k,
                num_v_heads * head_dim_v * head_dim_k,
                head_dim_v * head_dim_k,
                head_dim_k,
                1,
            ),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_v, head_dim_k),
            (num_v_heads * head_dim_v * head_dim_k, head_dim_v * head_dim_k, head_dim_k, 1),
        ),
    )

    program_id = al.block_id(0)
    lane_v = al.thread_id(0)
    total_programs = batch_size * num_v_heads * num_v_blocks

    if program_id < total_programs:
        v_block_idx = program_id % num_v_blocks
        value_head_idx = (program_id // num_v_blocks) % num_v_heads
        batch_idx = program_id // (num_v_heads * num_v_blocks)
        value_idx = v_block_idx * block_v + lane_v

        if value_idx < head_dim_v:
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat

            state = al.make_local((head_dim_k,), al.f32)
            for kk in al.range(head_dim_k):
                if has_initial_state:
                    state[kk] = initial_state[batch_idx, value_head_idx, value_idx, kk]
                else:
                    state[kk] = al.convert(0.0, al.f32)

            for chunk_idx in al.range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                last_token = chunk_start + chunk_size - 1
                if last_token >= num_tokens:
                    last_token = num_tokens - 1

                for kk in al.range(head_dim_k):
                    h[batch_idx, chunk_idx, value_head_idx, value_idx, kk] = state[kk]

                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        pred = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            pred = pred + w[batch_idx, token_idx, value_head_idx, kk] * state[kk]
                        vn[batch_idx, token_idx, value_head_idx, value_idx] = (
                            u[batch_idx, token_idx, value_head_idx, value_idx] - pred
                        )

                g_last = g[batch_idx, last_token, value_head_idx]
                g_last_exp = al.exp(g_last)
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] * g_last_exp
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        decay = al.exp(g_last - g[batch_idx, token_idx, value_head_idx])
                        v_new = vn[batch_idx, token_idx, value_head_idx, value_idx]
                        for kk in al.range(head_dim_k):
                            k_value = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                            state[kk] = state[kk] + k_value * decay * v_new

            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, value_idx, kk] = state[kk]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
    block_v: al.constexpr,
    block_k: al.constexpr,
    num_v_blocks: al.constexpr,
    k_items_per_lane: al.constexpr,
    k_reduce_steps: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_k),
            (num_tokens * num_v_heads * head_dim_k, num_v_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_v, head_dim_k),
            (num_v_heads * head_dim_v * head_dim_k, head_dim_v * head_dim_k, head_dim_k, 1),
        ),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_chunks, num_v_heads, head_dim_v, head_dim_k),
            (
                num_chunks * num_v_heads * head_dim_v * head_dim_k,
                num_v_heads * head_dim_v * head_dim_k,
                head_dim_v * head_dim_k,
                head_dim_k,
                1,
            ),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_v, head_dim_k),
            (num_v_heads * head_dim_v * head_dim_k, head_dim_v * head_dim_k, head_dim_k, 1),
        ),
    )

    program_id = al.block_id(0)
    lane_v = al.thread_id(0)
    lane_k = al.thread_id(1)
    v_block_idx = program_id % num_v_blocks
    value_head_idx = (program_id // num_v_blocks) % num_v_heads
    batch_idx = program_id // (num_v_heads * num_v_blocks)
    value_idx = v_block_idx * block_v + lane_v
    value_valid = value_idx < head_dim_v

    repeat = num_v_heads // num_k_heads
    key_head_idx = value_head_idx // repeat

    state = al.make_local((k_items_per_lane,), al.f32)
    partial_pred = al.make_shared((block_v, block_k), al.f32)
    chunk_vn = al.make_shared((block_v, chunk_size), al.f32)

    for item in al.range(k_items_per_lane):
        kk = lane_k + item * block_k
        if value_valid and kk < head_dim_k:
            if has_initial_state:
                state[item] = initial_state[batch_idx, value_head_idx, value_idx, kk]
            else:
                state[item] = al.convert(0.0, al.f32)
        else:
            state[item] = al.convert(0.0, al.f32)

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        last_token = chunk_start + chunk_size - 1
        if last_token >= num_tokens:
            last_token = num_tokens - 1

        for item in al.range(k_items_per_lane):
            kk = lane_k + item * block_k
            if value_valid and kk < head_dim_k:
                h[batch_idx, chunk_idx, value_head_idx, value_idx, kk] = state[item]

        for offset in al.range(chunk_size):
            token_idx = chunk_start + offset
            partial = al.convert(0.0, al.f32)
            if value_valid and token_idx < num_tokens:
                for item in al.range(k_items_per_lane):
                    kk = lane_k + item * block_k
                    if kk < head_dim_k:
                        partial = partial + w[batch_idx, token_idx, value_head_idx, kk] * state[item]

            partial_pred[lane_v, lane_k] = partial
            al.syncthreads()

            stride = block_k >> 1
            for _step in al.range(k_reduce_steps):
                if lane_k < stride:
                    partial_pred[lane_v, lane_k] = (
                        partial_pred[lane_v, lane_k] + partial_pred[lane_v, lane_k + stride]
                    )
                al.syncthreads()
                stride = stride >> 1

            if lane_k == 0:
                v_new = al.convert(0.0, al.f32)
                if value_valid and token_idx < num_tokens:
                    v_new = u[batch_idx, token_idx, value_head_idx, value_idx] - partial_pred[lane_v, 0]
                    vn[batch_idx, token_idx, value_head_idx, value_idx] = v_new
                chunk_vn[lane_v, offset] = v_new

        al.syncthreads()

        g_last = g[batch_idx, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)
        for item in al.range(k_items_per_lane):
            state[item] = state[item] * g_last_exp

        for offset in al.range(chunk_size):
            token_idx = chunk_start + offset
            if value_valid and token_idx < num_tokens:
                decay = al.exp(g_last - g[batch_idx, token_idx, value_head_idx])
                v_new = chunk_vn[lane_v, offset]
                for item in al.range(k_items_per_lane):
                    kk = lane_k + item * block_k
                    if kk < head_dim_k:
                        k_value = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                        state[item] = state[item] + k_value * decay * v_new

    for item in al.range(k_items_per_lane):
        kk = lane_k + item * block_k
        if value_valid and kk < head_dim_k:
            final_state[batch_idx, value_head_idx, value_idx, kk] = state[item]


def _validate_block_v(block_v: int) -> None:
    if block_v not in (32, 64):
        raise ValueError(f"block_v must be 32 or 64 for v8 vblock chunk_gdr, got {block_v}.")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _ilog2_power_of_two(value: int) -> int:
    if not _is_power_of_two(value):
        raise ValueError(f"value must be a power of two, got {value}.")
    return value.bit_length() - 1


def _validate_vk_params(block_v: int, block_k: int) -> None:
    if block_v not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"block_v must be 1, 2, 4, 8, 16, 32, or 64 for v8 vk chunk_gdr, got {block_v}.")
    if block_k not in (4, 8, 16, 32, 64, 128):
        raise ValueError(f"block_k must be 4, 8, 16, 32, 64, or 128 for v8 vk chunk_gdr, got {block_k}.")
    if block_v * block_k > 1024:
        raise ValueError(f"block_v * block_k must be <= 1024 on MI300, got {block_v * block_k}.")


def _default_vk_block_k(head_dim_k: int) -> int:
    if head_dim_k >= 64:
        return 64
    if head_dim_k >= 32:
        return 32
    if head_dim_k >= 16:
        return 16
    if head_dim_k >= 8:
        return 8
    return 4


def _validate_initial_state_vllm_layout(
    initial_state: torch.Tensor | None,
    *,
    state_shape: tuple[int, int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor | None, bool]:
    has_initial_state = initial_state is not None
    if initial_state is None:
        return None, has_initial_state
    _require_fp32_cuda_contiguous("initial_state", initial_state)
    if initial_state.shape != state_shape:
        raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
    if initial_state.device != device:
        raise ValueError(f"initial_state must be on device {device}, got {initial_state.device}.")
    return initial_state, has_initial_state


def qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 4,
    use_parallel_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    parallel_mode: str = "vk",
    block_v: int = 4,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run chunk_gdr.  BF16 uses v8 VxK parallel mapping by default."""
    if not use_parallel_chunk_gdr or not prefer_optimized or k.dtype != torch.bfloat16:
        return qwen_gdn_chunk_gdr_avelang_v6_standalone(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=chunk_size,
            prefer_optimized=True,
        )

    if parallel_mode not in ("vk", "vblock"):
        raise ValueError(f"parallel_mode must be 'vk' or 'vblock', got {parallel_mode!r}.")

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k,
        w,
        u,
        g,
        chunk_size,
    )
    if parallel_mode == "vblock":
        _validate_block_v(block_v)
    else:
        if block_k is None:
            block_k = _default_vk_block_k(head_dim_k)
        _validate_vk_params(block_v, block_k)

    state_shape = (batch_size, num_v_heads, head_dim_v, head_dim_k)
    initial_state_arg, has_initial_state = _validate_initial_state_vllm_layout(
        initial_state,
        state_shape=state_shape,
        device=k.device,
    )

    num_chunks = _num_chunks(num_tokens, chunk_size)
    num_v_blocks = (head_dim_v + block_v - 1) // block_v
    h = torch.empty((batch_size, num_chunks, num_v_heads, head_dim_v, head_dim_k), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty(state_shape, dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    grid_size = batch_size * num_v_heads * num_v_blocks
    if parallel_mode == "vblock":
        _qwen_gdn_chunk_gdr_bf16_kernel_v8_vblock[lambda: ((grid_size, 1, 1), (block_v, 1, 1))](
            k,
            w,
            u,
            g,
            initial_state_arg,
            h,
            vn,
            final_state,
            batch_size,
            num_tokens,
            num_k_heads,
            num_v_heads,
            head_dim_k,
            head_dim_v,
            chunk_size,
            num_chunks,
            block_v,
            num_v_blocks,
            has_initial_state,
        )
    else:
        k_items_per_lane = max(2, (head_dim_k + block_k - 1) // block_k)
        k_reduce_steps = _ilog2_power_of_two(block_k)
        _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk[lambda: ((grid_size, 1, 1), (block_v, block_k, 1))](
            k,
            w,
            u,
            g,
            initial_state_arg,
            h,
            vn,
            final_state,
            batch_size,
            num_tokens,
            num_k_heads,
            num_v_heads,
            head_dim_k,
            head_dim_v,
            chunk_size,
            num_chunks,
            block_v,
            block_k,
            num_v_blocks,
            k_items_per_lane,
            k_reduce_steps,
            has_initial_state,
        )
    return h, vn, final_state


def qwen_gdn_chunked_avelang_v8_vllm_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 4,
    use_parallel_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    parallel_mode: str = "vk",
    block_v: int = 4,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run full forward and return g_cumsum, output, A_solved, h, final_state."""
    _validate_chunk_size(chunk_size)
    if q.dtype == k.dtype == v.dtype == torch.bfloat16:
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_qkvgb(q, k, v, g, beta)
    elif q.dtype == k.dtype == v.dtype == torch.float32:
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_qkvgb(q, k, v, g, beta)
    else:
        raise ValueError(f"q/k/v must all be torch.bfloat16 or all torch.float32, got {q.dtype}, {k.dtype}, {v.dtype}.")

    state_shape = (batch_size, num_v_heads, head_dim_v, head_dim_k)
    _validate_initial_state_vllm_layout(initial_state, state_shape=state_shape, device=q.device)

    if q.dtype != torch.bfloat16 or not use_parallel_chunk_gdr or not prefer_optimized:
        return qwen_gdn_chunked_avelang_v6_standalone(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=True,
        )

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(
        k,
        g_cumsum,
        beta,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        prefer_optimized=prefer_optimized,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )
    output = qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    return g_cumsum, output, a_solved, h, final_state


def qwen_gdn_chunked_avelang_v8_vllm_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 4,
    use_parallel_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    parallel_mode: str = "vk",
    block_v: int = 4,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-style thin entry returning output and final_state."""
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v8_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=use_parallel_chunk_gdr,
        prefer_optimized=prefer_optimized,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )
    return output, final_state


def qwen_gdn_chunk_gated_delta_rule_vllm_compatible(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    scale: float | None = None,
    chunk_size: int = 4,
    use_parallel_chunk_gdr: bool = True,
    prefer_optimized: bool = True,
    parallel_mode: str = "vk",
    block_v: int = 4,
    block_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in vLLM prefill surface for equal-length head_first=False inputs."""
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens / packed variable-length input is not supported.")
    if head_first:
        raise NotImplementedError("head_first=True is not supported; expected vLLM layout [B, T, H, D].")
    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("Native in-kernel q/k L2 norm is not implemented.")
    if not output_final_state:
        raise NotImplementedError("output_final_state=False is not supported; this implementation always returns final_state.")
    return qwen_gdn_chunked_avelang_v8_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=use_parallel_chunk_gdr,
        prefer_optimized=prefer_optimized,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )


__all__ = [
    "V8_VLLM_LAYOUT_OPTIMIZATION_SUMMARY",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v8_vblock",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v8_vk",
    "qwen_gdn_chunk_gdr_avelang_v8_vllm_layout",
    "qwen_gdn_chunked_avelang_v8_vllm_layout_full",
    "qwen_gdn_chunked_avelang_v8_vllm_layout",
    "qwen_gdn_chunk_gated_delta_rule_vllm_compatible",
]

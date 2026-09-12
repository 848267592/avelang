"""Qwen GDN v7 vLLM-layout optimization.

This file keeps the v6 vLLM-layout implementation as the correctness baseline
and optimizes only the current hotspot: chunk_gdr.

Optimization:
    precompute per-chunk decay terms used by chunk_gdr:
    - last_exp[b, chunk, hv] = exp(g_cumsum at the chunk tail)
    - decay[b, token, hv] = exp(g_last - g_cumsum[token])

The chunk_gdr kernels then reuse those values for every value dimension V.
State tensors keep the vLLM/Qwen3Next value-major layout:
    initial_state/final_state: [B, Hv, V, K]
    chunk_states(h):          [B, C, Hv, V, K]

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


V7_VLLM_LAYOUT_OPTIMIZATION_SUMMARY = {
    "target": "chunk_gdr",
    "change": (
        "precompute chunk-tail exp and token-to-tail decay; value_tile>1 can be enabled explicitly "
        "to experiment with reusing k/decay across V dimensions"
    ),
    "state_layout": "vLLM value-major: state [B, Hv, V, K], h [B, C, Hv, V, K]",
    "fallback": "prefer_optimized=False calls the v6 vLLM-layout baseline",
}


@avelang.jit
def _qwen_gdn_chunk_decay_kernel_v7_vllm_layout(
    g_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    last_exp_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_v_heads: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
):
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    decay = al.make_tensor(
        decay_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    last_exp = al.make_tensor(
        last_exp_ptr,
        al.f32,
        al.make_layout((batch_size, num_chunks, num_v_heads), (num_chunks * num_v_heads, num_v_heads, 1)),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_chunks * num_v_heads

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_head_idx = program_id % num_v_heads
            chunk_idx = (program_id // num_v_heads) % num_chunks
            batch_idx = program_id // (num_chunks * num_v_heads)
            chunk_start = chunk_idx * chunk_size
            last_token = chunk_start + chunk_size - 1
            if last_token >= num_tokens:
                last_token = num_tokens - 1

            g_last = g[batch_idx, last_token, value_head_idx]
            last_exp[batch_idx, chunk_idx, value_head_idx] = al.exp(g_last)

            for offset in al.range(chunk_size):
                token_idx = chunk_start + offset
                if token_idx < num_tokens:
                    decay[batch_idx, token_idx, value_head_idx] = al.exp(
                        g_last - g[batch_idx, token_idx, value_head_idx]
                    )


@avelang.jit
def _qwen_gdn_chunk_gdr_fp32_decay_kernel_v7_vllm_layout(
    k_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    last_exp_ptr: al.Pointer(al.f32),
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
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.f32,
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
    decay = al.make_tensor(
        decay_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    last_exp = al.make_tensor(
        last_exp_ptr,
        al.f32,
        al.make_layout((batch_size, num_chunks, num_v_heads), (num_chunks * num_v_heads, num_v_heads, 1)),
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
    total_programs = batch_size * num_v_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            value_head_idx = (program_id // head_dim_v) % num_v_heads
            batch_idx = program_id // (num_v_heads * head_dim_v)
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

                g_last_exp = last_exp[batch_idx, chunk_idx, value_head_idx]
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] * g_last_exp
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        decay_value = decay[batch_idx, token_idx, value_head_idx]
                        v_new = vn[batch_idx, token_idx, value_head_idx, value_idx]
                        for kk in al.range(head_dim_k):
                            state[kk] = state[kk] + k[batch_idx, token_idx, key_head_idx, kk] * decay_value * v_new

            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, value_idx, kk] = state[kk]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_decay_kernel_v7_vllm_layout(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    last_exp_ptr: al.Pointer(al.f32),
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
    decay = al.make_tensor(
        decay_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    last_exp = al.make_tensor(
        last_exp_ptr,
        al.f32,
        al.make_layout((batch_size, num_chunks, num_v_heads), (num_chunks * num_v_heads, num_v_heads, 1)),
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
    total_programs = batch_size * num_v_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            value_head_idx = (program_id // head_dim_v) % num_v_heads
            batch_idx = program_id // (num_v_heads * head_dim_v)
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

                g_last_exp = last_exp[batch_idx, chunk_idx, value_head_idx]
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] * g_last_exp
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        decay_value = decay[batch_idx, token_idx, value_head_idx]
                        v_new = vn[batch_idx, token_idx, value_head_idx, value_idx]
                        for kk in al.range(head_dim_k):
                            k_value = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                            state[kk] = state[kk] + k_value * decay_value * v_new

            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, value_idx, kk] = state[kk]


@avelang.jit
def _qwen_gdn_chunk_gdr_fp32_decay_vtile_kernel_v7_vllm_layout(
    k_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    last_exp_ptr: al.Pointer(al.f32),
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
    value_tile: al.constexpr,
    num_value_tiles: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.f32,
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
    decay = al.make_tensor(
        decay_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    last_exp = al.make_tensor(
        last_exp_ptr,
        al.f32,
        al.make_layout((batch_size, num_chunks, num_v_heads), (num_chunks * num_v_heads, num_v_heads, 1)),
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
    total_programs = batch_size * num_v_heads * num_value_tiles

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_tile_idx = program_id % num_value_tiles
            value_head_idx = (program_id // num_value_tiles) % num_v_heads
            batch_idx = program_id // (num_v_heads * num_value_tiles)
            value_start = value_tile_idx * value_tile
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat

            state = al.make_local((value_tile, head_dim_k), al.f32)
            for vt in al.range(value_tile):
                value_idx = value_start + vt
                for kk in al.range(head_dim_k):
                    if value_idx < head_dim_v:
                        if has_initial_state:
                            state[vt, kk] = initial_state[batch_idx, value_head_idx, value_idx, kk]
                        else:
                            state[vt, kk] = al.convert(0.0, al.f32)
                    else:
                        state[vt, kk] = al.convert(0.0, al.f32)

            for chunk_idx in al.range(num_chunks):
                chunk_start = chunk_idx * chunk_size

                for vt in al.range(value_tile):
                    value_idx = value_start + vt
                    if value_idx < head_dim_v:
                        for kk in al.range(head_dim_k):
                            h[batch_idx, chunk_idx, value_head_idx, value_idx, kk] = state[vt, kk]

                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        for vt in al.range(value_tile):
                            value_idx = value_start + vt
                            if value_idx < head_dim_v:
                                pred = al.convert(0.0, al.f32)
                                for kk in al.range(head_dim_k):
                                    pred = pred + w[batch_idx, token_idx, value_head_idx, kk] * state[vt, kk]
                                vn[batch_idx, token_idx, value_head_idx, value_idx] = (
                                    u[batch_idx, token_idx, value_head_idx, value_idx] - pred
                                )

                g_last_exp = last_exp[batch_idx, chunk_idx, value_head_idx]
                for vt in al.range(value_tile):
                    value_idx = value_start + vt
                    if value_idx < head_dim_v:
                        for kk in al.range(head_dim_k):
                            state[vt, kk] = state[vt, kk] * g_last_exp
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        decay_value = decay[batch_idx, token_idx, value_head_idx]
                        for kk in al.range(head_dim_k):
                            k_decay = k[batch_idx, token_idx, key_head_idx, kk] * decay_value
                            for vt in al.range(value_tile):
                                value_idx = value_start + vt
                                if value_idx < head_dim_v:
                                    v_new = vn[batch_idx, token_idx, value_head_idx, value_idx]
                                    state[vt, kk] = state[vt, kk] + k_decay * v_new

            for vt in al.range(value_tile):
                value_idx = value_start + vt
                if value_idx < head_dim_v:
                    for kk in al.range(head_dim_k):
                        final_state[batch_idx, value_head_idx, value_idx, kk] = state[vt, kk]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_decay_vtile_kernel_v7_vllm_layout(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    last_exp_ptr: al.Pointer(al.f32),
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
    value_tile: al.constexpr,
    num_value_tiles: al.constexpr,
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
    decay = al.make_tensor(
        decay_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    last_exp = al.make_tensor(
        last_exp_ptr,
        al.f32,
        al.make_layout((batch_size, num_chunks, num_v_heads), (num_chunks * num_v_heads, num_v_heads, 1)),
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
    total_programs = batch_size * num_v_heads * num_value_tiles

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_tile_idx = program_id % num_value_tiles
            value_head_idx = (program_id // num_value_tiles) % num_v_heads
            batch_idx = program_id // (num_v_heads * num_value_tiles)
            value_start = value_tile_idx * value_tile
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat

            state = al.make_local((value_tile, head_dim_k), al.f32)
            for vt in al.range(value_tile):
                value_idx = value_start + vt
                for kk in al.range(head_dim_k):
                    if value_idx < head_dim_v:
                        if has_initial_state:
                            state[vt, kk] = initial_state[batch_idx, value_head_idx, value_idx, kk]
                        else:
                            state[vt, kk] = al.convert(0.0, al.f32)
                    else:
                        state[vt, kk] = al.convert(0.0, al.f32)

            for chunk_idx in al.range(num_chunks):
                chunk_start = chunk_idx * chunk_size

                for vt in al.range(value_tile):
                    value_idx = value_start + vt
                    if value_idx < head_dim_v:
                        for kk in al.range(head_dim_k):
                            h[batch_idx, chunk_idx, value_head_idx, value_idx, kk] = state[vt, kk]

                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        for vt in al.range(value_tile):
                            value_idx = value_start + vt
                            if value_idx < head_dim_v:
                                pred = al.convert(0.0, al.f32)
                                for kk in al.range(head_dim_k):
                                    pred = pred + w[batch_idx, token_idx, value_head_idx, kk] * state[vt, kk]
                                vn[batch_idx, token_idx, value_head_idx, value_idx] = (
                                    u[batch_idx, token_idx, value_head_idx, value_idx] - pred
                                )

                g_last_exp = last_exp[batch_idx, chunk_idx, value_head_idx]
                for vt in al.range(value_tile):
                    value_idx = value_start + vt
                    if value_idx < head_dim_v:
                        for kk in al.range(head_dim_k):
                            state[vt, kk] = state[vt, kk] * g_last_exp
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        decay_value = decay[batch_idx, token_idx, value_head_idx]
                        for kk in al.range(head_dim_k):
                            k_value = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                            k_decay = k_value * decay_value
                            for vt in al.range(value_tile):
                                value_idx = value_start + vt
                                if value_idx < head_dim_v:
                                    v_new = vn[batch_idx, token_idx, value_head_idx, value_idx]
                                    state[vt, kk] = state[vt, kk] + k_decay * v_new

            for vt in al.range(value_tile):
                value_idx = value_start + vt
                if value_idx < head_dim_v:
                    for kk in al.range(head_dim_k):
                        final_state[batch_idx, value_head_idx, value_idx, kk] = state[vt, kk]


def _validate_fp32_chunk_gdr_stage_vllm_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    _validate_chunk_size(chunk_size)
    for name, tensor in (("k", k), ("w", w), ("u", u), ("g", g)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
    if k.ndim != 4:
        raise ValueError(f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}.")
    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    if u.ndim != 4:
        raise ValueError(f"u must have shape [B, T, Hv, V], got {tuple(u.shape)}.")
    if u.shape[:2] != (batch_size, num_tokens):
        raise ValueError(f"u must share k's [B, T], got {tuple(u.shape[:2])}.")
    num_v_heads = u.shape[2]
    head_dim_v = u.shape[3]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")
    if w.shape != (batch_size, num_tokens, num_v_heads, head_dim_k):
        raise ValueError(
            f"w must have shape {(batch_size, num_tokens, num_v_heads, head_dim_k)}, got {tuple(w.shape)}."
        )
    if g.shape != (batch_size, num_tokens, num_v_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_v_heads)}, got {tuple(g.shape)}.")
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


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


def _validate_value_tile(value_tile: int) -> None:
    if value_tile <= 0:
        raise ValueError(f"value_tile must be > 0, got {value_tile}.")
    if value_tile > 4:
        raise ValueError(f"value_tile must be <= 4 for the current scalar-local kernel, got {value_tile}.")


def qwen_gdn_chunk_decay_avelang_v7_vllm_layout(
    g: torch.Tensor,
    *,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute chunk_gdr decay tensors from chunk-local cumulative g."""
    _validate_chunk_size(chunk_size)
    _require_fp32_cuda_contiguous("g", g)
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, Hv], got {tuple(g.shape)}.")
    batch_size, num_tokens, num_v_heads = g.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")

    num_chunks = _num_chunks(num_tokens, chunk_size)
    decay = torch.empty_like(g)
    last_exp = torch.empty((batch_size, num_chunks, num_v_heads), dtype=torch.float32, device=g.device)
    grid_size = batch_size * num_chunks * num_v_heads
    _qwen_gdn_chunk_decay_kernel_v7_vllm_layout[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        g,
        decay,
        last_exp,
        batch_size,
        num_tokens,
        num_v_heads,
        chunk_size,
        num_chunks,
    )
    return decay, last_exp


def qwen_gdn_chunk_gdr_avelang_v7_vllm_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
    value_tile: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run chunk_gdr with vLLM value-major state layout."""
    if not prefer_optimized:
        return qwen_gdn_chunk_gdr_avelang_v6_standalone(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=chunk_size,
            prefer_optimized=False,
        )
    _validate_value_tile(value_tile)

    if k.dtype == torch.float32:
        batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = (
            _validate_fp32_chunk_gdr_stage_vllm_layout(k, w, u, g, chunk_size)
        )
    else:
        batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
            k,
            w,
            u,
            g,
            chunk_size,
        )

    state_shape = (batch_size, num_v_heads, head_dim_v, head_dim_k)
    initial_state_arg, has_initial_state = _validate_initial_state_vllm_layout(
        initial_state,
        state_shape=state_shape,
        device=k.device,
    )
    num_chunks = _num_chunks(num_tokens, chunk_size)
    h = torch.empty((batch_size, num_chunks, num_v_heads, head_dim_v, head_dim_k), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty(state_shape, dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    decay, last_exp = qwen_gdn_chunk_decay_avelang_v7_vllm_layout(g, chunk_size=chunk_size)
    if value_tile == 1:
        grid_size = batch_size * num_v_heads * head_dim_v
        if k.dtype == torch.float32:
            _qwen_gdn_chunk_gdr_fp32_decay_kernel_v7_vllm_layout[lambda: ((grid_size, 1, 1), (1, 1, 1))](
                k,
                w,
                u,
                decay,
                last_exp,
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
                has_initial_state,
            )
        else:
            _qwen_gdn_chunk_gdr_bf16_decay_kernel_v7_vllm_layout[lambda: ((grid_size, 1, 1), (1, 1, 1))](
                k,
                w,
                u,
                decay,
                last_exp,
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
                has_initial_state,
            )
        return h, vn, final_state

    num_value_tiles = (head_dim_v + value_tile - 1) // value_tile
    grid_size = batch_size * num_v_heads * num_value_tiles
    if k.dtype == torch.float32:
        _qwen_gdn_chunk_gdr_fp32_decay_vtile_kernel_v7_vllm_layout[lambda: ((grid_size, 1, 1), (1, 1, 1))](
            k,
            w,
            u,
            decay,
            last_exp,
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
            value_tile,
            num_value_tiles,
            has_initial_state,
        )
    else:
        _qwen_gdn_chunk_gdr_bf16_decay_vtile_kernel_v7_vllm_layout[lambda: ((grid_size, 1, 1), (1, 1, 1))](
            k,
            w,
            u,
            decay,
            last_exp,
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
            value_tile,
            num_value_tiles,
            has_initial_state,
        )
    return h, vn, final_state


def qwen_gdn_chunked_avelang_v7_vllm_layout_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
    value_tile: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run full forward and return the debug tuple: g_cumsum, output, A_solved, h, final_state."""
    if not prefer_optimized:
        return qwen_gdn_chunked_avelang_v6_standalone(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=False,
        )

    if q.dtype == k.dtype == v.dtype == torch.float32:
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_qkvgb(q, k, v, g, beta)
    else:
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_qkvgb(q, k, v, g, beta)
    state_shape = (batch_size, num_v_heads, head_dim_v, head_dim_k)
    _validate_initial_state_vllm_layout(initial_state, state_shape=state_shape, device=q.device)

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
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v7_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=True,
        value_tile=value_tile,
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


def qwen_gdn_chunked_avelang_v7_vllm_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor | None = None,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
    value_tile: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM-style thin entry returning output and final_state."""
    if beta is None:
        beta = torch.ones_like(g, dtype=torch.float32, device=g.device).contiguous()
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v7_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
        value_tile=value_tile,
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
    prefer_optimized: bool = True,
    value_tile: int = 1,
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
    return qwen_gdn_chunked_avelang_v7_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
        value_tile=value_tile,
    )


__all__ = [
    "V7_VLLM_LAYOUT_OPTIMIZATION_SUMMARY",
    "qwen_gdn_chunk_decay_avelang_v7_vllm_layout",
    "qwen_gdn_chunk_gdr_avelang_v7_vllm_layout",
    "qwen_gdn_chunked_avelang_v7_vllm_layout_full",
    "qwen_gdn_chunked_avelang_v7_vllm_layout",
    "qwen_gdn_chunk_gated_delta_rule_vllm_compatible",
]

"""Qwen GDN v4：完整 correctness-first chunked forward。

当前版本实现了 Qwen-style chunked GDN forward 的完整前向流水线：
g 的 chunk 内 cumsum、KKT、solve、w/u、chunk_gdr、chunk_o，并返回
与 `qwen_gdn_forward_ref` 相同逻辑的五元组。
相比 v3，本版本从单独的 g cumsum 扩展到完整 chunked forward，但仍然使用多个
简单 Avelang kernel 串起来做 correctness 验证。
当前版本仍然不支持 bf16、backward、cu_seqlens、raw_buffer、shared memory、
MFMA、向量化、融合 kernel 或任何性能优化。
当前版本是 correctness-first 版本，不属于性能优化版。
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunk_cumsum_avelang_v3 import qwen_gdn_chunk_cumsum_avelang_v3


@avelang.jit
def _qwen_gdn_kkt_kernel_v4(
    k_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    chunk_size: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, chunk_size),
            (num_tokens * num_v_heads * chunk_size, num_v_heads * chunk_size, chunk_size, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_tokens * num_v_heads * chunk_size

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            d_idx = program_id % chunk_size
            value_head_idx = (program_id // chunk_size) % num_v_heads
            token_idx = (program_id // (chunk_size * num_v_heads)) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads * chunk_size)
            
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            local_pos = token_idx % chunk_size
            chunk_start = token_idx - local_pos
            source_token = chunk_start + d_idx

            # KKT 阶段只保留严格过去位置，当前 token 对角线写 0。
            value = al.convert(0.0, al.f32)
            if d_idx < local_pos:
                if source_token < num_tokens:
                    dot = al.convert(0.0, al.f32)
                    for kk in al.range(head_dim_k):
                        dot = dot + k[batch_idx, token_idx, key_head_idx, kk] * k[
                            batch_idx,
                            source_token,
                            key_head_idx,
                            kk,
                        ]
                    decay = al.exp(g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx])
                    value = beta[batch_idx, token_idx, value_head_idx] * dot * decay
            out[batch_idx, token_idx, value_head_idx, d_idx] = value


@avelang.jit
def _qwen_gdn_solve_kernel_v4(
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_heads: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
):
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, chunk_size),
            (num_tokens * num_heads * chunk_size, num_heads * chunk_size, chunk_size, 1),
        ),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, chunk_size),
            (num_tokens * num_heads * chunk_size, num_heads * chunk_size, chunk_size, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_chunks * num_heads

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            head_idx = program_id % num_heads
            chunk_idx = (program_id // num_heads) % num_chunks
            batch_idx = program_id // (num_chunks * num_heads)
            chunk_start = chunk_idx * chunk_size

            # solve 阶段用扁平 local matrix 保存一个小 chunk 的下三角矩阵。
            mat = al.make_local((chunk_size * chunk_size,), al.f32)
            row_buf = al.make_local((chunk_size,), al.f32)
            for row_idx in al.range(chunk_size):
                token_idx = chunk_start + row_idx
                for col_idx in al.range(chunk_size):
                    flat_idx = row_idx * chunk_size + col_idx
                    if token_idx < num_tokens:
                        mat[flat_idx] = al.convert(0.0, al.f32) - a[batch_idx, token_idx, head_idx, col_idx]
                    else:
                        mat[flat_idx] = al.convert(0.0, al.f32)

            # 对每一行顺序执行 reference 中的三角递推。
            for row_idx in al.range(1, chunk_size):
                for col_idx in al.range(chunk_size):
                    row_buf[col_idx] = mat[row_idx * chunk_size + col_idx]
                for col_idx in al.range(chunk_size):
                    if col_idx < row_idx:
                        acc = row_buf[col_idx]
                        for inner_idx in al.range(chunk_size):
                            if inner_idx < row_idx:
                                acc = acc + row_buf[inner_idx] * mat[inner_idx * chunk_size + col_idx]
                        mat[row_idx * chunk_size + col_idx] = acc

            # 加单位阵并只写真实 token 对应的行。
            for row_idx in al.range(chunk_size):
                token_idx = chunk_start + row_idx
                if token_idx < num_tokens:
                    for col_idx in al.range(chunk_size):
                        value = mat[row_idx * chunk_size + col_idx]
                        if col_idx == row_idx:
                            value = value + al.convert(1.0, al.f32)
                        out[batch_idx, token_idx, head_idx, col_idx] = value


@avelang.jit
def _qwen_gdn_w_kernel_v4(
    k_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    chunk_size: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, chunk_size),
            (num_tokens * num_v_heads * chunk_size, num_v_heads * chunk_size, chunk_size, 1),
        ),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_k),
            (num_tokens * num_v_heads * head_dim_k, num_v_heads * head_dim_k, head_dim_k, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_tokens * num_v_heads * head_dim_k

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            kk_idx = program_id % head_dim_k
            value_head_idx = (program_id // head_dim_k) % num_v_heads
            token_idx = (program_id // (head_dim_k * num_v_heads)) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads * head_dim_k)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            chunk_start = (token_idx // chunk_size) * chunk_size

            # w 阶段把 solved A 乘回 beta、exp(g) 和对应 source k。
            acc = al.convert(0.0, al.f32)
            for d_idx in al.range(chunk_size):
                source_token = chunk_start + d_idx
                if source_token < num_tokens:
                    acc = acc + a[batch_idx, token_idx, value_head_idx, d_idx] * k[
                        batch_idx,
                        source_token,
                        key_head_idx,
                        kk_idx,
                    ] * beta[batch_idx, source_token, value_head_idx] * al.exp(
                        g[batch_idx, source_token, value_head_idx]
                    )
            out[batch_idx, token_idx, value_head_idx, kk_idx] = acc


@avelang.jit
def _qwen_gdn_u_kernel_v4(
    v_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_heads: al.constexpr,
    head_dim_v: al.constexpr,
    chunk_size: al.constexpr,
):
    v = al.make_tensor(
        v_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_v),
            (num_tokens * num_heads * head_dim_v, num_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_heads), (num_tokens * num_heads, num_heads, 1)),
    )
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, chunk_size),
            (num_tokens * num_heads * chunk_size, num_heads * chunk_size, chunk_size, 1),
        ),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_v),
            (num_tokens * num_heads * head_dim_v, num_heads * head_dim_v, head_dim_v, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_tokens * num_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            head_idx = (program_id // head_dim_v) % num_heads
            token_idx = (program_id // (head_dim_v * num_heads)) % num_tokens
            batch_idx = program_id // (num_tokens * num_heads * head_dim_v)
            chunk_start = (token_idx // chunk_size) * chunk_size

            # u 阶段把 solved A 乘回 beta 和对应 source v。
            acc = al.convert(0.0, al.f32)
            for d_idx in al.range(chunk_size):
                source_token = chunk_start + d_idx
                if source_token < num_tokens:
                    acc = acc + a[batch_idx, token_idx, head_idx, d_idx] * v[
                        batch_idx,
                        source_token,
                        head_idx,
                        value_idx,
                    ] * beta[batch_idx, source_token, head_idx]
            out[batch_idx, token_idx, head_idx, value_idx] = acc


@avelang.jit
def _qwen_gdn_chunk_gdr_kernel_v4(
    k_ptr: al.Pointer(al.f32),
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
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            (num_v_heads * head_dim_k * head_dim_v, head_dim_k * head_dim_v, head_dim_v, 1),
        ),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_chunks, num_v_heads, head_dim_k, head_dim_v),
            (
                num_chunks * num_v_heads * head_dim_k * head_dim_v,
                num_v_heads * head_dim_k * head_dim_v,
                head_dim_k * head_dim_v,
                head_dim_v,
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
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            (num_v_heads * head_dim_k * head_dim_v, head_dim_k * head_dim_v, head_dim_v, 1),
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
                    state[kk] = initial_state[batch_idx, value_head_idx, kk, value_idx]
                else:
                    state[kk] = al.convert(0.0, al.f32)

            for chunk_idx in al.range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                last_token = chunk_start + chunk_size - 1
                if last_token >= num_tokens:
                    last_token = num_tokens - 1

                # 每个 chunk 开始前保存 h，也就是进入该 chunk 前的 recurrent state。
                for kk in al.range(head_dim_k):
                    h[batch_idx, chunk_idx, value_head_idx, kk, value_idx] = state[kk]

                # 先用 chunk 起点 state 计算每个 token 的 v_new/vn。
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        pred = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            pred = pred + w[batch_idx, token_idx, value_head_idx, kk] * state[kk]
                        vn[batch_idx, token_idx, value_head_idx, value_idx] = (
                            u[batch_idx, token_idx, value_head_idx, value_idx] - pred
                        )

                # 再把 state 推进到当前 chunk 末尾。
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
                            state[kk] = state[kk] + k[batch_idx, token_idx, key_head_idx, kk] * decay * v_new

            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, kk, value_idx] = state[kk]


@avelang.jit
def _qwen_gdn_chunk_o_kernel_v4(
    q_ptr: al.Pointer(al.f32),
    k_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
    chunk_size: al.constexpr,
    num_chunks: al.constexpr,
):
    q = al.make_tensor(
        q_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    k = al.make_tensor(
        k_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
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
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_chunks, num_v_heads, head_dim_k, head_dim_v),
            (
                num_chunks * num_v_heads * head_dim_k * head_dim_v,
                num_v_heads * head_dim_k * head_dim_v,
                head_dim_k * head_dim_v,
                head_dim_v,
                1,
            ),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_tokens * num_v_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            value_head_idx = (program_id // head_dim_v) % num_v_heads
            token_idx = (program_id // (head_dim_v * num_v_heads)) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads * head_dim_v)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            chunk_idx = token_idx // chunk_size
            local_pos = token_idx - chunk_idx * chunk_size
            chunk_start = chunk_idx * chunk_size
            scale_f32 = al.convert(scale, al.f32)

            # inter 是 chunk 入口 state 对当前 q 的贡献。
            inter = al.convert(0.0, al.f32)
            q_decay = al.exp(g[batch_idx, token_idx, value_head_idx])
            for kk in al.range(head_dim_k):
                inter = inter + q[batch_idx, token_idx, key_head_idx, kk] * scale_f32 * q_decay * h[
                    batch_idx,
                    chunk_idx,
                    value_head_idx,
                    kk,
                    value_idx,
                ]

            # intra 是当前 chunk 内从 source_token 到 token_idx 的因果贡献，包含对角线。
            intra = al.convert(0.0, al.f32)
            for d_idx in al.range(chunk_size):
                if d_idx <= local_pos:
                    source_token = chunk_start + d_idx
                    if source_token < num_tokens:
                        dot = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            dot = dot + q[batch_idx, token_idx, key_head_idx, kk] * scale_f32 * k[
                                batch_idx,
                                source_token,
                                key_head_idx,
                                kk,
                            ]
                        decay = al.exp(
                            g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx]
                        )
                        intra = intra + dot * decay * vn[batch_idx, source_token, value_head_idx, value_idx]

            out[batch_idx, token_idx, value_head_idx, value_idx] = inter + intra


def _require_fp32_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_qkvgb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")

    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, Hk, K], got {tuple(q.shape)}.")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}.")
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B, T, Hv, V], got {tuple(v.shape)}.")
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, Hv], got {tuple(g.shape)}.")
    if beta.shape != g.shape:
        raise ValueError(f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}.")

    batch_size, num_tokens, num_k_heads, head_dim_k = q.shape
    v_batch, v_tokens, num_v_heads, head_dim_v = v.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    if (v_batch, v_tokens) != (batch_size, num_tokens):
        raise ValueError(f"v must share q's [B, T], got {tuple(v.shape[:2])}.")
    if g.shape != (batch_size, num_tokens, num_v_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_v_heads)}, got {tuple(g.shape)}.")
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")

    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


def _validate_chunk_size(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")


def _num_chunks(num_tokens: int, chunk_size: int) -> int:
    return (num_tokens + chunk_size - 1) // chunk_size


def qwen_gdn_chunk_cumsum_avelang_v4(g: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """运行 v4 的 g chunk-local cumsum，目前直接复用 v3 的已验证实现。"""
    return qwen_gdn_chunk_cumsum_avelang_v3(g, chunk_size=chunk_size)


def qwen_gdn_kkt_avelang_v4(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """计算 KKT 阶段的严格过去注意力矩阵。"""
    _validate_chunk_size(chunk_size)
    for name, tensor in (("k", k), ("g", g), ("beta", beta)):
        _require_fp32_cuda_contiguous(name, tensor)
    if k.ndim != 4:
        raise ValueError(f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}.")
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, Hv], got {tuple(g.shape)}.")
    if beta.shape != g.shape:
        raise ValueError(f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}.")

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    if g.shape[:2] != (batch_size, num_tokens):
        raise ValueError(f"g must share k's [B, T], got {tuple(g.shape[:2])}.")
    num_v_heads = g.shape[2]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")

    out = torch.empty((batch_size, num_tokens, num_v_heads, chunk_size), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads * chunk_size
    _qwen_gdn_kkt_kernel_v4[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        k,
        g,
        beta,
        out,
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        chunk_size,
    )
    return out


def qwen_gdn_solve_avelang_v4(a: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """对每个 chunk/head 的小矩阵执行 reference 等价的三角 solve。"""
    _validate_chunk_size(chunk_size)
    _require_fp32_cuda_contiguous("a", a)
    if a.ndim != 4:
        raise ValueError(f"a must have shape [B, T, H, C], got {tuple(a.shape)}.")
    batch_size, num_tokens, num_heads, a_chunk_size = a.shape
    if a_chunk_size != chunk_size:
        raise ValueError(f"a last dim must equal chunk_size={chunk_size}, got {a_chunk_size}.")
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")

    num_chunks = _num_chunks(num_tokens, chunk_size)
    out = torch.empty_like(a)
    grid_size = batch_size * num_chunks * num_heads
    _qwen_gdn_solve_kernel_v4[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        a,
        out,
        batch_size,
        num_tokens,
        num_heads,
        chunk_size,
        num_chunks,
    )
    return out


def qwen_gdn_w_u_avelang_v4(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 chunk 内校正后的 w 和 u。"""
    _validate_chunk_size(chunk_size)
    for name, tensor in (("k", k), ("v", v), ("g", g), ("beta", beta), ("a_solved", a_solved)):
        _require_fp32_cuda_contiguous(name, tensor)
    if k.ndim != 4:
        raise ValueError(f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}.")
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B, T, Hv, V], got {tuple(v.shape)}.")
    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    if v.shape[:2] != (batch_size, num_tokens):
        raise ValueError(f"v must share k's [B, T], got {tuple(v.shape[:2])}.")
    num_v_heads = v.shape[2]
    head_dim_v = v.shape[3]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")
    if g.shape != (batch_size, num_tokens, num_v_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_v_heads)}, got {tuple(g.shape)}.")
    if beta.shape != g.shape:
        raise ValueError(f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}.")
    expected_a_shape = (batch_size, num_tokens, num_v_heads, chunk_size)
    if a_solved.shape != expected_a_shape:
        raise ValueError(f"a_solved must have shape {expected_a_shape}, got {tuple(a_solved.shape)}.")

    w = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_k), dtype=torch.float32, device=k.device)
    u = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_v), dtype=torch.float32, device=k.device)

    w_grid = batch_size * num_tokens * num_v_heads * head_dim_k
    _qwen_gdn_w_kernel_v4[lambda: ((w_grid, 1, 1), (1, 1, 1))](
        k,
        g,
        beta,
        a_solved,
        w,
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        chunk_size,
    )
    u_grid = batch_size * num_tokens * num_v_heads * head_dim_v
    _qwen_gdn_u_kernel_v4[lambda: ((u_grid, 1, 1), (1, 1, 1))](
        v,
        beta,
        a_solved,
        u,
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        chunk_size,
    )
    return w, u


def qwen_gdn_chunk_gdr_avelang_v4(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 chunk 入口 state 生成 h、vn 和 final_state。"""
    _validate_chunk_size(chunk_size)
    for name, tensor in (("k", k), ("w", w), ("u", u), ("g", g)):
        _require_fp32_cuda_contiguous(name, tensor)
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

    state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state_arg = None
    else:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
        if initial_state.device != k.device:
            raise ValueError(f"initial_state must be on device {k.device}, got {initial_state.device}.")
        initial_state_arg = initial_state

    num_chunks = _num_chunks(num_tokens, chunk_size)
    h = torch.empty((batch_size, num_chunks, num_v_heads, head_dim_k, head_dim_v), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty(state_shape, dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    grid_size = batch_size * num_v_heads * head_dim_v
    _qwen_gdn_chunk_gdr_kernel_v4[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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
        has_initial_state,
    )
    return h, vn, final_state


def qwen_gdn_chunk_o_avelang_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = 4,
) -> torch.Tensor:
    """根据 chunk state 和 chunk 内 vn 生成最终 output。"""
    _validate_chunk_size(chunk_size)
    for name, tensor in (("q", q), ("k", k), ("vn", vn), ("h", h), ("g", g)):
        _require_fp32_cuda_contiguous(name, tensor)
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, Hk, K], got {tuple(q.shape)}.")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}.")
    batch_size, num_tokens, num_k_heads, head_dim_k = q.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
    if vn.ndim != 4:
        raise ValueError(f"vn must have shape [B, T, Hv, V], got {tuple(vn.shape)}.")
    if vn.shape[:2] != (batch_size, num_tokens):
        raise ValueError(f"vn must share q's [B, T], got {tuple(vn.shape[:2])}.")
    num_v_heads = vn.shape[2]
    head_dim_v = vn.shape[3]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")
    num_chunks = _num_chunks(num_tokens, chunk_size)
    expected_h_shape = (batch_size, num_chunks, num_v_heads, head_dim_k, head_dim_v)
    if h.shape != expected_h_shape:
        raise ValueError(f"h must have shape {expected_h_shape}, got {tuple(h.shape)}.")
    if g.shape != (batch_size, num_tokens, num_v_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_v_heads)}, got {tuple(g.shape)}.")
    if scale is None:
        scale = head_dim_k**-0.5

    out = torch.empty_like(vn)
    grid_size = batch_size * num_tokens * num_v_heads * head_dim_v
    _qwen_gdn_chunk_o_kernel_v4[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        q,
        k,
        vn,
        h,
        g,
        out,
        float(scale),
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        chunk_size,
        num_chunks,
    )
    return out


def qwen_gdn_chunked_avelang_v4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """串起 v4 的所有 correctness-first chunked forward 阶段。"""
    _validate_chunk_size(chunk_size)
    batch_size, num_tokens, _, num_v_heads, head_dim_k, head_dim_v = _validate_qkvgb(q, k, v, g, beta)
    if initial_state is not None:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v4(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v4(k, g_cumsum, beta, chunk_size=chunk_size)
    a_solved = qwen_gdn_solve_avelang_v4(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v4(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v4(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output = qwen_gdn_chunk_o_avelang_v4(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk_size)
    return g_cumsum, output, a_solved, h, final_state

"""Qwen GDN v6：支持 BF16 输入的完整 chunked forward 正确性版本。

当前版本实现 `q/k/v` 为 BF16、其余状态和累加为 FP32 的 Qwen-style chunked
GDN 前向流水线，并继续返回 `(g_cumsum, output, A_solved, chunk_states, final_state)`。
相比 v5，本版本把 v2 已验证的 BF16 读取与 FP32 累加合同补入完整 chunked
pipeline；这一步以目标 dtype 的正确性为目的，不声称新增性能收益。
当前版本仍然不支持 backward、cu_seqlens、variable-length packed input、
raw_buffer、shared memory、MFMA 或不生成 chunk_states 的 benchmark 专用接口。
真正沿用的结构优化是 v5 的 KKT、融合 w/u 与 chunk_o 映射；新加入的 BF16
kernel 和复用的 cumsum/solve 均属于 correctness-first baseline。
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v4 import _num_chunks, _require_fp32_cuda_contiguous, _validate_chunk_size
from qwen_gdn_chunked_avelang_v5 import (
    qwen_gdn_chunk_cumsum_avelang_v5,
    qwen_gdn_chunk_gdr_avelang_v5,
    qwen_gdn_chunk_o_avelang_v5,
    qwen_gdn_chunked_avelang_v5,
    qwen_gdn_kkt_avelang_v5,
    qwen_gdn_solve_avelang_v5,
    qwen_gdn_w_u_avelang_v5,
)


V6_STAGE_STRATEGY = {
    "g_cumsum": "复用 v5：输入本来就是 FP32，不改变已验证的 chunk 内累加。",
    "kkt": "BF16 correctness 路径：沿用 v5 整行映射，读取 key 后转为 FP32 运算。",
    "solve": "复用 v5：A 保持 FP32，小矩阵求解顺序不变。",
    "w_u": "BF16 correctness 路径：沿用 v5 融合映射，k/v 读取后转为 FP32 累加。",
    "chunk_gdr": "BF16 correctness 路径：沿用 v4 状态推进，k 读取后转为 FP32。",
    "chunk_o": "BF16 correctness 路径：沿用 v5 输出映射，q/k 读取后转为 FP32。",
}


@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_v6(
    k_ptr: al.Pointer(al.bf16),
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
        al.bf16,
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
    total_programs = batch_size * num_tokens * num_v_heads

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_head_idx = program_id % num_v_heads
            token_idx = (program_id // num_v_heads) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            local_pos = token_idx % chunk_size
            chunk_start = token_idx - local_pos

            # KKT 阶段缓存当前 token 的 BF16 key，并立即提升为 FP32 参与点积。
            target_k = al.make_local((head_dim_k,), al.f32)
            for kk in al.range(head_dim_k):
                target_k[kk] = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)

            # 每个 program 写出当前 token 的整行，严格过去以外的位置保持为零。
            for d_idx in al.range(chunk_size):
                value = al.convert(0.0, al.f32)
                if d_idx < local_pos:
                    source_token = chunk_start + d_idx
                    if source_token < num_tokens:
                        dot = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            source_k = al.convert(k[batch_idx, source_token, key_head_idx, kk], al.f32)
                            dot = dot + target_k[kk] * source_k
                        decay = al.exp(
                            g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx]
                        )
                        value = beta[batch_idx, token_idx, value_head_idx] * dot * decay
                out[batch_idx, token_idx, value_head_idx, d_idx] = value


@avelang.jit
def _qwen_gdn_w_u_bf16_kernel_v6(
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.f32),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
    chunk_size: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    v = al.make_tensor(
        v_ptr,
        al.bf16,
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

    program_id = al.block_id(0)
    total_programs = batch_size * num_tokens * num_v_heads

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_head_idx = program_id % num_v_heads
            token_idx = (program_id // num_v_heads) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            chunk_start = (token_idx // chunk_size) * chunk_size

            # 融合阶段始终使用 FP32 local buffer 累加，避免把中间结果压回 BF16。
            w_acc = al.make_local((head_dim_k,), al.f32)
            u_acc = al.make_local((head_dim_v,), al.f32)
            for kk in al.range(head_dim_k):
                w_acc[kk] = al.convert(0.0, al.f32)
            for vv in al.range(head_dim_v):
                u_acc[vv] = al.convert(0.0, al.f32)

            # 单次扫描 solved A，同时将 BF16 k/v 扩展为 FP32 后分别累加到 w 与 u。
            for d_idx in al.range(chunk_size):
                source_token = chunk_start + d_idx
                if source_token < num_tokens:
                    a_value = a[batch_idx, token_idx, value_head_idx, d_idx]
                    beta_value = beta[batch_idx, source_token, value_head_idx]
                    g_value = g[batch_idx, source_token, value_head_idx]
                    for kk in al.range(head_dim_k):
                        k_value = al.convert(k[batch_idx, source_token, key_head_idx, kk], al.f32)
                        w_acc[kk] = w_acc[kk] + a_value * k_value * beta_value * al.exp(g_value)
                    for vv in al.range(head_dim_v):
                        v_value = al.convert(v[batch_idx, source_token, value_head_idx, vv], al.f32)
                        u_acc[vv] = u_acc[vv] + a_value * v_value * beta_value

            # 写回的 w/u 是后续状态递推使用的 FP32 中间张量。
            for kk in al.range(head_dim_k):
                w[batch_idx, token_idx, value_head_idx, kk] = w_acc[kk]
            for vv in al.range(head_dim_v):
                u[batch_idx, token_idx, value_head_idx, vv] = u_acc[vv]


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v6(
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

            # 状态始终保存在 FP32 local buffer 中，与 v2 的 BF16 累加合同一致。
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

                # 保存当前 chunk 入口的 FP32 state，供输出阶段计算跨 chunk 贡献。
                for kk in al.range(head_dim_k):
                    h[batch_idx, chunk_idx, value_head_idx, kk, value_idx] = state[kk]

                # 使用 chunk 入口 state 先生成每个 token 的校正后 value。
                for offset in al.range(chunk_size):
                    token_idx = chunk_start + offset
                    if token_idx < num_tokens:
                        pred = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            pred = pred + w[batch_idx, token_idx, value_head_idx, kk] * state[kk]
                        vn[batch_idx, token_idx, value_head_idx, value_idx] = (
                            u[batch_idx, token_idx, value_head_idx, value_idx] - pred
                        )

                # 先应用 chunk 末端衰减，再用转为 FP32 的 BF16 key 推进状态。
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

            # 最终状态以 FP32 输出，用于和 reference 以及 recurrent 路径逐张量比较。
            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, kk, value_idx] = state[kk]


@avelang.jit
def _qwen_gdn_chunk_o_bf16_kernel_v6(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
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
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
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
    total_programs = batch_size * num_tokens * num_v_heads

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_head_idx = program_id % num_v_heads
            token_idx = (program_id // num_v_heads) % num_tokens
            batch_idx = program_id // (num_tokens * num_v_heads)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat
            chunk_idx = token_idx // chunk_size
            local_pos = token_idx - chunk_idx * chunk_size
            chunk_start = chunk_idx * chunk_size
            scale_f32 = al.convert(scale, al.f32)

            # 输出阶段把 BF16 q 提升到 FP32，并缓存所有 value 维共享的因果权重。
            q_scaled = al.make_local((head_dim_k,), al.f32)
            intra_weight = al.make_local((chunk_size,), al.f32)
            for kk in al.range(head_dim_k):
                q_value = al.convert(q[batch_idx, token_idx, key_head_idx, kk], al.f32)
                q_scaled[kk] = q_value * scale_f32
            for d_idx in al.range(chunk_size):
                weight = al.convert(0.0, al.f32)
                if d_idx <= local_pos:
                    source_token = chunk_start + d_idx
                    if source_token < num_tokens:
                        dot = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            k_value = al.convert(k[batch_idx, source_token, key_head_idx, kk], al.f32)
                            dot = dot + q_scaled[kk] * k_value
                        decay = al.exp(
                            g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx]
                        )
                        weight = dot * decay
                intra_weight[d_idx] = weight

            # 每个输出 value 复用 FP32 权重，并合并入口 state 与 chunk 内贡献。
            q_decay = al.exp(g[batch_idx, token_idx, value_head_idx])
            for vv in al.range(head_dim_v):
                inter = al.convert(0.0, al.f32)
                for kk in al.range(head_dim_k):
                    inter = inter + q_scaled[kk] * q_decay * h[batch_idx, chunk_idx, value_head_idx, kk, vv]

                intra = al.convert(0.0, al.f32)
                for d_idx in al.range(chunk_size):
                    if d_idx <= local_pos:
                        source_token = chunk_start + d_idx
                        if source_token < num_tokens:
                            intra = intra + intra_weight[d_idx] * vn[batch_idx, source_token, value_head_idx, vv]
                out[batch_idx, token_idx, value_head_idx, vv] = inter + intra


def _require_bf16_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_bf16_qkvgb(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _require_bf16_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")
    for name, tensor in (("g", g), ("beta", beta)):
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


def _validate_bf16_k_stage(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
    _validate_chunk_size(chunk_size)
    _require_bf16_cuda_contiguous("k", k)
    for name, tensor in (("g", g), ("beta", beta)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k


def _validate_bf16_w_u_stage(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    _validate_chunk_size(chunk_size)
    for name, tensor in (("k", k), ("v", v)):
        _require_bf16_cuda_contiguous(name, tensor)
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
    for name, tensor in (("g", g), ("beta", beta), ("a_solved", a_solved)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


def _validate_bf16_chunk_gdr_stage(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    _validate_chunk_size(chunk_size)
    _require_bf16_cuda_contiguous("k", k)
    for name, tensor in (("w", w), ("u", u), ("g", g)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != k.device:
            raise ValueError(f"{name} must be on device {k.device}, got {tensor.device}.")
    if k.ndim != 4:
        raise ValueError(f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}.")
    if u.ndim != 4:
        raise ValueError(f"u must have shape [B, T, Hv, V], got {tuple(u.shape)}.")
    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    if num_tokens <= 0:
        raise ValueError("T must be > 0.")
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


def _validate_bf16_chunk_o_stage(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int, int]:
    _validate_chunk_size(chunk_size)
    for name, tensor in (("q", q), ("k", k)):
        _require_bf16_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")
    for name, tensor in (("vn", vn), ("h", h), ("g", g)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks


def qwen_gdn_chunk_cumsum_avelang_v6(g: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """运行 v6 的 FP32 chunk-local cumsum，直接复用已验证实现。"""
    return qwen_gdn_chunk_cumsum_avelang_v5(g, chunk_size=chunk_size)


def qwen_gdn_kkt_avelang_v6(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> torch.Tensor:
    """计算 KKT；FP32 委托给 v5，BF16 使用转为 FP32 累加的新路径。"""
    if k.dtype == torch.float32:
        return qwen_gdn_kkt_avelang_v5(k, g, beta, chunk_size=chunk_size, prefer_optimized=prefer_optimized)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k = _validate_bf16_k_stage(
        k,
        g,
        beta,
        chunk_size,
    )
    if not prefer_optimized:
        return qwen_gdn_kkt_avelang_v5(k.float(), g, beta, chunk_size=chunk_size, prefer_optimized=False)
    out = torch.empty((batch_size, num_tokens, num_v_heads, chunk_size), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_kkt_bf16_kernel_v6[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_solve_avelang_v6(a: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """运行 v6 的 FP32 小矩阵 solve，复用 v5 已验证递推顺序。"""
    return qwen_gdn_solve_avelang_v5(a, chunk_size=chunk_size)


def qwen_gdn_w_u_avelang_v6(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 w/u；BF16 输入在 kernel 内提升至 FP32 后完成融合累加。"""
    if k.dtype == v.dtype == torch.float32:
        return qwen_gdn_w_u_avelang_v5(
            k,
            v,
            g,
            beta,
            a_solved,
            chunk_size=chunk_size,
            prefer_optimized=prefer_optimized,
        )
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_w_u_stage(
        k,
        v,
        g,
        beta,
        a_solved,
        chunk_size,
    )
    if not prefer_optimized:
        return qwen_gdn_w_u_avelang_v5(
            k.float(),
            v.float(),
            g,
            beta,
            a_solved,
            chunk_size=chunk_size,
            prefer_optimized=False,
        )
    w = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_k), dtype=torch.float32, device=k.device)
    u = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_v), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_w_u_bf16_kernel_v6[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        k,
        v,
        g,
        beta,
        a_solved,
        w,
        u,
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        chunk_size,
    )
    return w, u


def qwen_gdn_chunk_gdr_avelang_v6(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """根据 chunk 入口状态生成 FP32 h、vn 和 final_state。"""
    if k.dtype == torch.float32:
        return qwen_gdn_chunk_gdr_avelang_v5(k, w, u, g, initial_state=initial_state, chunk_size=chunk_size)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k,
        w,
        u,
        g,
        chunk_size,
    )
    if not prefer_optimized:
        return qwen_gdn_chunk_gdr_avelang_v5(k.float(), w, u, g, initial_state=initial_state, chunk_size=chunk_size)
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
    _qwen_gdn_chunk_gdr_bf16_kernel_v6[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunk_o_avelang_v6(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> torch.Tensor:
    """生成 FP32 output；BF16 q/k 只在读取后以 FP32 参与计算。"""
    if q.dtype == k.dtype == torch.float32:
        return qwen_gdn_chunk_o_avelang_v5(
            q,
            k,
            vn,
            h,
            g,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=prefer_optimized,
        )
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks = (
        _validate_bf16_chunk_o_stage(q, k, vn, h, g, chunk_size)
    )
    if not prefer_optimized:
        return qwen_gdn_chunk_o_avelang_v5(
            q.float(),
            k.float(),
            vn,
            h,
            g,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=False,
        )
    if scale is None:
        scale = head_dim_k**-0.5
    out = torch.empty_like(vn)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_chunk_o_bf16_kernel_v6[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunked_avelang_v6(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """运行完整 v6 forward，支持统一的 FP32 或统一的 BF16 q/k/v 输入。"""
    if q.dtype == k.dtype == v.dtype == torch.float32:
        return qwen_gdn_chunked_avelang_v5(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=prefer_optimized,
        )
    _validate_chunk_size(chunk_size)
    batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_qkvgb(q, k, v, g, beta)
    if initial_state is not None:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
        if initial_state.device != q.device:
            raise ValueError(f"initial_state must be on device {q.device}, got {initial_state.device}.")

    # cumsum 与 solve 的输入输出原本即为 FP32，因此直接保持已验证的递推路径。
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6(
        k,
        g_cumsum,
        beta,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    a_solved = qwen_gdn_solve_avelang_v6(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v6(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    output = qwen_gdn_chunk_o_avelang_v6(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    return g_cumsum, output, a_solved, h, final_state

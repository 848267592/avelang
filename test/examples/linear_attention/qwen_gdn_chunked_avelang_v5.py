"""Qwen GDN v5：最终的半优化 chunked forward 实现。

当前版本实现完整 Qwen-style chunked GDN 前向流水线，并继续返回
`(g_cumsum, output, A_solved, chunk_states, final_state)` 五元组。
相比 v4，本版本真正优化了三个阶段：KKT 以 `(b, t, hv)` 为 program 粒度并缓存目标
key，w/u 合并为单 kernel，chunk_o 以 `(b, t, hv)` 为 program 粒度并复用 chunk 内
注意力权重；同时提供 `prefer_optimized=False` 的 v4 回退路径。
当前版本仍然不支持 bf16、backward、cu_seqlens、raw_buffer、shared memory、
MFMA 或并行 scan，也没有改变 chunk_gdr 的串行 value 维映射。
真正优化部分是 KKT、融合 w/u 和 chunk_o；cumsum、solve、chunk_gdr 仍然是
correctness-first fallback，因为当前 checkout 没有可直接迁移并验证的相应并行模式。
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v4 import (
    _num_chunks,
    _require_fp32_cuda_contiguous,
    _validate_chunk_size,
    _validate_qkvgb,
    qwen_gdn_chunk_cumsum_avelang_v4,
    qwen_gdn_chunk_gdr_avelang_v4,
    qwen_gdn_chunk_o_avelang_v4,
    qwen_gdn_kkt_avelang_v4,
    qwen_gdn_solve_avelang_v4,
    qwen_gdn_w_u_avelang_v4,
)


V5_STAGE_STRATEGY = {
    "g_cumsum": "复用 v3/v4：当前没有已验证的并行 scan 模式。",
    "kkt": "优化：一个 program 处理一个 token/head，并在 local buffer 中复用目标 key。",
    "solve": "复用 v4：小下三角矩阵递推优先保持求解顺序。",
    "w_u": "优化：单个 program 同时写出 w 和 u，复用 solved A、beta 与 source 遍历。",
    "chunk_gdr": "复用 v4：跨 chunk 状态依赖保持 correctness-first 的 value 维映射。",
    "chunk_o": "优化：一个 program 写出全部 value 维，并复用 q/k 的 chunk 内点积。",
}


@avelang.jit
def _qwen_gdn_kkt_kernel_v5(
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

            # KKT 优化阶段先缓存目标 token 的 key，使 chunk 中每个 d 复用同一行读取。
            target_k = al.make_local((head_dim_k,), al.f32)
            for kk in al.range(head_dim_k):
                target_k[kk] = k[batch_idx, token_idx, key_head_idx, kk]

            # KKT 仍严格屏蔽对角线和未来位置，并一次写完当前 token 的整行。
            for d_idx in al.range(chunk_size):
                value = al.convert(0.0, al.f32)
                if d_idx < local_pos:
                    source_token = chunk_start + d_idx
                    if source_token < num_tokens:
                        dot = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            dot = dot + target_k[kk] * k[batch_idx, source_token, key_head_idx, kk]
                        decay = al.exp(
                            g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx]
                        )
                        value = beta[batch_idx, token_idx, value_head_idx] * dot * decay
                out[batch_idx, token_idx, value_head_idx, d_idx] = value


@avelang.jit
def _qwen_gdn_w_u_kernel_v5(
    k_ptr: al.Pointer(al.f32),
    v_ptr: al.Pointer(al.f32),
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
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    v = al.make_tensor(
        v_ptr,
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

            # 融合阶段在 local buffer 中保存当前 token 的两类输出累加器。
            w_acc = al.make_local((head_dim_k,), al.f32)
            u_acc = al.make_local((head_dim_v,), al.f32)
            for kk in al.range(head_dim_k):
                w_acc[kk] = al.convert(0.0, al.f32)
            for vv in al.range(head_dim_v):
                u_acc[vv] = al.convert(0.0, al.f32)

            # 单次遍历 source token，同时累加 w 与 u，避免两个独立 kernel 重复扫描 A。
            for d_idx in al.range(chunk_size):
                source_token = chunk_start + d_idx
                if source_token < num_tokens:
                    a_value = a[batch_idx, token_idx, value_head_idx, d_idx]
                    beta_value = beta[batch_idx, source_token, value_head_idx]
                    g_value = g[batch_idx, source_token, value_head_idx]
                    for kk in al.range(head_dim_k):
                        w_acc[kk] = (
                            w_acc[kk]
                            + a_value
                            * k[batch_idx, source_token, key_head_idx, kk]
                            * beta_value
                            * al.exp(g_value)
                        )
                    for vv in al.range(head_dim_v):
                        u_acc[vv] = (
                            u_acc[vv] + a_value * v[batch_idx, source_token, value_head_idx, vv] * beta_value
                        )

            # 融合 kernel 结束时一次写回当前 token/head 的完整 w 与 u。
            for kk in al.range(head_dim_k):
                w[batch_idx, token_idx, value_head_idx, kk] = w_acc[kk]
            for vv in al.range(head_dim_v):
                u[batch_idx, token_idx, value_head_idx, vv] = u_acc[vv]


@avelang.jit
def _qwen_gdn_chunk_o_kernel_v5(
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

            # 输出优化阶段缓存缩放后的 q 和与 value 维无关的 chunk 内注意力权重。
            q_scaled = al.make_local((head_dim_k,), al.f32)
            intra_weight = al.make_local((chunk_size,), al.f32)
            for kk in al.range(head_dim_k):
                q_scaled[kk] = q[batch_idx, token_idx, key_head_idx, kk] * scale_f32
            for d_idx in al.range(chunk_size):
                weight = al.convert(0.0, al.f32)
                if d_idx <= local_pos:
                    source_token = chunk_start + d_idx
                    if source_token < num_tokens:
                        dot = al.convert(0.0, al.f32)
                        for kk in al.range(head_dim_k):
                            dot = dot + q_scaled[kk] * k[batch_idx, source_token, key_head_idx, kk]
                        decay = al.exp(
                            g[batch_idx, token_idx, value_head_idx] - g[batch_idx, source_token, value_head_idx]
                        )
                        weight = dot * decay
                intra_weight[d_idx] = weight

            # 每个 value 输出共享同一组 q/k 权重，只单独读取 state 与 vn。
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


def _validate_kkt_inputs(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int]:
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k


def _validate_w_u_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


def _validate_chunk_o_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int, int]:
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
    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks


def qwen_gdn_chunk_cumsum_avelang_v5(g: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """复用已验证的 chunk-local cumsum；当前没有可靠的并行 scan 替代路径。"""
    return qwen_gdn_chunk_cumsum_avelang_v4(g, chunk_size=chunk_size)


def qwen_gdn_kkt_avelang_v5(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> torch.Tensor:
    """计算 KKT，并允许切换到 v4 标量 program 回退路径。"""
    if not prefer_optimized:
        return qwen_gdn_kkt_avelang_v4(k, g, beta, chunk_size=chunk_size)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k = _validate_kkt_inputs(k, g, beta, chunk_size)
    out = torch.empty((batch_size, num_tokens, num_v_heads, chunk_size), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_kkt_kernel_v5[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_solve_avelang_v5(a: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """复用 v4 的小矩阵三角递推 solve。"""
    return qwen_gdn_solve_avelang_v4(a, chunk_size=chunk_size)


def qwen_gdn_w_u_avelang_v5(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 w/u，并允许切换到 v4 分离 kernel 回退路径。"""
    if not prefer_optimized:
        return qwen_gdn_w_u_avelang_v4(k, v, g, beta, a_solved, chunk_size=chunk_size)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_w_u_inputs(
        k,
        v,
        g,
        beta,
        a_solved,
        chunk_size,
    )
    w = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_k), dtype=torch.float32, device=k.device)
    u = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_v), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_w_u_kernel_v5[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunk_gdr_avelang_v5(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """复用 v4 的跨 chunk 状态推进路径。"""
    return qwen_gdn_chunk_gdr_avelang_v4(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )


def qwen_gdn_chunk_o_avelang_v5(
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
    """生成 output，并允许切换到 v4 单 value 输出回退路径。"""
    if not prefer_optimized:
        return qwen_gdn_chunk_o_avelang_v4(q, k, vn, h, g, scale=scale, chunk_size=chunk_size)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks = _validate_chunk_o_inputs(
        q,
        k,
        vn,
        h,
        g,
        chunk_size,
    )
    if scale is None:
        scale = head_dim_k**-0.5
    out = torch.empty_like(vn)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_chunk_o_kernel_v5[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunked_avelang_v5(
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
    """串联 v5 完整前向流水线，并可整体切换到 v4 stage 回退实现。"""
    _validate_chunk_size(chunk_size)
    batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_qkvgb(q, k, v, g, beta)
    if initial_state is not None:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v5(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v5(
        k,
        g_cumsum,
        beta,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    a_solved = qwen_gdn_solve_avelang_v5(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v5(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v5(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output = qwen_gdn_chunk_o_avelang_v5(
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

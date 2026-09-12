"""Qwen GDN v6 standalone：自包含的 chunked forward 正确性版本。

这是 v6 standalone 版本，用于后续优化前固定一份不依赖旧版本 wrapper 的基线。
本文件从 v3 内联了 chunk-local cumsum kernel，从 v4 内联了 correctness-first
KKT、solve、w/u、chunk_gdr 和 chunk_o fallback，从 v5 内联了 FP32 KKT、融合
w/u 与 chunk_o 的当前优化映射，并从 v6 内联了 BF16 q/k/v 读取后以 FP32 累加的
KKT、w/u、chunk_gdr 和 chunk_o 路径。
本文件没有新增性能优化，只做依赖消除和函数改名；数学行为、dtype 支持范围与
shape 支持范围保持和现有 v6 一致。
当前支持 q/k/v 全为 FP32，或 q/k/v 全为 BF16 且 g/beta/initial_state 为 FP32；
输出、A_solved、chunk_states 与 final_state 均为 FP32。
当前仍然不支持 backward、cu_seqlens、variable-length packed input、raw_buffer、
shared memory、MFMA、parallel scan，或 benchmark 专用的省略 chunk_states 接口。
"""

import torch

import avelang
import avelang.language as al


V6_STANDALONE_STAGE_STRATEGY = {
    "g_cumsum": "内联 v3 correctness-first chunk-local cumsum。",
    "kkt": "FP32 内联 v5 映射，BF16 内联 v6 映射，fallback 内联 v4 映射。",
    "solve": "内联 v4 correctness-first 小矩阵递推。",
    "w_u": "FP32 内联 v5 融合映射，BF16 内联 v6 融合映射，fallback 内联 v4 分离映射。",
    "chunk_gdr": "FP32 内联 v4 状态推进，BF16 内联 v6 状态推进。",
    "chunk_o": "FP32 内联 v5 输出映射，BF16 内联 v6 输出映射，fallback 内联 v4 标量输出。",
}

@avelang.jit
def _qwen_gdn_chunk_cumsum_kernel_v6_standalone(
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

@avelang.jit
def _qwen_gdn_kkt_fallback_kernel_v6_standalone(
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
def _qwen_gdn_solve_kernel_v6_standalone(
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
def _qwen_gdn_w_fallback_kernel_v6_standalone(
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
def _qwen_gdn_u_fallback_kernel_v6_standalone(
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
def _qwen_gdn_chunk_gdr_fp32_kernel_v6_standalone(
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
def _qwen_gdn_chunk_o_fallback_kernel_v6_standalone(
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

def qwen_gdn_chunk_cumsum_avelang_v6_standalone(g: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
    """运行 standalone 的 FP32 chunk-local cumsum。"""
    _require_fp32_cuda_contiguous("g", g)
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, H], got {tuple(g.shape)}.")
    _validate_chunk_size(chunk_size)

    batch_size, num_tokens, num_heads = g.shape
    out = torch.empty_like(g)
    num_chunks = _num_chunks(num_tokens, chunk_size)
    grid_size = batch_size * num_heads * num_chunks
    _qwen_gdn_chunk_cumsum_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        g,
        out,
        batch_size,
        num_tokens,
        num_heads,
        chunk_size,
        num_chunks,
    )
    return out

def _qwen_gdn_kkt_fallback_v6_standalone(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
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
    _qwen_gdn_kkt_fallback_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_solve_avelang_v6_standalone(a: torch.Tensor, *, chunk_size: int = 4) -> torch.Tensor:
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
    _qwen_gdn_solve_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        a,
        out,
        batch_size,
        num_tokens,
        num_heads,
        chunk_size,
        num_chunks,
    )
    return out


def _qwen_gdn_w_u_fallback_v6_standalone(
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
    _qwen_gdn_w_fallback_kernel_v6_standalone[lambda: ((w_grid, 1, 1), (1, 1, 1))](
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
    _qwen_gdn_u_fallback_kernel_v6_standalone[lambda: ((u_grid, 1, 1), (1, 1, 1))](
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


def _qwen_gdn_chunk_gdr_fp32_v6_standalone(
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
    _qwen_gdn_chunk_gdr_fp32_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def _qwen_gdn_chunk_o_fallback_v6_standalone(
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
    _qwen_gdn_chunk_o_fallback_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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

@avelang.jit
def _qwen_gdn_kkt_fp32_opt_kernel_v6_standalone(
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
def _qwen_gdn_w_u_fp32_opt_kernel_v6_standalone(
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
def _qwen_gdn_chunk_o_fp32_opt_kernel_v6_standalone(
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

@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_v6_standalone(
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
def _qwen_gdn_w_u_bf16_kernel_v6_standalone(
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
def _qwen_gdn_chunk_gdr_bf16_kernel_v6_standalone(
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
def _qwen_gdn_chunk_o_bf16_kernel_v6_standalone(
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

def qwen_gdn_kkt_avelang_v6_standalone(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> torch.Tensor:
    """计算 KKT；FP32 使用内联 v5/v4 路径，BF16 使用内联 v6/v4 路径。"""
    if k.dtype == torch.float32:
        if not prefer_optimized:
            return _qwen_gdn_kkt_fallback_v6_standalone(k, g, beta, chunk_size=chunk_size)
        batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k = _validate_kkt_inputs(
            k,
            g,
            beta,
            chunk_size,
        )
        out = torch.empty((batch_size, num_tokens, num_v_heads, chunk_size), dtype=torch.float32, device=k.device)
        grid_size = batch_size * num_tokens * num_v_heads
        _qwen_gdn_kkt_fp32_opt_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k = _validate_bf16_k_stage(
        k,
        g,
        beta,
        chunk_size,
    )
    if not prefer_optimized:
        return _qwen_gdn_kkt_fallback_v6_standalone(k.float(), g, beta, chunk_size=chunk_size)
    out = torch.empty((batch_size, num_tokens, num_v_heads, chunk_size), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_kkt_bf16_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_w_u_avelang_v6_standalone(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = 4,
    prefer_optimized: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 w/u；FP32 使用内联 v5/v4 路径，BF16 使用内联 v6/v4 路径。"""
    if k.dtype == v.dtype == torch.float32:
        if not prefer_optimized:
            return _qwen_gdn_w_u_fallback_v6_standalone(k, v, g, beta, a_solved, chunk_size=chunk_size)
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
        _qwen_gdn_w_u_fp32_opt_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_w_u_stage(
        k,
        v,
        g,
        beta,
        a_solved,
        chunk_size,
    )
    if not prefer_optimized:
        return _qwen_gdn_w_u_fallback_v6_standalone(
            k.float(),
            v.float(),
            g,
            beta,
            a_solved,
            chunk_size=chunk_size,
        )
    w = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_k), dtype=torch.float32, device=k.device)
    u = torch.empty((batch_size, num_tokens, num_v_heads, head_dim_v), dtype=torch.float32, device=k.device)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_w_u_bf16_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunk_gdr_avelang_v6_standalone(
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
    del prefer_optimized
    if k.dtype == torch.float32:
        return _qwen_gdn_chunk_gdr_fp32_v6_standalone(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=chunk_size,
        )

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k,
        w,
        u,
        g,
        chunk_size,
    )
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
    _qwen_gdn_chunk_gdr_bf16_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunk_o_avelang_v6_standalone(
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
    """生成 FP32 output；FP32 与 BF16 路径都保持现有 v6 行为。"""
    if q.dtype == k.dtype == torch.float32:
        if not prefer_optimized:
            return _qwen_gdn_chunk_o_fallback_v6_standalone(
                q,
                k,
                vn,
                h,
                g,
                scale=scale,
                chunk_size=chunk_size,
            )
        batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks = (
            _validate_chunk_o_inputs(q, k, vn, h, g, chunk_size)
        )
        if scale is None:
            scale = head_dim_k**-0.5
        out = torch.empty_like(vn)
        grid_size = batch_size * num_tokens * num_v_heads
        _qwen_gdn_chunk_o_fp32_opt_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, num_chunks = (
        _validate_bf16_chunk_o_stage(q, k, vn, h, g, chunk_size)
    )
    if not prefer_optimized:
        return _qwen_gdn_chunk_o_fallback_v6_standalone(
            q.float(),
            k.float(),
            vn,
            h,
            g,
            scale=scale,
            chunk_size=chunk_size,
        )
    if scale is None:
        scale = head_dim_k**-0.5
    out = torch.empty_like(vn)
    grid_size = batch_size * num_tokens * num_v_heads
    _qwen_gdn_chunk_o_bf16_kernel_v6_standalone[lambda: ((grid_size, 1, 1), (1, 1, 1))](
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


def qwen_gdn_chunked_avelang_v6_standalone(
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
    """运行完整 standalone v6 forward，支持统一 FP32 或统一 BF16 q/k/v 输入。"""
    if q.dtype == k.dtype == v.dtype == torch.float32:
        _validate_chunk_size(chunk_size)
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_qkvgb(q, k, v, g, beta)
        if initial_state is not None:
            _require_fp32_cuda_contiguous("initial_state", initial_state)
            state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
            if initial_state.shape != state_shape:
                raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
            if initial_state.device != q.device:
                raise ValueError(f"initial_state must be on device {q.device}, got {initial_state.device}.")
    else:
        _validate_chunk_size(chunk_size)
        batch_size, _, _, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_qkvgb(q, k, v, g, beta)
        if initial_state is not None:
            _require_fp32_cuda_contiguous("initial_state", initial_state)
            state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
            if initial_state.shape != state_shape:
                raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
            if initial_state.device != q.device:
                raise ValueError(f"initial_state must be on device {q.device}, got {initial_state.device}.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(
        k,
        g_cumsum,
        beta,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v6_standalone(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=prefer_optimized,
    )
    output = qwen_gdn_chunk_o_avelang_v6_standalone(
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

"""Clean Qwen GDN v12 MFMA prototype for the Qwen3Next TP4 per-rank target.

This version intentionally removes the v11 fallback/debug paths.  It only
supports the fixed production target used by the current experiments:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16, chunk_size=BT=16, BV=16, BK=64
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
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)

BT = 16
BV = 16
BK = 64


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    h_all_bf16 = al.make_shared((BV, 128), al.bf16)
    w_all_bf16 = al.make_shared((BT, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    h_all_vec = al.view(h_all_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_all_vec = al.view(w_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
            state[vv, kk + 64] = initial_state[0, value_head_idx, global_v, kk + 64]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)
            state[vv, kk + 64] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        last_token = chunk_start + 15
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            global_v = value_base + vv
            h[0, chunk_idx, value_head_idx, global_v, kk] = state[vv, kk]
            h[0, chunk_idx, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]

        for rep_all in al.range(32):
            idx_all = lane + rep_all * 64
            row_all = idx_all // 128
            col_all = idx_all - row_all * 128
            h_all_bf16[row_all, col_all] = al.convert(state[row_all, col_all], al.bf16)
            w_all_bf16[row_all, col_all] = al.convert(w[0, chunk_start + row_all, value_head_idx, col_all], al.bf16)
            k_all_t[col_all, row_all] = k[0, chunk_start + row_all, key_head_idx, col_all]

        al.syncthreads()

        # DEBUG isolate path: MFMA pred with the same contiguous [16,128]
        # shared/view layout as the passing standalone pred prototype.
        pred_acc = al.full((4,), 0.0, al.f32)
        for batch128 in al.range(4):
            k_vec128 = lane_group + batch128 * 4
            a_words128 = w_all_vec[lane_col, k_vec128]
            b_words128 = h_all_vec[lane_col, k_vec128]
            a_frag128 = al.view(a_words128, al.Tensor((2, 4, 1), al.bf16))
            b_frag128 = al.view(b_words128, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[0], b_frag128[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag128[1], b_frag128[1], pred_acc)




        for r in al.range(4):
            token_offset = lane_group * 4 + r
            value_offset = lane_col
            token_idx = chunk_start + token_offset
            global_v = value_base + value_offset
            v_new = u[0, token_idx, value_head_idx, global_v] - pred_acc[r]
            vn[0, token_idx, value_head_idx, global_v] = v_new
            decay = al.exp(g_last - g[0, token_idx, value_head_idx])
            v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for rep in al.range(16):
            idx = lane + rep * 64
            vv = idx // BK
            kk = idx - vv * BK
            state[vv, kk] = state[vv, kk] * g_last_exp
            state[vv, kk + 64] = state[vv, kk + 64] * g_last_exp

        al.syncthreads()

        # Integrated update MFMA: delta_H[BV,128] = v_decay[BT,BV]^T @ k_chunk[BT,128].
        # This follows the passing standalone staged-delta prototype exactly:
        # lane_group 0/1/2/3 select the four packed token fragments.
        for tile in al.range(8):
            base_k = tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words = vdecay_vec[lane_col, 0]
                b_words = kall_vec[base_k + lane_col, 0]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
            if lane_group == 1:
                a_words = vdecay_vec[lane_col, 0]
                b_words = kall_vec[base_k + lane_col, 0]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
            if lane_group == 2:
                a_words = vdecay_vec[lane_col, 1]
                b_words = kall_vec[base_k + lane_col, 1]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
            if lane_group == 3:
                a_words = vdecay_vec[lane_col, 1]
                b_words = kall_vec[base_k + lane_col, 1]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] + acc[r_up]

            al.syncthreads()

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // BK
        kk = idx - vv * BK
        global_v = value_base + vv
        final_state[0, value_head_idx, global_v, kk] = state[vv, kk]
        final_state[0, value_head_idx, global_v, kk + 64] = state[vv, kk + 64]


@avelang.jit
def _qwen_gdn_chunk_o_bf16_kernel_v12_mfma(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.f32,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, 8), (num_tokens * 8, 8, 1)))
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    chunk_idx = program_id // 64

    value_base = v_block_idx * 16
    chunk_start = chunk_idx * 16
    key_head_idx = value_head_idx // 2

    q_scaled_bf16 = al.make_shared((BT, 128), al.bf16)
    k_bf16 = al.make_shared((BT, 128), al.bf16)
    h_bf16 = al.make_shared((BV, 128), al.bf16)
    score_decay_bf16 = al.make_shared((BT, BT), al.bf16)
    vn_t_bf16 = al.make_shared((BV, BT), al.bf16)

    q_vec = al.view(q_scaled_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    h_vec = al.view(h_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    score_vec = al.view(score_decay_bf16, al.i32, al.make_layout((BT, 2, 4), (8, 4, 1)))
    vn_vec = al.view(vn_t_bf16, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))

    for rep_all in al.range(32):
        idx = lane + rep_all * 64
        row = idx // 128
        col = idx - row * 128
        token_idx = chunk_start + row
        global_v = value_base + row
        q_scaled_bf16[row, col] = al.convert(al.convert(q[0, token_idx, key_head_idx, col], al.f32) * scale, al.bf16)
        k_bf16[row, col] = k[0, token_idx, key_head_idx, col]
        h_bf16[row, col] = al.convert(h[0, chunk_idx, value_head_idx, global_v, col], al.bf16)

    for rep_small in al.range(4):
        idx_small = lane + rep_small * 64
        vv = idx_small // BT
        tok = idx_small - vv * BT
        token_idx = chunk_start + tok
        global_v = value_base + vv
        vn_t_bf16[vv, tok] = al.convert(vn[0, token_idx, value_head_idx, global_v], al.bf16)

    al.syncthreads()

    inter_acc = al.full((4,), 0.0, al.f32)
    for batch128 in al.range(4):
        k_vec128 = lane_group + batch128 * 4
        a_words = q_vec[lane_col, k_vec128]
        b_words = h_vec[lane_col, k_vec128]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], inter_acc)
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], inter_acc)

    score_acc = al.full((4,), 0.0, al.f32)
    for batch128_s in al.range(4):
        k_vec128_s = lane_group + batch128_s * 4
        a_words_s = q_vec[lane_col, k_vec128_s]
        b_words_s = k_vec[lane_col, k_vec128_s]
        a_frag_s = al.view(a_words_s, al.Tensor((2, 4, 1), al.bf16))
        b_frag_s = al.view(b_words_s, al.Tensor((2, 4, 1), al.bf16))
        score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[0], b_frag_s[0], score_acc)
        score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[1], b_frag_s[1], score_acc)

    for r in al.range(4):
        token_offset = lane_group * 4 + r
        source_offset = lane_col
        token_idx = chunk_start + token_offset
        source_idx = chunk_start + source_offset
        score_value = al.convert(0.0, al.f32)
        if source_offset <= token_offset:
            decay = al.exp(g[0, token_idx, value_head_idx] - g[0, source_idx, value_head_idx])
            score_value = score_acc[r] * decay
        score_decay_bf16[token_offset, source_offset] = al.convert(score_value, al.bf16)

    al.syncthreads()

    intra_acc = al.full((4,), 0.0, al.f32)
    if lane_group == 0:
        a_words_i = score_vec[lane_col, 0]
        b_words_i = vn_vec[lane_col, 0]
        a_frag_i = al.view(a_words_i, al.Tensor((2, 4, 1), al.bf16))
        b_frag_i = al.view(b_words_i, al.Tensor((2, 4, 1), al.bf16))
        intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_i[0], b_frag_i[0], intra_acc)
    if lane_group == 1:
        a_words_i = score_vec[lane_col, 0]
        b_words_i = vn_vec[lane_col, 0]
        a_frag_i = al.view(a_words_i, al.Tensor((2, 4, 1), al.bf16))
        b_frag_i = al.view(b_words_i, al.Tensor((2, 4, 1), al.bf16))
        intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_i[1], b_frag_i[1], intra_acc)
    if lane_group == 2:
        a_words_i = score_vec[lane_col, 1]
        b_words_i = vn_vec[lane_col, 1]
        a_frag_i = al.view(a_words_i, al.Tensor((2, 4, 1), al.bf16))
        b_frag_i = al.view(b_words_i, al.Tensor((2, 4, 1), al.bf16))
        intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_i[0], b_frag_i[0], intra_acc)
    if lane_group == 3:
        a_words_i = score_vec[lane_col, 1]
        b_words_i = vn_vec[lane_col, 1]
        a_frag_i = al.view(a_words_i, al.Tensor((2, 4, 1), al.bf16))
        b_frag_i = al.view(b_words_i, al.Tensor((2, 4, 1), al.bf16))
        intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_i[1], b_frag_i[1], intra_acc)

    for r_out in al.range(4):
        token_offset_o = lane_group * 4 + r_out
        value_offset_o = lane_col
        token_idx_o = chunk_start + token_offset_o
        global_v_o = value_base + value_offset_o
        q_decay = al.exp(g[0, token_idx_o, value_head_idx])
        out[0, token_idx_o, value_head_idx, global_v_o] = inter_acc[r_out] * q_decay + intra_acc[r_out]


def _is_v12_target_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor | None = None) -> bool:
    if tuple(q.shape) != (1, q.shape[1], 4, 128):
        return False
    if tuple(k.shape) != (1, q.shape[1], 4, 128):
        return False
    if tuple(v.shape) != (1, q.shape[1], 8, 128):
        return False
    if tuple(g.shape) != (1, q.shape[1], 8):
        return False
    if beta is not None and tuple(beta.shape) != (1, q.shape[1], 8):
        return False
    return True


def _require_v12_target_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor | None = None) -> None:
    if not _is_v12_target_shape(q, k, v, g, beta):
        raise ValueError(
            "v12 MFMA only supports Qwen3Next TP4 per-rank shape "
            "B=1,Hk=4,Hv=8,K=128,V=128,layout=[B,T,H,D]."
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("v12 MFMA requires q/k/v dtype torch.bfloat16.")
    if g.dtype != torch.float32:
        raise ValueError("v12 MFMA requires g dtype torch.float32.")
    if beta is not None and beta.dtype != torch.float32:
        raise ValueError("v12 MFMA requires beta dtype torch.float32.")


def _validate_initial_state_v12(initial_state: torch.Tensor | None, *, device: torch.device) -> tuple[torch.Tensor | None, bool]:
    has_initial_state = initial_state is not None
    if initial_state is None:
        return None, has_initial_state
    _require_fp32_cuda_contiguous("initial_state", initial_state)
    if tuple(initial_state.shape) != (1, 8, 128, 128):
        raise ValueError(f"initial_state must have shape (1, 8, 128, 128), got {tuple(initial_state.shape)}.")
    if initial_state.device != device:
        raise ValueError(f"initial_state must be on device {device}, got {initial_state.device}.")
    return initial_state, has_initial_state


def qwen_gdn_chunk_gdr_avelang_v12_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("v12 chunk_gdr MFMA only supports chunk_size=16.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(k, w, u, g, chunk_size)
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v12 chunk_gdr MFMA only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v12 chunk_gdr MFMA requires num_tokens divisible by 16.")
    initial_state_arg, has_initial_state = _validate_initial_state_v12(initial_state, device=k.device)

    num_chunks = _num_chunks(num_tokens, chunk_size)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    _qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update[
        lambda: ((64, 1, 1), (64, 1, 1))
    ](
        k,
        w,
        u,
        g,
        initial_state_arg,
        h,
        vn,
        final_state,
        num_tokens,
        num_chunks,
        has_initial_state,
    )
    return h, vn, final_state


def qwen_gdn_chunk_o_avelang_v12_mfma_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    vn: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    if scale is None:
        scale = 128 ** -0.5
    if chunk_size != BT:
        raise ValueError("v12 chunk_o MFMA only supports chunk_size=16.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("v12 chunk_o MFMA requires q/k dtype torch.bfloat16.")
    if vn.dtype != torch.float32 or h.dtype != torch.float32 or g.dtype != torch.float32:
        raise ValueError("v12 chunk_o MFMA requires vn/h/g dtype torch.float32.")
    if tuple(q.shape) != (1, q.shape[1], 4, 128):
        raise ValueError("q must have shape [1,T,4,128].")
    num_tokens = q.shape[1]
    if tuple(k.shape) != (1, num_tokens, 4, 128):
        raise ValueError("k must have shape [1,T,4,128].")
    if tuple(vn.shape) != (1, num_tokens, 8, 128):
        raise ValueError("vn must have shape [1,T,8,128].")
    if tuple(g.shape) != (1, num_tokens, 8):
        raise ValueError("g must have shape [1,T,8].")
    if num_tokens % BT != 0:
        raise ValueError("v12 chunk_o MFMA requires num_tokens divisible by 16.")
    num_chunks = _num_chunks(num_tokens, chunk_size)
    if tuple(h.shape) != (1, num_chunks, 8, 128, 128):
        raise ValueError(f"h must have shape [1,{num_chunks},8,128,128], got {tuple(h.shape)}.")

    out = torch.empty_like(vn)
    grid_size = num_chunks * 8 * 8
    _qwen_gdn_chunk_o_bf16_kernel_v12_mfma[lambda: ((grid_size, 1, 1), (64, 1, 1))](
        q,
        k,
        vn,
        h,
        g,
        out,
        float(scale),
        num_tokens,
        num_chunks,
    )
    return out


def qwen_gdn_chunked_avelang_v12_mfma_layout_full(
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
        raise ValueError("v12 full MFMA only supports chunk_size=16.")
    _validate_bf16_qkvgb(q, k, v, g, beta)
    _require_v12_target_shape(q, k, v, g, beta)
    _validate_initial_state_v12(initial_state, device=q.device)
    if q.shape[1] % BT != 0:
        raise ValueError("v12 full MFMA requires num_tokens divisible by 16.")

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size, prefer_optimized=True)
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v12_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    output = qwen_gdn_chunk_o_avelang_v12_mfma_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
    )
    return g_cumsum, output, a_solved, h, final_state


def qwen_gdn_chunked_avelang_v12_mfma_layout(
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
    _, output, _, _, final_state = qwen_gdn_chunked_avelang_v12_mfma_layout_full(
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
    "_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update",
    "_qwen_gdn_chunk_o_bf16_kernel_v12_mfma",
    "qwen_gdn_chunk_gdr_avelang_v12_mfma_layout",
    "qwen_gdn_chunk_o_avelang_v12_mfma_layout",
    "qwen_gdn_chunked_avelang_v12_mfma_layout_full",
    "qwen_gdn_chunked_avelang_v12_mfma_layout",
]

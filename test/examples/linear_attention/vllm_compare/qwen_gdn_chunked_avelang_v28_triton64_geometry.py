"""Qwen GDN v28 chunk_gdr-only BT64/BV32 Triton-like geometry experiment.

This experiment keeps the BT64/BV32 target from v26, but changes the
persistent state geometry to track the vLLM Triton kernel more closely:

    h1: [BV, 64] for K 0:64
    h2: [BV, 64] for K 64:128

Unlike v26/v27, the state is not split across 4 waves by Kq=32 quarters.
Instead, each wave owns one 16x64 state tile:

    wave0: V  0:16, K  0:64 (h1 upper)
    wave1: V 16:32, K  0:64 (h1 lower)
    wave2: V  0:16, K 64:128 (h2 upper)
    wave3: V 16:32, K 64:128 (h2 lower)

The main FP32 state remains persistent in MFMA accumulator/register fragments.
LDS is only used for BF16 operand staging and partial reductions.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    _validate_bf16_chunk_gdr_stage,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import _validate_initial_state_v14

BT = 64
BV = 32
KDIM = 128
WORKGROUP = 256
GRID_SIZE = 32

MODE_FULL_V28 = "full_v28"
MODE_NO_H_STORE = "no_h_store"
MODE_NO_VN_STORE = "no_vn_store"
MODE_NO_DECAY = "no_decay"
MODE_PRED_ONLY = "pred_only"
MODE_UPDATE_ONLY = "update_only"

_VARIANTS = {
    MODE_FULL_V28: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=True, store_vn=True, store_final=True, enable_initial=True),
    MODE_NO_H_STORE: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=False, store_vn=True, store_final=True, enable_initial=True),
    MODE_NO_VN_STORE: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=True, store_vn=False, store_final=True, enable_initial=True),
    MODE_NO_DECAY: dict(do_pred=True, do_update=True, enable_g_decay=False, store_h=True, store_vn=True, store_final=True, enable_initial=True),
    MODE_PRED_ONLY: dict(do_pred=True, do_update=False, enable_g_decay=False, store_h=False, store_vn=True, store_final=False, enable_initial=False),
    MODE_UPDATE_ONLY: dict(do_pred=False, do_update=True, enable_g_decay=False, store_h=False, store_vn=False, store_final=True, enable_initial=False),
}


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v28_triton64_geometry(
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
    do_pred: al.constexpr,
    do_update: al.constexpr,
    enable_g_decay: al.constexpr,
    store_h: al.constexpr,
    store_vn: al.constexpr,
    store_final: al.constexpr,
    enable_initial: al.constexpr,
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

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 4
    value_head_idx = program_id // 4
    value_base = v_block_idx * 32
    key_head_idx = value_head_idx // 2

    value_tile = wave_id & 1
    k_block = wave_id >> 1
    value_row_base = value_tile * 16
    k_col_base = k_block * 64

    # One wave owns one 16x64 state tile represented as four 16x16 accumulator tiles.
    s0 = al.full((4,), 0.0, al.f32)
    s1 = al.full((4,), 0.0, al.f32)
    s2 = al.full((4,), 0.0, al.f32)
    s3 = al.full((4,), 0.0, al.f32)

    for r_init in al.range(4):
        row = value_row_base + lane_group * 4 + r_init
        global_v = value_base + row
        if has_initial_state and enable_initial:
            s0[r_init] = initial_state[0, value_head_idx, global_v, k_col_base + lane_col]
            s1[r_init] = initial_state[0, value_head_idx, global_v, k_col_base + 16 + lane_col]
            s2[r_init] = initial_state[0, value_head_idx, global_v, k_col_base + 32 + lane_col]
            s3[r_init] = initial_state[0, value_head_idx, global_v, k_col_base + 48 + lane_col]

    state1_bf16 = al.make_shared((BV, 64), al.bf16)
    state2_bf16 = al.make_shared((BV, 64), al.bf16)
    w1_tile_bf16 = al.make_shared((16, 64), al.bf16)
    w2_tile_bf16 = al.make_shared((16, 64), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k1_all_t = al.make_shared((64, BT), al.bf16)
    k2_all_t = al.make_shared((64, BT), al.bf16)
    pred_partial = al.make_shared((4, 16, 16), al.f32)

    state1_vec = al.view(state1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    state2_vec = al.view(state2_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_tile_bf16, al.i32, al.make_layout((16, 8, 4), (32, 4, 1)))
    w2_vec = al.view(w2_tile_bf16, al.i32, al.make_layout((16, 8, 4), (32, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    k1_vec = al.view(k1_all_t, al.i32, al.make_layout((64, 8, 4), (32, 4, 1)))
    k2_vec = al.view(k2_all_t, al.i32, al.make_layout((64, 8, 4), (32, 4, 1)))

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 64
        last_token = chunk_start + 63
        g_last_exp = al.convert(1.0, al.f32)
        if enable_g_decay:
            g_last_exp = al.exp(g[0, last_token, value_head_idx])

        if store_h or do_pred:
            for r_stage in al.range(4):
                row = value_row_base + lane_group * 4 + r_stage
                global_v = value_base + row
                col0 = k_col_base + lane_col
                col1 = k_col_base + 16 + lane_col
                col2 = k_col_base + 32 + lane_col
                col3 = k_col_base + 48 + lane_col
                if store_h:
                    h[0, chunk_idx, value_head_idx, global_v, col0] = s0[r_stage]
                    h[0, chunk_idx, value_head_idx, global_v, col1] = s1[r_stage]
                    h[0, chunk_idx, value_head_idx, global_v, col2] = s2[r_stage]
                    h[0, chunk_idx, value_head_idx, global_v, col3] = s3[r_stage]
                if do_pred:
                    if k_block == 0:
                        state1_bf16[row, lane_col] = al.convert(s0[r_stage], al.bf16)
                        state1_bf16[row, 16 + lane_col] = al.convert(s1[r_stage], al.bf16)
                        state1_bf16[row, 32 + lane_col] = al.convert(s2[r_stage], al.bf16)
                        state1_bf16[row, 48 + lane_col] = al.convert(s3[r_stage], al.bf16)
                    else:
                        state2_bf16[row, lane_col] = al.convert(s0[r_stage], al.bf16)
                        state2_bf16[row, 16 + lane_col] = al.convert(s1[r_stage], al.bf16)
                        state2_bf16[row, 32 + lane_col] = al.convert(s2[r_stage], al.bf16)
                        state2_bf16[row, 48 + lane_col] = al.convert(s3[r_stage], al.bf16)

        if do_update:
            for rep_k in al.range(16):
                idx_k = tid + rep_k * 256
                col_k = idx_k // 64
                tok_k = idx_k - col_k * 64
                token_k = chunk_start + tok_k
                k1_all_t[col_k, tok_k] = k[0, token_k, key_head_idx, col_k]
                k2_all_t[col_k, tok_k] = k[0, token_k, key_head_idx, 64 + col_k]

        al.syncthreads()

        if do_pred:
            for token_tile in al.range(4):
                token_base = token_tile * 16

                for rep_w in al.range(4):
                    idx_w = tid + rep_w * 256
                    row_w = idx_w // 64
                    col_w = idx_w - row_w * 64
                    token_w = chunk_start + token_base + row_w
                    w1_tile_bf16[row_w, col_w] = al.convert(w[0, token_w, value_head_idx, col_w], al.bf16)
                    w2_tile_bf16[row_w, col_w] = al.convert(w[0, token_w, value_head_idx, 64 + col_w], al.bf16)

                al.syncthreads()

                pred_acc = al.full((4,), 0.0, al.f32)
                value_row = value_row_base + lane_col
                if k_block == 0:
                    for seg32 in al.range(2):
                        vec_idx = lane_group + seg32 * 4
                        a_words = w1_vec[lane_col, vec_idx]
                        b_words = state1_vec[value_row, vec_idx]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)
                else:
                    for seg32 in al.range(2):
                        vec_idx = lane_group + seg32 * 4
                        a_words = w2_vec[lane_col, vec_idx]
                        b_words = state2_vec[value_row, vec_idx]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

                for r_pred in al.range(4):
                    token_offset_p = lane_group * 4 + r_pred
                    value_offset_p = lane_col
                    pred_partial[wave_id, token_offset_p, value_offset_p] = pred_acc[r_pred]

                al.syncthreads()

                for value_tile_compute in al.range(2):
                    linear = tid
                    token_offset = linear // 16
                    value_offset = linear - token_offset * 16
                    token_idx = chunk_start + token_base + token_offset
                    local_v = value_tile_compute * 16 + value_offset
                    global_v = value_base + local_v
                    pred_value = pred_partial[value_tile_compute, token_offset, value_offset] + pred_partial[2 + value_tile_compute, token_offset, value_offset]
                    v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
                    if store_vn:
                        vn[0, token_idx, value_head_idx, global_v] = v_new
                    if do_update:
                        decay = al.convert(1.0, al.f32)
                        if enable_g_decay:
                            decay = al.exp(g[0, last_token, value_head_idx] - g[0, token_idx, value_head_idx])
                        v_decay_t[local_v, token_base + token_offset] = al.convert(v_new * decay, al.bf16)

                al.syncthreads()
        else:
            # update_only ablation: use u as v_new and skip the pred dot.
            for rep_vd in al.range(8):
                idx_vd = tid + rep_vd * 256
                local_v_vd = idx_vd // 64
                tok_vd = idx_vd - local_v_vd * 64
                token_vd = chunk_start + tok_vd
                global_v_vd = value_base + local_v_vd
                v_new_vd = u[0, token_vd, value_head_idx, global_v_vd]
                decay_vd = al.convert(1.0, al.f32)
                if enable_g_decay:
                    decay_vd = al.exp(g[0, last_token, value_head_idx] - g[0, token_vd, value_head_idx])
                v_decay_t[local_v_vd, tok_vd] = al.convert(v_new_vd * decay_vd, al.bf16)
            al.syncthreads()

        if do_update:
            for local_tile in al.range(4):
                base_k = local_tile * 16
                acc = al.full((4,), 0.0, al.f32)
                for token_pack_base in al.range(4):
                    pack0 = token_pack_base * 2
                    pack1 = pack0 + 1
                    if lane_group == 0:
                        a_words_u = vdecay_vec[value_row_base + lane_col, pack0]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        if k_block == 0:
                            b_words_u0 = k1_vec[base_k + lane_col, pack0]
                            b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u0[0], acc)
                        else:
                            b_words_u0 = k2_vec[base_k + lane_col, pack0]
                            b_frag_u0 = al.view(b_words_u0, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u0[0], acc)
                    if lane_group == 1:
                        a_words_u = vdecay_vec[value_row_base + lane_col, pack0]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        if k_block == 0:
                            b_words_u1 = k1_vec[base_k + lane_col, pack0]
                            b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u1[1], acc)
                        else:
                            b_words_u1 = k2_vec[base_k + lane_col, pack0]
                            b_frag_u1 = al.view(b_words_u1, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u1[1], acc)
                    if lane_group == 2:
                        a_words_u = vdecay_vec[value_row_base + lane_col, pack1]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        if k_block == 0:
                            b_words_u2 = k1_vec[base_k + lane_col, pack1]
                            b_frag_u2 = al.view(b_words_u2, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u2[0], acc)
                        else:
                            b_words_u2 = k2_vec[base_k + lane_col, pack1]
                            b_frag_u2 = al.view(b_words_u2, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u2[0], acc)
                    if lane_group == 3:
                        a_words_u = vdecay_vec[value_row_base + lane_col, pack1]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        if k_block == 0:
                            b_words_u3 = k1_vec[base_k + lane_col, pack1]
                            b_frag_u3 = al.view(b_words_u3, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u3[1], acc)
                        else:
                            b_words_u3 = k2_vec[base_k + lane_col, pack1]
                            b_frag_u3 = al.view(b_words_u3, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u3[1], acc)

                for r_up in al.range(4):
                    updated = acc[r_up]
                    if local_tile == 0:
                        s0[r_up] = s0[r_up] * g_last_exp + updated
                    if local_tile == 1:
                        s1[r_up] = s1[r_up] * g_last_exp + updated
                    if local_tile == 2:
                        s2[r_up] = s2[r_up] * g_last_exp + updated
                    if local_tile == 3:
                        s3[r_up] = s3[r_up] * g_last_exp + updated

            al.syncthreads()

    if store_final:
        for r_final in al.range(4):
            row = value_row_base + lane_group * 4 + r_final
            global_v = value_base + row
            final_state[0, value_head_idx, global_v, k_col_base + lane_col] = s0[r_final]
            final_state[0, value_head_idx, global_v, k_col_base + 16 + lane_col] = s1[r_final]
            final_state[0, value_head_idx, global_v, k_col_base + 32 + lane_col] = s2[r_final]
            final_state[0, value_head_idx, global_v, k_col_base + 48 + lane_col] = s3[r_final]


def _validate_v28_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("v28_triton64_geometry only supports chunk_size=64.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k, w, u, g, chunk_size
    )
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v28_triton64_geometry only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v28_triton64_geometry requires num_tokens divisible by 64.")
    return num_tokens, _num_chunks(num_tokens, chunk_size)


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown v28 variant {variant!r}; expected one of {sorted(_VARIANTS)}")
    return _VARIANTS[variant]


def qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
    variant: str = MODE_FULL_V28,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_v28_inputs(k, w, u, g, chunk_size)
    flags = _variant_flags(variant)
    initial_state_arg, has_initial_state = _validate_initial_state_v14(initial_state, device=k.device)

    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    _qwen_gdn_chunk_gdr_bf16_kernel_v28_triton64_geometry[lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))](
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
        flags["do_pred"],
        flags["do_update"],
        flags["enable_g_decay"],
        flags["store_h"],
        flags["store_vn"],
        flags["store_final"],
        flags["enable_initial"],
    )
    return h, vn, final_state


__all__ = [
    "BT",
    "BV",
    "MODE_FULL_V28",
    "MODE_NO_H_STORE",
    "MODE_NO_VN_STORE",
    "MODE_NO_DECAY",
    "MODE_PRED_ONLY",
    "MODE_UPDATE_ONLY",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v28_triton64_geometry",
    "qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry",
]

"""Qwen GDN v25 chunk_gdr-only BT64/BV32 MFMA experiment.

This file intentionally does not implement a full forward path.  It isolates
the vLLM-style larger-chunk delta_h recurrence:

    B=1, Hk=4, Hv=8, K=128, V=128, BF16 k, FP32 w/u/g/state
    chunk_size=BT=64, BV=32

Unsupported shapes raise ValueError; there is no silent fallback.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    _require_fp32_cuda_contiguous,
    _validate_bf16_chunk_gdr_stage,
)
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import _validate_initial_state_v14

BT = 64
BV32 = 32
BK_HALF = 64


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v25_bt64_bv32_mfma(
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

    state = al.make_shared((BV32, 128), al.f32)
    state_bf16 = al.make_shared((BV32, 128), al.bf16)
    w_tile_bf16 = al.make_shared((16, 128), al.bf16)
    v_decay_t = al.make_shared((BV32, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    pred_partial = al.make_shared((4, 16, 16), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV32, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_tile_bf16, al.i32, al.make_layout((16, 16, 4), (64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV32, 8, 4), (32, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (32, 4, 1)))

    for rep_init in al.range(16):
        idx_init = tid + rep_init * 256
        vv_init = idx_init // 128
        kk_init = idx_init - vv_init * 128
        global_v_init = value_base + vv_init
        if has_initial_state:
            state[vv_init, kk_init] = initial_state[0, value_head_idx, global_v_init, kk_init]
        else:
            state[vv_init, kk_init] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 64
        last_token = chunk_start + 63
        g_last = g[0, last_token, value_head_idx]
        g_last_exp = al.exp(g_last)

        for rep_h in al.range(16):
            idx_h = tid + rep_h * 256
            vv_h = idx_h // 128
            kk_h = idx_h - vv_h * 128
            global_v_h = value_base + vv_h
            h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]
            state_bf16[vv_h, kk_h] = al.convert(state[vv_h, kk_h], al.bf16)

        for rep_k in al.range(32):
            idx_k = tid + rep_k * 256
            col_k = idx_k // 64
            tok_k = idx_k - col_k * 64
            token_k = chunk_start + tok_k
            k_all_t[col_k, tok_k] = k[0, token_k, key_head_idx, col_k]

        al.syncthreads()

        for token_tile in al.range(4):
            token_base = token_tile * 16

            for rep_w in al.range(8):
                idx_w = tid + rep_w * 256
                row_w = idx_w // 128
                col_w = idx_w - row_w * 128
                token_w = chunk_start + token_base + row_w
                w_tile_bf16[row_w, col_w] = al.convert(w[0, token_w, value_head_idx, col_w], al.bf16)

            al.syncthreads()

            for value_tile in al.range(2):
                pred_acc = al.full((4,), 0.0, al.f32)
                k_vec32 = lane_group + wave_id * 4
                a_words = w_vec[lane_col, k_vec32]
                b_words = state_vec[value_tile * 16 + lane_col, k_vec32]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
                pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

                for r_pred in al.range(4):
                    token_offset_p = lane_group * 4 + r_pred
                    value_offset_p = lane_col
                    pred_partial[wave_id, token_offset_p, value_offset_p] = pred_acc[r_pred]

                al.syncthreads()

                linear_vn = tid
                token_offset = linear_vn // 16
                value_offset = linear_vn - token_offset * 16
                token_idx = chunk_start + token_base + token_offset
                local_v = value_tile * 16 + value_offset
                global_v = value_base + local_v
                pred_value = (
                    pred_partial[0, token_offset, value_offset]
                    + pred_partial[1, token_offset, value_offset]
                    + pred_partial[2, token_offset, value_offset]
                    + pred_partial[3, token_offset, value_offset]
                )
                v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
                vn[0, token_idx, value_head_idx, global_v] = v_new
                decay = al.exp(g_last - g[0, token_idx, value_head_idx])
                v_decay_t[local_v, token_base + token_offset] = al.convert(v_new * decay, al.bf16)

                al.syncthreads()

        for value_tile_u in al.range(2):
            for local_tile in al.range(2):
                global_tile = wave_id * 2 + local_tile
                base_k = global_tile * 16
                acc = al.full((4,), 0.0, al.f32)
                for token_pack_base in al.range(4):
                    pack0 = token_pack_base * 2
                    pack1 = pack0 + 1
                    if lane_group == 0:
                        a_words_u = vdecay_vec[value_tile_u * 16 + lane_col, pack0]
                        b_words_u = kall_vec[base_k + lane_col, pack0]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                    if lane_group == 1:
                        a_words_u = vdecay_vec[value_tile_u * 16 + lane_col, pack0]
                        b_words_u = kall_vec[base_k + lane_col, pack0]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
                    if lane_group == 2:
                        a_words_u = vdecay_vec[value_tile_u * 16 + lane_col, pack1]
                        b_words_u = kall_vec[base_k + lane_col, pack1]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                    if lane_group == 3:
                        a_words_u = vdecay_vec[value_tile_u * 16 + lane_col, pack1]
                        b_words_u = kall_vec[base_k + lane_col, pack1]
                        a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

                out_col = base_k + lane_col
                for r_up in al.range(4):
                    vv_up = value_tile_u * 16 + lane_group * 4 + r_up
                    state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc[r_up]

            al.syncthreads()

    for rep_final in al.range(16):
        idx_final = tid + rep_final * 256
        vv_final = idx_final // 128
        kk_final = idx_final - vv_final * 128
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


def _validate_v25_bt64_bv32_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("v25 BT64/BV32 chunk_gdr only supports chunk_size=64.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k, w, u, g, chunk_size
    )
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v25 BT64/BV32 chunk_gdr only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v25 BT64/BV32 chunk_gdr requires num_tokens divisible by 64.")
    return num_tokens, _num_chunks(num_tokens, chunk_size)


def qwen_gdn_chunk_gdr_avelang_v25_bt64_bv32_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_v25_bt64_bv32_inputs(k, w, u, g, chunk_size)
    initial_state_arg, has_initial_state = _validate_initial_state_v14(initial_state, device=k.device)

    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    _qwen_gdn_chunk_gdr_bf16_kernel_v25_bt64_bv32_mfma[lambda: ((32, 1, 1), (256, 1, 1))](
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


def qwen_gdn_chunk_gdr_torch_ref_bt64(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("torch reference only supports chunk_size=64.")
    if tuple(k.shape) != (1, k.shape[1], 4, 128) or tuple(w.shape) != (1, k.shape[1], 8, 128):
        raise ValueError("torch reference only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    num_tokens = k.shape[1]
    if num_tokens % BT != 0:
        raise ValueError("torch reference requires T divisible by 64.")

    num_chunks = num_tokens // BT
    state = torch.zeros((1, 8, 128, 128), device=k.device, dtype=torch.float32)
    if initial_state is not None:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        state.copy_(initial_state)

    h = torch.empty((1, num_chunks, 8, 128, 128), device=k.device, dtype=torch.float32)
    vn = torch.empty_like(u)
    kf = k.float()
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BT
        end = start + BT
        h[:, chunk_idx].copy_(state)
        g_last = g[:, end - 1, :]
        g_last_exp = torch.exp(g_last)
        for vh in range(8):
            # The Avelang kernel stages w/state/v_decay through BF16 before
            # MFMA. Mirror that here so checks compare the same math, not a
            # stricter FP32-only recurrence.
            w_b = w[0, start:end, vh].to(torch.bfloat16).float()
            state_b = state[0, vh].to(torch.bfloat16).float()
            pred = w_b @ state_b.transpose(0, 1)
            v_new = u[0, start:end, vh].float() - pred
            vn[0, start:end, vh].copy_(v_new)
            decay = torch.exp(g_last[0, vh] - g[0, start:end, vh]).view(BT, 1)
            v_decay = (v_new * decay).to(torch.bfloat16).float()
            kh = vh // 2
            delta = v_decay.transpose(0, 1) @ kf[0, start:end, kh]
            state[0, vh] = state[0, vh] * g_last_exp[0, vh] + delta
    return h, vn, state


__all__ = [
    "_qwen_gdn_chunk_gdr_bf16_kernel_v25_bt64_bv32_mfma",
    "qwen_gdn_chunk_gdr_avelang_v25_bt64_bv32_mfma_layout",
    "qwen_gdn_chunk_gdr_torch_ref_bt64",
]

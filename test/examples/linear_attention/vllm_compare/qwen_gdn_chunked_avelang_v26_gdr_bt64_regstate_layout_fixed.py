"""Qwen GDN v26 chunk_gdr-only BT64/BV32 register-state experiment.

This file intentionally does not implement a full forward path.  It is an
experimental answer to the v25 negative result: v25 kept the cross-chunk state
in LDS as FP32, while v26 keeps the main state in MFMA accumulator/register
fragments across the chunk loop.

Fixed target shape:
    B=1, Hk=4, Hv=8, K=128, V=128, BF16 k, FP32 w/u/g/state
    chunk_size=BT=64, BV=32, workgroup=4 waves = 256 threads

The current Avelang MFMA path consumes BF16 operands, so pred still needs a
temporary BF16 materialization of the accumulator state into LDS.  That staging
is deliberately not the authoritative cross-chunk state; the FP32 state lives
in accumulator fragments and is updated in place by the update MFMA.
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
BV = 32
KDIM = 128
WORKGROUP = 256
GRID_SIZE = 32

MODE_FULL_V26C = "full_v26c"
MODE_PRED_ONLY = "pred_only"
MODE_UPDATE_ONLY = "update_only"
MODE_NO_H_STORE = "no_h_store"
MODE_NO_VN_STORE = "no_vn_store"
MODE_NO_DECAY = "no_decay"
MODE_NO_PRED_STATE_CONVERT = "no_pred_state_convert"
MODE_V26A_CORE = "v26a_core"
MODE_V26B_IO = "v26b_io"

_VARIANTS = {
    MODE_FULL_V26C: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=True, store_vn=True, store_final=True, enable_initial=True, stage_pred_state=True),
    MODE_PRED_ONLY: dict(do_pred=True, do_update=False, enable_g_decay=False, store_h=False, store_vn=True, store_final=False, enable_initial=False, stage_pred_state=True),
    MODE_UPDATE_ONLY: dict(do_pred=False, do_update=True, enable_g_decay=False, store_h=False, store_vn=False, store_final=True, enable_initial=False, stage_pred_state=False),
    MODE_NO_H_STORE: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=False, store_vn=True, store_final=True, enable_initial=True, stage_pred_state=True),
    MODE_NO_VN_STORE: dict(do_pred=True, do_update=True, enable_g_decay=True, store_h=True, store_vn=False, store_final=True, enable_initial=True, stage_pred_state=True),
    MODE_NO_DECAY: dict(do_pred=True, do_update=True, enable_g_decay=False, store_h=True, store_vn=True, store_final=True, enable_initial=True, stage_pred_state=True),
    MODE_NO_PRED_STATE_CONVERT: dict(do_pred=True, do_update=True, enable_g_decay=False, store_h=False, store_vn=False, store_final=True, enable_initial=False, stage_pred_state=False),
    MODE_V26A_CORE: dict(do_pred=True, do_update=True, enable_g_decay=False, store_h=False, store_vn=False, store_final=False, enable_initial=False, stage_pred_state=True),
    MODE_V26B_IO: dict(do_pred=True, do_update=True, enable_g_decay=False, store_h=True, store_vn=True, store_final=True, enable_initial=False, stage_pred_state=True),
}


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v26_bt64_bv32_regstate_mfma(
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
    stage_pred_state: al.constexpr,
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
    wave_k_base = wave_id * 32

    # Main state is held in four accumulator/register tiles per wave:
    #   s00: V  0:16, Kq  0:16 within this wave's 32-wide K quarter
    #   s01: V  0:16, Kq 16:32
    #   s10: V 16:32, Kq  0:16
    #   s11: V 16:32, Kq 16:32
    s00 = al.full((4,), 0.0, al.f32)
    s01 = al.full((4,), 0.0, al.f32)
    s10 = al.full((4,), 0.0, al.f32)
    s11 = al.full((4,), 0.0, al.f32)

    for r_init in al.range(4):
        row0 = lane_group * 4 + r_init
        row1 = 16 + row0
        global_v0 = value_base + row0
        global_v1 = value_base + row1
        col0 = wave_k_base + lane_col
        col1 = wave_k_base + 16 + lane_col
        if has_initial_state and enable_initial:
            s00[r_init] = initial_state[0, value_head_idx, global_v0, col0]
            s01[r_init] = initial_state[0, value_head_idx, global_v0, col1]
            s10[r_init] = initial_state[0, value_head_idx, global_v1, col0]
            s11[r_init] = initial_state[0, value_head_idx, global_v1, col1]

    # LDS is now only operand staging, not the authoritative FP32 state.
    state_bf16 = al.make_shared((BV, 128), al.bf16)
    w_tile_bf16 = al.make_shared((16, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    pred_partial = al.make_shared((4, 16, 16), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_tile_bf16, al.i32, al.make_layout((16, 16, 4), (64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (32, 4, 1)))

    if not stage_pred_state:
        for rep_zero in al.range(16):
            idx_zero = tid + rep_zero * 256
            row_zero = idx_zero // 128
            col_zero = idx_zero - row_zero * 128
            state_bf16[row_zero, col_zero] = al.convert(0.0, al.bf16)
        al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 64
        last_token = chunk_start + 63
        g_last = al.convert(0.0, al.f32)
        g_last_exp = al.convert(1.0, al.f32)
        if enable_g_decay:
            g_last = g[0, last_token, value_head_idx]
            g_last_exp = al.exp(g_last)

        for r_stage in al.range(4):
            row0_s = lane_group * 4 + r_stage
            row1_s = 16 + row0_s
            global_v0_s = value_base + row0_s
            global_v1_s = value_base + row1_s
            col0_s = wave_k_base + lane_col
            col1_s = wave_k_base + 16 + lane_col

            if store_h:
                h[0, chunk_idx, value_head_idx, global_v0_s, col0_s] = s00[r_stage]
                h[0, chunk_idx, value_head_idx, global_v0_s, col1_s] = s01[r_stage]
                h[0, chunk_idx, value_head_idx, global_v1_s, col0_s] = s10[r_stage]
                h[0, chunk_idx, value_head_idx, global_v1_s, col1_s] = s11[r_stage]

            if stage_pred_state:
                state_bf16[row0_s, col0_s] = al.convert(s00[r_stage], al.bf16)
                state_bf16[row0_s, col1_s] = al.convert(s01[r_stage], al.bf16)
                state_bf16[row1_s, col0_s] = al.convert(s10[r_stage], al.bf16)
                state_bf16[row1_s, col1_s] = al.convert(s11[r_stage], al.bf16)

        for rep_k in al.range(32):
            idx_k = tid + rep_k * 256
            col_k = idx_k // 64
            tok_k = idx_k - col_k * 64
            token_k = chunk_start + tok_k
            k_all_t[col_k, tok_k] = k[0, token_k, key_head_idx, col_k]

        al.syncthreads()

        if do_pred:
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
                    if store_vn:
                        vn[0, token_idx, value_head_idx, global_v] = v_new
                    decay = al.convert(1.0, al.f32)
                    if enable_g_decay:
                        decay = al.exp(g_last - g[0, token_idx, value_head_idx])
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
                    decay_vd = al.exp(g_last - g[0, token_vd, value_head_idx])
                v_decay_t[local_v_vd, tok_vd] = al.convert(v_new_vd * decay_vd, al.bf16)
            al.syncthreads()

        if do_update:
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

                    for r_up in al.range(4):
                        if value_tile_u == 0 and local_tile == 0:
                            s00[r_up] = s00[r_up] * g_last_exp + acc[r_up]
                        if value_tile_u == 0 and local_tile == 1:
                            s01[r_up] = s01[r_up] * g_last_exp + acc[r_up]
                        if value_tile_u == 1 and local_tile == 0:
                            s10[r_up] = s10[r_up] * g_last_exp + acc[r_up]
                        if value_tile_u == 1 and local_tile == 1:
                            s11[r_up] = s11[r_up] * g_last_exp + acc[r_up]

            al.syncthreads()

    if store_final:
        for r_final in al.range(4):
            row0_f = lane_group * 4 + r_final
            row1_f = 16 + row0_f
            global_v0_f = value_base + row0_f
            global_v1_f = value_base + row1_f
            col0_f = wave_k_base + lane_col
            col1_f = wave_k_base + 16 + lane_col
            final_state[0, value_head_idx, global_v0_f, col0_f] = s00[r_final]
            final_state[0, value_head_idx, global_v0_f, col1_f] = s01[r_final]
            final_state[0, value_head_idx, global_v1_f, col0_f] = s10[r_final]
            final_state[0, value_head_idx, global_v1_f, col1_f] = s11[r_final]


def _validate_v26_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("v26 BT64/BV32 chunk_gdr only supports chunk_size=64.")
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k, w, u, g, chunk_size
    )
    if (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v26 BT64/BV32 chunk_gdr only supports B=1,Hk=4,Hv=8,K=128,V=128.")
    if num_tokens % BT != 0:
        raise ValueError("v26 BT64/BV32 chunk_gdr requires num_tokens divisible by 64.")
    return num_tokens, _num_chunks(num_tokens, chunk_size)


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown v26 variant {variant!r}; expected one of {sorted(_VARIANTS)}")
    return _VARIANTS[variant]


def qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
    variant: str = MODE_FULL_V26C,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_v26_inputs(k, w, u, g, chunk_size)
    flags = _variant_flags(variant)
    initial_state_arg, has_initial_state = _validate_initial_state_v14(initial_state, device=k.device)

    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state

    _qwen_gdn_chunk_gdr_bf16_kernel_v26_bt64_bv32_regstate_mfma[lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))](
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
        flags["stage_pred_state"],
    )
    return h, vn, final_state


def qwen_gdn_chunk_gdr_torch_ref_bt64_regstate(
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
            # Match the Avelang MFMA operand staging: w/state/v_decay are BF16
            # inputs for MFMA, while accumulators and output state are FP32.
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
    "BT",
    "BV",
    "MODE_FULL_V26C",
    "MODE_PRED_ONLY",
    "MODE_UPDATE_ONLY",
    "MODE_NO_H_STORE",
    "MODE_NO_VN_STORE",
    "MODE_NO_DECAY",
    "MODE_NO_PRED_STATE_CONVERT",
    "MODE_V26A_CORE",
    "MODE_V26B_IO",
    "_qwen_gdn_chunk_gdr_bf16_kernel_v26_bt64_bv32_regstate_mfma",
    "qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout",
    "qwen_gdn_chunk_gdr_torch_ref_bt64_regstate",
]

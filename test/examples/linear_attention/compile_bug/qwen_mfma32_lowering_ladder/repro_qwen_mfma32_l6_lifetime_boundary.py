#!/usr/bin/env python3
"""Qwen-shaped L6 lifetime-boundary prototype.

This is an isolated compiler/backend evidence repro.  It keeps the Qwen-ish
MFMA32 pred/v_decay shape from the lowering ladder, then inserts
al.end_lifetime() after v_decay_t is fully staged and before update MFMA.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
WORKGROUP = 128
GRID = 32

VARIANTS = {
    "L6_baseline_no_lifetime": 0,
    "L6_with_end_lifetime_after_vdecay": 1,
    "L6_memref_only_end_lifetime": 10,
    "L6_subtile_no_lifetime": 6,
    "L6_subtile_with_end_lifetime_after_vdecay": 7,
    "L6_subtile_memref_only_end_lifetime": 11,
}


@avelang.jit
def _qwen_mfma32_l6_kstage_update_variants_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    state_in_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    sink_ptr: al.Pointer(al.f32),
    variant: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, BT, 4, 128), (BT * 4 * 128, 4 * 128, 128, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, BT, 8, 128), (BT * 8 * 128, 8 * 128, 128, 1)))
    u_flat = al.make_tensor(u_ptr, al.f32, al.make_layout((BT * 8 * 128,), (1,)))
    state_in = al.make_tensor(state_in_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    decay = al.make_tensor(decay_ptr, al.f32, al.make_layout((1, 1, 8, BT), (8 * BT, 8 * BT, BT, 1)))
    sink = al.make_tensor(sink_ptr, al.f32, al.make_layout((GRID, 4096), (4096, 1)))

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_mod32 = lane & 31
    lane_col = lane & 15
    lane_group = lane >> 4

    program_id = al.block_id(0)
    v_block_idx = program_id % 4
    value_head_idx = program_id // 4
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx // 2

    for sink_rep in al.range(32):
        sink[program_id, tid + sink_rep * WORKGROUP] = al.convert(0.0, al.f32)

    if variant < 9:
        v_decay_t = al.make_shared((BV, BT), al.bf16)
        vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (8 * 4, 4, 1)))

        if variant < 8:
            state_bf16 = al.make_shared((2, BV, 64), al.bf16)
            w_bf16 = al.make_shared((2, 32, 64), al.bf16)
            state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
            w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))

            for rep_state in al.range(32):
                linear_s = tid + rep_state * WORKGROUP
                kb_s = linear_s // (BV * 64)
                rem_s = linear_s - kb_s * (BV * 64)
                row_s = rem_s // 64
                col_s = rem_s - row_s * 64
                global_v_s = value_base + row_s
                global_k_s = kb_s * 64 + col_s
                state_bf16[kb_s, row_s, col_s] = al.convert(state_in[0, value_head_idx, global_v_s, global_k_s], al.bf16)

            al.syncthreads()

            for token_tile in al.range(2):
                token_base = token_tile * 32
                for rep_w in al.range(32):
                    linear_w = tid + rep_w * WORKGROUP
                    kb_w = linear_w // (32 * 64)
                    rem_w = linear_w - kb_w * (32 * 64)
                    token_off_w = rem_w // 64
                    col_w = rem_w - token_off_w * 64
                    global_k_w = kb_w * 64 + col_w
                    w_bf16[kb_w, token_off_w, col_w] = al.convert(w[0, token_base + token_off_w, value_head_idx, global_k_w], al.bf16)

                al.syncthreads()

                pred_acc = al.full((16,), 0.0, al.f32)
                for kpack in al.range(4):
                    a_words = w_vec[wave_id, lane_mod32, kpack]
                    b_words = state_vec[wave_id, lane_mod32, kpack]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

                pred_partial = al.make_shared((2, 32, BV), al.f32)
                row_base = lane_col & 7
                col_base = ((lane_col >> 3) * 4) + lane_group
                for acc_i in al.range(16):
                    out_row = ((acc_i & 3) * 8) + row_base
                    out_col = ((acc_i >> 2) * 8) + col_base
                    pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]

                al.syncthreads()

                for rep_vd in al.range(8):
                    linear_vd = tid + rep_vd * WORKGROUP
                    tok_vd = linear_vd // BV
                    vv_vd = linear_vd - tok_vd * BV
                    token_idx_vd = token_base + tok_vd
                    offset_vd = token_idx_vd * (8 * 128) + value_head_idx * 128 + value_base + vv_vd
                    pred_vd = pred_partial[0, tok_vd, vv_vd] + pred_partial[1, tok_vd, vv_vd]
                    u_corr_vd = u_flat[offset_vd] - pred_vd
                    decay_v = decay[0, 0, value_head_idx, token_idx_vd]
                    v_decay_t[vv_vd, token_idx_vd] = al.convert(u_corr_vd * decay_v, al.bf16)

                al.syncthreads()

                if variant == 1 or variant == 7:
                    al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
                    al.syncthreads()

                if variant == 10 or variant == 11:
                    al.end_lifetime(pred_partial, state_bf16, w_bf16, state_vec, w_vec)
                    al.syncthreads()

                if variant == 0 or variant == 1 or variant == 10:
                    k_all_t = al.make_shared((128, BT), al.bf16)
                    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
                    for rep_k in al.range(64):
                        linear_k = tid + rep_k * WORKGROUP
                        kk = linear_k // BT
                        tok_k = linear_k - kk * BT
                        k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
                    al.syncthreads()

                    for tile in al.range(8):
                        acc16 = al.full((4,), 0.0, al.f32)
                        pack_base = token_tile * 4
                        if lane_group == 0:
                            a_words16 = vdecay_vec[lane_col, pack_base]
                            b_words16 = kall_vec[tile * 16 + lane_col, pack_base]
                            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                            acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], acc16)
                        if lane_group == 1:
                            a_words16 = vdecay_vec[lane_col, pack_base]
                            b_words16 = kall_vec[tile * 16 + lane_col, pack_base]
                            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                            acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[1], b_frag16[1], acc16)
                        if lane_group == 2:
                            a_words16 = vdecay_vec[lane_col, pack_base + 1]
                            b_words16 = kall_vec[tile * 16 + lane_col, pack_base + 1]
                            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                            acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], acc16)
                        if lane_group == 3:
                            a_words16 = vdecay_vec[lane_col, pack_base + 1]
                            b_words16 = kall_vec[tile * 16 + lane_col, pack_base + 1]
                            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                            acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[1], b_frag16[1], acc16)
                        for r in al.range(4):
                            sink[program_id, token_tile * 1024 + tile * 128 + lane_group * 16 + r * 4 + (lane_col & 3)] = acc16[r]

                if variant >= 2 and variant <= 3:
                    half_count = 1
                    if variant == 3:
                        half_count = 2
                    for k_half in al.range(half_count):
                        k_half_t = al.make_shared((64, BT), al.bf16)
                        khalf_vec = al.view(k_half_t, al.i32, al.make_layout((64, 8, 4), (8 * 4, 4, 1)))
                        for rep_half in al.range(32):
                            linear_half = tid + rep_half * WORKGROUP
                            kk_half = linear_half // BT
                            tok_half = linear_half - kk_half * BT
                            global_k_half = k_half * 64 + kk_half
                            k_half_t[kk_half, tok_half] = k[0, tok_half, key_head_idx, global_k_half]
                        al.syncthreads()

                        local_tile_count = 1
                        if variant == 3:
                            local_tile_count = 4
                        for local_tile in al.range(local_tile_count):
                            acc16_h = al.full((4,), 0.0, al.f32)
                            pack_base_h = token_tile * 4
                            row_h = local_tile * 16 + lane_col
                            if lane_group == 0:
                                a_words_h = vdecay_vec[lane_col, pack_base_h]
                                b_words_h = khalf_vec[row_h, pack_base_h]
                                a_frag_h = al.view(a_words_h, al.Tensor((2, 4, 1), al.bf16))
                                b_frag_h = al.view(b_words_h, al.Tensor((2, 4, 1), al.bf16))
                                acc16_h = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_h[0], b_frag_h[0], acc16_h)
                            if variant != 1:
                                if lane_group == 1:
                                    a_words_h = vdecay_vec[lane_col, pack_base_h]
                                    b_words_h = khalf_vec[row_h, pack_base_h]
                                    a_frag_h = al.view(a_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    b_frag_h = al.view(b_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    acc16_h = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_h[1], b_frag_h[1], acc16_h)
                                if lane_group == 2:
                                    a_words_h = vdecay_vec[lane_col, pack_base_h + 1]
                                    b_words_h = khalf_vec[row_h, pack_base_h + 1]
                                    a_frag_h = al.view(a_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    b_frag_h = al.view(b_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    acc16_h = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_h[0], b_frag_h[0], acc16_h)
                                if lane_group == 3:
                                    a_words_h = vdecay_vec[lane_col, pack_base_h + 1]
                                    b_words_h = khalf_vec[row_h, pack_base_h + 1]
                                    a_frag_h = al.view(a_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    b_frag_h = al.view(b_words_h, al.Tensor((2, 4, 1), al.bf16))
                                    acc16_h = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_h[1], b_frag_h[1], acc16_h)
                            for rh in al.range(4):
                                sink[program_id, token_tile * 1024 + k_half * 512 + local_tile * 128 + lane_group * 16 + rh * 4 + (lane_col & 3)] = acc16_h[rh]
                        al.syncthreads()

                if variant == 6 or variant == 7 or variant == 11:
                    # K-subtile helper/source pattern:
                    #
                    # Stage only K[128,16] for the 16-token subtile consumed by
                    # the current update tile.  This avoids broad
                    # k_all_t[128,BT] materialization and the wide kall_vec view.
                    k_sub_t = al.make_shared((128, 16), al.bf16)
                    ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))
                    for rep_sub in al.range(16):
                        linear_sub = tid + rep_sub * WORKGROUP
                        kk_sub = linear_sub // 16
                        tok_local = linear_sub - kk_sub * 16
                        tok_sub = token_base + tok_local
                        k_sub_t[kk_sub, tok_local] = k[0, tok_sub, key_head_idx, kk_sub]
                    al.syncthreads()

                    sub_tile_count = 1
                    if variant == 6 or variant == 7 or variant == 11:
                        sub_tile_count = 8
                    for tile_s in al.range(sub_tile_count):
                        acc16_s = al.full((4,), 0.0, al.f32)
                        pack_base_s = token_tile * 4
                        if lane_group == 0:
                            a_words_s = vdecay_vec[lane_col, pack_base_s]
                            b_words_s = ksub_vec[tile_s * 16 + lane_col, 0]
                            a_frag_s = al.view(a_words_s, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_s = al.view(b_words_s, al.Tensor((2, 4, 1), al.bf16))
                            acc16_s = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[0], b_frag_s[0], acc16_s)
                        if variant != 4:
                            if lane_group == 1:
                                a_words_s = vdecay_vec[lane_col, pack_base_s]
                                b_words_s = ksub_vec[tile_s * 16 + lane_col, 0]
                                a_frag_s = al.view(a_words_s, al.Tensor((2, 4, 1), al.bf16))
                                b_frag_s = al.view(b_words_s, al.Tensor((2, 4, 1), al.bf16))
                                acc16_s = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[1], b_frag_s[1], acc16_s)
                            if lane_group == 2:
                                a_words_s = vdecay_vec[lane_col, pack_base_s + 1]
                                b_words_s = ksub_vec[tile_s * 16 + lane_col, 1]
                                a_frag_s = al.view(a_words_s, al.Tensor((2, 4, 1), al.bf16))
                                b_frag_s = al.view(b_words_s, al.Tensor((2, 4, 1), al.bf16))
                                acc16_s = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[0], b_frag_s[0], acc16_s)
                            if lane_group == 3:
                                a_words_s = vdecay_vec[lane_col, pack_base_s + 1]
                                b_words_s = ksub_vec[tile_s * 16 + lane_col, 1]
                                a_frag_s = al.view(a_words_s, al.Tensor((2, 4, 1), al.bf16))
                                b_frag_s = al.view(b_words_s, al.Tensor((2, 4, 1), al.bf16))
                                acc16_s = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_s[1], b_frag_s[1], acc16_s)
                        for rs in al.range(4):
                            sink[program_id, token_tile * 1024 + tile_s * 128 + lane_group * 16 + rs * 4 + (lane_col & 3)] = acc16_s[rs]

                if variant == 99:
                    k_tile_t = al.make_shared((16, BT), al.bf16)
                    ktile_vec = al.view(k_tile_t, al.i32, al.make_layout((16, 8, 4), (8 * 4, 4, 1)))
                    for rep_tile in al.range(8):
                        linear_tile = tid + rep_tile * WORKGROUP
                        kk_tile = linear_tile // BT
                        tok_tile = linear_tile - kk_tile * BT
                        k_tile_t[kk_tile, tok_tile] = k[0, tok_tile, key_head_idx, kk_tile]
                    al.syncthreads()

                    acc16_d = al.full((4,), 0.0, al.f32)
                    pack_base_d = token_tile * 4
                    if lane_group == 0:
                        a_words_d = vdecay_vec[lane_col, pack_base_d]
                        b_words_d = ktile_vec[lane_col, pack_base_d]
                        a_frag_d = al.view(a_words_d, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_d = al.view(b_words_d, al.Tensor((2, 4, 1), al.bf16))
                        acc16_d = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_d[0], b_frag_d[0], acc16_d)
                    if lane_group == 1:
                        a_words_d = vdecay_vec[lane_col, pack_base_d]
                        b_words_d = ktile_vec[lane_col, pack_base_d]
                        a_frag_d = al.view(a_words_d, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_d = al.view(b_words_d, al.Tensor((2, 4, 1), al.bf16))
                        acc16_d = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_d[1], b_frag_d[1], acc16_d)
                    if lane_group == 2:
                        a_words_d = vdecay_vec[lane_col, pack_base_d + 1]
                        b_words_d = ktile_vec[lane_col, pack_base_d + 1]
                        a_frag_d = al.view(a_words_d, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_d = al.view(b_words_d, al.Tensor((2, 4, 1), al.bf16))
                        acc16_d = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_d[0], b_frag_d[0], acc16_d)
                    if lane_group == 3:
                        a_words_d = vdecay_vec[lane_col, pack_base_d + 1]
                        b_words_d = ktile_vec[lane_col, pack_base_d + 1]
                        a_frag_d = al.view(a_words_d, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_d = al.view(b_words_d, al.Tensor((2, 4, 1), al.bf16))
                        acc16_d = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_d[1], b_frag_d[1], acc16_d)
                    for rd in al.range(4):
                        sink[program_id, token_tile * 512 + lane_group * 16 + rd * 4 + (lane_col & 3)] = acc16_d[rd]

                al.syncthreads()

        if variant == 8:
            for rep_syn in al.range(16):
                linear_syn = tid + rep_syn * WORKGROUP
                vv_syn = linear_syn // BT
                tok_syn = linear_syn - vv_syn * BT
                v_decay_t[vv_syn, tok_syn] = al.convert(
                    al.convert((tok_syn + 1), al.f32) * al.convert(0.00390625, al.f32)
                    + al.convert((vv_syn + 1), al.f32) * al.convert(0.001953125, al.f32),
                    al.bf16,
                )

            al.syncthreads()

            k_all_t_np = al.make_shared((128, BT), al.bf16)
            kall_vec_np = al.view(k_all_t_np, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
            for rep_k_np in al.range(64):
                linear_k_np = tid + rep_k_np * WORKGROUP
                kk_np = linear_k_np // BT
                tok_np = linear_k_np - kk_np * BT
                k_all_t_np[kk_np, tok_np] = k[0, tok_np, key_head_idx, kk_np]

            al.syncthreads()

            for tile_np in al.range(8):
                acc16_np = al.full((4,), 0.0, al.f32)
                if lane_group == 0:
                    a_words_np = vdecay_vec[lane_col, 0]
                    b_words_np = kall_vec_np[tile_np * 16 + lane_col, 0]
                    a_frag_np = al.view(a_words_np, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_np = al.view(b_words_np, al.Tensor((2, 4, 1), al.bf16))
                    acc16_np = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_np[0], b_frag_np[0], acc16_np)
                if lane_group == 1:
                    a_words_np = vdecay_vec[lane_col, 0]
                    b_words_np = kall_vec_np[tile_np * 16 + lane_col, 0]
                    a_frag_np = al.view(a_words_np, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_np = al.view(b_words_np, al.Tensor((2, 4, 1), al.bf16))
                    acc16_np = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_np[1], b_frag_np[1], acc16_np)
                if lane_group == 2:
                    a_words_np = vdecay_vec[lane_col, 1]
                    b_words_np = kall_vec_np[tile_np * 16 + lane_col, 1]
                    a_frag_np = al.view(a_words_np, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_np = al.view(b_words_np, al.Tensor((2, 4, 1), al.bf16))
                    acc16_np = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_np[0], b_frag_np[0], acc16_np)
                if lane_group == 3:
                    a_words_np = vdecay_vec[lane_col, 1]
                    b_words_np = kall_vec_np[tile_np * 16 + lane_col, 1]
                    a_frag_np = al.view(a_words_np, al.Tensor((2, 4, 1), al.bf16))
                    b_frag_np = al.view(b_words_np, al.Tensor((2, 4, 1), al.bf16))
                    acc16_np = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_np[1], b_frag_np[1], acc16_np)
                for rn in al.range(4):
                    sink[program_id, tile_np * 128 + lane_group * 16 + rn * 4 + (lane_col & 3)] = acc16_np[rn]

    if variant == 9:
        v_decay_min = al.make_shared((16, 16), al.bf16)
        k_min = al.make_shared((16, 16), al.bf16)
        vmin_vec = al.view(v_decay_min, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))
        kmin_vec = al.view(k_min, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))
        for rep_min in al.range(2):
            linear_min = tid + rep_min * WORKGROUP
            row_min = linear_min // 16
            col_min = linear_min - row_min * 16
            v_decay_min[row_min, col_min] = al.convert(
                al.convert((row_min + 1), al.f32) * al.convert(0.0078125, al.f32)
                + al.convert((col_min + 1), al.f32) * al.convert(0.00390625, al.f32),
                al.bf16,
            )
            k_min[row_min, col_min] = k[0, col_min, key_head_idx, row_min]

        al.syncthreads()

        acc_min = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words_m = vmin_vec[lane_col, 0]
            b_words_m = kmin_vec[lane_col, 0]
            a_frag_m = al.view(a_words_m, al.Tensor((2, 4, 1), al.bf16))
            b_frag_m = al.view(b_words_m, al.Tensor((2, 4, 1), al.bf16))
            acc_min = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_m[0], b_frag_m[0], acc_min)
        if lane_group == 1:
            a_words_m = vmin_vec[lane_col, 0]
            b_words_m = kmin_vec[lane_col, 0]
            a_frag_m = al.view(a_words_m, al.Tensor((2, 4, 1), al.bf16))
            b_frag_m = al.view(b_words_m, al.Tensor((2, 4, 1), al.bf16))
            acc_min = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_m[1], b_frag_m[1], acc_min)
        if lane_group == 2:
            a_words_m = vmin_vec[lane_col, 1]
            b_words_m = kmin_vec[lane_col, 1]
            a_frag_m = al.view(a_words_m, al.Tensor((2, 4, 1), al.bf16))
            b_frag_m = al.view(b_words_m, al.Tensor((2, 4, 1), al.bf16))
            acc_min = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_m[0], b_frag_m[0], acc_min)
        if lane_group == 3:
            a_words_m = vmin_vec[lane_col, 1]
            b_words_m = kmin_vec[lane_col, 1]
            a_frag_m = al.view(a_words_m, al.Tensor((2, 4, 1), al.bf16))
            b_frag_m = al.view(b_words_m, al.Tensor((2, 4, 1), al.bf16))
            acc_min = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_m[1], b_frag_m[1], acc_min)
        for r_min in al.range(4):
            sink[program_id, lane_group * 16 + r_min * 4 + (lane_col & 3)] = acc_min[r_min]


def make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn((1, BT, 4, 128), device=device, dtype=torch.bfloat16)
    w = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    u = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    state = torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.05
    decay = torch.rand((1, 1, 8, BT), device=device, dtype=torch.float32) * 0.5 + 0.75
    sink = torch.empty((GRID, 4096), device=device, dtype=torch.float32)
    return k, w, u, state, decay, sink


def launch_variant(variant_name: str, tensors: tuple[torch.Tensor, ...]) -> None:
    if variant_name not in VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}")
    _qwen_mfma32_l6_kstage_update_variants_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
        *tensors,
        VARIANTS[variant_name],
        num_warps=2,
    )


def run_variant(variant_name: str, *, seed: int, warmup: int, repeat: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    tensors = make_inputs(seed)
    for _ in range(warmup):
        launch_variant(variant_name, tensors)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch_variant(variant_name, tensors)
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    sink = tensors[-1]
    finite = bool(torch.isfinite(sink).all().item())
    checksum = float(torch.nan_to_num(sink.float()).abs().sum().item())
    return {
        "variant": variant_name,
        "latency_ms": statistics.median(times),
        "sink_finite": finite,
        "sink_checksum_abs": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    rows = [run_variant(name, seed=args.seed, warmup=args.warmup, repeat=args.repeat) for name in names]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}")
        print(f"device={torch.cuda.get_device_name(0)}")
        for row in rows:
            print(
                "variant={variant},latency_ms={latency_ms:.6f},sink_finite={sink_finite},sink_checksum_abs={sink_checksum_abs:.9g}".format(
                    **row
                )
            )


if __name__ == "__main__":
    main()

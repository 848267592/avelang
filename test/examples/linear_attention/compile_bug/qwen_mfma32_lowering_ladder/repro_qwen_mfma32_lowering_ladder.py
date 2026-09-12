#!/usr/bin/env python3
"""Qwen-shaped MFMA32 lowering ladder.

This is a compiler/backend evidence repro, not a production Qwen GDN kernel.
It grows a v29-like BT64/BV32 MFMA32 pred path one source feature at a time so
rocprof/ISA deltas can identify which Qwen-shaped construct causes resource
growth.
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
KDIM = 128
WORKGROUP = 128
GRID = 32  # 8 value heads * 4 V-blocks

LEVELS = {
    "L0_pred32_mfma_only": 0,
    "L1_pred32_unpack_to_pred_partial": 1,
    "L2_pred32_unpack_reduce": 2,
    "L3_pred32_ucorr_epilogue_no_store": 3,
    "L4_pred32_vdecay_shared_stage": 4,
    "L5_pred32_vdecay_plus_k_stage": 5,
    "L5_alt_token_major_k_stage": -5,
    "L6_pred32_vdecay_update16_one_ktile": 6,
    "L7_pred32_vdecay_update16_full_k_no_feedback": 7,
    "L8_pred32_update16_full_k_state_writeback": 8,
    "L9_pred32_update16_with_h_vn_global_materialization": 9,
    "L9_alt_grouped_v4_materialization": 19,
}
LEVEL_NAMES = {v: k for k, v in LEVELS.items()}


@avelang.jit
def _qwen_mfma32_lowering_ladder_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    state_in_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    state_out_ptr: al.Pointer(al.f32),
    sink_ptr: al.Pointer(al.f32),
    level: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, BT, 4, 128), (BT * 4 * 128, 4 * 128, 128, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, BT, 8, 128), (BT * 8 * 128, 8 * 128, 128, 1)))
    u = al.make_tensor(u_ptr, al.f32, al.make_layout((1, BT, 8, 128), (BT * 8 * 128, 8 * 128, 128, 1)))
    u_flat = al.make_tensor(u_ptr, al.f32, al.make_layout((BT * 8 * 128,), (1,)))
    state_in = al.make_tensor(state_in_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    decay = al.make_tensor(decay_ptr, al.f32, al.make_layout((1, 1, 8, BT), (8 * BT, 8 * BT, BT, 1)))
    h = al.make_tensor(h_ptr, al.f32, al.make_layout((1, 1, 8, 128, 128), (8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.f32, al.make_layout((1, BT, 8, 128), (BT * 8 * 128, 8 * 128, 128, 1)))
    vn_flat = al.make_tensor(vn_ptr, al.f32, al.make_layout((BT * 8 * 128,), (1,)))
    state_out = al.make_tensor(state_out_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    sink = al.make_tensor(sink_ptr, al.f32, al.make_layout((GRID, 512), (512, 1)))

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

    for sink_rep in al.range(4):
        sink[program_id, tid + sink_rep * WORKGROUP] = al.convert(0.0, al.f32)

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

    if level >= 8 or level == 19:
        for rep_state_out0 in al.range(32):
            linear_so0 = tid + rep_state_out0 * WORKGROUP
            vv_so0 = linear_so0 // 128
            kk_so0 = linear_so0 - vv_so0 * 128
            global_v_so0 = value_base + vv_so0
            state_out[0, value_head_idx, global_v_so0, kk_so0] = state_in[0, value_head_idx, global_v_so0, kk_so0]

    # Two 32-token tiles inside BT64.
    for token_tile in al.range(2):
        token_base = token_tile * 32
        for rep_w in al.range(32):
            linear_w = tid + rep_w * WORKGROUP
            kb_w = linear_w // (32 * 64)
            rem_w = linear_w - kb_w * (32 * 64)
            token_off_w = rem_w // 64
            col_w = rem_w - token_off_w * 64
            global_k_w = kb_w * 64 + col_w
            w_bf16[kb_w, token_off_w, col_w] = al.convert(
                w[0, token_base + token_off_w, value_head_idx, global_k_w],
                al.bf16,
            )

        al.syncthreads()

        pred_acc = al.full((16,), 0.0, al.f32)
        for kpack in al.range(4):
            a_words = w_vec[wave_id, lane_mod32, kpack]
            b_words = state_vec[wave_id, lane_mod32, kpack]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

        if level == 0:
            if lane < 16:
                sink[program_id, token_tile * 64 + lane] = pred_acc[lane & 15]

        if level >= 1 or level == 19 or level == -5:
            pred_partial = al.make_shared((2, 32, BV), al.f32)
            row_base = lane_col & 7
            col_base = ((lane_col >> 3) * 4) + lane_group
            for acc_i in al.range(16):
                out_row = ((acc_i & 3) * 8) + row_base
                out_col = ((acc_i >> 2) * 8) + col_base
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]

            al.syncthreads()

            if level == 1:
                if lane < 16:
                    sink[program_id, token_tile * 64 + lane] = pred_partial[0, lane & 31, lane_col]

            if level >= 2 or level == 19 or level == -5:
                if level == 2:
                    for rep_red in al.range(8):
                        linear_r = tid + rep_red * WORKGROUP
                        tok_r = linear_r // BV
                        vv_r = linear_r - tok_r * BV
                        pred_r = pred_partial[0, tok_r, vv_r] + pred_partial[1, tok_r, vv_r]
                        sink[program_id, token_tile * 128 + (linear_r & 127)] = pred_r

                if level >= 3 or level == 19 or level == -5:
                    if level <= 3:
                        for rep_u in al.range(8):
                            linear_u = tid + rep_u * WORKGROUP
                            tok_u = linear_u // BV
                            vv_u = linear_u - tok_u * BV
                            token_idx_u = token_base + tok_u
                            offset_u = token_idx_u * (8 * 128) + value_head_idx * 128 + value_base + vv_u
                            pred_u = pred_partial[0, tok_u, vv_u] + pred_partial[1, tok_u, vv_u]
                            u_corr = u_flat[offset_u] - pred_u
                            sink[program_id, token_tile * 128 + (linear_u & 127)] = u_corr

                    if level >= 4 or level == 19 or level == -5:
                        v_decay_t = al.make_shared((BV, BT), al.bf16)
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
                            if level == 4:
                                sink[program_id, token_tile * 128 + (linear_vd & 127)] = al.convert(v_decay_t[vv_vd, token_idx_vd], al.f32)

                        al.syncthreads()

                        if level == -5:
                            k_token_major = al.make_shared((BT, 128), al.bf16)
                            for rep_kt in al.range(64):
                                linear_kt = tid + rep_kt * WORKGROUP
                                tok_kt = linear_kt // 128
                                kk_kt = linear_kt - tok_kt * 128
                                k_token_major[tok_kt, kk_kt] = k[0, tok_kt, key_head_idx, kk_kt]
                                vv_keep_alt = tok_kt & 31
                                sink[program_id, (linear_kt & 511)] = (
                                    al.convert(k_token_major[tok_kt, kk_kt], al.f32)
                                    + al.convert(v_decay_t[vv_keep_alt, tok_kt], al.f32)
                                )

                            al.syncthreads()

                        if level >= 5 or level == 19:
                            k_all_t = al.make_shared((128, BT), al.bf16)
                            vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (8 * 4, 4, 1)))
                            kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
                            for rep_k in al.range(64):
                                linear_k = tid + rep_k * WORKGROUP
                                kk = linear_k // BT
                                tok_k = linear_k - kk * BT
                                k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
                                if level == 5:
                                    # Keep both the newly staged K tile and the
                                    # earlier pred->u_corr->v_decay path live.
                                    vv_keep = tok_k & 31
                                    sink[program_id, (linear_k & 511)] = (
                                        al.convert(k_all_t[kk, tok_k], al.f32)
                                        + al.convert(v_decay_t[vv_keep, tok_k], al.f32)
                                    )

                            al.syncthreads()

                            if level >= 6 or level == 19:
                                tile_count = 1
                                if level >= 7 or level == 19:
                                    tile_count = 8
                                for tile in al.range(tile_count):
                                    acc16 = al.full((4,), 0.0, al.f32)
                                    # Use token_tile-local 32-token half. This isolates first dependent update without full BT64 complexity.
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

                                    if level == 6:
                                        for r6 in al.range(4):
                                            sink[program_id, tile * 64 + lane_group * 16 + r6 * 4 + (lane_col & 3)] = acc16[r6]

                                    if level >= 7 or level == 19:
                                        out_col = tile * 16 + lane_col
                                        for r7 in al.range(4):
                                            out_v = lane_group * 4 + r7
                                            state_delta = acc16[r7]
                                            if level == 7:
                                                sink[program_id, tile * 64 + lane_group * 16 + r7 * 4 + (lane_col & 3)] = state_delta
                                            if level >= 8 or level == 19:
                                                global_v_o = value_base + out_v
                                                state_out[0, value_head_idx, global_v_o, out_col] = (
                                                    state_in[0, value_head_idx, global_v_o, out_col] + state_delta
                                                )

                                if level >= 9:
                                    for rep_mat in al.range(8):
                                        linear_m = tid + rep_mat * WORKGROUP
                                        tok_m = linear_m // BV
                                        vv_m = linear_m - tok_m * BV
                                        token_idx_m = token_base + tok_m
                                        global_v_m = value_base + vv_m
                                        pred_m = pred_partial[0, tok_m, vv_m] + pred_partial[1, tok_m, vv_m]
                                        vn_val = u[0, token_idx_m, value_head_idx, global_v_m] - pred_m
                                        vn[0, token_idx_m, value_head_idx, global_v_m] = vn_val
                                        h[0, 0, value_head_idx, global_v_m, lane & 127] = state_in[0, value_head_idx, global_v_m, lane & 127]

                                if level == 19:
                                    # Workaround candidate: grouped-v4 flat materialization to reduce address-generation pressure.
                                    for rep4 in al.range(2):
                                        linear4 = tid + rep4 * WORKGROUP
                                        tok4 = linear4 // 8
                                        v4 = (linear4 - tok4 * 8) * 4
                                        token_idx4 = token_base + tok4
                                        base_offset = token_idx4 * (8 * 128) + value_head_idx * 128 + value_base + v4
                                        p0 = pred_partial[0, tok4, v4 + 0] + pred_partial[1, tok4, v4 + 0]
                                        p1 = pred_partial[0, tok4, v4 + 1] + pred_partial[1, tok4, v4 + 1]
                                        p2 = pred_partial[0, tok4, v4 + 2] + pred_partial[1, tok4, v4 + 2]
                                        p3 = pred_partial[0, tok4, v4 + 3] + pred_partial[1, tok4, v4 + 3]
                                        vn_flat[base_offset + 0] = u_flat[base_offset + 0] - p0
                                        vn_flat[base_offset + 1] = u_flat[base_offset + 1] - p1
                                        vn_flat[base_offset + 2] = u_flat[base_offset + 2] - p2
                                        vn_flat[base_offset + 3] = u_flat[base_offset + 3] - p3

        al.syncthreads()


def make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn((1, BT, 4, 128), device=device, dtype=torch.bfloat16)
    w = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    u = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    state = torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.05
    decay = torch.rand((1, 1, 8, BT), device=device, dtype=torch.float32) * 0.5 + 0.75
    h = torch.empty((1, 1, 8, 128, 128), device=device, dtype=torch.float32)
    vn = torch.empty((1, BT, 8, 128), device=device, dtype=torch.float32)
    state_out = torch.empty((1, 8, 128, 128), device=device, dtype=torch.float32)
    sink = torch.empty((GRID, 512), device=device, dtype=torch.float32)
    return k, w, u, state, decay, h, vn, state_out, sink


def launch_level(level_name: str, tensors: tuple[torch.Tensor, ...]) -> None:
    if level_name not in LEVELS:
        raise ValueError(f"unknown level {level_name!r}")
    _qwen_mfma32_lowering_ladder_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
        *tensors,
        LEVELS[level_name],
        num_warps=2,
    )


def run_level(level_name: str, *, seed: int, warmup: int, repeat: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    tensors = make_inputs(seed)

    def launch() -> None:
        launch_level(level_name, tensors)

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    sink = tensors[-1]
    checksum = float(torch.nan_to_num(sink.float()).abs().sum().item())
    finite = bool(torch.isfinite(sink).all().item())
    return {
        "level": level_name,
        "latency_ms": statistics.median(times),
        "sink_finite": finite,
        "sink_checksum_abs": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=sorted(LEVELS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    names = sorted(LEVELS, key=lambda name: LEVELS[name]) if args.level == "all" else [args.level]
    rows = [run_level(name, seed=args.seed, warmup=args.warmup, repeat=args.repeat) for name in names]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}")
        print(f"device={torch.cuda.get_device_name(0)}")
        for row in rows:
            print(
                "level={level},latency_ms={latency_ms:.6f},sink_finite={sink_finite},sink_checksum_abs={sink_checksum_abs:.9g}".format(
                    **row
                )
            )


if __name__ == "__main__":
    main()

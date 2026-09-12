#!/usr/bin/env python3
"""Focused Qwen-shaped L5 K-staging variants.

This is a compiler/backend evidence repro.  It starts from the existing
Qwen-shaped lowering ladder L4 pred/u_corr/v_decay staging and varies only the
K staging pattern that was responsible for the L4 -> L5 jump.
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
    "baseline_L5": 5,
    "baseline_L6": 6,
    "L5_token_major_no_kall_vec": -5,
    "L5_transposed_prepacked_input": -6,
    "L5_subtile_k_stage_16token": -7,
    "L5_khalf_stage_64": -8,
    "L5_direct_global_k_update_probe": -9,
    "L5_packed_i32_contiguous_load": -10,
    "L5_alt_token_major_k_stage_existing": -11,
}


@avelang.jit
def _qwen_mfma32_l5_kstaging_variants_kernel(
    k_ptr: al.Pointer(al.bf16),
    k_t_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    state_in_ptr: al.Pointer(al.f32),
    decay_ptr: al.Pointer(al.f32),
    sink_ptr: al.Pointer(al.f32),
    variant: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, BT, 4, 128), (BT * 4 * 128, 4 * 128, 128, 1)))
    k_t = al.make_tensor(k_t_ptr, al.bf16, al.make_layout((4, 128, BT), (128 * BT, BT, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, BT, 8, 128), (BT * 8 * 128, 8 * 128, 128, 1)))
    u_flat = al.make_tensor(u_ptr, al.f32, al.make_layout((BT * 8 * 128,), (1,)))
    state_in = al.make_tensor(state_in_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    decay = al.make_tensor(decay_ptr, al.f32, al.make_layout((1, 1, 8, BT), (8 * BT, 8 * BT, BT, 1)))
    sink = al.make_tensor(sink_ptr, al.f32, al.make_layout((GRID, 1024), (1024, 1)))

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

    for sink_rep in al.range(8):
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

        al.syncthreads()

        if variant == 5 or variant == 6:
            k_all_t = al.make_shared((128, BT), al.bf16)
            vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (8 * 4, 4, 1)))
            kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
            for rep_k in al.range(64):
                linear_k = tid + rep_k * WORKGROUP
                kk = linear_k // BT
                tok_k = linear_k - kk * BT
                k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
                if variant == 5:
                    vv_keep = tok_k & 31
                    sink[program_id, linear_k & 1023] = al.convert(k_all_t[kk, tok_k], al.f32) + al.convert(
                        v_decay_t[vv_keep, tok_k],
                        al.f32,
                    )

            al.syncthreads()

            if variant == 6:
                acc16 = al.full((4,), 0.0, al.f32)
                pack_base = token_tile * 4
                if lane_group == 0:
                    a_words16 = vdecay_vec[lane_col, pack_base]
                    b_words16 = kall_vec[lane_col, pack_base]
                    a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                    b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                    acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], acc16)
                if lane_group == 1:
                    a_words16 = vdecay_vec[lane_col, pack_base]
                    b_words16 = kall_vec[lane_col, pack_base]
                    a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                    b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                    acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[1], b_frag16[1], acc16)
                if lane_group == 2:
                    a_words16 = vdecay_vec[lane_col, pack_base + 1]
                    b_words16 = kall_vec[lane_col, pack_base + 1]
                    a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                    b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                    acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], acc16)
                if lane_group == 3:
                    a_words16 = vdecay_vec[lane_col, pack_base + 1]
                    b_words16 = kall_vec[lane_col, pack_base + 1]
                    a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
                    b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
                    acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[1], b_frag16[1], acc16)
                for r6 in al.range(4):
                    sink[program_id, token_tile * 256 + lane_group * 16 + r6 * 4 + (lane_col & 3)] = acc16[r6]

        if variant == -5 or variant == -11:
            k_token_major = al.make_shared((BT, 128), al.bf16)
            for rep_kt in al.range(64):
                linear_kt = tid + rep_kt * WORKGROUP
                tok_kt = linear_kt // 128
                kk_kt = linear_kt - tok_kt * 128
                k_token_major[tok_kt, kk_kt] = k[0, tok_kt, key_head_idx, kk_kt]
                vv_keep_alt = tok_kt & 31
                sink[program_id, linear_kt & 1023] = al.convert(k_token_major[tok_kt, kk_kt], al.f32) + al.convert(
                    v_decay_t[vv_keep_alt, tok_kt],
                    al.f32,
                )
            al.syncthreads()

        if variant == -6:
            for rep_pre in al.range(64):
                linear_pre = tid + rep_pre * WORKGROUP
                kk_pre = linear_pre // BT
                tok_pre = linear_pre - kk_pre * BT
                vv_pre = tok_pre & 31
                sink[program_id, linear_pre & 1023] = al.convert(k_t[key_head_idx, kk_pre, tok_pre], al.f32) + al.convert(
                    v_decay_t[vv_pre, tok_pre],
                    al.f32,
                )

        if variant == -7:
            k_sub_t = al.make_shared((128, 16), al.bf16)
            for rep_sub in al.range(16):
                linear_sub = tid + rep_sub * WORKGROUP
                kk_sub = linear_sub // 16
                tok_local = linear_sub - kk_sub * 16
                tok_sub = token_base + tok_local
                k_sub_t[kk_sub, tok_local] = k[0, tok_sub, key_head_idx, kk_sub]
                vv_sub = tok_sub & 31
                sink[program_id, linear_sub & 1023] = al.convert(k_sub_t[kk_sub, tok_local], al.f32) + al.convert(
                    v_decay_t[vv_sub, tok_sub],
                    al.f32,
                )
            al.syncthreads()

        if variant == -8:
            k_half_t = al.make_shared((64, BT), al.bf16)
            for rep_half in al.range(32):
                linear_half = tid + rep_half * WORKGROUP
                kk_half = linear_half // BT
                tok_half = linear_half - kk_half * BT
                k_half_t[kk_half, tok_half] = k[0, tok_half, key_head_idx, kk_half]
                vv_half = tok_half & 31
                sink[program_id, linear_half & 1023] = al.convert(k_half_t[kk_half, tok_half], al.f32) + al.convert(
                    v_decay_t[vv_half, tok_half],
                    al.f32,
                )
            al.syncthreads()

        if variant == -9:
            for rep_gl in al.range(8):
                linear_gl = tid + rep_gl * WORKGROUP
                kk_gl = lane_col
                tok_gl = linear_gl & 63
                vv_gl = tok_gl & 31
                sink[program_id, linear_gl & 1023] = al.convert(k[0, tok_gl, key_head_idx, kk_gl], al.f32) + al.convert(
                    v_decay_t[vv_gl, tok_gl],
                    al.f32,
                )

        if variant == -10:
            k_packed = al.make_shared((BT, 128), al.bf16)
            k_pack_vec = al.view(k_packed, al.i32, al.make_layout((BT, 16, 4), (16 * 4, 4, 1)))
            for rep_pack in al.range(64):
                linear_pack = tid + rep_pack * WORKGROUP
                tok_pack = linear_pack // 128
                kk_pack = linear_pack - tok_pack * 128
                k_packed[tok_pack, kk_pack] = k[0, tok_pack, key_head_idx, kk_pack]
            al.syncthreads()
            for rep_word in al.range(16):
                linear_word = tid + rep_word * WORKGROUP
                tok_word = linear_word // 16
                pack_word = linear_word - tok_word * 16
                word = k_pack_vec[tok_word, pack_word]
                frag = al.view(word, al.Tensor((2, 4, 1), al.bf16))
                vv_word = tok_word & 31
                sink[program_id, linear_word & 1023] = al.convert(frag[0, 0, 0], al.f32) + al.convert(
                    v_decay_t[vv_word, tok_word],
                    al.f32,
                )

        al.syncthreads()


def make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn((1, BT, 4, 128), device=device, dtype=torch.bfloat16)
    k_t = k[0].permute(1, 2, 0).contiguous()
    w = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    u = torch.randn((1, BT, 8, 128), device=device, dtype=torch.float32)
    state = torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.05
    decay = torch.rand((1, 1, 8, BT), device=device, dtype=torch.float32) * 0.5 + 0.75
    sink = torch.empty((GRID, 1024), device=device, dtype=torch.float32)
    return k, k_t, w, u, state, decay, sink


def launch_variant(variant_name: str, tensors: tuple[torch.Tensor, ...]) -> None:
    if variant_name not in VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}")
    _qwen_mfma32_l5_kstaging_variants_kernel[lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))](
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
    return {
        "variant": variant_name,
        "latency_ms": statistics.median(times),
        "sink_finite": bool(torch.isfinite(sink).all().item()),
        "sink_checksum_abs": float(torch.nan_to_num(sink.float()).abs().sum().item()),
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

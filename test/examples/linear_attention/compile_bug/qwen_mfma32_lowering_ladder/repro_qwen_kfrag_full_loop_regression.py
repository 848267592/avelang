#!/usr/bin/env python3
"""Reduced BT64 loop ladder for the full v29 K-fragment rewrite regression.

This is deliberately a diagnostic sink kernel.  It keeps the [128,64] shared
K producer plus four persistent MFMA16 B consumers, while adding only the
full-v29 control/liveness structures needed to locate the first regression.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
NT = 32
T = BT * NT
WORKGROUP = 128
GRID = 32

VARIANTS = {
    "R0_isolated_l6_rewrite": 0,
    "R1_rewrite_plus_bt64_window_loop": 1,
    "R2_rewrite_plus_pred_vdecay_live": 2,
    "R3_rewrite_plus_state_update_writeback": 3,
    "R4_full_loop_skeleton_rewrite": 4,
    "A_current_fused": 4,
    "B_hard_shared_phase_boundary": 5,
    "C_no_pred_accumulator_control": 1,
}


@avelang.jit
def _qwen_kfrag_full_loop_regression_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    state_in_ptr: al.Pointer(al.f32),
    sink_ptr: al.Pointer(al.f32),
    variant: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, T, 4, 128), (T * 4 * 128, 4 * 128, 128, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout((1, T, 8, 128), (T * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, T, 8, 128), (T * 8 * 128, 8 * 128, 128, 1)),
    )
    state_in = al.make_tensor(
        state_in_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    sink = al.make_tensor(
        sink_ptr,
        al.f32,
        al.make_layout(
            (GRID, NT, 2, 4, 4, 16, 4),
            (
                NT * 2 * 4 * 4 * 16 * 4,
                2 * 4 * 4 * 16 * 4,
                4 * 4 * 16 * 4,
                4 * 16 * 4,
                16 * 4,
                4,
                1,
            ),
        ),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    value_head_idx = program_id // 4
    value_base = (program_id % 4) * BV
    key_head_idx = value_head_idx // 2

    # This is intentionally outside the chunk loop, matching original full
    # v29. The rewrite pass inserts its replacement before the producer loop.
    k_all_t = al.make_shared((128, BT), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    vdecay_vec = al.view(
        v_decay_t,
        al.i32,
        al.make_layout((BV, 8, 4), (8 * 4, 4, 1)),
    )
    state = al.make_shared((BV, 128), al.f32)
    # R2+ uses the actual v29 MFMA32 pred schedule, not a scalar stand-in.
    pred_w = al.make_shared((2, 32, 64), al.bf16)
    pred_state = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, 32), al.f32)
    pred_w_vec = al.view(pred_w, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))
    pred_state_vec = al.view(pred_state, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))
    for rep_state in al.range(32):
        linear_state = tid + rep_state * WORKGROUP
        vv_state = linear_state // 128
        kk_state = linear_state - vv_state * 128
        state[vv_state, kk_state] = state_in[
            0, value_head_idx, value_base + vv_state, kk_state
        ]
    # All waves consume the initialized state in the first pred phase.
    al.syncthreads()

    # R0 is the one-window anchor. R1-R4 use the full BT64 chunk-loop count.
    trip_count = 1
    if variant >= 1:
        trip_count = NT

    for chunk_idx in al.range(trip_count):
        chunk_start = chunk_idx * BT

        # Keep the real MFMA32 pred accumulator and unpacked pred_partial live
        # across v_decay/update, matching the full-v29 pressure composition.
        pred_live = al.full((4,), 0.0, al.f32)
        if variant >= 2:
            # Each wave owns a complete [32,64] pred tile. Using tid/128
            # filled only alternating rows of each wave-private tile and left
            # the remaining LDS values undefined.
            for rep_pred in al.range(32):
                linear_pred = lane + rep_pred * 64
                row_pred = linear_pred // 64
                col_pred = linear_pred - row_pred * 64
                pred_w[wave_id, row_pred, col_pred] = al.convert(
                    w[0, chunk_start + row_pred, value_head_idx, wave_id * 64 + col_pred], al.bf16)
                pred_state[wave_id, row_pred, col_pred] = al.convert(
                    state[row_pred, wave_id * 64 + col_pred], al.bf16)
            al.syncthreads()
            pred_acc = al.full((16,), 0.0, al.f32)
            for kpack_pred in al.range(4):
                a_words_pred = pred_w_vec[wave_id, lane & 31, kpack_pred]
                b_words_pred = pred_state_vec[wave_id, lane & 31, kpack_pred]
                a_frag_pred = al.view(a_words_pred, al.Tensor((2, 4, 1), al.bf16))
                b_frag_pred = al.view(b_words_pred, al.Tensor((2, 4, 1), al.bf16))
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag_pred[0], a_frag_pred[0], pred_acc)
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag_pred[1], a_frag_pred[1], pred_acc)
            row_base = lane_col & 7
            col_base = ((lane_col >> 3) * 4) + lane_group
            for acc_i in al.range(16):
                out_row = ((acc_i & 3) * 8) + row_base
                out_col = ((acc_i >> 2) * 8) + col_base
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
            if variant != 5:
                for pred_rep in al.range(4):
                    pred_live[pred_rep] = pred_partial[wave_id, lane_group * 4 + pred_rep, lane_col]
            # v_decay reloads both waves' pred_partial tiles.
            al.syncthreads()

        # B is a real dataflow cut: pred MFMA32 only writes pred_partial to
        # workgroup memory. All subsequent pred consumers reload after this
        # barrier; no pred_acc/pred_live SSA value crosses into update.
        if variant == 5:
            al.syncthreads()

        for rep_vdecay in al.range(16):
            linear_vdecay = tid + rep_vdecay * WORKGROUP
            vv_vdecay = linear_vdecay // BT
            tok_vdecay = linear_vdecay - vv_vdecay * BT
            correction = u[
                0, chunk_start + tok_vdecay, value_head_idx,
                value_base + vv_vdecay
            ]
            if variant >= 2:
                correction = correction - pred_partial[0, tok_vdecay & 31, vv_vdecay] - pred_partial[1, tok_vdecay & 31, vv_vdecay]
            v_decay_t[vv_vdecay, tok_vdecay] = al.convert(correction, al.bf16)

        # Original broad producer. The pass must erase this loop and replace
        # the four persistent consumers below.
        for rep_k in al.range(64):
            linear_k = tid + rep_k * WORKGROUP
            kk = linear_k // BT
            tok = linear_k - kk * BT
            k_all_t[kk, tok] = k[0, chunk_start + tok, key_head_idx, kk]

        al.syncthreads()

        for local_tile in al.range(4):
            base_k = wave_id * 64 + local_tile * 16
            update_acc = al.full((4,), 0.0, al.f32)
            for token_sub in al.range(4):
                pack_base = token_sub * 2
                if lane_group == 0:
                    a_words = vdecay_vec[lane_col, pack_base]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
                        k_all_t, k, tid, key_head_idx, chunk_start,
                        base_k + lane_col, chunk_start + token_sub * 16,
                    )
                    update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a_frag[0], b_frag, update_acc
                    )
                if lane_group == 1:
                    a_words = vdecay_vec[lane_col, pack_base]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
                        k_all_t, k, tid, key_head_idx, chunk_start,
                        base_k + lane_col, chunk_start + token_sub * 16 + 4,
                    )
                    update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a_frag[1], b_frag, update_acc
                    )
                if lane_group == 2:
                    a_words = vdecay_vec[lane_col, pack_base + 1]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
                        k_all_t, k, tid, key_head_idx, chunk_start,
                        base_k + lane_col, chunk_start + token_sub * 16 + 8,
                    )
                    update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a_frag[0], b_frag, update_acc
                    )
                if lane_group == 3:
                    a_words = vdecay_vec[lane_col, pack_base + 1]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
                        k_all_t, k, tid, key_head_idx, chunk_start,
                        base_k + lane_col, chunk_start + token_sub * 16 + 12,
                    )
                    update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(
                        a_frag[1], b_frag, update_acc
                    )

            if variant >= 3:
                for r in al.range(4):
                    out_v = lane_group * 4 + r
                    old_state = state[out_v, base_k + lane_col]
                    state[out_v, base_k + lane_col] = old_state + update_acc[r] * al.convert(1.0e-20, al.f32)
            if variant == 4:
                for r in al.range(4):
                    # Sink the real pred result. The update MFMA remains in
                    # the live region, but this avoids treating its synthetic
                    # diagnostic recurrence as a numerical reference.
                    sink[
                        program_id, chunk_idx, wave_id, local_tile,
                        lane_group, lane_col, r
                    ] = pred_live[r]
            if variant == 5:
                for r in al.range(4):
                    sink[
                        program_id, chunk_idx, wave_id, local_tile,
                        lane_group, lane_col, r
                    ] = pred_partial[wave_id, lane_group * 4 + r, lane_col]
            if variant < 4:
                for r in al.range(4):
                    sink[
                        program_id, chunk_idx, wave_id, local_tile,
                        lane_group, lane_col, r
                    ] = update_acc[r]
        al.syncthreads()


def make_inputs(seed: int, input_path: Path | None = None):
    if input_path is not None:
        saved = torch.load(input_path, weights_only=True)
        if len(saved) != 4:
            raise ValueError("--input-path must contain exactly k/w/u/state")
        k, w, u, state = tuple(t.to(device="cuda").contiguous() for t in saved)
        sink = torch.empty((GRID, NT, 2, 4, 4, 16, 4), device="cuda", dtype=torch.float32)
        return k, w, u, state, sink

    # Generate on CPU before the one-way host-to-device copy. This makes a
    # saved input bundle portable across separate HIP processes for lowering
    # A/B tests; GPU-side RNG state is not part of that contract.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Keep the 32-window recurrence finite; this is a lowering repro, not a
    # stress test for numerical stability of an unnormalized recurrence.
    k = (torch.randn((1, T, 4, 128), device="cpu", dtype=torch.float32, generator=generator) * 0.02).to(device="cuda", dtype=torch.bfloat16).contiguous()
    w = (torch.randn((1, T, 8, 128), device="cpu", dtype=torch.float32, generator=generator) * 0.002).to(device="cuda").contiguous()
    u = (torch.randn((1, T, 8, 128), device="cpu", dtype=torch.float32, generator=generator) * 0.02).to(device="cuda").contiguous()
    state = (torch.randn((1, 8, 128, 128), device="cpu", dtype=torch.float32, generator=generator) * 0.02).to(device="cuda").contiguous()
    sink = torch.empty((GRID, NT, 2, 4, 4, 16, 4), device="cuda", dtype=torch.float32)
    return k, w, u, state, sink


def launch_variant(variant: str, tensors):
    k, w, u, state, sink = tensors
    _qwen_kfrag_full_loop_regression_kernel[
        lambda: ((GRID, 1, 1), (WORKGROUP, 1, 1))
    ](k, w, u, state, sink, VARIANTS[variant], num_warps=2)
    return sink


def time_variant(variant: str, tensors, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        launch_variant(variant, tensors)
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        launch_variant(variant, tensors)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--save-sink", type=Path)
    parser.add_argument("--save-inputs", type=Path)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP")
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    rows = []
    for variant in variants:
        tensors = make_inputs(args.seed, args.input_path)
        if args.save_inputs:
            if len(variants) != 1:
                raise ValueError("--save-inputs requires exactly one variant")
            args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tuple(t.detach().cpu() for t in tensors[:4]), args.save_inputs)
        sink = launch_variant(variant, tensors)
        torch.cuda.synchronize()
        if args.save_sink:
            if len(variants) != 1:
                raise ValueError("--save-sink requires exactly one variant")
            args.save_sink.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sink.detach().cpu(), args.save_sink)
        rows.append(
            {
                "variant": variant,
                "latency_ms": time_variant(variant, tensors, args.warmup, args.repeat),
                "sink_finite": bool(torch.isfinite(sink).all().item()),
                "sink_checksum_abs": float(sink.abs().sum().item()),
                "input_checksums_abs": [
                    float(t.abs().sum().item()) for t in tensors[:4]
                ],
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()

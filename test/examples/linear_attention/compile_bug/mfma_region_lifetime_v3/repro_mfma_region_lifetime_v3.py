#!/usr/bin/env python3
"""Minimal sequential-MFMA region lifetime repro for Avelang AMDGPU lowering.

This is intentionally Qwen-free.  It isolates whether a pred MFMA region keeps
accumulator/register state live across a later independent update MFMA region.

The update16 path is the correctness anchor:

    out[16,128] = init[16,128] * scale + v_decay[16,16].T @ k_chunk[16,128]

Pred regions write tiny sinks so they cannot be DCE'd, but they are independent
of the update output.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


BT16 = 16
BV16 = 16
KDIM = 128

MODE_UPDATE16_ONLY = 0
MODE_PRED16_ONLY_SINK = 1
MODE_PRED32_ONLY_SINK = 2
MODE_PRED16_THEN_UPDATE16 = 3
MODE_PRED32_THEN_UPDATE16 = 4
MODE_PRED32_THEN_UPDATE32 = 5
MODE_PRED32_UNPACKED_TO_LDS_THEN_UPDATE16 = 6
MODE_PRED32_TWO_REGIONS_THEN_UPDATE16 = 7
MODE_PRED32_THEN_UPDATE16_WITH_DUMMY_BARRIER = 8
MODE_PRED32_THEN_UPDATE16_WITH_SOURCE_LIFETIME_HINT = 9

MODE_NAMES = {
    MODE_UPDATE16_ONLY: "update16_only",
    MODE_PRED16_ONLY_SINK: "pred16_only_sink",
    MODE_PRED32_ONLY_SINK: "pred32_only_sink",
    MODE_PRED16_THEN_UPDATE16: "pred16_then_update16",
    MODE_PRED32_THEN_UPDATE16: "pred32_then_update16",
    MODE_PRED32_THEN_UPDATE32: "pred32_then_update32",
    MODE_PRED32_UNPACKED_TO_LDS_THEN_UPDATE16: "pred32_unpacked_to_lds_then_update16",
    MODE_PRED32_TWO_REGIONS_THEN_UPDATE16: "pred32_two_regions_then_update16",
    MODE_PRED32_THEN_UPDATE16_WITH_DUMMY_BARRIER: "pred32_then_update16_with_dummy_barrier",
    MODE_PRED32_THEN_UPDATE16_WITH_SOURCE_LIFETIME_HINT: "pred32_then_update16_with_source_lifetime_hint",
}
NAME_TO_MODE = {v: k for k, v in MODE_NAMES.items()}


@avelang.jit
def _mfma_region_lifetime_v3_kernel(
    v16_ptr: al.Pointer(al.bf16),
    k16_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    a32_ptr: al.Pointer(al.bf16),
    b32_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    sink_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    mode: al.constexpr,
):
    v16 = al.make_tensor(v16_ptr, al.bf16, al.make_layout((BT16, BV16), (BV16, 1)))
    k16 = al.make_tensor(k16_ptr, al.bf16, al.make_layout((BT16, KDIM), (KDIM, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV16, KDIM), (KDIM, 1)))
    a32 = al.make_tensor(a32_ptr, al.bf16, al.make_layout((32, 64), (64, 1)))
    b32 = al.make_tensor(b32_ptr, al.bf16, al.make_layout((32, 64), (64, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV16, KDIM), (KDIM, 1)))
    sink = al.make_tensor(sink_ptr, al.f32, al.make_layout((512,), (1,)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    lane_mod32 = lane & 31

    state = al.make_shared((BV16, KDIM), al.f32)
    v_decay_t = al.make_shared((BV16, BT16), al.bf16)
    k_all_t = al.make_shared((KDIM, BT16), al.bf16)
    pred16_a = al.make_shared((BT16, BT16), al.bf16)
    pred16_b = al.make_shared((BV16, BT16), al.bf16)
    pred32_a = al.make_shared((32, 64), al.bf16)
    pred32_b = al.make_shared((32, 64), al.bf16)
    pred32_lds = al.make_shared((64, 16), al.f32)

    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV16, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((KDIM, 2, 4), (8, 4, 1)))
    pred16_a_vec = al.view(pred16_a, al.i32, al.make_layout((BT16, 2, 4), (8, 4, 1)))
    pred16_b_vec = al.view(pred16_b, al.i32, al.make_layout((BV16, 2, 4), (8, 4, 1)))
    pred32_a_vec = al.view(pred32_a, al.i32, al.make_layout((32, 8, 4), (32, 4, 1)))
    pred32_b_vec = al.view(pred32_b, al.i32, al.make_layout((32, 8, 4), (32, 4, 1)))

    for rep_sink in al.range(8):
        sink_idx = lane + rep_sink * 64
        sink[sink_idx] = al.convert(0.0, al.f32)

    for rep_state in al.range(32):
        idx = lane + rep_state * 64
        row = idx // KDIM
        col = idx - row * KDIM
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    for rep_v in al.range(4):
        idx_v = lane + rep_v * 64
        vv = idx_v // BT16
        tok = idx_v - vv * BT16
        v_decay_t[vv, tok] = v16[tok, vv]

    for rep_k in al.range(32):
        idx_k = lane + rep_k * 64
        kk = idx_k // BT16
        tok_k = idx_k - kk * BT16
        k_all_t[kk, tok_k] = k16[tok_k, kk]

    for rep_small in al.range(4):
        idx_s = lane + rep_small * 64
        row_s = idx_s // 16
        col_s = idx_s - row_s * 16
        pred16_a[row_s, col_s] = a32[row_s, col_s]
        pred16_b[row_s, col_s] = b32[row_s, col_s]

    for rep32 in al.range(32):
        idx32 = lane + rep32 * 64
        row32 = idx32 // 64
        col32 = idx32 - row32 * 64
        pred32_a[row32, col32] = a32[row32, col32]
        pred32_b[row32, col32] = b32[row32, col32]

    al.syncthreads()

    if mode == MODE_PRED16_ONLY_SINK or mode == MODE_PRED16_THEN_UPDATE16:
        pred16_acc = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words16 = pred16_a_vec[lane_col, 0]
            b_words16 = pred16_b_vec[lane_col, 0]
            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
            pred16_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], pred16_acc)
            pred16_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[1], b_frag16[1], pred16_acc)
        if lane < 4:
            sink[lane] = pred16_acc[lane]

    if (
        mode == MODE_PRED32_ONLY_SINK
        or mode == MODE_PRED32_THEN_UPDATE16
        or mode == MODE_PRED32_THEN_UPDATE32
        or mode == MODE_PRED32_UNPACKED_TO_LDS_THEN_UPDATE16
        or mode == MODE_PRED32_TWO_REGIONS_THEN_UPDATE16
        or mode == MODE_PRED32_THEN_UPDATE16_WITH_DUMMY_BARRIER
        or mode == MODE_PRED32_THEN_UPDATE16_WITH_SOURCE_LIFETIME_HINT
    ):
        pred32_acc = al.full((16,), 0.0, al.f32)
        a_row32 = lane_mod32
        b_row32 = lane_mod32
        for kpack32 in al.range(8):
            a_words32 = pred32_a_vec[a_row32, kpack32]
            b_words32 = pred32_b_vec[b_row32, kpack32]
            a_frag32 = al.view(a_words32, al.Tensor((2, 4, 1), al.bf16))
            b_frag32 = al.view(b_words32, al.Tensor((2, 4, 1), al.bf16))
            pred32_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag32[0], a_frag32[0], pred32_acc)
            pred32_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag32[1], a_frag32[1], pred32_acc)

        for acc_i in al.range(16):
            if mode == MODE_PRED32_UNPACKED_TO_LDS_THEN_UPDATE16:
                pred32_lds[lane, acc_i] = pred32_acc[acc_i]
            if lane < 32 and acc_i < 8:
                sink[lane * 8 + acc_i] = pred32_acc[acc_i]

        if mode == MODE_PRED32_TWO_REGIONS_THEN_UPDATE16:
            pred32_acc_b = al.full((16,), 0.0, al.f32)
            for kpack32_b in al.range(8):
                a_words32_b = pred32_a_vec[a_row32, kpack32_b]
                b_words32_b = pred32_b_vec[b_row32, kpack32_b]
                a_frag32_b = al.view(a_words32_b, al.Tensor((2, 4, 1), al.bf16))
                b_frag32_b = al.view(b_words32_b, al.Tensor((2, 4, 1), al.bf16))
                pred32_acc_b = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag32_b[0], a_frag32_b[0], pred32_acc_b)
                pred32_acc_b = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag32_b[1], a_frag32_b[1], pred32_acc_b)
            for acc_j in al.range(8):
                if lane < 32:
                    sink[256 + lane * 8 + acc_j] = pred32_acc_b[acc_j]

        if mode == MODE_PRED32_THEN_UPDATE16_WITH_SOURCE_LIFETIME_HINT:
            # Source-level lifetime hint attempt: overwrite the local SSA name
            # after the sink.  This may or may not affect lowering.
            pred32_acc = al.full((16,), 0.0, al.f32)
            if lane == 63:
                sink[511] = pred32_acc[0]

    if mode == MODE_PRED32_THEN_UPDATE16_WITH_DUMMY_BARRIER:
        al.syncthreads()
        al.syncthreads()

    al.syncthreads()

    if mode == MODE_PRED32_THEN_UPDATE32:
        update32_acc = al.full((16,), 0.0, al.f32)
        for kpack_u32 in al.range(8):
            a_words_u32 = pred32_a_vec[lane_mod32, kpack_u32]
            b_words_u32 = pred32_b_vec[lane_mod32, kpack_u32]
            a_frag_u32 = al.view(a_words_u32, al.Tensor((2, 4, 1), al.bf16))
            b_frag_u32 = al.view(b_words_u32, al.Tensor((2, 4, 1), al.bf16))
            update32_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag_u32[0], a_frag_u32[0], update32_acc)
            update32_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag_u32[1], a_frag_u32[1], update32_acc)
        for acc_u in al.range(8):
            if lane < 32:
                sink[128 + lane * 8 + acc_u] = update32_acc[acc_u]

    if (
        mode == MODE_UPDATE16_ONLY
        or mode == MODE_PRED16_THEN_UPDATE16
        or mode == MODE_PRED32_THEN_UPDATE16
        or mode == MODE_PRED32_UNPACKED_TO_LDS_THEN_UPDATE16
        or mode == MODE_PRED32_TWO_REGIONS_THEN_UPDATE16
        or mode == MODE_PRED32_THEN_UPDATE16_WITH_DUMMY_BARRIER
        or mode == MODE_PRED32_THEN_UPDATE16_WITH_SOURCE_LIFETIME_HINT
    ):
        for tile in al.range(8):
            base = tile * 16
            update_acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], update_acc)
            if lane_group == 1:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], update_acc)
            if lane_group == 2:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], update_acc)
            if lane_group == 3:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], update_acc)

            for r in al.range(4):
                out_row = lane_group * 4 + r
                out_col = base + lane_col
                state[out_row, out_col] = state[out_row, out_col] + update_acc[r]

            al.syncthreads()

    for rep_out in al.range(32):
        idx_o = lane + rep_out * 64
        row_o = idx_o // KDIM
        col_o = idx_o - row_o * KDIM
        out[row_o, col_o] = state[row_o, col_o]


def make_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = "cuda"
    v16 = torch.randn((BT16, BV16), device=device, dtype=torch.bfloat16)
    k16 = torch.randn((BT16, KDIM), device=device, dtype=torch.bfloat16)
    init = torch.randn((BV16, KDIM), device=device, dtype=torch.float32) * 0.01
    a32 = torch.randn((32, 64), device=device, dtype=torch.bfloat16)
    b32 = torch.randn((32, 64), device=device, dtype=torch.bfloat16)
    return v16, k16, init, a32, b32


def torch_update_ref(v16: torch.Tensor, k16: torch.Tensor, init: torch.Tensor, scale: float) -> torch.Tensor:
    return init.float() * scale + v16.float().T @ k16.float()


def run_variant(mode_name: str, *, seed: int, scale: float, warmup: int, repeat: int, atol: float, rtol: float) -> dict[str, object]:
    if mode_name not in NAME_TO_MODE:
        raise ValueError(f"unknown mode {mode_name!r}; choose one of {sorted(NAME_TO_MODE)}")
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    mode = NAME_TO_MODE[mode_name]
    v16, k16, init, a32, b32 = make_inputs(seed)
    out = torch.empty((BV16, KDIM), device="cuda", dtype=torch.float32)
    sink = torch.empty((512,), device="cuda", dtype=torch.float32)

    def launch() -> None:
        _mfma_region_lifetime_v3_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
            v16,
            k16,
            init,
            a32,
            b32,
            out,
            sink,
            float(scale),
            mode,
            num_warps=1,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    ref = torch_update_ref(v16, k16, init, scale)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    max_rel = float((diff / ref.float().abs().clamp_min(1e-6)).max().item())
    update_expected = mode_name in {
        "update16_only",
        "pred16_then_update16",
        "pred32_then_update16",
        "pred32_unpacked_to_lds_then_update16",
        "pred32_two_regions_then_update16",
        "pred32_then_update16_with_dummy_barrier",
        "pred32_then_update16_with_source_lifetime_hint",
    }
    finite_sink = bool(torch.isfinite(sink).all().item())
    checksum_sink = float(sink.float().abs().sum().item())
    ok = bool(torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)) if update_expected else finite_sink
    return {
        "mode": mode_name,
        "latency_ms": statistics.median(times),
        "update_expected": update_expected,
        "ok": ok,
        "max_abs": max_abs if update_expected else None,
        "mean_abs": mean_abs if update_expected else None,
        "max_rel": max_rel if update_expected else None,
        "sink_finite": finite_sink,
        "sink_checksum_abs": checksum_sink,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(NAME_TO_MODE) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--scale", type=float, default=0.73)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    modes = sorted(NAME_TO_MODE) if args.mode == "all" else [args.mode]
    rows = [
        run_variant(
            mode,
            seed=args.seed,
            scale=args.scale,
            warmup=args.warmup,
            repeat=args.repeat,
            atol=args.atol,
            rtol=args.rtol,
        )
        for mode in modes
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(f"torch={torch.__version__}, hip={getattr(torch.version, 'hip', None)}")
        print(f"device={torch.cuda.get_device_name(0)}")
        for row in rows:
            print(
                "mode={mode},ok={ok},latency_ms={latency_ms:.6f},"
                "max_abs={max_abs},mean_abs={mean_abs},max_rel={max_rel},"
                "sink_checksum_abs={sink_checksum_abs:.9g}".format(**row)
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 7A minimal workgroup-barrier provenance repros.

These kernels are deliberately Qwen-free.  They retain the relevant gfx942
source primitives from Stage 6Z (BF16 shared memory, 32x32x8 MFMA, and a
256-thread CTA) while changing exactly one synchronization relationship per
variant.  They are evidence for barrier provenance, not candidates for the
full graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


WORKGROUP = 256
ROWS = 64
COLS = 32

MODE_A_DIRECT_SHARED_TO_MFMA = 0
MODE_A_FRAGMENT_SHARED_TO_MFMA = 1
MODE_B_SCORE_REUSE_SPLIT_BARRIERS = 2
MODE_B_SCORE_REUSE_MERGED_BARRIER = 3
MODE_C_ALL_CTA_STAGE_OWNER_WAVE = 4

MODES = {
    "A_direct_shared_to_mfma": MODE_A_DIRECT_SHARED_TO_MFMA,
    "A_fragment_shared_to_mfma": MODE_A_FRAGMENT_SHARED_TO_MFMA,
    "B_score_reuse_split_barriers": MODE_B_SCORE_REUSE_SPLIT_BARRIERS,
    "B_score_reuse_merged_barrier": MODE_B_SCORE_REUSE_MERGED_BARRIER,
    "C_all_cta_stage_owner_wave": MODE_C_ALL_CTA_STAGE_OWNER_WAVE,
}


@avelang.jit
def _qwen_bt64_chunko_barrier_stage7a_kernel(
    lhs_ptr: al.Pointer(al.bf16),
    rhs_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    mode: al.constexpr,
):
    lhs = al.make_tensor(lhs_ptr, al.bf16, al.make_layout((ROWS, COLS), (COLS, 1)))
    rhs = al.make_tensor(rhs_ptr, al.bf16, al.make_layout((ROWS, COLS), (COLS, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((WORKGROUP,), (1,)))

    tid = al.thread_id(0)
    lane = tid & 63
    lane_mod32 = lane & 31
    lane_group = lane >> 5
    wave_id = tid >> 6

    if mode == MODE_A_DIRECT_SHARED_TO_MFMA or mode == MODE_A_FRAGMENT_SHARED_TO_MFMA:
        # A: CTA-wide producer -> MFMA consumer.  The fragment form only adds
        # a per-lane shared round-trip, so its extra barrier should be source
        # scheduling rather than a mathematical cross-wave requirement.
        phase = al.make_shared((ROWS, COLS), al.bf16)
        phase_vec = al.view(phase, al.Tensor((ROWS, 4, 4), al.i32))
        frag_words = al.make_shared((WORKGROUP * 2, 4), al.i32)
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // COLS
            col = idx - row * COLS
            phase[row, col] = lhs[row, col]
        al.syncthreads()

        # Both A modes use the same per-lane fragment serialization.  The
        # only variable is the extra CTA barrier after these lane-private
        # writes.  This avoids a Python/AveLang type-join artifact while
        # retaining the exact synchronization question.
        frag_words[tid] = phase_vec[lane_mod32, lane_group]
        frag_words[WORKGROUP + tid] = phase_vec[32 + lane_mod32, lane_group]
        if mode == MODE_A_FRAGMENT_SHARED_TO_MFMA:
            al.syncthreads()
        acc = al.full((16,), 0.0, al.f32)
        a_words = frag_words[tid]
        b_words = frag_words[WORKGROUP + tid]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        out[tid] = acc[0]

    if mode == MODE_B_SCORE_REUSE_SPLIT_BARRIERS or mode == MODE_B_SCORE_REUSE_MERGED_BARRIER:
        # B: lower and upper physical halves have disjoint ownership.  The
        # only consumer is after both stores.  A split barrier between the
        # writes is therefore deliberately redundant; the merged variant is
        # a direct safety/control comparison with bit-identical output.
        phase = al.make_shared((128, COLS), al.bf16)
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // COLS
            col = idx - row * COLS
            phase[row, col] = lhs[row, col]
        if mode == MODE_B_SCORE_REUSE_SPLIT_BARRIERS:
            al.syncthreads()
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // COLS
            col = idx - row * COLS
            phase[64 + row, col] = rhs[row, col]
        al.syncthreads()
        out[tid] = al.convert(phase[lane_mod32, lane_group], al.f32) + al.convert(
            phase[64 + lane_mod32, lane_group], al.f32
        )

    if mode == MODE_C_ALL_CTA_STAGE_OWNER_WAVE:
        # C: exactly the Z1 ownership shape: every wave participates in
        # staging, but one owner wave consumes the packed LDS operands.  This
        # establishes the one producer->owner MFMA barrier that cannot be
        # removed merely because only one wave owns an accumulator.
        phase = al.make_shared((ROWS, COLS), al.bf16)
        phase_vec = al.view(phase, al.Tensor((ROWS, 4, 4), al.i32))
        frag_words = al.make_shared((WORKGROUP * 2, 4), al.i32)
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // COLS
            col = idx - row * COLS
            phase[row, col] = lhs[row, col]
        al.syncthreads()
        frag_words[tid] = phase_vec[lane_mod32, lane_group]
        frag_words[WORKGROUP + tid] = phase_vec[32 + lane_mod32, lane_group]
        if wave_id == 0:
            owner_acc = al.full((16,), 0.0, al.f32)
            a_words = frag_words[tid]
            b_words = frag_words[WORKGROUP + tid]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            owner_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], owner_acc)
            out[tid] = owner_acc[0]


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    lhs = torch.randn((ROWS, COLS), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    rhs = torch.randn((ROWS, COLS), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    return lhs, rhs


def launch(mode: str, lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown Stage 7A mode: {mode}")
    _qwen_bt64_chunko_barrier_stage7a_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](lhs, rhs, out, MODES[mode])


def _time(fn: Callable[[], None], repeat: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def run_mode(mode: str, *, seed: int, warmup: int, repeat: int) -> dict[str, object]:
    lhs, rhs = _inputs(seed)
    out = torch.zeros((WORKGROUP,), device="cuda", dtype=torch.float32)
    fn = lambda: launch(mode, lhs, rhs, out)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = _time(fn, repeat)
    torch.cuda.synchronize()
    return {
        "mode": mode,
        "finite": bool(torch.isfinite(out).all().item()),
        "sum": float(out.float().sum().item()),
        "abs_mean": float(out.float().abs().mean().item()),
        "hip_ms_median": statistics.median(times),
        "hip_ms_p10": sorted(times)[max(0, int(repeat * 0.1) - 1)],
        "hip_ms_p90": sorted(times)[min(repeat - 1, int(repeat * 0.9))],
    }


def _equal_modes(first: str, second: str, seed: int) -> dict[str, object]:
    lhs, rhs = _inputs(seed)
    a = torch.zeros((WORKGROUP,), device="cuda", dtype=torch.float32)
    b = torch.zeros_like(a)
    launch(first, lhs, rhs, a)
    launch(second, lhs, rhs, b)
    torch.cuda.synchronize()
    diff = (a - b).abs()
    return {
        "first": first,
        "second": second,
        "bit_exact": bool(torch.equal(a, b)),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--seed", type=int, default=2026072207)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7A requires the gfx942 HIP runtime")
    modes = tuple(MODES) if args.mode == "all" else (args.mode,)
    rows = [run_mode(mode, seed=args.seed, warmup=args.warmup, repeat=args.repeat) for mode in modes]
    checks = []
    if args.mode == "all":
        checks = [
            _equal_modes("A_direct_shared_to_mfma", "A_fragment_shared_to_mfma", args.seed),
            _equal_modes("B_score_reuse_split_barriers", "B_score_reuse_merged_barrier", args.seed),
        ]
        if not all(bool(row["finite"]) for row in rows):
            raise RuntimeError("Stage 7A barrier repro produced non-finite output")
        if not all(bool(check["bit_exact"]) for check in checks):
            raise AssertionError(f"Stage 7A paired repro mismatch: {checks}")
    payload = {"rows": rows, "paired_checks": checks}
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else payload)


if __name__ == "__main__":
    main()

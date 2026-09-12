"""Minimal Avelang probes for BF16 32x32 MFMA and raw_buffer_store_x4.

Purpose
-------
This file is NOT a full Qwen GDN v29 kernel.  It is a source-level feasibility
probe used before rewriting v28/v29.

It answers two questions:

1. Can Avelang source code explicitly generate BF16 32x32 MFMA?
   Expected ISA mnemonic:
       v_mfma_f32_32x32x8_bf16
   or equivalent 32x32 BF16 form.

2. Can Avelang source code explicitly generate vectorized 16B stores?
   Expected ISA:
       buffer_store_dwordx4
   or global/store dwordx4 equivalent.

Why this exists
---------------
v28 currently calls mfma_16x16x16_bf16_f32 explicitly.  Triton selected
BV=32, num_warps=2 and uses a 32x32 MFMA form.  Before writing a full v29
Qwen GDN kernel, we need a tiny repro that proves Avelang can express the
same primitive and lets us dump ISA/counters.

This file intentionally stores raw per-lane accumulator fragments for the
MFMA32 probe.  It does not yet reconstruct row-major C[32,32], because the
32x32 accumulator lane layout must be verified first.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


# ---------------------------------------------------------------------------
# Probe A: BF16 32x32x8 MFMA
# ---------------------------------------------------------------------------


@avelang.jit
def _mfma32_bf16_fragment_probe_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_frag_ptr: al.Pointer(al.f32),
):
    """Run a small BF16 32x32x64 dot-like MFMA sequence.

    Inputs:
        A:      [32, 64] BF16 row-major
        Bpack:  [32, 64] BF16 row-major

    Output:
        out_frag: [64, 16] FP32
            out_frag[lane, acc_idx] stores the raw accumulator fragment for
            that lane.  This is a layout probe, not a reconstructed C matrix.

    Notes:
        The language reference says mfma_32x32x8_bf16_f32 consumes fragments
        where each lane contributes 4 BF16 from A and 4 BF16 from B and returns
        16 FP32 accumulator values per lane.

        We follow the documented packed view pattern:
            words -> Tensor((2, 4, 1), bf16)
            two MFMA calls cover 8 BF16 values along K.
    """
    A = al.make_tensor(a_ptr, al.bf16, al.make_layout((32, 64), (64, 1)))
    B = al.make_tensor(b_ptr, al.bf16, al.make_layout((32, 64), (64, 1)))
    O = al.make_tensor(out_frag_ptr, al.f32, al.make_layout((64, 16), (16, 1)))

    # 64 BF16 per row = 128 bytes = 32 i32 words.
    # Shape: [row, vec16B, word_in_vec]
    # vec16B count = 8, word_in_vec = 4.
    A_vec = al.view(A, al.i32, al.make_layout((32, 8, 4), (32, 4, 1)))
    B_vec = al.view(B, al.i32, al.make_layout((32, 8, 4), (32, 4, 1)))

    lane = al.thread_id(0)  # 0..63
    lane_col = lane & 15
    lane_group = lane >> 4

    # 32x32 MFMA accumulator: 16 FP32 per lane.
    acc = al.full((16,), 0.0, al.f32)

    # This is a conservative source-level probe.  The exact lane-to-row mapping
    # is intentionally simple and may need to be adjusted after comparing
    # accumulator fragments against torch reference.
    #
    # The goal of this first probe is to force the 32x32 MFMA instruction to
    # appear in ISA, then let a follow-up layout probe derive the exact mapping.
    a_row = lane_col
    b_row = (lane_col + lane_group * 4) & 31

    for kpack in al.range(8):
        a_words = A_vec[a_row, kpack]  # Tensor((4,), i32)
        b_words = B_vec[b_row, kpack]  # Tensor((4,), i32)

        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

        # Two 4-BF16 halves cover K=8.
        #
        # Depending on Avelang's exact wrapper convention, Codex may need to
        # swap operand order.  Keep this version close to the language reference.
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)

    for i in al.range(16):
        O[lane, i] = acc[i]


def run_mfma32_probe(*, warmup: int, repeat: int) -> dict[str, float]:
    device = "cuda"
    torch.manual_seed(0)

    a = torch.randn((32, 64), device=device, dtype=torch.bfloat16)
    b = torch.randn((32, 64), device=device, dtype=torch.bfloat16)
    out = torch.empty((64, 16), device=device, dtype=torch.float32)

    def launch() -> None:
        _mfma32_bf16_fragment_probe_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
            a,
            b,
            out,
            num_warps=1,
        )

    _maybe_dump_hsaco(launch, "_mfma32_bf16_fragment_probe_kernel")
    latency_ms = _benchmark_cuda(launch, warmup=warmup, repeat=repeat)

    # Basic sanity: output should be finite and nonzero for random inputs.
    torch.cuda.synchronize()
    finite = torch.isfinite(out).all().item()
    checksum = out.float().abs().sum().item()
    max_abs = out.float().abs().max().item()

    return {
        "latency_ms": latency_ms,
        "finite": float(finite),
        "checksum": checksum,
        "max_abs": max_abs,
    }


# ---------------------------------------------------------------------------
# Probe B: raw_buffer_store_x4
# ---------------------------------------------------------------------------


@avelang.jit
def _raw_buffer_store_x4_probe_kernel(
    src_ptr: al.Pointer(al.i32),
    dst_ptr: al.Pointer(al.i32),
    num_i32: al.constexpr,
):
    """Copy int32 data using raw_buffer_load_x4 + raw_buffer_store_x4.

    Input/output:
        src: [num_i32] int32
        dst: [num_i32] int32

    Requirements:
        num_i32 must be divisible by 4.

    Expected ISA:
        raw_buffer_load_x4 should lower to dwordx4-style load.
        raw_buffer_store_x4 should lower to buffer_store_dwordx4 or equivalent.
    """
    src = al.make_tensor(src_ptr, al.i32, al.make_layout((num_i32,), (1,)))
    dst = al.make_tensor(dst_ptr, al.i32, al.make_layout((num_i32,), (1,)))

    tid = al.thread_id(0)
    bid = al.block_id(0)
    block_threads = al.block_dim(0)

    vec_idx = bid * block_threads + tid
    word_base = vec_idx * 4

    zero = al.convert(0, al.i32)

    src_rsrc = al.amdgpu.make_rsrc(src, num_i32 * 4)
    dst_rsrc = al.amdgpu.make_rsrc(dst, num_i32 * 4)

    byte_offset = al.convert(word_base * 4, al.i32)

    if word_base + 3 < num_i32:
        words = al.amdgpu.raw_buffer_load_x4(src_rsrc, zero, byte_offset, 0)
        al.amdgpu.raw_buffer_store_x4(words, dst_rsrc, zero, byte_offset, 0)


def run_store_x4_probe(*, n_i32: int, warmup: int, repeat: int) -> dict[str, float]:
    if n_i32 % 4 != 0:
        raise ValueError("n_i32 must be divisible by 4 for x4 stores")

    device = "cuda"
    src = torch.arange(n_i32, device=device, dtype=torch.int32)
    dst = torch.empty_like(src)

    block = 256
    grid = math.ceil((n_i32 // 4) / block)

    def launch() -> None:
        _raw_buffer_store_x4_probe_kernel[lambda: ((grid, 1, 1), (block, 1, 1))](
            src,
            dst,
            n_i32,
            num_warps=4,
        )

    _maybe_dump_hsaco(launch, "_raw_buffer_store_x4_probe_kernel")
    latency_ms = _benchmark_cuda(launch, warmup=warmup, repeat=repeat)
    torch.cuda.synchronize()

    ok = torch.equal(src, dst)
    checksum = dst.long().sum().item()

    return {
        "latency_ms": latency_ms,
        "correct": float(ok),
        "checksum": float(checksum),
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _benchmark_cuda(fn: Callable[[], None], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    return statistics.median(times)


_HSACO_DUMP_DIR: Path | None = None


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_substr: str) -> None:
    if _HSACO_DUMP_DIR is None:
        return

    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if not dumped and kernel_substr in kernel_name:
            path = dump_dir / f"{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped = True
            print(f"dumped_hsaco: {path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile

    if not dumped:
        raise RuntimeError(f"no compiled kernel matched {kernel_substr!r} for hsaco dump")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["mfma32", "store_x4", "all"],
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--n-i32", type=int, default=1 << 20)
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()

    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir

    if args.mode in ("mfma32", "all"):
        print("=== MFMA32 BF16 fragment probe ===")
        res = run_mfma32_probe(warmup=args.warmup, repeat=args.repeat)
        for k, v in res.items():
            print(f"{k}: {v}")

    if args.mode in ("store_x4", "all"):
        print("=== raw_buffer_store_x4 probe ===")
        res = run_store_x4_probe(n_i32=args.n_i32, warmup=args.warmup, repeat=args.repeat)
        for k, v in res.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()

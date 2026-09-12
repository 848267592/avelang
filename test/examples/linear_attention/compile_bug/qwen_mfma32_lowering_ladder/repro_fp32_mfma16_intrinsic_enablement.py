"""Minimal gfx942 JIT/ISA repro for ``mfma_16x16x4_f32_f32``.

The kernel deliberately uses all-one FP32 fragments.  A 16x16x4 MFMA with a
zero accumulator must therefore produce four exactly-four accumulator values
per lane, independent of the hardware accumulator lane layout.  This makes
the repro suitable for validating the newly registered Avelang intrinsic
without introducing solve or Qwen-specific indexing.
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


@avelang.jit
def _fp32_mfma16x4_intrinsic_probe_kernel(
    src_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
):
    src = al.make_tensor(src_ptr, al.f32, al.make_layout((64,), (1,)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((64, 4), (4, 1)))

    lane = al.thread_id(0)
    # A rank-one ``al.full`` is a scalar expression in the current frontend.
    # Give the hardware intrinsic an explicit one-element vector view instead.
    src_fragment = al.view(src, al.f32, al.make_layout((64, 1), (1, 1)))
    a_fragment = src_fragment[lane]
    b_fragment = src_fragment[(lane + 1) & 63]
    acc = al.full((4,), 0.0, al.f32)
    acc = al.amdgpu.mfma_16x16x4_f32_f32(a_fragment, b_fragment, acc)

    for acc_index in al.range(4):
        out[lane, acc_index] = acc[acc_index]


def _median_ms(fn: Callable[[], None], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(begin.elapsed_time(end)))
    return statistics.median(times)


def _dump_hsaco(launch: Callable[[], None], dump_dir: Path | None) -> Path | None:
    if dump_dir is None:
        return None

    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped_path: Path | None = None

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped_path
        binary = original_compile(self, src, target, options)
        if dumped_path is None and "_fp32_mfma16x4_intrinsic_probe_kernel" in src.fn.fn.__name__:
            dumped_path = dump_dir / "fp32_mfma16x4_intrinsic_probe.hsaco"
            dumped_path.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile

    if dumped_path is None:
        raise RuntimeError("failed to capture the FP32 MFMA16x4 code object")
    return dumped_path


def run(*, warmup: int, repeat: int, dump_hsaco_dir: Path | None) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this repro requires a CUDA/HIP device")

    src = torch.ones((64,), device="cuda", dtype=torch.float32)
    out = torch.empty((64, 4), device="cuda", dtype=torch.float32)

    def launch() -> None:
        _fp32_mfma16x4_intrinsic_probe_kernel[lambda: ((1, 1, 1), (64, 1, 1))](src, out)

    hsaco = _dump_hsaco(launch, dump_hsaco_dir)
    latency_ms = _median_ms(launch, warmup=warmup, repeat=repeat)
    torch.cuda.synchronize()

    expected = torch.full_like(out, 4.0)
    error = (out - expected).abs()
    return {
        "correct": bool(torch.equal(out, expected)),
        "finite": bool(torch.isfinite(out).all().item()),
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
        "median_ms": latency_ms,
        "hsaco": None if hsaco is None else str(hsaco),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dump-hsaco-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = run(warmup=args.warmup, repeat=args.repeat, dump_hsaco_dir=args.dump_hsaco_dir)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")


if __name__ == "__main__":
    main()

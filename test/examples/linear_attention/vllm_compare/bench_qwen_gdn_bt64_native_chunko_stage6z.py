#!/usr/bin/env python3
"""Fresh-process isolated body benchmark for the single Stage 6Z Z1 kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w,
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z3_wg128 import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z3,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


IMPLEMENTATIONS = ("stage6w", "z1", "z2", "z3")
KERNELS = {
    "stage6w": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w.fn.__name__,
    "z1": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1.fn.__name__,
    "z2": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2.fn.__name__,
    "z3": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z3.fn.__name__,
}


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, 2026072600 + t, "random", True)
    torch.manual_seed(2026072700 + t)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // 64, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


def _launch(implementation: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    q, k, vn, h, g = tensors
    if implementation == "stage6w":
        return lambda: qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into(q, k, vn, h, g, output)
    if implementation == "z2":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(q, k, vn, h, g, output)
    if implementation == "z3":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z3_launch_into(q, k, vn, h, g, output)
    return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1_launch_into(q, k, vn, h, g, output)


def _sample(fn: Callable[[], None]) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    before = time.perf_counter_ns()
    start.record()
    fn()
    end.record()
    end.synchronize()
    after = time.perf_counter_ns()
    return float(start.elapsed_time(end)), (after - before) / 1.0e6


def _capture_hsaco(fn: Callable[[], None], kernel: str, path: Path) -> str:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    path.parent.mkdir(parents=True, exist_ok=True)
    original = amdgpu_compiler.AmdgpuCompiler.compile
    captured = False

    def wrapped(self, src, target, options=None):
        nonlocal captured
        binary = original(self, src, target, options)
        if src.fn.fn.__name__ == kernel and not captured:
            path.write_bytes(binary)
            captured = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
    if not captured:
        raise RuntimeError(f"did not capture {kernel}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--dump-hsaco", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be >=64 and divisible by 64")
    tensors = _inputs(args.T)
    output = torch.empty_like(tensors[2])
    fn = _launch(args.implementation, tensors, output)
    if args.dump_hsaco is not None:
        _capture_hsaco(fn, KERNELS[args.implementation], args.dump_hsaco)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    hip_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(args.repeat):
        hip, wall = _sample(fn)
        hip_ms.append(hip)
        wall_ms.append(wall)
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("non-finite chunk-o output")
    result = {
        "scope": "caller_owned_isolated_body_diagnostic",
        "T": args.T,
        "chunks": args.T // 64,
        "implementation": args.implementation,
        "kernel": KERNELS[args.implementation],
        "cta": (args.T // 64) * 8 * (8 if args.implementation == "stage6w" else 2),
        "workgroup": 128 if args.implementation == "z3" else 256,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "hip_ms": {"median": statistics.median(hip_ms), "p10": sorted(hip_ms)[max(0, int(len(hip_ms) * 0.1) - 1)], "p90": sorted(hip_ms)[min(len(hip_ms) - 1, int(len(hip_ms) * 0.9))]},
        "wall_ms": {"median": statistics.median(wall_ms), "p10": sorted(wall_ms)[max(0, int(len(wall_ms) * 0.1) - 1)], "p90": sorted(wall_ms)[min(len(wall_ms) - 1, int(len(wall_ms) * 0.9))]},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()

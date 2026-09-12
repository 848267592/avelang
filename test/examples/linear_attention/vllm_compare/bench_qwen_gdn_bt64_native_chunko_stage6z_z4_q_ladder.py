#!/usr/bin/env python3
"""Fresh-process caller-owned body benchmark for one Z4 Q-ladder arm."""

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

from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into,
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into,
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


ARMS = {
    "z2": qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
    "z4a": qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into,
    "z4b": qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into,
    "z4c": qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into,
}
KERNELS = {
    "z2": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2.fn.__name__,
    "z4a": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q.fn.__name__,
    "z4b": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency.fn.__name__,
    "z4c": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion.fn.__name__,
}


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, 2026080800 + t, "random", True)
    torch.manual_seed(2026080900 + t)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // 64, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


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
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--dump-hsaco", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be >=64 and divisible by 64")

    q, k, vn, h, g = _inputs(args.T)
    output = torch.empty_like(vn)
    launch = ARMS[args.arm]
    fn = lambda: launch(q, k, vn, h, g, output)
    if args.dump_hsaco is not None:
        _capture_hsaco(fn, KERNELS[args.arm], args.dump_hsaco)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    hip_ms = []
    wall_ms = []
    for _ in range(args.repeat):
        hip, wall = _sample(fn)
        hip_ms.append(hip)
        wall_ms.append(wall)
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("non-finite output")
    result = {
        "scope": "caller_owned_isolated_body_diagnostic",
        "arm": args.arm,
        "kernel": KERNELS[args.arm],
        "T": args.T,
        "chunks": args.T // 64,
        "workgroup": 256,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "hip_ms": {"median": statistics.median(hip_ms), "p10": min(hip_ms), "p90": max(hip_ms)},
        "wall_ms": {"median": statistics.median(wall_ms), "p10": min(wall_ms), "p90": max(wall_ms)},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()

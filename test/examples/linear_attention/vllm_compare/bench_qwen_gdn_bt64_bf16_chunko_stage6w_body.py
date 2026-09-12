#!/usr/bin/env python3
"""Isolated current-versus-Stage-6W BT64 chunk-o body benchmark and HSACO capture."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w,
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w,
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0,
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
)
from stage2_runner import make_inputs  # noqa: E402


def _inputs(t: int):
    q, k, _, g, _, _ = make_inputs(t, 2026074500 + t, "random", True)
    torch.manual_seed(2026074600 + t)
    v_new_bf16 = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h_bf16 = (torch.randn((1, t // 64, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, v_new_bf16, h_bf16, g


def _median_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return statistics.median(values)


def _capture_hsaco(launch: Callable[[], object], kernel_name: str, destination: Path) -> str:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    destination.parent.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    captured: list[Path] = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if src.fn.fn.__name__ == kernel_name and not captured:
            destination.write_bytes(binary)
            captured.append(destination)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not captured:
        raise RuntimeError(f"did not capture HSACO for {kernel_name}")
    return str(captured[0])


def _run_t(t: int, warmup: int, repeat: int, dump_dir: Path | None, implementation: str) -> dict[str, object]:
    q, k, v_new_bf16, h_bf16, g = _inputs(t)
    num_chunks = t // 64
    v_new_fp32 = v_new_bf16.float()
    current_out = torch.empty_like(v_new_fp32)
    stage6w_out = torch.empty_like(v_new_bf16)
    current = lambda: _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0[
        lambda: ((num_chunks * 8 * 8, 1, 1), (256, 1, 1))
    ](q, k, v_new_fp32, h_bf16, g, current_out, 128 ** -0.5, t, num_chunks)
    stage6w = lambda: qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into(
        q, k, v_new_bf16, h_bf16, g, stage6w_out
    )
    result: dict[str, object] = {"T": t, "chunks": t // 64}
    # Capture before Avelang can reuse a cached module in this process.
    if dump_dir is not None and t == 2048:
        result["current_hsaco"] = _capture_hsaco(
            current,
            _qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0.fn.__name__,
            dump_dir / "stage6u_current_chunk_o.hsaco",
        )
        result["stage6w_hsaco"] = _capture_hsaco(
            stage6w,
            _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w.fn.__name__,
            dump_dir / "stage6w_bf16_chunk_o.hsaco",
        )
    current()
    stage6w()
    torch.cuda.synchronize()
    expected = current_out.to(torch.bfloat16)
    actual = stage6w_out
    result.update({
        "current_kernel_ms": _median_ms(current, warmup, repeat) if implementation in ("both", "current") else None,
        "stage6w_kernel_ms": _median_ms(stage6w, warmup, repeat) if implementation in ("both", "stage6w") else None,
        "output_bf16_bit_exact": bool(torch.equal(expected, actual)),
        "output_bf16_mismatch": int((expected.view(torch.int16) != actual.view(torch.int16)).sum().item()),
        "output_bf16_max_abs": float((expected.float() - actual.float()).abs().max().item()),
    })
    result["speedup"] = (
        float(result["current_kernel_ms"]) / float(result["stage6w_kernel_ms"])
        if implementation == "both" else None
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--implementation", choices=("both", "current", "stage6w"), default="both")
    parser.add_argument("--dump-hsaco-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires the gfx942 HIP runtime")
    rows = [_run_t(t, args.warmup, args.repeat, args.dump_hsaco_dir, args.implementation) for t in args.T]
    payload = json.dumps(rows, indent=2, sort_keys=True)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Preallocated body benchmarks for Stage 6X KKT and KKT-to-solve phases."""

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

from qwen_gdn_bt64_kkt_solve_handoff_stage6x import (  # noqa: E402
    _qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1,
    _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2,
    _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into,
    qwen_gdn_kkt_bt64_one_cta_stage6x_x1,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    H_V,
    _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (  # noqa: E402
    _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into,
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
)


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


def _inputs(t: int):
    _, k, _, g, beta, _ = make_inputs(t, 2026072500 + t, "random", True)
    return k, qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64), beta


def _run_t(t: int, warmup: int, repeat: int, dump_hsaco_dir: Path | None) -> dict[str, object]:
    k, g_cumsum, beta = _inputs(t)
    chunks = t // 64
    current_a = torch.empty((1, t, H_V, 64), dtype=torch.float32, device=k.device)
    current_solved = torch.empty((1, t, H_V, 64), dtype=torch.bfloat16, device=k.device)
    x2_solved = torch.empty_like(current_solved)

    current = lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    x1 = lambda: qwen_gdn_kkt_bt64_one_cta_stage6x_x1(k, g_cumsum, beta)
    current_chain = lambda: (
        _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0[
            lambda: ((chunks * H_V * 16, 1, 1), (64, 1, 1))
        ](k, g_cumsum, beta, current_a, t, chunks),
        _qwen_gdn_solve_hierarchical_bt64_bf16_stage6u_launch_into(current_a, current_solved),
    )
    x2_chain = lambda: _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into(
        k, g_cumsum, beta, x2_solved
    )
    result: dict[str, object] = {"T": t, "chunks": chunks}
    if dump_hsaco_dir is not None and t == 2048:
        result["current_hsaco"] = _capture_hsaco(
            current, _qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0.fn.__name__, dump_hsaco_dir / "current_kkt.hsaco"
        )
        result["x1_hsaco"] = _capture_hsaco(
            x1, _qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1.fn.__name__, dump_hsaco_dir / "stage6x_x1_kkt.hsaco"
        )
        result["x2_hsaco"] = _capture_hsaco(
            x2_chain,
            _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2.fn.__name__,
            dump_hsaco_dir / "stage6x_x2_kkt_solve.hsaco",
        )
    expected = current()
    actual = x1()
    expected_solved = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(expected)
    actual_solved = x2_chain()
    torch.cuda.synchronize()
    result.update(
        current_kkt_ms=_median_ms(current, warmup, repeat),
        x1_kkt_ms=_median_ms(x1, warmup, repeat),
        bit_exact=bool(torch.equal(actual, expected)),
        max_abs=float((actual - expected).abs().max().item()),
        mean_abs=float((actual - expected).abs().mean().item()),
        current_kkt_solve_ms=_median_ms(current_chain, warmup, repeat),
        x2_kkt_solve_ms=_median_ms(x2_chain, warmup, repeat),
        x2_bit_exact=bool(torch.equal(actual_solved, expected_solved)),
        x2_max_abs=float((actual_solved.float() - expected_solved.float()).abs().max().item()),
        x2_mean_abs=float((actual_solved.float() - expected_solved.float()).abs().mean().item()),
    )
    result["speedup_current_over_x1"] = float(result["current_kkt_ms"]) / float(result["x1_kkt_ms"])
    result["speedup_current_chain_over_x2"] = float(result["current_kkt_solve_ms"]) / float(result["x2_kkt_solve_ms"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 1024, 2048, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dump-hsaco-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires the gfx942 HIP runtime")
    rows = [_run_t(t, args.warmup, args.repeat, args.dump_hsaco_dir) for t in args.T]
    payload = json.dumps(rows, indent=2, sort_keys=True)
    print(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")


if __name__ == "__main__":
    main()

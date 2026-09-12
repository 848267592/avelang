#!/usr/bin/env python3
"""Isolated Stage 6V V0 C0-body benchmark and HSACO capture."""

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

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    _qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u,
    qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u,
)
from qwen_gdn_bt64_predicate_collapse_stage6v import (  # noqa: E402
    _qwen_gdn_wu_kernel_bt64_predicate_collapse_v0,
    qwen_gdn_w_u_bt64_predicate_collapse_v0,
)
from stage2_runner import make_inputs  # noqa: E402


def _inputs(t: int):
    _, k, v, g, beta, _ = make_inputs(t, 2026073200 + t, "random", True)
    torch.manual_seed(2026073300 + t)
    a_solved = (torch.randn((1, t, 8, 64), device=k.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    return k, v, g, beta, a_solved


def _median_ms(fn: Callable[[], object], warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


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


def _run_t(t: int, warmup: int, repeat: int, dump_dir: Path | None) -> dict[str, object]:
    inputs = _inputs(t)
    baseline = lambda: qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(*inputs)
    collapsed = lambda: qwen_gdn_w_u_bt64_predicate_collapse_v0(*inputs)
    result: dict[str, object] = {"T": t, "chunks": t // 64}
    # Capture before either launch can populate Avelang's in-process cache.
    if dump_dir is not None and t == 2048:
        result["baseline_hsaco"] = _capture_hsaco(
            baseline,
            _qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u.fn.__name__,
            dump_dir / "stage6u_c0.hsaco",
        )
        result["v0_hsaco"] = _capture_hsaco(
            collapsed,
            _qwen_gdn_wu_kernel_bt64_predicate_collapse_v0.fn.__name__,
            dump_dir / "stage6v_v0.hsaco",
        )
    baseline_w, baseline_u = baseline()
    collapsed_w, collapsed_u = collapsed()
    torch.cuda.synchronize()
    baseline_ms = _median_ms(baseline, warmup, repeat)
    collapsed_ms = _median_ms(collapsed, warmup, repeat)
    result.update({
        "baseline_c0_ms": baseline_ms,
        "predicate_collapse_v0_ms": collapsed_ms,
        "speedup": baseline_ms / collapsed_ms,
        "w_bit_exact": bool(torch.equal(baseline_w, collapsed_w)),
        "u_bit_exact": bool(torch.equal(baseline_u, collapsed_u)),
        "w_bf16_bit_mismatch": int((baseline_w.view(torch.int16) != collapsed_w.view(torch.int16)).sum().item()),
        "u_bf16_bit_mismatch": int((baseline_u.view(torch.int16) != collapsed_u.view(torch.int16)).sum().item()),
        "w_max_abs": float((baseline_w.float() - collapsed_w.float()).abs().max().item()),
        "u_max_abs": float((baseline_u.float() - collapsed_u.float()).abs().max().item()),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--dump-hsaco-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires the gfx942 HIP runtime")
    rows = [_run_t(t, args.warmup, args.repeat, args.dump_hsaco_dir) for t in args.T]
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

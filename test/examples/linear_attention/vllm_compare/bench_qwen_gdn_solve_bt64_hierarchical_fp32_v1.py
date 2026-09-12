#!/usr/bin/env python3
"""Solve-only benchmark for the opt-in BT64 FP32 hierarchical S0 kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import (
    _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1,
    qwen_gdn_solve_bt64_hierarchical_fp32_v1,
)


BT = 64


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _make_a(t: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    k = _l2norm(torch.randn((1, t, 4, 128), device="cuda", dtype=torch.bfloat16))
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, 8), device="cuda")) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn((1, t, 8), device="cuda")).contiguous()
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=BT, prefer_optimized=True)


def _median_ms(fn: Callable[[], torch.Tensor], warmup: int, repeat: int) -> float:
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


def _residual_inf(a: torch.Tensor, solved: torch.Tensor) -> float:
    chunks = a.shape[1] // BT
    identity = torch.eye(BT, device=a.device, dtype=torch.float32)
    worst = torch.zeros((), device=a.device, dtype=torch.float32)
    for chunk_idx in range(chunks):
        start = chunk_idx * BT
        lhs = a[0, start : start + BT].permute(1, 0, 2).float() + identity
        rhs = solved[0, start : start + BT].permute(1, 0, 2).float()
        worst = torch.maximum(worst, ((lhs @ rhs) - identity).abs().max())
    return float(worst.item())


def _dump_hsaco(a: torch.Tensor, dump_dir: Path | None) -> str | None:
    if dump_dir is None:
        return None
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped: Path | None = None

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        if dumped is None and src.fn.fn.__name__ == _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1.fn.__name__:
            dumped = dump_dir / "qwen_bt64_hierarchical_fp32_s0.hsaco"
            dumped.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if dumped is None:
        raise RuntimeError("failed to capture S0 HSACO")
    return str(dumped)


def _run_t(t: int, warmup: int, repeat: int, dump_hsaco_dir: Path | None) -> dict[str, object]:
    a = _make_a(t, seed=20260900 + t)
    hsaco = _dump_hsaco(a, dump_hsaco_dir) if dump_hsaco_dir is not None else None
    candidate = qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    v18 = qwen_gdn_solve_avelang_v18_bt64_layout(a)
    v6 = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=BT)
    torch.cuda.synchronize()
    error = (candidate - v6).abs()
    v1_ms = _median_ms(lambda: qwen_gdn_solve_bt64_hierarchical_fp32_v1(a), warmup, repeat)
    v18_ms = _median_ms(lambda: qwen_gdn_solve_avelang_v18_bt64_layout(a), warmup, repeat)
    v6_ms = _median_ms(lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=BT), warmup, repeat)
    return {
        "T": t,
        "v1_ms": v1_ms,
        "v18_ms": v18_ms,
        "v6_ms": v6_ms,
        "speedup_vs_v18": v18_ms / v1_ms,
        "speedup_vs_v6": v6_ms / v1_ms,
        "max_abs_vs_v6": float(error.max().item()),
        "mean_abs_vs_v6": float(error.mean().item()),
        "max_abs_vs_v18": float((candidate - v18).abs().max().item()),
        "residual_inf": _residual_inf(a, candidate),
        "hsaco": hsaco,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--dump-hsaco-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    results = [_run_t(t, args.warmup, args.repeat, args.dump_hsaco_dir) for t in args.T]
    for row in results:
        print(
            "solve_bt64_hierarchical,"
            + ",".join(f"{key}={value}" for key, value in row.items() if key != "hsaco")
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

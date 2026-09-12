#!/usr/bin/env python3
"""Benchmark the fixed BT64 recurrence artifact without module load/allocation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
FULLSEQ_AUDIT = HERE.parent / "codex_triton_fullseq_asm_opt_audit"
VLLM_COMPARE = HERE.parents[4] / "vllm_compare"
sys.path.insert(0, str(FULLSEQ_AUDIT))
sys.path.insert(0, str(VLLM_COMPARE))

from capture_long_sequence import make_inputs, patch_rocm_autotune  # noqa: E402
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32,
    qwen_gdn_gdr_decay_bt64_reference as v29_decay,
)
from qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred,
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL_HSACO = FULLSEQ_AUDIT / "golden_fullseq/shared/original.hsaco"
REBUILT_HSACO = FULLSEQ_AUDIT / "golden_fullseq/shared/rebuilt.hsaco"
ASM_V0_HSACO = HERE / "assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco"
HARNESS = HERE / "asm_v0_harness"
ROUND_ROBIN_HARNESS = HERE / "round_robin_harness"
PREPARE = FULLSEQ_AUDIT / "standalone_fullseq/prepare_case.py"
TRITON_SYMBOL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
ASM_V0_SYMBOL = "qwen_gdn_bt64_gfx942_asm_v0"


def _time(fn: Callable[[], None], warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    values.sort()
    return {
        "median_ms": float(statistics.median(values)),
        "p10_ms": values[(len(values) - 1) // 10],
        "p90_ms": values[(len(values) - 1) * 9 // 10],
    }


def _prepare_harness_input(t: int) -> Path:
    input_dir = HERE / "benchmark_inputs" / f"t{t}"
    if not (input_dir / "k_bf16.bin").is_file():
        input_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(PREPARE), "--T", str(t), "--out", str(input_dir)], check=True)
    return input_dir


def _time_module_harness(t: int, hsaco: Path, symbol: str, warmup: int, repeat: int, session: int) -> dict[str, float]:
    if not HARNESS.is_file():
        subprocess.run(["sh", str(HERE / "build_harness.sh")], check=True)
    input_dir = _prepare_harness_input(t)
    output_dir = HERE / "benchmark_outputs" / f"t{t}_{symbol}_{session}"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(HARNESS), str(hsaco), symbol, str(t), str(input_dir), str(output_dir), str(warmup), str(repeat)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return {key: float(payload[key]) for key in ("median_ms", "p10_ms", "p90_ms")}


def _time_round_robin(t: int, warmup: int, repeat: int) -> list[dict[str, object]]:
    if not ROUND_ROBIN_HARNESS.is_file():
        subprocess.run(["sh", str(HERE / "build_round_robin_harness.sh")], check=True)
    input_dir = _prepare_harness_input(t)
    result = subprocess.run(
        [
            str(ROUND_ROBIN_HARNESS),
            str(ORIGINAL_HSACO),
            str(REBUILT_HSACO),
            str(ASM_V0_HSACO),
            str(t),
            str(input_dir),
            str(warmup),
            str(repeat),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])["rows"]


def _context_fns(values: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], None]]:
    k, w, u, g, h0 = values
    decay, last = v29_decay(g)
    return {
        "old_full_v29_context": lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32(k, w, u, decay, last, h0),
        "v31_context": lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred(k, w, u, decay, last, h0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for t in args.T:
        values = make_inputs(t, 20263000 + t)
        context = _context_fns(values)
        for session in range(args.sessions):
            try:
                for timing in _time_round_robin(t, args.warmup, args.repeat):
                    name = str(timing.pop("implementation"))
                    rows.append({
                        "T": t,
                        "implementation": name,
                        "session": session,
                        "comparison_class": "strict_same_raw_contract",
                        "status": "ok",
                        **timing,
                    })
                    print(f"T={t} impl={name} session={session} median_ms={timing['median_ms']:.6f}")
            except Exception as error:
                for name in ("golden_triton_original_hsaco", "golden_triton_rebuilt_hsaco", "avelang_asm_v0"):
                    rows.append({"T": t, "implementation": name, "session": session, "comparison_class": "strict_same_raw_contract", "status": f"error: {error}"})
                    print(f"T={t} impl={name} session={session} error={error}")
        for name, fn in context.items():
            for session in range(args.sessions):
                try:
                    timing = _time(fn, args.warmup, args.repeat)
                    rows.append({
                        "T": t,
                        "implementation": name,
                        "session": session,
                        "comparison_class": "context_different_visible_outputs",
                        "status": "ok",
                        **timing,
                    })
                    print(f"T={t} impl={name} session={session} median_ms={timing['median_ms']:.6f}")
                except Exception as error:
                    rows.append({"T": t, "implementation": name, "session": session, "comparison_class": "context_different_visible_outputs", "status": f"error: {error}"})
                    print(f"T={t} impl={name} session={session} error={error}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage_benchmark.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = ["T", "implementation", "session", "comparison_class", "status", "median_ms", "p10_ms", "p90_ms"]
    with (args.out / "stage_benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Primary eager-public-API leaderboard for v24, Stage 6S, and vLLM.

This intentionally does not use CUDA/HIP Graph replay. Each timed sample calls
the public full wrapper so launch count, materialized intermediates, and the
cached allocator behaviour of each implementation stay inside the same HIP
event interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import (  # noqa: E402
    qwen_gdn_chunked_avelang_v24_kkt_mfma_layout,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT = 64
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
IMPLEMENTATIONS = ("v24_bt16", "stage6s_bt64", "vllm_native")


def event_ms(fn: Callable[[], object], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))]


def error(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    delta = (lhs.float() - rhs.float()).abs()
    return float(delta.max().item()), float(delta.mean().item())


def calls(inputs: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]]:
    q, k, v, g, beta, h0 = inputs
    return {
        "v24_bt16": lambda: qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(
            q, k, v, g, beta, initial_state=h0, scale=128 ** -0.5
        ),
        "stage6s_bt64": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(
            q, k, v, g, beta, initial_state=h0, scale=128 ** -0.5, output_final_state=True
        ),
        "vllm_native": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def paired_order(repeat: int) -> list[tuple[str, str]]:
    sequence: list[tuple[str, str]] = []
    for _ in range(repeat):
        sequence.extend((
            ("ABBA", "v24_bt16"), ("ABBA", "stage6s_bt64"),
            ("ABBA", "stage6s_bt64"), ("ABBA", "v24_bt16"),
            ("BCCB", "stage6s_bt64"), ("BCCB", "vllm_native"),
            ("BCCB", "vllm_native"), ("BCCB", "stage6s_bt64"),
            ("ACCA", "v24_bt16"), ("ACCA", "vllm_native"),
            ("ACCA", "vllm_native"), ("ACCA", "v24_bt16"),
        ))
    return sequence


def correctness_for_t(t: int) -> dict[str, object]:
    inputs = make_inputs(t, 2026072300 + t, "random", True)
    functions = calls(inputs)
    results = {name: fn() for name, fn in functions.items()}
    torch.cuda.synchronize()
    reference_out, reference_state = results["vllm_native"]
    if reference_state is None:
        raise AssertionError("vLLM did not return final_state")
    row: dict[str, object] = {"T": t, "reference": "vllm_native"}
    for name in ("v24_bt16", "stage6s_bt64"):
        output, final_state = results[name]
        if final_state is None:
            raise AssertionError(f"{name} did not return final_state")
        output_max, output_mean = error(output, reference_out)
        state_max, state_mean = error(final_state, reference_state)
        accepted = output_max <= OUTPUT_ATOL and state_max <= STATE_ATOL
        row.update({
            f"{name}_output_max_abs": output_max,
            f"{name}_output_mean_abs": output_mean,
            f"{name}_state_max_abs": state_max,
            f"{name}_state_mean_abs": state_mean,
            f"{name}_accepted": accepted,
        })
        if not accepted:
            raise AssertionError(
                f"{name} correctness failed at T={t}: output={output_max}, state={state_max}"
            )
    return row


def benchmark_for_t(t: int, args: argparse.Namespace) -> list[dict[str, object]]:
    inputs = make_inputs(t, 2026072300 + t, "random", True)
    functions = calls(inputs)
    # Module load, JIT compilation, autotune and allocator warmup are outside
    # timing. Public wrappers retain their ordinary cached-allocator behaviour.
    for fn in functions.values():
        fn()
    torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    for session in range(args.sessions):
        for _ in range(args.warmup):
            for name in IMPLEMENTATIONS:
                functions[name]()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for sequence, (order, name) in enumerate(paired_order(args.repeat)):
            rows.append({
                "T": t,
                "chunks": t // BT,
                "session": session,
                "order": order,
                "sequence": sequence,
                "implementation": name,
                "latency_ms": event_ms(functions[name], start, end),
            })
    return rows


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[float]] = {}
    for row in raw:
        grouped.setdefault((int(row["T"]), int(row["session"]), str(row["implementation"])), []).append(float(row["latency_ms"]))
    per_t: dict[tuple[int, str], list[float]] = {}
    result: list[dict[str, object]] = []
    for (t, session, name), values in sorted(grouped.items()):
        median = statistics.median(values)
        per_t.setdefault((t, name), []).append(median)
        result.append({
            "T": t, "chunks": t // BT, "session": session, "implementation": name,
            "median_ms": median, "p10_ms": quantile(values, 0.1), "p90_ms": quantile(values, 0.9),
            "samples": len(values),
        })
    for (t, name), values in sorted(per_t.items()):
        result.append({
            "T": t, "chunks": t // BT, "session": "aggregate", "implementation": name,
            "median_ms": statistics.median(values), "p10_ms": min(values), "p90_ms": max(values),
            "samples": len(values),
        })
    return result


def ratio_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    aggregate = [row for row in summary if row["session"] == "aggregate"]
    rows: list[dict[str, object]] = []
    for t in sorted({int(row["T"]) for row in aggregate}):
        values = {str(row["implementation"]): float(row["median_ms"]) for row in aggregate if int(row["T"]) == t}
        vllm_ms = values["vllm_native"]
        for name in ("v24_bt16", "stage6s_bt64"):
            latency = values[name]
            rows.append({
                "T": t, "chunks": t // BT, "implementation": name,
                "avelang_ms": latency, "vllm_eager_ms": vllm_ms,
                "avelang_over_vllm_x": latency / vllm_ms,
                "slowdown_percent": (latency / vllm_ms - 1.0) * 100.0,
                "gap_us": (latency - vllm_ms) * 1000.0,
            })
    return rows


def fit_slope(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name in ("v24_bt16", "stage6s_bt64"):
        points = [row for row in rows if row["implementation"] == name]
        if len(points) < 2:
            result.append({"implementation": name, "ratio_intercept": math.nan,
                           "ratio_slope_per_chunk": math.nan, "points": len(points)})
            continue
        xs = [float(row["chunks"]) for row in points]
        ys = [float(row["avelang_over_vllm_x"]) for row in points]
        mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum((x - mean_x) ** 2 for x in xs)
        result.append({"implementation": name, "ratio_intercept": mean_y - slope * mean_x,
                       "ratio_slope_per_chunk": slope, "points": len(points)})
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=HERE / "eager_public_leaderboard")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    if any(t % BT for t in args.T):
        raise ValueError("all T values must be divisible by 64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    patch_rocm_autotune()

    correctness = [correctness_for_t(t) for t in args.T]
    raw: list[dict[str, object]] = []
    for t in args.T:
        print(f"benchmark eager public API T={t}", flush=True)
        raw.extend(benchmark_for_t(t, args))
    summary = summarize(raw)
    ratios = ratio_rows(summary)
    write_csv(args.out_dir / "correctness.csv", correctness)
    write_csv(args.out_dir / "raw.csv", raw)
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "ratios.csv", ratios)
    write_csv(args.out_dir / "ratio_slopes.csv", fit_slope(ratios))
    (args.out_dir / "measurement_contract.json").write_text(json.dumps({
        "timing_scope": "direct eager public API; no CUDA/HIP Graph replay",
        "implementations": list(IMPLEMENTATIONS), "T": args.T, "warmup": args.warmup,
        "repeat": args.repeat, "sessions": args.sessions,
        "orders": ["ABBA", "BCCB", "ACCA"],
        "outputs_and_intermediates": "public wrappers use their existing cached allocator behaviour after untimed warmup",
        "correctness_reference": "vllm_native",
        "output_atol": OUTPUT_ATOL, "state_atol": STATE_ATOL,
    }, indent=2) + "\n")
    print(json.dumps(ratios, indent=2), flush=True)


if __name__ == "__main__":
    main()

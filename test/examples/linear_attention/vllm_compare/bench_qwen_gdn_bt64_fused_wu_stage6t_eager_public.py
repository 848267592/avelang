#!/usr/bin/env python3
"""Stage 6T primary eager public-API full benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
DEFAULT_OUT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_fused_wu_eager_stage6t"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_fused_wu_eager_stage6t import (  # noqa: E402
    qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager,
    qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT = 64
NAMES = ("stage6s", "f0", "f1", "vllm")


def public_calls(inputs: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor | None]]]:
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f0": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def measure_one(fn: Callable[[], object], start: torch.cuda.Event, end: torch.cuda.Event) -> tuple[float, float]:
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1.0e6
    event_ms = float(start.elapsed_time(end))
    # Keep the public API result alive through completion of its complete work.
    if result is None:
        raise AssertionError("public API returned no result")
    return event_ms, wall_ms


def timed_order(repeat_index: int, rng: random.Random) -> tuple[str, list[str]]:
    if repeat_index % 3 == 0:
        return "S-F0-F1-V", ["stage6s", "f0", "f1", "vllm", "vllm", "f1", "f0", "stage6s"]
    if repeat_index % 3 == 1:
        return "V-F1-F0-S", ["vllm", "f1", "f0", "stage6s", "stage6s", "f0", "f1", "vllm"]
    shuffled = list(NAMES)
    rng.shuffle(shuffled)
    return "random-balanced", shuffled + list(reversed(shuffled))


def qtile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))]


def bootstrap_delta_ci(lhs: list[float], rhs: list[float], seed: int = 20260717) -> tuple[float, float]:
    if len(lhs) != len(rhs) or not lhs:
        return float("nan"), float("nan")
    deltas = [left - right for left, right in zip(lhs, rhs)]
    rng = random.Random(seed)
    values = []
    for _ in range(4000):
        values.append(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    return qtile(values, 0.025), qtile(values, 0.975)


def fit_slope(rows: list[dict[str, object]], name: str) -> tuple[float, float]:
    points = [(float(row["chunks"]), float(row["event_median_ms"])) for row in rows if row["implementation"] == name]
    count = len(points)
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    if count == 1:
        return mean_y, float("nan")
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    return mean_y - slope * mean_x, slope * 1000.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_t(t: int, sessions: int, warmup: int, repeat: int) -> list[dict[str, object]]:
    inputs = make_inputs(t, 2026071700 + t, "random", True)
    calls = public_calls(inputs)
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    raw: list[dict[str, object]] = []
    for session in range(sessions):
        for _ in range(warmup):
            for name in NAMES:
                calls[name]()
        torch.cuda.synchronize()
        rng = random.Random(202607170000 + t * 10 + session)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        sequence = 0
        for repeat_index in range(repeat):
            order, names = timed_order(repeat_index, rng)
            for name in names:
                event_ms, wall_ms = measure_one(calls[name], start, end)
                raw.append({
                    "timing_contract": "eager_public_api", "cuda_graph_used": False,
                    "T": t, "chunks": t // BT, "session": session, "repeat": repeat_index,
                    "sequence": sequence, "order": order, "implementation": name,
                    "event_ms": event_ms, "wall_ms": wall_ms,
                })
                sequence += 1
    return raw


def summarize(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for row in raw:
        grouped.setdefault((int(row["T"]), int(row["session"]), str(row["implementation"])), []).append(row)
    output: list[dict[str, object]] = []
    aggregates: dict[tuple[int, str], list[dict[str, float]]] = {}
    for (t, session, name), rows in sorted(grouped.items()):
        event_values = [float(row["event_ms"]) for row in rows]
        wall_values = [float(row["wall_ms"]) for row in rows]
        item = {
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "chunks": t // BT, "session": session, "implementation": name,
            "samples": len(rows), "event_median_ms": statistics.median(event_values),
            "event_p10_ms": qtile(event_values, 0.1), "event_p90_ms": qtile(event_values, 0.9),
            "wall_median_ms": statistics.median(wall_values), "wall_p10_ms": qtile(wall_values, 0.1),
            "wall_p90_ms": qtile(wall_values, 0.9),
        }
        output.append(item)
        aggregates.setdefault((t, name), []).append({"event": item["event_median_ms"], "wall": item["wall_median_ms"]})
    for (t, name), values in sorted(aggregates.items()):
        event_values = [float(value["event"]) for value in values]
        wall_values = [float(value["wall"]) for value in values]
        output.append({
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "chunks": t // BT, "session": "aggregate", "implementation": name,
            "samples": len(event_values), "event_median_ms": statistics.median(event_values),
            "event_p10_ms": qtile(event_values, 0.1), "event_p90_ms": qtile(event_values, 0.9),
            "wall_median_ms": statistics.median(wall_values), "wall_p10_ms": qtile(wall_values, 0.1),
            "wall_p90_ms": qtile(wall_values, 0.9),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.sessions < 5 or args.warmup < 30 or args.repeat < 200:
        raise ValueError("authoritative eager runs require sessions>=5, warmup>=30, repeat>=200")
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = [row for t in args.T for row in run_t(t, args.sessions, args.warmup, args.repeat)]
    summary = summarize(raw)
    aggregate = [row for row in summary if row["session"] == "aggregate"]
    slopes = []
    for name in NAMES:
        intercept, slope = fit_slope(aggregate, name)
        slopes.append({
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "implementation": name, "intercept_ms": intercept, "slope_us_per_chunk": slope,
        })
    slope_by_name = {str(row["implementation"]): float(row["slope_us_per_chunk"]) for row in slopes}
    gap_slopes = [
        {
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "implementation": name, "vs": "vllm",
            "gap_slope_us_per_chunk": slope_by_name[name] - slope_by_name["vllm"],
        }
        for name in ("stage6s", "f0", "f1")
    ]
    write_csv(args.out_dir / "eager_full_raw.csv", raw)
    write_csv(args.out_dir / "eager_full_summary.csv", summary)
    write_csv(args.out_dir / "eager_full_slopes.csv", slopes)
    write_csv(args.out_dir / "eager_full_gap_slopes.csv", gap_slopes)
    wall = [row for row in summary if row["session"] == "aggregate"]
    write_csv(args.out_dir / "eager_wallclock_summary.csv", wall)
    metadata = {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "sessions": args.sessions, "warmup": args.warmup, "repeat": args.repeat,
        "device": torch.cuda.get_device_name(), "T": args.T,
        "api_names": list(NAMES), "paired_bootstrap_sessions": args.sessions,
    }
    (args.out_dir / "eager_benchmark_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"summary": [row for row in aggregate], "slopes": slopes, "gap_slopes": gap_slopes}, indent=2))


if __name__ == "__main__":
    main()

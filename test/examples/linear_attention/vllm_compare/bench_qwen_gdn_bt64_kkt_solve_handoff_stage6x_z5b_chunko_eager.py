#!/usr/bin/env python3
"""Paired Eager public-API benchmark for X2 and its Z5B chunk-o replacement."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_kkt_solve_handoff_stage6x_full import qwen_gdn_full_bt64_stage6x_kkt_solve_eager  # noqa: E402
from qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full import (  # noqa: E402
    qwen_gdn_full_bt64_stage6x_z5b_chunko_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


NAMES = ("x2", "x2_z5b", "vllm")
WILLIAMS_ORDERS = (
    ("x2", "x2_z5b", "vllm"),
    ("x2_z5b", "vllm", "x2"),
    ("vllm", "x2", "x2_z5b"),
    ("x2", "vllm", "x2_z5b"),
    ("vllm", "x2_z5b", "x2"),
    ("x2_z5b", "x2", "vllm"),
)


def calls_for_t(t: int):
    q, k, v, g, beta, h0 = make_inputs(t, 2026083100 + t, "random", True)
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "x2": lambda: qwen_gdn_full_bt64_stage6x_kkt_solve_eager(q, k, v, g, beta, **common),
        "x2_z5b": lambda: qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def measure(fn) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    if value is None:
        raise AssertionError("public API returned None")
    return float(start.elapsed_time(end)), (time.perf_counter_ns() - wall_start) / 1e6


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def cluster_bootstrap(session_values: dict[int, list[float]], seed: int, samples: int = 4000) -> tuple[float, float]:
    rng = random.Random(seed)
    sessions = sorted(session_values)
    draws = []
    for _ in range(samples):
        means = []
        for _ in sessions:
            values = session_values[sessions[rng.randrange(len(sessions))]]
            means.append(statistics.mean(values[rng.randrange(len(values))] for _ in values))
        draws.append(statistics.mean(means))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup-blocks", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026083117)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires the gfx942 HIP runtime")
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    rng = random.Random(args.seed)
    for session in range(args.sessions):
        t_order = list(args.T)
        rng.shuffle(t_order)
        for t in t_order:
            calls = calls_for_t(t)
            for fn in calls.values():
                fn()
            torch.cuda.synchronize()
            for _ in range(args.warmup_blocks):
                orders = list(WILLIAMS_ORDERS)
                rng.shuffle(orders)
                for order in orders:
                    for name in order:
                        calls[name]()
            torch.cuda.synchronize()

            for block in range(args.blocks):
                orders = list(WILLIAMS_ORDERS)
                rng.shuffle(orders)
                event_by_name: dict[str, list[float]] = defaultdict(list)
                wall_by_name: dict[str, list[float]] = defaultdict(list)
                for order_index, order in enumerate(orders):
                    for position, name in enumerate(order):
                        event_ms, wall_ms = measure(calls[name])
                        event_by_name[name].append(event_ms)
                        wall_by_name[name].append(wall_ms)
                        raw.append({
                            "timing_contract": "eager_public_api", "cuda_graph_used": False,
                            "T": t, "chunks": t // 64, "session": session, "block": block,
                            "order_index": order_index, "order": "-".join(order), "position": position,
                            "implementation": name, "event_ms": event_ms, "wall_ms": wall_ms,
                        })
                event_medians = {name: statistics.median(event_by_name[name]) for name in NAMES}
                wall_medians = {name: statistics.median(wall_by_name[name]) for name in NAMES}
                for reference in ("x2", "vllm"):
                    block_rows.append({
                        "T": t, "chunks": t // 64, "session": session, "block": block,
                        "candidate": "x2_z5b", "reference": reference,
                        "event_gain_us": 1000 * (event_medians[reference] - event_medians["x2_z5b"]),
                        "wall_gain_us": 1000 * (wall_medians[reference] - wall_medians["x2_z5b"]),
                        "x2_z5b_event_ms": event_medians["x2_z5b"],
                        "reference_event_ms": event_medians[reference],
                    })
        print(f"complete session={session} t_order={t_order}", flush=True)

    aggregate: list[dict[str, object]] = []
    for t in args.T:
        for name in NAMES:
            values = [row["event_ms"] for row in raw if row["T"] == t and row["implementation"] == name]
            aggregate.append({
                "timing_contract": "eager_public_api", "cuda_graph_used": False,
                "T": t, "chunks": t // 64, "implementation": name, "samples": len(values),
                "event_median_ms": statistics.median(values), "event_p10_ms": percentile(values, 0.1),
                "event_p90_ms": percentile(values, 0.9),
            })
    pairs: list[dict[str, object]] = []
    for t in args.T:
        for reference in ("x2", "vllm"):
            rows = [row for row in block_rows if row["T"] == t and row["reference"] == reference]
            by_session: dict[int, list[float]] = defaultdict(list)
            wall_by_session: dict[int, list[float]] = defaultdict(list)
            for row in rows:
                by_session[int(row["session"])].append(float(row["event_gain_us"]))
                wall_by_session[int(row["session"])].append(float(row["wall_gain_us"]))
            event_ci = cluster_bootstrap(by_session, args.seed + t + len(reference))
            wall_ci = cluster_bootstrap(wall_by_session, args.seed + 1000 + t + len(reference))
            session_means = [statistics.mean(values) for values in by_session.values()]
            pairs.append({
                "T": t, "chunks": t // 64, "candidate": "x2_z5b", "reference": reference,
                "sessions": len(by_session), "blocks": len(rows),
                "event_gain_us_mean": statistics.mean(row["event_gain_us"] for row in rows),
                "event_gain_us_median": statistics.median(row["event_gain_us"] for row in rows),
                "event_cluster_ci_low_us": event_ci[0], "event_cluster_ci_high_us": event_ci[1],
                "wall_gain_us_mean": statistics.mean(row["wall_gain_us"] for row in rows),
                "wall_cluster_ci_low_us": wall_ci[0], "wall_cluster_ci_high_us": wall_ci[1],
                "majority_sessions_positive": sum(value > 0 for value in session_means) > len(session_means) / 2,
            })

    slopes = {}
    for name in NAMES:
        points = [(row["chunks"], row["event_median_ms"]) for row in aggregate if row["implementation"] == name]
        x_mean = statistics.mean(point[0] for point in points)
        y_mean = statistics.mean(point[1] for point in points)
        denominator = sum((x - x_mean) ** 2 for x, _ in points)
        slopes[name] = None if denominator == 0.0 else 1000 * sum(
            (x - x_mean) * (y - y_mean) for x, y in points
        ) / denominator

    write_csv(args.out_dir / "eager_public_raw.csv", raw)
    write_csv(args.out_dir / "eager_public_block_pairs.csv", block_rows)
    write_csv(args.out_dir / "eager_public_aggregate.csv", aggregate)
    write_csv(args.out_dir / "eager_public_pairs.csv", pairs)
    summary = {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "aggregate": aggregate,
        "pairs": pairs,
        "slope_us_per_chunk": slopes,
    }
    (args.out_dir / "eager_public_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

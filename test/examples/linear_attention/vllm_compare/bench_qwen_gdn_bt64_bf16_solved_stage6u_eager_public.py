#!/usr/bin/env python3
"""Authoritative complete eager public-API benchmark for Stage 6U."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = ROOT / "codex_qwen_bt64_full_pipeline_stage2"
DEFAULT_OUT = ROOT / "codex_qwen_bt64_bf16_solved_boundary_stage6u"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager,
    qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager,
)
from qwen_gdn_bt64_fused_wu_eager_stage6t import qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT = 64
NAMES = ("stage6s", "f1", "u0", "u1", "vllm")


def public_calls(inputs: tuple[torch.Tensor, ...]) -> dict[str, Callable[[], object]]:
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "u0": lambda: qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager(q, k, v, g, beta, **common),
        "u1": lambda: qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
            output_final_state=True, scale=128 ** -0.5, head_first=False,
            use_qk_l2norm_in_kernel=False,
        ),
    }


def measure(fn, start, end):
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    if result is None:
        raise AssertionError("public API returned no result")
    return float(start.elapsed_time(end)), (time.perf_counter_ns() - wall_start) / 1.0e6


def order_for(session: int, repeat_index: int):
    shift = (session + repeat_index) % len(NAMES)
    order = list(NAMES[shift:] + NAMES[:shift])
    if ((repeat_index // len(NAMES)) + session) & 1:
        order.reverse()
    return order


def qtile(values, q):
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, round((len(ordered) - 1) * q)))]


def bootstrap_ci(values, seed):
    rng = random.Random(seed)
    means = [statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(4000)]
    return qtile(means, 0.025), qtile(means, 0.975)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_t(t: int, sessions: int, warmup: int, repeat: int):
    calls = public_calls(make_inputs(t, 2026072700 + t, "random", True))
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    raw = []
    for session in range(sessions):
        for warmup_index in range(warmup):
            for name in order_for(session, warmup_index):
                calls[name]()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for repeat_index in range(repeat):
            order = order_for(session, repeat_index)
            for position, name in enumerate(order):
                event_ms, wall_ms = measure(calls[name], start, end)
                raw.append({
                    "timing_contract": "eager_public_api", "cuda_graph_used": False,
                    "T": t, "chunks": t // BT, "session": session,
                    "repeat": repeat_index, "position": position,
                    "order": "-".join(order), "implementation": name,
                    "event_ms": event_ms, "wall_ms": wall_ms,
                })
        print(f"benchmark complete: T={t} session={session}", flush=True)
    return raw


def summarize(raw):
    grouped = {}
    for row in raw:
        grouped.setdefault((int(row["T"]), int(row["session"]), row["implementation"]), []).append(row)
    session_rows = []
    for (t, session, name), rows in sorted(grouped.items()):
        events = [float(row["event_ms"]) for row in rows]
        walls = [float(row["wall_ms"]) for row in rows]
        session_rows.append({
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "chunks": t // BT, "session": session, "implementation": name,
            "samples": len(rows), "event_median_ms": statistics.median(events),
            "event_p10_ms": qtile(events, 0.1), "event_p90_ms": qtile(events, 0.9),
            "wall_median_ms": statistics.median(walls), "wall_p10_ms": qtile(walls, 0.1),
            "wall_p90_ms": qtile(walls, 0.9),
        })
    aggregate = []
    for t in sorted({int(row["T"]) for row in raw}):
        for name in NAMES:
            rows = [row for row in session_rows if row["T"] == t and row["implementation"] == name]
            events = [float(row["event_median_ms"]) for row in rows]
            walls = [float(row["wall_median_ms"]) for row in rows]
            aggregate.append({
                "timing_contract": "eager_public_api", "cuda_graph_used": False,
                "T": t, "chunks": t // BT, "session": "aggregate", "implementation": name,
                "samples": len(events), "event_median_ms": statistics.median(events),
                "event_p10_ms": min(events), "event_p90_ms": max(events),
                "wall_median_ms": statistics.median(walls), "wall_p10_ms": min(walls),
                "wall_p90_ms": max(walls),
            })
    return session_rows + aggregate, aggregate


def fit(aggregate, name):
    points = [(float(row["chunks"]), float(row["event_median_ms"])) for row in aggregate if row["implementation"] == name]
    mx = statistics.mean(x for x, _ in points)
    my = statistics.mean(y for _, y in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return my - slope * mx, slope * 1000.0


def pairwise(raw):
    indexed = {}
    for row in raw:
        indexed[(row["T"], row["session"], row["repeat"], row["implementation"])] = float(row["event_ms"])
    rows = []
    for t in sorted({int(row["T"]) for row in raw}):
        for reference in ("stage6s", "f1", "vllm"):
            deltas = []
            for session in sorted({int(row["session"]) for row in raw if int(row["T"]) == t}):
                repeats = sorted({int(row["repeat"]) for row in raw if int(row["T"]) == t and int(row["session"]) == session})
                for repeat_index in repeats:
                    deltas.append(1000.0 * (indexed[(t, session, repeat_index, reference)] - indexed[(t, session, repeat_index, "u1")]))
            low, high = bootstrap_ci(deltas, 2026072800 + t + len(reference))
            rows.append({
                "timing_contract": "eager_public_api", "cuda_graph_used": False,
                "T": t, "candidate": "u1", "reference": reference,
                "gain_us_positive_is_u1_faster": statistics.mean(deltas),
                "ci95_low_us": low, "ci95_high_us": high, "paired_samples": len(deltas),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.sessions < 5 or args.warmup < 30 or args.repeat < 200:
        raise ValueError("authoritative run requires sessions>=5, warmup>=30, repeat>=200")
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = []
    for t in args.T:
        raw.extend(run_t(t, args.sessions, args.warmup, args.repeat))
    summary, aggregate = summarize(raw)
    slopes = []
    for name in NAMES:
        intercept, slope = fit(aggregate, name)
        slopes.append({"timing_contract": "eager_public_api", "cuda_graph_used": False, "implementation": name, "intercept_ms": intercept, "slope_us_per_chunk": slope})
    slope_map = {row["implementation"]: row["slope_us_per_chunk"] for row in slopes}
    gap_slopes = [{"timing_contract": "eager_public_api", "cuda_graph_used": False, "implementation": name, "vs": "vllm", "gap_slope_us_per_chunk": slope_map[name] - slope_map["vllm"]} for name in ("stage6s", "f1", "u0", "u1")]
    pairs = pairwise(raw)
    write_csv(args.out_dir / "eager_full_raw.csv", raw)
    write_csv(args.out_dir / "eager_full_summary.csv", summary)
    write_csv(args.out_dir / "eager_full_pairwise.csv", pairs)
    write_csv(args.out_dir / "eager_full_slopes.csv", slopes)
    write_csv(args.out_dir / "eager_full_gap_slopes.csv", gap_slopes)
    write_csv(args.out_dir / "eager_wallclock_summary.csv", aggregate)
    (args.out_dir / "eager_order_analysis.md").write_text("# Eager Order Analysis\n\nFive cyclic rotations with alternating direction balance every implementation across all five positions. Session offset rotates the first position.\n")
    (args.out_dir / "eager_methodology.md").write_text("# Eager Methodology\n\nEvery timed sample is one complete public API call. Allocations, casts, dispatches and wrapper glue are included. HIP events are authoritative; synchronized wall time is supplemental. No CUDA/HIP Graph is used.\n")
    print(json.dumps({"aggregate": aggregate, "pairwise": pairs, "slopes": slopes, "gap_slopes": gap_slopes}, indent=2))


if __name__ == "__main__":
    main()

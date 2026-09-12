#!/usr/bin/env python3
"""ABBA Eager public-API benchmark for U1, Stage 6W, and native vLLM."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import qwen_gdn_full_bt64_stage6w_bf16_chunko_eager  # noqa: E402
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


NAMES = ("u1", "w1", "vllm")


def calls_for_t(t: int):
    q, k, v, g, beta, h0 = make_inputs(t, 2026074300 + t, "random", True)
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "u1": lambda: qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **common),
        "w1": lambda: qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def order_for(session: int, repeat: int) -> list[str]:
    shift = (session + repeat) % len(NAMES)
    order = list(NAMES[shift:] + NAMES[:shift])
    return order if ((session + repeat // len(NAMES)) & 1) == 0 else list(reversed(order))


def measure(fn) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = fn()
    end.record()
    end.synchronize()
    if result is None:
        raise AssertionError("public API returned None")
    return float(begin.elapsed_time(end))


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [statistics.mean(values[rng.randrange(len(values))] for _ in values) for _ in range(2000)]
    return percentile(means, 0.025), percentile(means, 0.975)


def slope_us_per_chunk(aggregate: list[dict[str, object]], name: str) -> float | None:
    points = [(float(row["chunks"]), float(row["event_median_ms"])) for row in aggregate if row["implementation"] == name]
    if len(points) < 2:
        return None
    x_mean = statistics.mean(x for x, _ in points)
    y_mean = statistics.mean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0.0:
        return None
    return 1000.0 * sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_t(t: int, sessions: int, warmup: int, repeat: int) -> list[dict[str, object]]:
    calls = calls_for_t(t)
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    raw = []
    for session in range(sessions):
        for warmup_index in range(warmup):
            for name in order_for(session, warmup_index):
                calls[name]()
        torch.cuda.synchronize()
        for repeat_index in range(repeat):
            order = order_for(session, repeat_index)
            for position, name in enumerate(order):
                raw.append({
                    "timing_contract": "eager_public_api", "cuda_graph_used": False,
                    "T": t, "chunks": t // 64, "session": session, "repeat": repeat_index,
                    "position": position, "order": "-".join(order), "implementation": name,
                    "event_ms": measure(calls[name]),
                })
        print(f"complete T={t} session={session}", flush=True)
    return raw


def summarize(raw: list[dict[str, object]]):
    by_session: dict[tuple[int, int, str], list[float]] = {}
    for row in raw:
        by_session.setdefault((int(row["T"]), int(row["session"]), str(row["implementation"])), []).append(float(row["event_ms"]))
    session_rows = [
        {
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "chunks": t // 64, "session": session, "implementation": name,
            "samples": len(values), "event_median_ms": statistics.median(values),
            "event_p10_ms": percentile(values, 0.1), "event_p90_ms": percentile(values, 0.9),
        }
        for (t, session, name), values in sorted(by_session.items())
    ]
    aggregate = []
    for t in sorted({int(row["T"]) for row in raw}):
        for name in NAMES:
            values = [float(row["event_median_ms"]) for row in session_rows if row["T"] == t and row["implementation"] == name]
            aggregate.append({
                "timing_contract": "eager_public_api", "cuda_graph_used": False,
                "T": t, "chunks": t // 64, "session": "aggregate", "implementation": name,
                "samples": len(values), "event_median_ms": statistics.median(values),
                "event_p10_ms": min(values), "event_p90_ms": max(values),
            })
    indexed = {(int(row["T"]), int(row["session"]), int(row["repeat"]), str(row["implementation"])): float(row["event_ms"]) for row in raw}
    pairs = []
    for t in sorted({int(row["T"]) for row in raw}):
        for reference in ("u1", "vllm"):
            values = [
                1000.0 * (indexed[(t, session, repeat, reference)] - indexed[(t, session, repeat, "w1")])
                for session in sorted({int(row["session"]) for row in raw if int(row["T"]) == t})
                for repeat in sorted({int(row["repeat"]) for row in raw if int(row["T"]) == t and int(row["session"]) == session})
            ]
            ci_low, ci_high = bootstrap_ci(values, 2026074400 + t + len(reference))
            pairs.append({
                "T": t, "candidate": "w1", "reference": reference,
                "paired_gain_us": statistics.median(values), "paired_mean_gain_us": statistics.mean(values),
                "bootstrap_ci_low_us": ci_low, "bootstrap_ci_high_us": ci_high,
            })
    return session_rows, aggregate, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = [row for t in args.T for row in run_t(t, args.sessions, args.warmup, args.repeat)]
    session_rows, aggregate, pairs = summarize(raw)
    slopes = {name: slope_us_per_chunk(aggregate, name) for name in NAMES}
    write_csv(args.out_dir / "eager_public_raw.csv", raw)
    write_csv(args.out_dir / "eager_public_session_medians.csv", session_rows)
    write_csv(args.out_dir / "eager_public_aggregate.csv", aggregate)
    write_csv(args.out_dir / "eager_public_pairs.csv", pairs)
    summary = {"timing_contract": "eager_public_api", "cuda_graph_used": False, "aggregate": aggregate, "pairs": pairs, "slope_us_per_chunk": slopes}
    (args.out_dir / "eager_public_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

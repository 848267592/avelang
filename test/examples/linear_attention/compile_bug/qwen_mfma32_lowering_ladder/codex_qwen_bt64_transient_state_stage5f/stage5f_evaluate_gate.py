#!/usr/bin/env python3
"""Apply the predeclared Stage 5F low-perturbation gate to CSV summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(rows_: list[dict[str, str]], operation: str, t: int) -> dict[str, float]:
    # Stage 5E's aggregate row intentionally stores only the paired delta.
    # Reconstruct A/B aggregate medians from the five independent sessions.
    selected = [row for row in rows_ if row["operation"] == operation and int(row["T"]) == t and row["session"] != "aggregate"]
    a_values = [float(row["median_ms"]) for row in selected if row["solve_impl"] == "v18"]
    b_values = [float(row["median_ms"]) for row in selected if row["solve_impl"] == "hierarchical_fp32_v1"]
    paired = [float(row["v1_minus_v18_us"]) for row in selected if row["solve_impl"] == "v18"]
    if not a_values or not b_values or not paired:
        raise RuntimeError(f"missing Stage 5E session rows for {operation=} {t=}")
    return {"a_ms": statistics.median(a_values), "b_ms": statistics.median(b_values), "penalty_us": statistics.median(paired)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--operation", default="tail")
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = aggregate(rows(args.baseline), args.operation, args.T)
    candidate = aggregate(rows(args.candidate), args.operation, args.T)
    baseline_latency_us = max(base["a_ms"], base["b_ms"]) * 1000.0
    latency_distortion_us = max(abs(candidate["a_ms"] - base["a_ms"]), abs(candidate["b_ms"] - base["b_ms"])) * 1000.0
    penalty_distortion_us = abs(candidate["penalty_us"] - base["penalty_us"])
    latency_limit_us = max(5.0, baseline_latency_us * 0.05)
    penalty_limit_us = max(5.0, abs(base["penalty_us"]) * 0.10)
    result = {
        "operation": args.operation,
        "T": args.T,
        "baseline": base,
        "candidate": candidate,
        "latency_distortion_us": latency_distortion_us,
        "latency_limit_us": latency_limit_us,
        "penalty_distortion_us": penalty_distortion_us,
        "penalty_limit_us": penalty_limit_us,
        "latency_gate_pass": latency_distortion_us <= latency_limit_us,
        "penalty_gate_pass": penalty_distortion_us <= penalty_limit_us,
    }
    result["observability_gate_pass"] = result["latency_gate_pass"] and result["penalty_gate_pass"]
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

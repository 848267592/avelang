#!/usr/bin/env python3
"""Fresh-process R4-tail versus the three requested last-use d1 W granularities."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_microtile as microtile
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as tail_issue


BT = microtile.BT
HERE = Path(__file__).resolve().parent
IMPLEMENTATIONS: tuple[tuple[str, str | None], ...] = (
    ("r4_tail_issue", None),
    ("micro_w1_k1_lastuse_d1", "gfx942_bt64_bv32_microtile_experimental_w1_k1_lastuse_d1"),
    ("micro_w2_k1_lastuse_d1", "gfx942_bt64_bv32_microtile_experimental_w2_k1_lastuse_d1"),
    ("micro_w4_k1_lastuse_d1", "gfx942_bt64_bv32_microtile_experimental_w4_k1_lastuse_d1"),
)


def _launch_for(
    implementation: str, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor,
    g: torch.Tensor, initial_state: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    if implementation == "r4_tail_issue":
        return tail_issue.run_body(k, w, u, g, initial_state)
    plan = dict(IMPLEMENTATIONS)[implementation]
    assert plan is not None
    return microtile.run_body(plan, k, w, u, g, initial_state)


def _event_ms(launch: Callable[[], None], start: torch.cuda.Event,
              end: torch.cuda.Event) -> float:
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    t = args.T[0]
    k, w, u, g, initial_state = p2._make_long_case(t, args.seed + t)
    launch, h, v_new, final_state = _launch_for(args.implementation, k, w, u, g, initial_state)
    torch.cuda.synchronize()
    finite = all(bool(torch.isfinite(value.float()).all()) for value in (h, v_new, final_state))
    if not finite:
        raise RuntimeError(f"non-finite {args.implementation}")
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [_event_ms(launch, start, end) for _ in range(args.repeat)]
    return {
        "implementation": args.implementation,
        "T": t,
        "chunks": t // BT,
        "session": args.session,
        "median_ms": statistics.median(samples),
        "p10_ms": sorted(samples)[max(0, len(samples) // 10 - 1)],
        "p90_ms": sorted(samples)[min(len(samples) - 1, 9 * len(samples) // 10)],
        "repeat": args.repeat,
        "warmup": args.warmup,
        "fresh_process": True,
        "graph_capture": False,
        "finite": finite,
    }


def _child(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=True)
    line = next((line for line in reversed(completed.stdout.splitlines())
                 if line.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"worker emitted no JSON:\n{completed.stdout}")
    return json.loads(line)


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    for t in args.T:
        for session in range(args.sessions):
            for implementation, _ in IMPLEMENTATIONS:
                raw.append(_child([
                    sys.executable, str(HERE / Path(__file__).name), "--worker",
                    "--implementation", implementation, "--T", str(t),
                    "--seed", str(args.seed), "--session", str(session),
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                ]))
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in raw:
        buckets.setdefault((int(row["T"]), str(row["implementation"])), []).append(
            float(row["median_ms"]))
    summary = [{
        "T": t, "chunks": t // BT, "implementation": implementation,
        "median_of_session_medians_ms": statistics.median(values),
        "session_count": len(values),
    } for (t, implementation), values in sorted(buckets.items())]
    by_key = {(row["T"], row["implementation"]): row for row in summary}
    ratios = []
    for t in args.T:
        baseline = by_key[(t, "r4_tail_issue")]["median_of_session_medians_ms"]
        for implementation, _ in IMPLEMENTATIONS[1:]:
            value = by_key[(t, implementation)]["median_of_session_medians_ms"]
            ratios.append({
                "T": t, "implementation": implementation,
                "over_r4_tail": value / baseline,
                "speedup_vs_r4_tail": baseline / value,
            })
    low, high = min(args.T), max(args.T)
    slopes = []
    for implementation, _ in IMPLEMENTATIONS:
        low_ms = by_key[(low, implementation)]["median_of_session_medians_ms"]
        high_ms = by_key[(high, implementation)]["median_of_session_medians_ms"]
        slopes.append({
            "implementation": implementation,
            "interval": f"{low}->{high}",
            "ms_per_chunk": (high_ms - low_ms) / ((high - low) // BT),
        })
    return {
        "contract": {
            "fresh_process_per_implementation": True,
            "source_mode_cache_isolation": True,
            "timing": "HIP event",
            "graph_capture": False,
            "allocation_or_compile_in_timing": False,
        },
        "raw": raw,
        "summary": summary,
        "ratios": ratios,
        "slopes": slopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation", choices=[name for name, _ in IMPLEMENTATIONS])
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = _worker(args) if args.worker else _parent(args)
    if args.out is not None and not args.worker:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True) if args.worker else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fresh-process body benchmark: modulo scheduler versus R4 and Triton.

One worker owns exactly one implementation.  This is important because the
AveLang JIT cache specialization does not include the experimental lowering
environment in its key; loading R4 and the software-pipeline candidate in one
interpreter would not be a valid A/B experiment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

import bench_qwen_gdn_direct_k64_bv32_full_sequence_b0 as b0_bench
import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_r4_joint_v4 as r4
import repro_qwen_gdn_persistent_recurrence_software_pipeline as swp


BT = swp.BT
HERE = Path(__file__).resolve().parent


def _call_for(
    implementation: str,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
) -> b0_bench.Call:
    if implementation == "avelang_software_pipeline":
        launch, h, v_new, final_state = swp.run_body(k, w, u, g, initial_state)
        return b0_bench.Call(implementation, launch, h, v_new, final_state)
    if implementation == "avelang_r4_joint_v4":
        launch, h, v_new, final_state = r4.run_body(k, w, u, g, initial_state)
        return b0_bench.Call(implementation, launch, h, v_new, final_state)
    if implementation == "current_triton_direct":
        triton = b0_bench._direct_triton_call(k, w, u, g, initial_state)
        return b0_bench.Call(implementation, triton.launch, triton.h, triton.v_new, triton.final_state)
    raise ValueError(f"unknown implementation: {implementation}")


def _event_ms(launch: Callable[[], None], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    t = args.T[0]
    if t % BT:
        raise ValueError(f"T={t} is not divisible by BT={BT}")
    k, w, u, g, initial_state = p2._make_long_case(t, args.seed + t)
    call = _call_for(args.implementation, k, w, u, g, initial_state)
    torch.cuda.synchronize()  # construction/JIT is strictly outside timing
    finite = all(bool(torch.isfinite(value.float()).all().item()) for value in (call.h, call.v_new, call.final_state))
    if not finite:
        raise RuntimeError(f"non-finite {args.implementation} result")
    for _ in range(args.warmup):
        call.launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [_event_ms(call.launch, start, end) for _ in range(args.repeat)]
    return {
        "T": t,
        "chunks": t // BT,
        "session": args.session,
        "implementation": args.implementation,
        "median_ms": statistics.median(samples),
        "p10_ms": sorted(samples)[max(0, len(samples) // 10 - 1)],
        "p90_ms": sorted(samples)[min(len(samples) - 1, (len(samples) * 9) // 10)],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "finite": finite,
        "graph_capture": False,
        "fresh_process": True,
    }


def _summary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((int(row["T"]), str(row["implementation"])), []).append(float(row["median_ms"]))
    summary = [
        {
            "T": t,
            "chunks": t // BT,
            "implementation": impl,
            "median_of_session_medians_ms": statistics.median(values),
            "session_count": len(values),
        }
        for (t, impl), values in sorted(buckets.items())
    ]
    by_t = {int(row["T"]): row for row in summary if row["implementation"] == "avelang_software_pipeline"}
    ratios = []
    for row in summary:
        if row["implementation"] == "avelang_software_pipeline":
            continue
        candidate = by_t[int(row["T"])]["median_of_session_medians_ms"]
        ratios.append({
            "T": row["T"],
            "against": row["implementation"],
            "software_pipeline_over_reference": candidate / row["median_of_session_medians_ms"],
            "software_pipeline_speedup": row["median_of_session_medians_ms"] / candidate,
        })
    return summary, ratios


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    implementations = ("avelang_software_pipeline", "avelang_r4_joint_v4", "current_triton_direct")
    for t in args.T:
        for session in range(args.sessions):
            for implementation in implementations:
                command = [
                    sys.executable, str(HERE / Path(__file__).name), "--worker", "--implementation", implementation,
                    "--T", str(t), "--seed", str(args.seed), "--session", str(session),
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                ]
                completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
                line = next((item for item in reversed(completed.stdout.splitlines()) if item.startswith("{")), None)
                if line is None:
                    raise RuntimeError(f"worker emitted no JSON payload:\n{completed.stdout}")
                rows.append(json.loads(line))
    summary, ratios = _summary(rows)
    return {
        "contract": {
            "fresh_process_per_implementation": True,
            "source_mode_cache_isolation": True,
            "graph_capture": False,
            "timing": "HIP event",
            "allocation_or_compile_in_timing": False,
        },
        "raw": rows,
        "summary": summary,
        "ratios": ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--implementation", choices=("avelang_software_pipeline", "avelang_r4_joint_v4", "current_triton_direct"))
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = _worker(args) if args.worker else _parent(args)
    if args.out is not None and not args.worker:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True) if args.worker else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

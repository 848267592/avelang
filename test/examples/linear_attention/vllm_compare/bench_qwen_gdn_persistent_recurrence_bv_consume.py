#!/usr/bin/env python3
"""Fresh-process body benchmark for R4-tail and logical BV-consume tiles.

Each (implementation, T, session) uses a new interpreter, so the lowering
mode cannot reuse a code object compiled by another BV candidate.  Launches
operate on preallocated tensors and use HIP events; compilation/allocation are
outside the timing region.
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

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_bv_consume as bv
import repro_qwen_gdn_persistent_recurrence_r4_tail_issue as tail


HERE = Path(__file__).resolve().parent
BT = bv.BT
BASELINE = "r4_tail_bv32"
IMPLEMENTATIONS = (BASELINE, "bv16", "bv32", "bv64")


def _launch_for(
    implementation: str,
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor, torch.Tensor, torch.Tensor]:
    if implementation == BASELINE:
        return tail.run_body(k, w, u, g, initial_state)
    return bv.run_body(k, w, u, g, initial_state, bv_consume=int(implementation[2:]))


def _event_ms(launch: Callable[[], None], start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def worker(args: argparse.Namespace) -> dict[str, Any]:
    t = args.T[0]
    k, w, u, g, initial_state = p2._make_long_case(t, args.seed + t)
    launch, h, v_new, final_state = _launch_for(args.implementation, k, w, u, g, initial_state)
    torch.cuda.synchronize()
    if not all(bool(torch.isfinite(x.float()).all()) for x in (h, v_new, final_state)):
        raise RuntimeError(f"non-finite result: {args.implementation}")
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [_event_ms(launch, start, end) for _ in range(args.repeat)]
    values = sorted(samples)
    return {
        "implementation": args.implementation, "T": t, "chunks": t // BT,
        "session": args.session, "median_ms": statistics.median(values),
        "p10_ms": values[max(0, len(values) // 10 - 1)],
        "p90_ms": values[min(len(values) - 1, (len(values) * 9) // 10)],
        "warmup": args.warmup, "repeat": args.repeat, "fresh_process": True,
        "graph_capture": False,
    }


def _child(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=True)
    line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"worker emitted no JSON payload:\n{completed.stdout}")
    return json.loads(line)


def parent(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for t in args.T:
        for session in range(args.sessions):
            # Alternate the first arm; every arm remains a new interpreter.
            order = IMPLEMENTATIONS if session % 2 == 0 else tuple(reversed(IMPLEMENTATIONS))
            for implementation in order:
                rows.append(_child([
                    sys.executable, str(HERE / Path(__file__).name), "--worker",
                    "--implementation", implementation, "--T", str(t),
                    "--seed", str(args.seed), "--session", str(session),
                    "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                ]))
    buckets: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((int(row["T"]), str(row["implementation"])), []).append(float(row["median_ms"]))
    summary = [{
        "T": t, "chunks": t // BT, "implementation": impl,
        "median_of_session_medians_ms": statistics.median(values),
        "session_count": len(values),
    } for (t, impl), values in sorted(buckets.items())]
    table = {(int(r["T"]), str(r["implementation"])): r for r in summary}
    ratios = [{
        "T": t, "implementation": impl,
        "over_r4_tail": table[(t, impl)]["median_of_session_medians_ms"] /
                        table[(t, BASELINE)]["median_of_session_medians_ms"],
    } for t in args.T for impl in IMPLEMENTATIONS if impl != BASELINE]
    slopes = []
    if len(args.T) >= 2:
        low, high = min(args.T), max(args.T)
        for impl in IMPLEMENTATIONS:
            slopes.append({
                "implementation": impl, "interval": f"{low}->{high}",
                "ms_per_chunk": (table[(high, impl)]["median_of_session_medians_ms"] -
                                 table[(low, impl)]["median_of_session_medians_ms"]) /
                                ((high - low) // BT),
            })
    return {
        "contract": {
            "fresh_process_per_implementation_and_length": True,
            "source_mode_cache_isolation": True, "timing": "HIP event",
            "graph_capture": False, "allocation_or_compile_in_timing": False,
        },
        "raw": rows, "summary": summary, "ratios": ratios, "slopes": slopes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[1024, 2048, 8192])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.worker:
        if args.implementation is None:
            parser.error("--worker requires --implementation")
        result = worker(args)
    else:
        result = parent(args)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True) if args.worker else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

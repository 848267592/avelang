#!/usr/bin/env python3
"""C26 paired fresh-process timing for the selected chunk-o bodies.

The native worker seeds the real public selector before direct tail timing.
This driver only compares Z5B and that selected native tail; it does not run a
CUDA graph and it does not make a public-API performance claim.
"""

from __future__ import annotations

import argparse
import json
import random
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
WORKER = HERE / "run_qwen_gfx942_c26_selected_body.py"
ARMS = ("z5b", "native_selected")
ORDERS = (("z5b", "native_selected"), ("native_selected", "z5b"))


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _bootstrap_median(values: list[float], seed: int, samples: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    medians = [statistics.median([rng.choice(values) for _ in values]) for _ in range(samples)]
    return [_quantile(medians, 0.025), _quantile(medians, 0.975)]


def _run_worker(arm: str, t: int, session: int, warmup: int, repeat: int) -> dict[str, object]:
    command = [
        sys.executable, str(WORKER), "--arm", arm, "--T", str(t), "--warmup", str(warmup),
        "--repeat", str(repeat), "--timing",
    ]
    environment = os.environ.copy()
    # The native current selector is part of the C26 identity contract.  A
    # shared cache can pin a historical autotuner entry (for example W4 at
    # T2048) even when a clean current selection is W2.  Isolate this worker
    # before Python imports Triton/vLLM, exactly as the profiler capture does.
    with tempfile.TemporaryDirectory(prefix=f"c26_t{t}_s{session}_{arm}_") as cache:
        environment["TRITON_CACHE_DIR"] = cache
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if completed.returncode:
        raise RuntimeError(f"C26 T={t} arm={arm} failed:\n{completed.stdout}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be a positive BT64 multiple")

    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = ORDERS[session % len(ORDERS)]
        arms = {arm: _run_worker(arm, args.T, session, args.warmup, args.repeat) for arm in order}
        raw.append({"session": session, "order": list(order), "arms": arms})

    samples = {
        arm: [float(row["arms"][arm]["measurement"]["hip_ms"]["median"]) for row in raw]
        for arm in ARMS
    }
    summary = {
        arm: {
            "session_medians_ms": values,
            "median_of_session_medians_ms": statistics.median(values),
            "mean_of_session_medians_ms": statistics.fmean(values),
        }
        for arm, values in samples.items()
    }
    paired_us = [(n - z) * 1.0e3 for z, n in zip(samples["z5b"], samples["native_selected"])]
    payload = {
        "schema": "qwen.gfx942.stage6z.c26.formal_body.v1",
        "scope": "caller_owned_isolated_body_diagnostic",
        "T": args.T,
        "chunks": args.T // 64,
        "contract": {
            "sessions": args.sessions,
            "fresh_python_process_per_arm": True,
            "current_stream": True,
            "cuda_graph_used": False,
            "preallocated_output": True,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "balanced_orders": [list(ORDERS[index % len(ORDERS)]) for index in range(args.sessions)],
            "native_selector_seed": "current vLLM public API before direct selected tail",
        },
        "raw_sessions": raw,
        "summary": summary,
        "paired_native_minus_z5b_us": {
            "samples": paired_us,
            "median_us": statistics.median(paired_us),
            "bootstrap_ci95_us": _bootstrap_median(paired_us, 20260826 + args.T),
        },
        "derived": {
            "z5b_over_native": summary["z5b"]["median_of_session_medians_ms"] / summary["native_selected"]["median_of_session_medians_ms"],
        },
    }
    out = args.out or LADDER / f"stage6z_c26_formal_body_T{args.T}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

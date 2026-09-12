#!/usr/bin/env python3
"""Paired fresh-process body benchmark for repaired Z2 and selected native WG256.

This is a read-only audit harness.  It deliberately does not call the public
chunk-o wrapper in the measured worker: the exact fresh-public WG256 config is
seeded into the native autotuner cache before the first direct body launch.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BT = 64
ARMS = ("z2", "native")


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def worker(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(HERE))
    sys.path.insert(
        0,
        str(
            REPO
            / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
        ),
    )
    from bench_qwen_gdn_bt64_stage6z_native_selected_wg256 import (  # noqa: E402
        inputs,
        launch as native_launch,
        select_and_pin,
    )
    from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
    )

    tensors = inputs(args.T)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in ARMS}
    native_config = select_and_pin(tensors, args.T)
    launches = {
        "z2": lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(
            *tensors, outputs["z2"]
        ),
        "native": lambda: native_launch(tensors, outputs["native"], args.T),
    }

    # Compilation, module loading, and the pinned native body are all outside
    # the measured interval.  The benchmark still uses the current stream.
    for arm in args.order.split(","):
        launches[arm]()
    torch.cuda.synchronize()

    measurements: dict[str, object] = {}
    for arm in args.order.split(","):
        for _ in range(args.warmup):
            launches[arm]()
        torch.cuda.synchronize()
        hip: list[float] = []
        wall: list[float] = []
        for _ in range(args.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter_ns()
            start.record()
            launches[arm]()
            end.record()
            end.synchronize()
            hip.append(float(start.elapsed_time(end)))
            wall.append((time.perf_counter_ns() - wall_start) / 1.0e6)
        measurements[arm] = {
            "hip_ms": {
                "median": statistics.median(hip),
                "p10": quantile(hip, 0.1),
                "p90": quantile(hip, 0.9),
            },
            "wall_ms": {
                "median": statistics.median(wall),
                "p10": quantile(wall, 0.1),
                "p90": quantile(wall, 0.9),
            },
            "finite": bool(torch.isfinite(outputs[arm]).all().item()),
        }
    if not all(bool(measurements[arm]["finite"]) for arm in ARMS):
        raise RuntimeError(f"non-finite output: {measurements}")
    print(
        json.dumps(
            {
                "scope": "fixed_z2_vs_native_selected_wg256_body",
                "fresh_process_worker": True,
                "cuda_graph_used": False,
                "current_stream": True,
                "T": args.T,
                "chunks": args.T // BT,
                "order": args.order.split(","),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "native_config": native_config,
                "arms": measurements,
            },
            sort_keys=True,
        )
    )


def parent(args: argparse.Namespace) -> None:
    orders = ["z2,native", "native,z2", "native,z2", "z2,native", "native,z2"]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = orders[session % len(orders)]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--T",
            str(args.T),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--order",
            order,
        ]
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode:
            raise RuntimeError(f"session {session} order {order} failed:\n{completed.stdout}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        row = json.loads(lines[-1])
        row["session"] = session
        raw.append(row)
        print(json.dumps(row, sort_keys=True))

    medians = {
        arm: [float(row["arms"][arm]["hip_ms"]["median"]) for row in raw]
        for arm in ARMS
    }
    summary = []
    for arm in ARMS:
        values = medians[arm]
        summary.append(
            {
                "arm": arm,
                "session_count": len(values),
                "median_of_session_medians_ms": statistics.median(values),
                "mean_of_session_medians_ms": statistics.fmean(values),
                "session_medians_ms": values,
            }
        )
    paired = {
        "z2_minus_native_us": [
            (z2 - native) * 1.0e3 for z2, native in zip(medians["z2"], medians["native"])
        ],
        "z2_over_native": [z2 / native for z2, native in zip(medians["z2"], medians["native"])],
    }
    payload = {
        "scope": "fixed_z2_vs_native_selected_wg256_body",
        "contract": {
            "fresh_process_sessions": True,
            "sessions": args.sessions,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "current_stream": True,
            "cuda_graph_used": False,
            "preallocated_outputs": True,
            "native_selection": "fresh public WG256/BK32/BV64/stage2 config pinned before direct body",
        },
        "T": args.T,
        "raw": raw,
        "summary": summary,
        "paired_differences": paired,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--order", default="z2,native")
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    if args.worker:
        worker(args)
    else:
        parent(args)


if __name__ == "__main__":
    main()

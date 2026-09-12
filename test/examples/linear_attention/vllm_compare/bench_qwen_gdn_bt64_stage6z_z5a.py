#!/usr/bin/env python3
"""Fresh-process paired body benchmark for fixed Z2, Z4C and Z5A."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5a_dedicated_q_lds import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5a_dedicated_q_lds_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
ARMS = ("z2", "z4c", "z5a")


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


def _make_launch(arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    if arm == "z2":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(*tensors, output)
    if arm == "z4c":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into(
            *tensors, output
        )
    return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5a_dedicated_q_lds_launch_into(*tensors, output)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def _measure(fn: Callable[[], None], warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    hips: list[float] = []
    walls: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        end.synchronize()
        hips.append(float(start.elapsed_time(end)))
        walls.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    return {
        "samples": repeat,
        "hip_ms": {"median": statistics.median(hips), "p10": _quantile(hips, 0.1), "p90": _quantile(hips, 0.9)},
        "wall_ms": {"median": statistics.median(walls), "p10": _quantile(walls, 0.1), "p90": _quantile(walls, 0.9)},
    }


def _worker(args: argparse.Namespace) -> None:
    tensors = _inputs(args.T, 2026081000 + args.T)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in ARMS}
    launches = {arm: _make_launch(arm, tensors, outputs[arm]) for arm in ARMS}
    for arm in args.order.split(","):
        launches[arm]()
    torch.cuda.synchronize()
    measurements: dict[str, object] = {}
    for arm in args.order.split(","):
        measurements[arm] = _measure(launches[arm], args.warmup, args.repeat)
        if not bool(torch.isfinite(outputs[arm]).all().item()):
            raise RuntimeError(f"non-finite output for {arm}")
    print(
        json.dumps(
            {
                "scope": "caller_owned_isolated_body_diagnostic",
                "fresh_process_worker": True,
                "cuda_graph_used": False,
                "current_stream": True,
                "T": args.T,
                "chunks": args.T // BT,
                "order": args.order.split(","),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "workgroup": 256,
                "arms": measurements,
            },
            sort_keys=True,
        )
    )


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z2,z4c,z5a",
        "z4c,z5a,z2",
        "z5a,z2,z4c",
        "z4c,z2,z5a",
        "z5a,z4c,z2",
        "z2,z5a,z4c",
    ]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = orders[session % len(orders)]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--T",
            str(args.T),
            "--sessions",
            "1",
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--order",
            order,
        ]
        completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"session {session} order {order} failed:\n{completed.stdout}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        row = json.loads(lines[-1])
        row["session"] = session
        raw.append(row)
        print(json.dumps(row, sort_keys=True))

    medians = {arm: [float(row["arms"][arm]["hip_ms"]["median"]) for row in raw] for arm in ARMS}
    summary = [
        {
            "arm": arm,
            "session_count": len(medians[arm]),
            "median_of_session_medians_ms": statistics.median(medians[arm]),
            "mean_of_session_medians_ms": statistics.fmean(medians[arm]),
            "session_medians_ms": medians[arm],
        }
        for arm in ARMS
    ]
    paired = {
        "z4c_minus_z2_us": [(z4c - z2) * 1.0e3 for z2, z4c in zip(medians["z2"], medians["z4c"])],
        "z5a_minus_z2_us": [(z5a - z2) * 1.0e3 for z2, z5a in zip(medians["z2"], medians["z5a"])],
        "z5a_minus_z4c_us": [(z5a - z4c) * 1.0e3 for z4c, z5a in zip(medians["z4c"], medians["z5a"])],
    }
    payload = {
        "scope": "caller_owned_isolated_body_diagnostic",
        "contract": {
            "fresh_process_sessions": True,
            "sessions": args.sessions,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "current_stream": True,
            "cuda_graph_used": False,
            "preallocated_outputs": True,
            "rotating_orders": orders[: args.sessions],
        },
        "T": args.T,
        "raw": raw,
        "summary": summary,
        "paired_differences_us": paired,
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
    parser.add_argument("--order", default="z2,z4c,z5a")
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fresh-process paired body benchmark for frozen Z5B, C22, and native."""

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

from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_c22_z5b_schedule_preserving_physical import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c22_z5b_schedule_preserving_physical_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128
ARMS = ("z5b", "c22", "native")


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


def _native_direct(tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    q, k, vn, h, g = tensors
    t = int(q.shape[1])
    chunk_o.chunk_fwd_kernel_o[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)](
        q, k, vn, h, g, output, None, None, K ** -0.5,
        T=t, H=HV, Hg=HK, K=K, V=V, BT=BT,
    )


def _launch(arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    if arm == "z5b":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, output)
    if arm == "c22":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c22_z5b_schedule_preserving_physical_launch_into(
            *tensors, output
        )
    return lambda: _native_direct(tensors, output)


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
    tensors = _inputs(args.T, 2026081300 + args.T)
    output = torch.empty_like(tensors[2])
    if args.arm == "native":
        # Selector/autotuning/module load is deliberately outside timing.
        warm = chunk_o.chunk_fwd_o(
            q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4],
            scale=K ** -0.5, chunk_size=BT,
        )
        torch.cuda.synchronize()
        del warm
    launch = _launch(args.arm, tensors, output)
    measured = _measure(launch, args.warmup, args.repeat)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"non-finite output for {args.arm}")
    print(json.dumps({
        "scope": "caller_owned_isolated_body_diagnostic",
        "fresh_process_worker": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "T": args.T,
        "chunks": args.T // BT,
        "arm": args.arm,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "workgroup": 256,
        "measurement": measured,
    }, sort_keys=True))


def _parent(args: argparse.Namespace) -> None:
    orders = [
        ("z5b", "c22", "native"),
        ("native", "c22", "z5b"),
        ("c22", "z5b", "native"),
        ("c22", "native", "z5b"),
        ("native", "z5b", "c22"),
        ("z5b", "native", "c22"),
        ("c22", "z5b", "native"),
    ]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = orders[session % len(orders)]
        for rank, arm in enumerate(order):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--T", str(args.T), "--arm", arm,
                "--warmup", str(args.warmup), "--repeat", str(args.repeat),
            ]
            completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"session {session} arm {arm} failed:\n{completed.stdout}")
            lines = [line for line in completed.stdout.splitlines() if line.strip() and line.lstrip().startswith("{")]
            row = json.loads(lines[-1])
            row["session"] = session
            row["order"] = list(order)
            row["order_rank"] = rank
            raw.append(row)
            print(json.dumps(row, sort_keys=True))

    medians = {
        arm: [float(row["measurement"]["hip_ms"]["median"]) for row in raw if row["arm"] == arm]
        for arm in ARMS
    }
    session_summary = [
        {
            "arm": arm,
            "session_count": len(medians[arm]),
            "median_of_session_medians_ms": statistics.median(medians[arm]),
            "mean_of_session_medians_ms": statistics.fmean(medians[arm]),
            "session_medians_ms": medians[arm],
        }
        for arm in ARMS
    ]
    paired = {}
    for arm in ("c22",):
        paired[f"{arm}_minus_z5b_us"] = [(value - base) * 1.0e3 for value, base in zip(medians[arm], medians["z5b"])]
        paired[f"{arm}_minus_native_us"] = [(value - base) * 1.0e3 for value, base in zip(medians[arm], medians["native"])]
        paired[f"{arm}_over_native"] = [value / base for value, base in zip(medians[arm], medians["native"])]
    paired["z5b_minus_native_us"] = [(value - base) * 1.0e3 for value, base in zip(medians["z5b"], medians["native"])]
    payload = {
        "scope": "caller_owned_isolated_body_diagnostic",
        "contract": {
            "fresh_process_per_arm": True,
            "sessions": args.sessions,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "current_stream": True,
            "cuda_graph_used": False,
            "preallocated_outputs": True,
            "rotating_orders": [list(orders[i % len(orders)]) for i in range(args.sessions)],
        },
        "T": args.T,
        "raw": raw,
        "summary": session_summary,
        "paired_differences_us": paired,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    if args.worker and args.arm is None:
        raise ValueError("--worker requires --arm")
    if not args.worker and args.arm is not None:
        raise ValueError("--arm is only valid with --worker")
    (_worker if args.worker else _parent)(args)


if __name__ == "__main__":
    main()

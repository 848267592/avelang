#!/usr/bin/env python3
"""Fresh-process paired body benchmark for Z5B, Z6G-S, Z6G-I and native WG256."""

from __future__ import annotations

import argparse
import json
import random
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
from qwen_gdn_bt64_native_chunko_stage6z_z6g_g_residency import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128
ARMS = ("z5b", "z6g_s", "z6g_i", "native")


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


def _make_launch(arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    if arm == "z5b":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(
            *tensors, output
        )
    if arm == "z6g_s":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into(
            *tensors, output
        )
    if arm == "z6g_i":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into(
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


def _bootstrap_ci(values: list[float], seed: int, rounds: int = 4000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = []
    for _ in range(rounds):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(statistics.fmean(draw))
    samples.sort()
    return samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]


def _worker(args: argparse.Namespace) -> None:
    active_arms = tuple(args.only_arm.split(",")) if args.only_arm else ARMS
    if not active_arms or any(arm not in ARMS for arm in active_arms):
        raise ValueError(f"unknown arm in --only-arm: {active_arms}")
    tensors = _inputs(args.T, 2026081000 + args.T)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in active_arms}
    launches = {arm: _make_launch(arm, tensors, outputs[arm]) for arm in active_arms}

    # Compile/selector warmup is outside measured samples.  The native call is
    # only the same-shape diagnostic body; it is not changed by this experiment.
    if "native" in active_arms:
        native_public = chunk_o.chunk_fwd_o(
            q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4],
            scale=K ** -0.5, chunk_size=BT,
        )
        torch.cuda.synchronize()
        del native_public

    order = args.order.split(",")
    if set(order) != set(active_arms) or len(order) != len(active_arms):
        raise ValueError(f"worker order must match active arms {active_arms}: {order}")
    for arm in order:
        launches[arm]()
    torch.cuda.synchronize()
    measurements: dict[str, object] = {}
    for arm in order:
        measurements[arm] = _measure(launches[arm], args.warmup, args.repeat)
        if not bool(torch.isfinite(outputs[arm]).all().item()):
            raise RuntimeError(f"non-finite output for {arm}")
    print(json.dumps({
        "scope": "caller_owned_isolated_body_diagnostic",
        "fresh_process_worker": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "T": args.T,
        "chunks": args.T // BT,
        "order": order,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "active_arms": active_arms,
        "workgroup": {arm: 256 for arm in active_arms},
        "arms": measurements,
    }, sort_keys=True))


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z5b,z6g_s,z6g_i,native",
        "native,z6g_i,z6g_s,z5b",
        "z6g_s,z5b,native,z6g_i",
        "z6g_i,native,z5b,z6g_s",
        "native,z5b,z6g_i,z6g_s",
        "z6g_s,z6g_i,native,z5b",
        "z5b,native,z6g_s,z6g_i",
    ]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = orders[session % len(orders)]
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker", "--T", str(args.T),
            "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--order", order,
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
    summary = []
    for arm in ARMS:
        summary.append({
            "arm": arm,
            "session_count": len(medians[arm]),
            "median_of_session_medians_ms": statistics.median(medians[arm]),
            "mean_of_session_medians_ms": statistics.fmean(medians[arm]),
            "session_medians_ms": medians[arm],
        })
    paired = {}
    for arm in ("z6g_s", "z6g_i", "native"):
        values = [(value - base) * 1.0e3 for base, value in zip(medians["z5b"], medians[arm])]
        paired[f"{arm}_minus_z5b_us"] = {
            "samples": values,
            "mean_us": statistics.fmean(values),
            "bootstrap95_us": _bootstrap_ci(values, 2026080700 + args.T + len(arm)),
        }
    for arm in ("z6g_s", "z6g_i"):
        ratios = [value / base for base, value in zip(medians["native"], medians[arm])]
        paired[f"{arm}_over_native"] = ratios
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
            "rotating_orders": orders[:args.sessions],
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
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--order", default="z5b,z6g_s,z6g_i,native")
    parser.add_argument("--only-arm", default="")
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    requested = args.order.split(",")
    if args.worker:
        active_arms = tuple(args.only_arm.split(",")) if args.only_arm else ARMS
        if not active_arms or any(arm not in ARMS for arm in active_arms):
            raise ValueError(f"unknown arm in --only-arm: {active_arms}")
        if len(requested) != len(active_arms) or set(requested) != set(active_arms):
            raise ValueError(f"worker order must match active arms {active_arms}: {requested}")
    elif args.sessions < 1:
        raise ValueError("sessions must be positive")
    if args.worker:
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

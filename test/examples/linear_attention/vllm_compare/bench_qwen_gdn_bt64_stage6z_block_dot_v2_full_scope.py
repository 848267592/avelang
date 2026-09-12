#!/usr/bin/env python3
"""Fresh-process paired body benchmark for BDV2 and frozen diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE)]

from bench_qwen_gdn_bt64_stage6z_z7b_dot_v2 import (  # noqa: E402
    _inputs,
    _measure,
    _native_direct,
    _quantile,
)
from qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
K = 128
V = 128
ARMS = (
    "z5b",
    "bdv2_generic",
    "bdv2_specialized",
    "bdv2_p1_generic",
    "bdv2_p1_specialized",
    "bdv2_p2_specialized",
    "bdv2_p3_specialized",
    "bdv2_p4_specialized",
    "native",
)


def launch_for(arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    if arm == "z5b":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(
            *tensors, output
        )
    if arm == "bdv2_generic":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="generic", planner="legacy"
        )
    if arm == "bdv2_specialized":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="specialized", planner="legacy"
        )
    if arm == "bdv2_p1_generic":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="generic", planner="bdv2_p1_affine"
        )
    if arm == "bdv2_p1_specialized":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="specialized", planner="bdv2_p1_affine"
        )
    if arm == "bdv2_p2_specialized":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="specialized", planner="bdv2_p1_affine",
            preservation="p2_first_class"
        )
    if arm == "bdv2_p3_specialized":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="specialized", planner="bdv2_p1_affine",
            preservation="p3_packed_reuse"
        )
    if arm == "bdv2_p4_specialized":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors, output, lowering="specialized", planner="bdv2_p1_affine",
            preservation="p4_accumulator_reuse"
        )
    return lambda: _native_direct(tensors, output)


def worker(args: argparse.Namespace) -> None:
    tensors = _inputs(args.T, 2026081500 + args.T)
    output = torch.empty_like(tensors[2])
    if args.arm == "native":
        warm = chunk_o.chunk_fwd_o(
            q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4],
            scale=K ** -0.5, chunk_size=BT,
        )
        torch.cuda.synchronize()
        del warm
    launch = launch_for(args.arm, tensors, output)
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
        "planner": "legacy" if args.arm in {"bdv2_generic", "bdv2_specialized"} else (
            "bdv2_p1_affine" if args.arm.startswith("bdv2_p1_") or args.arm in {"bdv2_p2_specialized", "bdv2_p3_specialized", "bdv2_p4_specialized"} else "native"
        ),
        "operand_preservation": (
            "p2_first_class" if args.arm == "bdv2_p2_specialized" else
            "p3_packed_reuse" if args.arm == "bdv2_p3_specialized" else
            "p4_accumulator_reuse" if args.arm == "bdv2_p4_specialized" else
            "none"
        ),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "workgroup": 256,
        "measurement": measured,
    }, sort_keys=True))


def parent(args: argparse.Namespace) -> None:
    orders = [
        ("z5b", "bdv2_generic", "bdv2_specialized", "bdv2_p1_generic", "bdv2_p1_specialized", "bdv2_p2_specialized", "bdv2_p3_specialized", "bdv2_p4_specialized", "native"),
        ("native", "bdv2_p4_specialized", "bdv2_p3_specialized", "bdv2_p2_specialized", "bdv2_p1_specialized", "bdv2_p1_generic", "bdv2_specialized", "bdv2_generic", "z5b"),
        ("bdv2_p1_generic", "z5b", "native", "bdv2_p4_specialized", "bdv2_p3_specialized", "bdv2_p2_specialized", "bdv2_p1_specialized", "bdv2_generic", "bdv2_specialized"),
        ("bdv2_specialized", "native", "z5b", "bdv2_p3_specialized", "bdv2_p4_specialized", "bdv2_p2_specialized", "bdv2_generic", "bdv2_p1_specialized", "bdv2_p1_generic"),
        ("bdv2_p1_specialized", "bdv2_generic", "bdv2_p4_specialized", "bdv2_p3_specialized", "bdv2_p2_specialized", "bdv2_p1_generic", "z5b", "native", "bdv2_specialized"),
        ("bdv2_generic", "bdv2_p1_generic", "bdv2_p3_specialized", "bdv2_p4_specialized", "bdv2_p2_specialized", "bdv2_specialized", "native", "z5b", "bdv2_p1_specialized"),
    ]
    selected_arms = tuple(args.arms)
    if len(selected_arms) < 2:
        raise ValueError("parent benchmark needs at least two arms")
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = tuple(arm for arm in orders[session % len(orders)] if arm in selected_arms)
        for rank, arm in enumerate(order):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--T", str(args.T), "--arm", arm,
                "--warmup", str(args.warmup), "--repeat", str(args.repeat),
            ]
            completed = subprocess.run(
                command, cwd=REPO, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise RuntimeError(f"session {session} arm {arm} failed:\n{completed.stdout}")
            lines = [line for line in completed.stdout.splitlines() if line.strip() and line.lstrip().startswith("{")]
            row = json.loads(lines[-1])
            row.update({"session": session, "order": list(order), "order_rank": rank})
            raw.append(row)
            print(json.dumps(row, sort_keys=True))

    medians = {
        arm: [float(row["measurement"]["hip_ms"]["median"]) for row in raw if row["arm"] == arm]
        for arm in selected_arms
    }
    summary = [
        {
            "arm": arm,
            "session_count": len(medians[arm]),
            "median_of_session_medians_ms": statistics.median(medians[arm]),
            "mean_of_session_medians_ms": statistics.fmean(medians[arm]),
            "session_medians_ms": medians[arm],
        }
        for arm in selected_arms
    ]
    paired = {}
    for arm in selected_arms:
        if arm in {"z5b", "native"}:
            continue
        paired[f"{arm}_minus_z5b_us"] = [
            (value - base) * 1.0e3 for value, base in zip(medians[arm], medians["z5b"])
        ]
        paired[f"{arm}_minus_native_us"] = [
            (value - base) * 1.0e3 for value, base in zip(medians[arm], medians["native"])
        ]
        paired[f"{arm}_over_native"] = [value / base for value, base in zip(medians[arm], medians["native"])]
    if "z5b" in medians and "native" in medians:
        paired["z5b_minus_native_us"] = [
            (value - base) * 1.0e3
            for value, base in zip(medians["z5b"], medians["native"])
        ]
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
            "rotating_orders": [
                [arm for arm in orders[i % len(orders)] if arm in selected_arms]
                for i in range(args.sessions)
            ],
        },
        "T": args.T,
        "raw": raw,
        "summary": summary,
        "paired_differences_us": paired,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
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
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    if args.worker and args.arm is None:
        raise ValueError("--worker requires --arm")
    if not args.worker and args.arm is not None:
        raise ValueError("--arm is only valid with --worker")
    (worker if args.worker else parent)(args)


if __name__ == "__main__":
    main()

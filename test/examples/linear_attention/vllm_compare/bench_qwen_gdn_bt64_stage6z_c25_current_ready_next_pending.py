#!/usr/bin/env python3
"""Fresh-process formal body benchmark for C25 current-ready/next-pending.

Each arm runs in its own Python worker so C21 and C25 cannot share a JIT cache
entry while their late compiler schedule mode differs.  The parent forms seven
balanced sessions by rotating those independent arm workers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128
ARMS = ("z5b", "c21_frozen", "c25", "native_selected")
_C25_ENV = (
    "AVELANG_STAGE6Z_PENDING_PACKET_INFRA",
    "AVELANG_STAGE6Z_CURRENT_READY_NEXT_PENDING",
)


def _set_c25(enabled: bool) -> None:
    if enabled:
        os.environ[_C25_ENV[0]] = "c24"
        os.environ[_C25_ENV[1]] = "c25"
    else:
        for key in _C25_ENV:
            os.environ.pop(key, None)


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _native_direct(tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    q, k, v_new, h, g = tensors
    t = int(q.shape[1])
    chunk_o.chunk_fwd_kernel_o[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)](
        q, k, v_new, h, g, output, None, None, K ** -0.5, T=t, H=HV, Hg=HK, K=K, V=V, BT=BT
    )


def _native_identity() -> dict[str, object]:
    best = getattr(chunk_o.chunk_fwd_kernel_o, "best_config", None)
    if best is None:
        return {"selected": "not exposed after selector seed"}
    return {
        "kwargs": dict(getattr(best, "kwargs", {})),
        "num_warps": int(getattr(best, "num_warps", -1)),
        "num_stages": int(getattr(best, "num_stages", -1)),
        "num_ctas": int(getattr(best, "num_ctas", -1)),
    }


def _make_launch(arm: str, tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> Callable[[], None]:
    if arm == "z5b":
        _set_c25(False)
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, output)
    if arm == "c21_frozen":
        _set_c25(False)
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(*tensors, output)
    if arm == "c25":
        _set_c25(True)
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(*tensors, output)
    _set_c25(False)
    return lambda: _native_direct(tensors, output)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure(launch: Callable[[], None], warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    hips: list[float] = []
    walls: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        launch()
        end.record()
        end.synchronize()
        hips.append(float(start.elapsed_time(end)))
        walls.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    return {
        "samples": repeat,
        "hip_ms": {"median": statistics.median(hips), "p25": _quantile(hips, 0.25), "p75": _quantile(hips, 0.75), "min": min(hips), "max": max(hips)},
        "wall_ms": {"median": statistics.median(walls), "p25": _quantile(walls, 0.25), "p75": _quantile(walls, 0.75)},
    }


def _worker(args: argparse.Namespace) -> None:
    tensors = _inputs(args.T, 2026082800 + args.T)
    output = torch.empty_like(tensors[2])
    # Autotune/module-load native outside timing.  This also exposes the actual
    # selected native configuration for this individual sequence length.
    if args.arm == "native_selected":
        _set_c25(False)
        native = chunk_o.chunk_fwd_o(q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4], scale=K ** -0.5, chunk_size=BT)
        torch.cuda.synchronize()
        del native
    launch = _make_launch(args.arm, tensors, output)
    measurement = _measure(launch, args.warmup, args.repeat)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError(f"non-finite output for {args.arm}")
    print(json.dumps({
        "scope": "caller_owned_isolated_body_diagnostic",
        "fresh_python_process": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "preallocated_output": True,
        "T": args.T,
        "chunks": args.T // BT,
        "arm": args.arm,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measurement": measurement,
        "native_selected_identity": _native_identity() if args.arm == "native_selected" else None,
    }, sort_keys=True))


def _bootstrap_median(values: list[float], samples: int = 20000) -> list[float]:
    if not values:
        return []
    generator = __import__("random").Random(20260825)
    medians = [statistics.median([generator.choice(values) for _ in values]) for _ in range(samples)]
    return [_quantile(medians, 0.025), _quantile(medians, 0.975)]


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z5b,c21_frozen,c25,native_selected",
        "native_selected,c25,c21_frozen,z5b",
        "c21_frozen,z5b,native_selected,c25",
        "c25,native_selected,z5b,c21_frozen",
        "z5b,c25,native_selected,c21_frozen",
        "native_selected,c21_frozen,c25,z5b",
        "c21_frozen,c25,z5b,native_selected",
    ]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        order = orders[session % len(orders)].split(",")
        rows: list[dict[str, object]] = []
        for arm in order:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--arm", arm, "--T", str(args.T), "--warmup", str(args.warmup), "--repeat", str(args.repeat)]
            completed = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"C25 T={args.T} session={session} arm={arm} failed:\n{completed.stdout}")
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            rows.append(json.loads(lines[-1]))
        raw.append({"session": session, "order": order, "arms": {row["arm"]: row for row in rows}})

    values = {
        arm: [float(row["arms"][arm]["measurement"]["hip_ms"]["median"]) for row in raw]
        for arm in ARMS
    }
    summary = {
        arm: {
            "session_medians_ms": samples,
            "median_of_session_medians_ms": statistics.median(samples),
            "mean_of_session_medians_ms": statistics.fmean(samples),
        }
        for arm, samples in values.items()
    }
    paired_c25_z5b = [(c25 - z5b) * 1.0e3 for c25, z5b in zip(values["c25"], values["z5b"])]
    payload = {
        "schema": "qwen.gfx942.stage6z.c25.formal_body.v1",
        "scope": "caller_owned_isolated_body_diagnostic",
        "contract": {"sessions": args.sessions, "fresh_python_process_per_arm": True, "current_stream": True, "cuda_graph_used": False, "preallocated_output": True, "warmup": args.warmup, "repeat": args.repeat, "balanced_orders": orders[:args.sessions]},
        "T": args.T,
        "chunks": args.T // BT,
        "raw_sessions": raw,
        "summary": summary,
        "paired_c25_minus_z5b_us": {"samples": paired_c25_z5b, "median_us": statistics.median(paired_c25_z5b), "bootstrap_ci95_us": _bootstrap_median(paired_c25_z5b)},
        "derived": {
            "c25_speedup_vs_z5b": summary["z5b"]["median_of_session_medians_ms"] / summary["c25"]["median_of_session_medians_ms"],
            "c25_over_native": summary["c25"]["median_of_session_medians_ms"] / summary["native_selected"]["median_of_session_medians_ms"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be a positive multiple of 64")
    if args.worker:
        if args.arm is None:
            raise ValueError("worker requires --arm")
        _worker(args)
    else:
        if args.arm is not None:
            raise ValueError("parent does not accept --arm")
        if args.out is None:
            args.out = LADDER / f"stage6z_c25_formal_body_T{args.T}.json"
        _parent(args)


if __name__ == "__main__":
    main()

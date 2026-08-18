#!/usr/bin/env python3
"""Fresh-process body benchmark for standalone Z5B chunk-o and Triton.

The timed operation is one preallocated caller-owned chunk-o body on the
current HIP stream.  Compilation, Triton autotuning, module loading and
allocation happen before the HIP events.  This is deliberately not a public
full-operator benchmark and does not use CUDA/HIP graph replay.
"""

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


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[6]
sys.path.insert(0, str(HERE / "avelang"))

from chunko_z5b_minimal import (  # noqa: E402
    BT,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _make_inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    q = torch.randn((1, t, H_K, K_DIM), device=device, dtype=torch.float32).to(torch.bfloat16)
    k = torch.randn((1, t, H_K, K_DIM), device=device, dtype=torch.float32).to(torch.bfloat16)
    v_new = torch.randn((1, t, H_V, V_DIM), device=device, dtype=torch.float32).to(torch.bfloat16)
    h = (
        torch.randn((1, t // BT, H_V, V_DIM, K_DIM), device=device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    g = torch.randn((1, t, H_V), device=device, dtype=torch.float32) * 0.01
    return tuple(value.contiguous() for value in (q, k, v_new, h, g))


def _selected_config() -> dict[str, object]:
    result: dict[str, object] = {}
    wrapper = chunk_o.chunk_fwd_kernel_o
    best = getattr(wrapper, "best_config", None)
    # Triton 3.6 wraps Autotuner in a Heuristics object.  In that version the
    # selected Config is recorded in wrapper.fn.cache rather than exposed as
    # wrapper.best_config.  A fresh worker has one cache entry for this shape.
    if best is None:
        tuner = getattr(wrapper, "fn", None)
        cache = getattr(tuner, "cache", {}) if tuner is not None else {}
        if cache:
            _, best = next(reversed(cache.items()))
    if best is not None:
        result = {
            "kwargs": dict(getattr(best, "kwargs", {})),
            "num_warps": int(getattr(best, "num_warps", -1)),
            "num_stages": int(getattr(best, "num_stages", -1)),
            "num_ctas": int(getattr(best, "num_ctas", -1)),
            "workgroup": int(getattr(best, "num_warps", -1)) * 64,
        }
    return result


def _launch_triton(
    tensors: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    q, k, v_new, h, g = tensors
    t = int(q.shape[1])
    nt = t // BT

    def grid(meta):
        return (_cdiv(V_DIM, meta["BV"]), nt, H_V)

    chunk_o.chunk_fwd_kernel_o[grid](
        q,
        k,
        v_new,
        h,
        g,
        output,
        None,
        None,
        K_DIM**-0.5,
        T=t,
        H=H_V,
        Hg=H_K,
        K=K_DIM,
        V=V_DIM,
        BT=BT,
    )


def _launch_z5b(tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    q, k, v_new, h, g = tensors
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(
        q, k, v_new, h, g, output
    )


def _measure_pair(
    launches: dict[str, Callable[[], None]],
    order: tuple[str, str],
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    for _ in range(warmup):
        for name in order:
            launches[name]()
    torch.cuda.synchronize()
    hip_ms = {name: [] for name in launches}
    wall_ms = {name: [] for name in launches}
    for _ in range(repeat):
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter_ns()
            start.record()
            launches[name]()
            end.record()
            end.synchronize()
            hip_ms[name].append(float(start.elapsed_time(end)))
            wall_ms[name].append((time.perf_counter_ns() - wall_start) / 1.0e6)

    def summarize(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "median": statistics.median(values),
            "p10": ordered[max(0, round((len(ordered) - 1) * 0.10))],
            "p90": ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.90))],
        }

    return {
        name: {"hip_ms": summarize(hip_ms[name]), "wall_ms": summarize(wall_ms[name]), "samples": repeat}
        for name in launches
    }


def _worker(t: int, seed: int, warmup: int, repeat: int, session: int) -> dict[str, object]:
    tensors = _make_inputs(t, seed)
    z5b_output = torch.empty_like(tensors[2])
    triton_output = torch.empty_like(tensors[2])

    # This public call selects/compiles the Triton implementation outside the
    # timed body.  The timed arm below uses that exact selected kernel with a
    # caller-owned output tensor.
    chunk_o.chunk_fwd_o(q=tensors[0], k=tensors[1], v=tensors[2], h=tensors[3], g=tensors[4])
    torch.cuda.synchronize()
    selected = _selected_config()

    launches = {
        "z5b": lambda: _launch_z5b(tensors, z5b_output),
        "triton": lambda: _launch_triton(tensors, triton_output),
    }
    for name in ("z5b", "triton"):
        launches[name]()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(z5b_output).all().item()):
        raise RuntimeError("Z5B produced non-finite output")
    if not bool(torch.isfinite(triton_output).all().item()):
        raise RuntimeError("Triton produced non-finite output")

    order = ("z5b", "triton") if session % 2 == 0 else ("triton", "z5b")
    samples = _measure_pair(launches, order, warmup, repeat)
    z5b_ms = float(samples["z5b"]["hip_ms"]["median"])
    triton_ms = float(samples["triton"]["hip_ms"]["median"])
    return {
        "T": t,
        "chunks": t // BT,
        "session": session,
        "order": order,
        "shape": {"B": 1, "H": H_V, "Hg": H_K, "K": K_DIM, "V": V_DIM, "BT": BT},
        "contract": {
            "current_stream": True,
            "cuda_graph_used": False,
            "caller_owned_output": True,
            "warmup": warmup,
            "repeat": repeat,
        },
        "triton_selected_config": selected,
        "arms": samples,
        "diagnostic": {
            "z5b_minus_triton_us": (z5b_ms - triton_ms) * 1000.0,
            "z5b_over_triton": z5b_ms / triton_ms,
            "bf16_output_max_abs_difference": float(
                (z5b_output.float() - triton_output.float()).abs().max().item()
            ),
            "finite": True,
        },
    }


def _slope(rows: list[dict[str, object]], arm: str) -> float:
    points = [
        (float(row["chunks"]), float(row["arms"][arm]["hip_ms"]["median"]))
        for row in rows
    ]
    x_bar = statistics.mean(x for x, _ in points)
    y_bar = statistics.mean(y for _, y in points)
    denominator = sum((x - x_bar) ** 2 for x, _ in points)
    return sum((x - x_bar) * (y - y_bar) for x, y in points) / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048, 8192])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("every T must be >=64 and divisible by 64")
    if args.worker:
        print(json.dumps(_worker(args.T[0], args.seed, args.warmup, args.repeat, args.session)))
        return

    rows: list[dict[str, object]] = []
    for t in args.T:
        for session in range(args.sessions):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--T",
                str(t),
                "--sessions",
                "1",
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--seed",
                str(args.seed + t),
                "--session",
                str(session),
            ]
            output = subprocess.check_output(command, text=True)
            rows.append(json.loads(output.splitlines()[-1]))

    summary = []
    for t in args.T:
        selected = [row for row in rows if row["T"] == t]
        summary.append(
            {
                "T": t,
                "chunks": t // BT,
                "z5b_median_of_session_medians_ms": statistics.median(
                    float(row["arms"]["z5b"]["hip_ms"]["median"]) for row in selected
                ),
                "triton_median_of_session_medians_ms": statistics.median(
                    float(row["arms"]["triton"]["hip_ms"]["median"]) for row in selected
                ),
            }
        )
    result = {
        "benchmark": "standalone_chunk_o_z5b_vs_current_vllm_triton_body",
        "timing": "HIP event on current stream; no graph; preallocated output",
        "sessions": args.sessions,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "summary": summary,
        "slope_us_per_chunk": {
            "z5b": _slope(rows, "z5b") * 1000.0,
            "triton": _slope(rows, "triton") * 1000.0,
        },
        "rows": rows,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()

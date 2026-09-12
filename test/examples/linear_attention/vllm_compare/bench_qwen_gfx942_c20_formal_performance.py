#!/usr/bin/env python3
"""C20 frozen-C19 formal isolated-body benchmark.

This driver is deliberately an analysis harness.  It imports the already
frozen Z5B, P2, C18 and C19 source entry points and never edits compiler or
kernel state.  The native arm first executes the real public ``chunk_fwd_o``
selector outside timing; its subsequent direct call uses the selected cache
entry and a caller-owned output.
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


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = (
    REPO
    / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
    / "codex_qwen_bt64_full_pipeline_stage2"
)
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_c18_full_physical_region import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c18_full_physical_region_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c19_full_physical_region_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_block_dot_v2_full_scope import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into,
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
ARMS = ("z5b", "p2", "c18", "c19", "native_selected")


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    v_new = (
        torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    h = (
        torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32)
        * 0.01
    ).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _config_dict(config: object | None) -> dict[str, object]:
    if config is None:
        return {"available": False}
    return {
        "available": True,
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": int(getattr(config, "num_warps", -1)),
        "num_stages": int(getattr(config, "num_stages", -1)),
        "num_ctas": int(getattr(config, "num_ctas", -1)),
        "maxnreg": getattr(config, "maxnreg", None),
    }


def _native_selected_config() -> dict[str, object]:
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    best = getattr(tuner, "best_config", None)
    cache = getattr(tuner, "cache", {})
    return {
        "best_config": _config_dict(best),
        "cache_size": len(cache),
        "cache_entries": [_config_dict(value) for value in cache.values()],
        "selection_source": "actual public chunk_fwd_o call before timed direct body",
    }


def _native_direct(
    tensors: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    q, k, v_new, h, g = tensors
    t = int(q.shape[1])
    chunk_o.chunk_fwd_kernel_o[
        lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)
    ](
        q,
        k,
        v_new,
        h,
        g,
        output,
        None,
        None,
        K**-0.5,
        T=t,
        H=HV,
        Hg=HK,
        K=K,
        V=V,
        BT=BT,
    )


def _launch_for(
    arm: str,
    tensors: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> Callable[[], None]:
    if arm == "z5b":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(
            *tensors, output
        )
    if arm == "p2":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
            *tensors,
            output,
            lowering="specialized",
            planner="bdv2_p1_affine",
            preservation="p2_first_class",
        )
    if arm == "c18":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c18_full_physical_region_launch_into(
            *tensors,
            output,
            lowering="specialized",
            planner="bdv2_p1_affine",
            preservation="p2_first_class",
        )
    if arm == "c19":
        return lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c19_full_physical_region_launch_into(
            *tensors,
            output,
            lowering="specialized",
            planner="bdv2_p1_affine",
            preservation="p2_first_class",
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
    hip: list[float] = []
    wall: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        fn()
        end.record()
        end.synchronize()
        hip.append(float(start.elapsed_time(end)))
        wall.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    return {
        "samples": repeat,
        "hip_ms": {
            "median": statistics.median(hip),
            "p25": _quantile(hip, 0.25),
            "p75": _quantile(hip, 0.75),
            "min": min(hip),
            "max": max(hip),
        },
        "wall_ms": {
            "median": statistics.median(wall),
            "p25": _quantile(wall, 0.25),
            "p75": _quantile(wall, 0.75),
            "min": min(wall),
            "max": max(wall),
        },
    }


def _worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("C20 requires the gfx942 HIP runtime")
    tensors = _inputs(args.T, 2026081000 + args.T)
    selected_arms = tuple(args.arms)
    outputs = {arm: torch.empty_like(tensors[2]) for arm in selected_arms}

    # The measured native arm is the actual public selection for this shape.
    # It is not forced to WG256; the separate machine audit does that.
    selected_public = chunk_o.chunk_fwd_o(
        q=tensors[0],
        k=tensors[1],
        v=tensors[2],
        h=tensors[3],
        g=tensors[4],
        scale=K**-0.5,
        chunk_size=BT,
    )
    torch.cuda.synchronize()
    if not bool(torch.isfinite(selected_public).all().item()):
        raise RuntimeError("native public selector produced non-finite output")
    native_config = _native_selected_config()
    del selected_public

    launches = {arm: _launch_for(arm, tensors, outputs[arm]) for arm in selected_arms}
    order = args.order.split(",")
    for arm in order:
        launches[arm]()
    torch.cuda.synchronize()
    measurements: dict[str, object] = {}
    for arm in order:
        measurements[arm] = _measure(launches[arm], args.warmup, args.repeat)
        if not bool(torch.isfinite(outputs[arm]).all().item()):
            raise RuntimeError(f"non-finite output for {arm}")

    print(
        json.dumps(
            {
                "scope": "C20_formal_caller_owned_isolated_body",
                "fresh_process_worker": True,
                "cuda_graph_used": False,
                "current_stream": True,
                "T": args.T,
                "chunks": args.T // BT,
                "order": order,
                "selected_arms": list(selected_arms),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "native_selection": native_config,
                "arms": measurements,
            },
            sort_keys=True,
        )
    )


def _parent(args: argparse.Namespace) -> None:
    orders = [
        "z5b,p2,c18,c19,native_selected",
        "native_selected,c19,c18,p2,z5b",
        "p2,z5b,native_selected,c19,c18",
        "c18,native_selected,z5b,c19,p2",
        "c19,c18,p2,native_selected,z5b",
        "native_selected,z5b,c19,p2,c18",
        "c18,p2,c19,z5b,native_selected",
    ]
    raw: list[dict[str, object]] = []
    for session in range(args.sessions):
        selected_arms = tuple(args.arms)
        full_order = orders[session % len(orders)].split(",")
        order = [arm for arm in full_order if arm in selected_arms]
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
            ",".join(order),
            "--arms",
            *selected_arms,
        ]
        completed = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode:
            raise RuntimeError(
                f"C20 session {session} order {order} failed:\n{completed.stdout}"
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        row = json.loads(lines[-1])
        row["session"] = session
        raw.append(row)
        print(json.dumps(row, sort_keys=True))

    medians = {
        arm: [float(row["arms"][arm]["hip_ms"]["median"]) for row in raw]
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
    paired = {
        f"{arm}_minus_z5b_us": [
            (value - baseline) * 1.0e3
            for value, baseline in zip(medians[arm], medians["z5b"])
        ]
        for arm in selected_arms
        if arm != "z5b" and "z5b" in selected_arms
    }
    payload = {
        "scope": "C20_formal_caller_owned_isolated_body",
        "contract": {
            "fresh_process_sessions": True,
            "sessions": args.sessions,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "current_stream": True,
            "cuda_graph_used": False,
            "preallocated_outputs": True,
            "rotating_orders": [
                [
                    arm
                    for arm in orders[i % len(orders)].split(",")
                    if arm in selected_arms
                ]
                for i in range(args.sessions)
            ],
            "selected_arms": list(selected_arms),
            "native_performance_arm": "actual public selector followed by direct body",
            "native_same_shape_machine_arm": "separate fixed WG256 audit only",
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
    parser.add_argument("--order", default=",")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    if args.worker:
        requested = args.order.split(",")
        if set(requested) != set(args.arms) or len(requested) != len(args.arms):
            raise ValueError(f"worker order must contain each selected arm once: {args.arms}")
        _worker(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()

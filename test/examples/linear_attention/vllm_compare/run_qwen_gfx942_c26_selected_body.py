#!/usr/bin/env python3
"""C26 worker: launch one selected chunk-o body without changing any kernel.

The native arm deliberately warms the Triton autotuner before its measured
direct launches.  A surrounding rocprof invocation can therefore attribute
only the final dispatch tail rather than the autotune search population.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = LADDER / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as native_public  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, 2026081200 + t, "random", True)
    torch.manual_seed(2026082200 + t)
    v_new = (torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v_new.contiguous(), h.contiguous(), g.contiguous()


def _autotuner(kernel: object) -> object:
    current = kernel
    while current is not None:
        if hasattr(current, "configs") and hasattr(current, "best_config"):
            return current
        current = getattr(current, "fn", None)
    raise RuntimeError("could not find chunk_fwd_kernel_o autotuner")


def _config() -> dict[str, object]:
    config = getattr(_autotuner(chunk_o.chunk_fwd_kernel_o), "best_config", None)
    if config is None:
        return {"available": False}
    return {
        "available": True,
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": int(getattr(config, "num_warps", -1)),
        "num_stages": int(getattr(config, "num_stages", -1)),
        "num_ctas": int(getattr(config, "num_ctas", -1)),
    }


def _native(tensors: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
    q, k, v_new, h, g = tensors
    t = int(q.shape[1])
    chunk_o.chunk_fwd_kernel_o[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)](
        q, k, v_new, h, g, out, None, None, K ** -0.5, T=t, H=HV, Hg=HK, K=K, V=V, BT=BT
    )


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _measure(launch: object, warmup: int, repeat: int) -> dict[str, object]:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    hip_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        launch()
        end.record()
        end.synchronize()
        hip_ms.append(float(start.elapsed_time(end)))
        wall_ms.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    return {
        "samples": repeat,
        "hip_ms": {
            "median": statistics.median(hip_ms),
            "p25": _quantile(hip_ms, 0.25),
            "p75": _quantile(hip_ms, 0.75),
            "min": min(hip_ms),
            "max": max(hip_ms),
        },
        "wall_ms": {
            "median": statistics.median(wall_ms),
            "p25": _quantile(wall_ms, 0.25),
            "p75": _quantile(wall_ms, 0.75),
        },
    }


def _seed_native_public(t: int) -> None:
    """Select native chunk-o through the same public graph before tail timing."""
    q, k, v, g, beta, initial_state = make_inputs(t, 2026083200 + t, "random", True)
    native_public(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        scale=K ** -0.5,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("z5b", "native_selected"), required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--identity-out", type=Path)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be a positive BT64 multiple")
    tensors = _inputs(args.T)
    out = torch.empty((1, args.T, HV, V), device=tensors[0].device, dtype=torch.bfloat16)
    if args.arm == "z5b":
        launch = lambda: qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache_launch_into(*tensors, out)
        identity: dict[str, object] = {
            "arm": args.arm,
            "grid_ctas_expected": (args.T // BT) * HV * 2,
            "workgroup_expected": 256,
            "BT": BT,
            "BV": 64,
            "BK": 32,
        }
    else:
        # Match the native public-path environment used by the frozen capture
        # harness before consulting the selected Triton specialization.
        patch_rocm_autotune()
        # Seed via the real public API.  The later direct calls then reuse the
        # same selected cache key, while the profiler parser takes only their
        # final repeat tail and excludes public graph/autotune dispatches.
        _seed_native_public(args.T)
        torch.cuda.synchronize()
        launch = lambda: _native(tensors, out)
        config = _config()
        bv = int(dict(config.get("kwargs", {})).get("BV", -1))
        identity = {
            "arm": args.arm,
            "selected_config": config,
            "grid_ctas_expected": (V + bv - 1) // bv * (args.T // BT) * HV if bv > 0 else None,
            "workgroup_expected": int(config.get("num_warps", -1)) * 64,
            "BT": BT,
            "BV": bv,
            "BK": int(dict(config.get("kwargs", {})).get("BK", -1)),
        }
    measurement = _measure(launch, args.warmup, args.repeat) if args.timing else None
    if not args.timing:
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize()
        for _ in range(args.repeat):
            launch()
        torch.cuda.synchronize()
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError(f"non-finite output for {args.arm}")
    if args.identity_out:
        args.identity_out.parent.mkdir(parents=True, exist_ok=True)
        args.identity_out.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "arm": args.arm,
        "T": args.T,
        "chunks": args.T // BT,
        "identity": identity,
        "scope": "caller_owned_isolated_body_diagnostic" if args.timing else "selected_body_capture",
        "fresh_python_process": True,
        "current_stream": True,
        "cuda_graph_used": False,
        "preallocated_output": True,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "measurement": measurement,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

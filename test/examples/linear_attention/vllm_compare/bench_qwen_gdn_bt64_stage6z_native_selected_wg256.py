#!/usr/bin/env python3
"""Preallocated body benchmark for the freshly selected native WG256 config.

The public vLLM call selects the config first.  The selected BK/BV/warps/stages
Config is then pinned in the Autotuner cache before the direct body launch so
the body measurement cannot silently fall back to another specialization.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
K = 128
V = 128
HV = 8
HK = 4


def inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, 2026080600 + t, "random", True)
    torch.manual_seed(2026080700 + t)
    vn = (torch.randn((1, t, HV, V), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


def selected_config() -> object:
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    for config in tuner.configs:
        if (
            dict(getattr(config, "kwargs", {})) == {"BK": 32, "BV": 64}
            and int(getattr(config, "num_warps", -1)) == 4
            and int(getattr(config, "num_stages", -1)) == 2
        ):
            return config
    raise RuntimeError("fresh native tuner has no BK32/BV64/WG256/stage2 config")


def select_and_pin(tensors: tuple[torch.Tensor, ...], t: int) -> dict[str, object]:
    q, k, vn, h, g = tensors
    tuner = chunk_o.chunk_fwd_kernel_o.fn
    config = selected_config()
    # This is the cache key printed by a fresh public selection for this
    # exact shape/dtype contract.  Seeding it before the first profiled call
    # prevents autotune candidate dispatches from entering the PMC stream.
    key = (
        HV,
        K,
        V,
        BT,
        str(q.dtype),
        str(k.dtype),
        str(vn.dtype),
        str(h.dtype),
        str(g.dtype),
        str(vn.dtype),
    )
    tuner.cache.clear()
    tuner.cache[key] = config
    return {
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": int(getattr(config, "num_warps", -1)),
        "num_stages": int(getattr(config, "num_stages", -1)),
        "num_ctas": int(getattr(config, "num_ctas", -1)),
        "cache_key": list(key),
        "pin_reason": "fresh public selected config seeded before profiled direct body",
    }


def launch(tensors: tuple[torch.Tensor, ...], output: torch.Tensor, t: int) -> None:
    q, k, vn, h, g = tensors
    chunk_o.chunk_fwd_kernel_o[lambda meta: ((V + meta["BV"] - 1) // meta["BV"], t // BT, HV)](
        q, k, vn, h, g, output, None, None, K ** -0.5, T=t, H=HV, Hg=HK, K=K, V=V, BT=BT
    )


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    tensors = inputs(args.T)
    output = torch.empty_like(tensors[2])
    config = select_and_pin(tensors, args.T)
    for _ in range(args.warmup):
        launch(tensors, output, args.T)
    torch.cuda.synchronize()
    hip: list[float] = []
    wall: list[float] = []
    for _ in range(args.repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        launch(tensors, output, args.T)
        end.record()
        end.synchronize()
        hip.append(float(start.elapsed_time(end)))
        wall.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    result = {
        "T": args.T,
        "chunks": args.T // BT,
        "fresh_process": True,
        "cuda_graph_used": False,
        "current_stream": True,
        "selected_config": config,
        "workgroup": 256,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "finite": bool(torch.isfinite(output).all().item()),
        "hip_ms": {"median": statistics.median(hip), "p10": quantile(hip, 0.1), "p90": quantile(hip, 0.9)},
        "wall_ms": {"median": statistics.median(wall), "p10": quantile(wall, 0.1), "p90": quantile(wall, 0.9)},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    if not result["finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

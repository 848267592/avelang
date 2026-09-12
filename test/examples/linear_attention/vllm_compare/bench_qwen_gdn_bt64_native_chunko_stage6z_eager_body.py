#!/usr/bin/env python3
"""Measure the selected native vLLM chunk-o in a fresh process.

The direct-body path preallocates ``o`` and invokes the already-selected
``chunk_fwd_kernel_o``.  The public path calls ``chunk_fwd_o`` and therefore
includes its normal output allocation.  Neither path uses CUDA graph replay.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from stage2_runner import make_inputs  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_o  # noqa: E402


BT = 64
HK = 4
HV = 8
K = 128
V = 128


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[max(0, min(len(ordered) - 1, index))]


def _config_summary() -> dict[str, object]:
    tuner = chunk_o.chunk_fwd_kernel_o
    result: dict[str, object] = {}
    best = getattr(tuner, "best_config", None)
    if best is not None:
        result["best_config"] = {
            "kwargs": dict(getattr(best, "kwargs", {})),
            "num_warps": int(getattr(best, "num_warps", -1)),
            "num_stages": int(getattr(best, "num_stages", -1)),
            "num_ctas": int(getattr(best, "num_ctas", -1)),
        }
    cache = getattr(tuner, "cache", None)
    if cache is not None:
        result["cache_type"] = type(cache).__name__
        result["cache_size"] = len(cache)
    return result


def _inputs(t: int) -> tuple[torch.Tensor, ...]:
    q, k, v, g, _, _ = make_inputs(t, 2026080600 + t, "random", True)
    torch.manual_seed(2026080700 + t)
    h = (torch.randn((1, t // BT, HV, V, K), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), v.contiguous(), h.contiguous(), g.contiguous()


def _direct_call(tensors: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
    q, k, v, h, g = tensors
    t = int(q.shape[1])
    nt = t // BT
    kernel = chunk_o.chunk_fwd_kernel_o
    kernel[lambda meta: (triton_cdiv(V, meta["BV"]), nt, HV)](
        q,
        k,
        v,
        h,
        g,
        output,
        None,
        None,
        K ** -0.5,
        T=t,
        H=HV,
        Hg=HK,
        K=K,
        V=V,
        BT=BT,
    )


def triton_cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def _public_call(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    q, k, v, h, g = tensors
    return chunk_o.chunk_fwd_o(q=q, k=k, v=v, h=h, g=g, scale=K ** -0.5, chunk_size=BT)


def _measure(fn, warmup: int, repeat: int) -> dict[str, object]:
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
        result = fn()
        end.record()
        end.synchronize()
        if result is not None:
            del result
        hips.append(float(start.elapsed_time(end)))
        walls.append((time.perf_counter_ns() - wall_start) / 1.0e6)
    return {
        "samples": repeat,
        "hip_ms": {"median": statistics.median(hips), "p10": _quantile(hips, 0.1), "p90": _quantile(hips, 0.9)},
        "wall_ms": {"median": statistics.median(walls), "p10": _quantile(walls, 0.1), "p90": _quantile(walls, 0.9)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("T must be divisible by 64")
    tensors = _inputs(args.T)
    output = torch.empty_like(tensors[2])

    # This first public call performs/loads Triton's autotuning and compilation.
    # It is outside both timing contracts.
    public_result = _public_call(tensors)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(public_result).all().item()):
        raise RuntimeError("native public chunk-o produced non-finite output")
    config = _config_summary()
    direct = _measure(lambda: _direct_call(tensors, output), args.warmup, args.repeat)
    public = _measure(lambda: _public_call(tensors), args.warmup, args.repeat)
    torch.cuda.synchronize()
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("native direct chunk-o produced non-finite output")
    result = {
        "timing_contract": "native_vllm_chunk_o_eager_public_and_preallocated_body",
        "cuda_graph_used": False,
        "T": args.T,
        "chunks": args.T // BT,
        "shape": {"B": 1, "H": HV, "Hg": HK, "K": K, "V": V, "BT": BT},
        "selected_config": config,
        "direct_preallocated_body": direct,
        "public_chunk_fwd_o": public,
        "correctness": {"finite": True},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()

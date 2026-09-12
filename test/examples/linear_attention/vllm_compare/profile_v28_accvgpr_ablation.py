#!/usr/bin/env python3
"""Benchmark/profiling workload for v28 chunk_gdr AccVGPR ablations.

This script intentionally does not define or modify kernels.  It only calls the
existing v28 ablation variants with the fixed Qwen GDN profiling shape.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from bench_qwen_gdn_v28_triton64_geometry import make_inputs
from qwen_gdn_chunked_avelang_v28_triton64_geometry import (
    MODE_FULL_V28,
    MODE_NO_DECAY,
    MODE_NO_H_STORE,
    MODE_NO_VN_STORE,
    MODE_PRED_ONLY,
    MODE_UPDATE_ONLY,
    qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry,
)


VARIANTS = [
    MODE_FULL_V28,
    MODE_NO_H_STORE,
    MODE_NO_VN_STORE,
    MODE_NO_DECAY,
    MODE_PRED_ONLY,
    MODE_UPDATE_ONLY,
]


def time_fn(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def run_variant(variant: str, warmup: int, repeat: int, seed: int, t: int, with_initial_state: bool) -> float:
    k, w, u, g, initial_state = make_inputs(t, with_initial_state, seed=seed + t)

    def fn():
        return qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=64,
            variant=variant,
        )

    # Force compilation/JIT before timing/profiling.
    h, vn, final_state = fn()
    torch.cuda.synchronize()
    checksum = float(h.float().abs().mean() + vn.float().abs().mean() + final_state.float().abs().mean())
    latency_ms = time_fn(fn, warmup=warmup, repeat=repeat)
    print(
        f"result,T={t},variant={variant},with_initial_state={with_initial_state},"
        f"v28_ms={latency_ms:.6f},checksum={checksum:.9g}"
    )
    return latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=280000)
    parser.add_argument("--without-initial-state", dest="with_initial_state", action="store_false")
    parser.set_defaults(with_initial_state=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("target=v28_chunk_gdr_ablation,B=1,T=2048,Hk=4,Hv=8,K=128,V=128,BF16,chunk_size=64")

    variants = VARIANTS if args.variant == "all" else [args.variant]
    rows = [
        (variant, run_variant(variant, args.warmup, args.repeat, args.seed, args.T, args.with_initial_state))
        for variant in variants
    ]

    print("summary_table")
    print("variant,v28_ms")
    for variant, latency_ms in rows:
        print(f"{variant},{latency_ms:.6f}")


if __name__ == "__main__":
    main()

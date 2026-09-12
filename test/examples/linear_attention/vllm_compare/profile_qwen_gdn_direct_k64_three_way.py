#!/usr/bin/env python3
"""Launch one preallocated arm of the direct-K64 three-way diagnostic.

Use this entry under rocprofv3.  It intentionally performs no allocation,
JIT compilation, or module load inside the measured repeat loop.
"""

from __future__ import annotations

import argparse
import json

import torch

import bench_qwen_gdn_direct_k64_three_way as bench
import repro_qwen_gdn_direct_k64_update_current_abi as mfma32
import repro_qwen_gdn_direct_k64_update_mfma16_current_abi as mfma16


def make_call(implementation: str, t: int, seed: int) -> bench.Call:
    k, v_new, g, initial_state = mfma32._make_inputs(t, seed)
    if implementation == "mfma32":
        return bench._raw_avelang_call(
            mfma32, "_qwen_gdn_direct_k64_update_current_abi_kernel", k, v_new, g, initial_state
        )
    if implementation == "mfma16":
        return bench._raw_avelang_call(
            mfma16, "_qwen_gdn_direct_k64_update_mfma16_current_abi_kernel", k, v_new, g, initial_state
        )
    if implementation == "native":
        return bench._native_call(k, v_new, g, initial_state)
    raise ValueError(f"unsupported implementation {implementation}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["mfma32", "mfma16", "native"], required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    if args.T <= 0 or args.T % 64:
        raise ValueError("T must be a positive multiple of 64")
    call = make_call(args.implementation, args.T, args.seed)
    for _ in range(args.warmup):
        call.launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(args.repeat):
        start.record()
        call.launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    print(json.dumps({
        "implementation": args.implementation,
        "T": args.T,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "median_ms": sorted(samples)[len(samples) // 2],
        "h_finite": bool(torch.isfinite(call.h.float()).all().item()),
        "final_state_finite": bool(torch.isfinite(call.final_state).all().item()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()

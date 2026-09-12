#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from qwen_gdn_chunked_avelang_v28_triton64_geometry import (
    MODE_PRED_ONLY,
    qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry,
)
from qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only import (
    qwen_gdn_pred_only_avelang_v29_mfma32,
    qwen_gdn_pred_only_torch_reference,
)


def _make_inputs(t: int, seed: int):
    torch.manual_seed(seed)
    w = (torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    k = torch.randn((1, t, 4, 128), device="cuda", dtype=torch.bfloat16).contiguous()
    g = torch.randn((1, t, 8), device="cuda", dtype=torch.float32).contiguous()
    return k, w, u, g, initial_state


def _time_fn(fn: Callable[[], object], warmup: int, repeat: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=290000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("target=v29_pred_only,B=1,Hv=8,K=128,V=128,BT=64,BV=32,WG=128,num_warps=2")

    print("summary_table")
    print("T,v29_ms,v28_pred_only_ms,max_abs_vs_ref,mean_abs_vs_ref,max_abs_vs_v28,speedup_v29_vs_v28")

    for t in args.T:
        k, w, u, g, initial_state = _make_inputs(t, seed=args.seed + t)

        def v29_fn():
            return qwen_gdn_pred_only_avelang_v29_mfma32(w, u, initial_state)

        def v28_fn():
            return qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry(
                k,
                w,
                u,
                g,
                initial_state=initial_state,
                chunk_size=64,
                variant=MODE_PRED_ONLY,
            )[1]

        vn29 = v29_fn()
        vn28 = v28_fn()
        ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)

        err_ref = (vn29.float() - ref.float()).abs()
        err_v28 = (vn29.float() - vn28.float()).abs()
        max_abs_ref = err_ref.max().item()
        mean_abs_ref = err_ref.mean().item()
        max_abs_v28 = err_v28.max().item()

        v29_ms = _time_fn(v29_fn, args.warmup, args.repeat)
        v28_ms = _time_fn(v28_fn, args.warmup, args.repeat)
        speedup = v28_ms / v29_ms

        print(
            f"{t},{v29_ms:.6f},{v28_ms:.6f},{max_abs_ref:.8e},{mean_abs_ref:.8e},{max_abs_v28:.8e},{speedup:.4f}"
        )


if __name__ == "__main__":
    main()


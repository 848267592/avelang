#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

from qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only import (
    qwen_gdn_pred_only_avelang_v29_mfma32 as qwen_gdn_pred_only_original_v29,
    qwen_gdn_pred_only_torch_reference,
)
from qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best import (
    MODE_BASELINE,
    MODE_GROUPED_V4_EPILOGUE,
    qwen_gdn_pred_only_avelang_v29_mfma32_grouped_v4_best,
)


def _make_pred_inputs(t: int, seed: int):
    torch.manual_seed(seed)
    w = (torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    return w, u, initial_state


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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=293000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("target=v29_smallfix_grouped_v4_pred_only,B=1,Hv=8,K=128,V=128,BT=64,BV=32,WG=128")
    print("summary_table")
    print("T,variant,latency_ms,max_abs,mean_abs,speedup_vs_original")

    for t in args.T:
        w, u, initial_state = _make_pred_inputs(t, args.seed + t)
        ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)

        original_ms = _time_fn(lambda: qwen_gdn_pred_only_original_v29(w, u, initial_state), args.warmup, args.repeat)
        original_out = qwen_gdn_pred_only_original_v29(w, u, initial_state)
        err = (original_out.float() - ref.float()).abs()
        print(f"{t},original_v29,{original_ms:.6f},{err.max().item():.8e},{err.mean().item():.8e},1.0000")

        for variant in (MODE_BASELINE, MODE_GROUPED_V4_EPILOGUE):
            fn = lambda v=variant: qwen_gdn_pred_only_avelang_v29_mfma32_grouped_v4_best(w, u, initial_state, variant=v)
            out = fn()
            torch.cuda.synchronize()
            err = (out.float() - ref.float()).abs()
            ms = _time_fn(fn, args.warmup, args.repeat)
            print(f"{t},{variant},{ms:.6f},{err.max().item():.8e},{err.mean().item():.8e},{original_ms / ms:.4f}")


if __name__ == "__main__":
    main()

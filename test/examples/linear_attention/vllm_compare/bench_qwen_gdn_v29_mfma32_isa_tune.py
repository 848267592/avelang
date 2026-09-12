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
from qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_isa_tune import (
    MODE_BASELINE,
    MODE_CONSTANT_STATE,
    MODE_NO_ACC_UNPACK,
    MODE_OPTIMIZED,
    MODE_PRECOMPUTED_MAPPING,
    MODE_REDUCE_ONLY_NO_VN_STORE,
    MODE_UNPACK_ONLY_NO_REDUCE,
    qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune,
    qwen_gdn_pred_only_constant_state_reference,
)


DEFAULT_VARIANTS = [
    MODE_BASELINE,
    MODE_NO_ACC_UNPACK,
    MODE_UNPACK_ONLY_NO_REDUCE,
    MODE_REDUCE_ONLY_NO_VN_STORE,
    MODE_CONSTANT_STATE,
    MODE_PRECOMPUTED_MAPPING,
    MODE_OPTIMIZED,
]

CORRECTNESS_VARIANTS = {MODE_BASELINE, MODE_CONSTANT_STATE, MODE_PRECOMPUTED_MAPPING, MODE_OPTIMIZED}


def _make_inputs(t: int, seed: int):
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


def _error_for_variant(variant: str, out: torch.Tensor, w: torch.Tensor, u: torch.Tensor, initial_state: torch.Tensor):
    if variant not in CORRECTNESS_VARIANTS:
        return None, None
    if variant == MODE_CONSTANT_STATE:
        ref = qwen_gdn_pred_only_constant_state_reference(w, u)
    else:
        ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)
    err = (out.float() - ref.float()).abs()
    return err.max().item(), err.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=291000)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--skip-original", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("target=v29_isa_tune_pred_only,B=1,Hv=8,K=128,V=128,BT=64,BV=32,WG=128,num_warps=2")
    print("summary_table")
    print("T,variant,latency_ms,max_abs,mean_abs,speedup_vs_original")

    for t in args.T:
        w, u, initial_state = _make_inputs(t, seed=args.seed + t)

        original_ms = float("nan")
        if not args.skip_original:
            original_ms = _time_fn(
                lambda: qwen_gdn_pred_only_original_v29(w, u, initial_state),
                args.warmup,
                args.repeat,
            )
            original_out = qwen_gdn_pred_only_original_v29(w, u, initial_state)
            max_abs, mean_abs = _error_for_variant(MODE_BASELINE, original_out, w, u, initial_state)
            print(f"{t},original_v29,{original_ms:.6f},{max_abs:.8e},{mean_abs:.8e},1.0000")

        for variant in args.variants:
            def run_variant(v=variant):
                return qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune(w, u, initial_state, variant=v)

            out = run_variant()
            torch.cuda.synchronize()
            max_abs, mean_abs = _error_for_variant(variant, out, w, u, initial_state)
            latency_ms = _time_fn(run_variant, args.warmup, args.repeat)
            speedup = original_ms / latency_ms if original_ms == original_ms else float("nan")
            max_abs_text = "NA" if max_abs is None else f"{max_abs:.8e}"
            mean_abs_text = "NA" if mean_abs is None else f"{mean_abs:.8e}"
            print(f"{t},{variant},{latency_ms:.6f},{max_abs_text},{mean_abs_text},{speedup:.4f}")


if __name__ == "__main__":
    main()

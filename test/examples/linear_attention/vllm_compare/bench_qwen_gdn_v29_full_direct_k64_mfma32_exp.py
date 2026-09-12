#!/usr/bin/env python3
"""Matched body timing for compact-K MFMA16 versus direct-K64 MFMA32.

This is a resource experiment, not a promotion benchmark.  Both raw kernels
receive preallocated outputs and identical current-ABI BF16 inputs.
"""

from __future__ import annotations

import argparse
import statistics

import torch

import qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_current_abi_exp as compact
import qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_direct_k64_mfma32_exp as direct


def _median_ms(launch, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeat):
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _make_raw_launch(module, kernel_name: str, tensors, t: int):
    k, w, u, decay, g_last, initial = tensors
    chunks = t // direct.BT
    h = torch.empty((1, chunks, 8, 128, 128), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    kernel = getattr(module, kernel_name)

    def launch() -> None:
        kernel[lambda: ((direct.GRID_SIZE, 1, 1), (direct.WORKGROUP, 1, 1))](
            k,
            w,
            u,
            decay,
            g_last,
            initial,
            h,
            v_new,
            final_state,
            t,
            chunks,
            True,
            num_warps=2,
        )

    return launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP device")
    for t in args.T:
        tensors = direct._make_inputs(t)
        compact_launch = _make_raw_launch(
            compact,
            "_qwen_gdn_fused_chunk_gdr_full_current_abi_exp_bf16_kernel_v29_mfma32",
            tensors,
            t,
        )
        direct_launch = _make_raw_launch(
            direct,
            "_qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32",
            tensors,
            t,
        )
        compact_ms = _median_ms(compact_launch, warmup=args.warmup, repeat=args.repeat)
        direct_ms = _median_ms(direct_launch, warmup=args.warmup, repeat=args.repeat)
        print(
            f"T={t} compact_mfma16_ms={compact_ms:.6f} "
            f"direct_k64_mfma32_ms={direct_ms:.6f} speedup={compact_ms / direct_ms:.4f}x"
        )


if __name__ == "__main__":
    main()

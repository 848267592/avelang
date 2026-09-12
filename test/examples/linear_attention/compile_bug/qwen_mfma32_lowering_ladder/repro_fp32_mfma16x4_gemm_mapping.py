"""Deterministic mapping probe for FP32 16x16x4 MFMA on gfx942.

The Stage 5B solve needs a verified source mapping before it can use the
new high-level FP32 MFMA intrinsic for 16x16 block products.  This probe
computes one FP32 16x16 GEMM through four 16x16x4 MFMA operations and checks
the lane/accumulator mapping against ``torch.matmul``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import avelang
import avelang.language as al


@avelang.jit
def _fp32_mfma16x4_gemm_mapping_kernel(
    a_ptr: al.Pointer(al.f32),
    b_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
):
    a = al.make_tensor(a_ptr, al.f32, al.make_layout((16, 16), (16, 1)))
    b = al.make_tensor(b_ptr, al.f32, al.make_layout((16, 16), (16, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((16, 16), (16, 1)))

    a_frag = al.view(a, al.f32, al.make_layout((16, 16, 1), (16, 1, 1)))
    b_frag = al.view(b, al.f32, al.make_layout((16, 16, 1), (16, 1, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    acc = al.full((4,), 0.0, al.f32)

    # A lane owns A[row=lane_col, k=4*lane_group+piece]; B owns
    # B[k=4*lane_group+piece, col=lane_col].  Each result lane owns four
    # consecutive rows at one column.
    for piece in al.range(4):
        k_idx = lane_group + piece * 4
        acc = al.amdgpu.mfma_16x16x4_f32_f32(a_frag[lane_col, k_idx], b_frag[k_idx, lane_col], acc)

    for row_in_group in al.range(4):
        out[lane_group * 4 + row_in_group, lane_col] = acc[row_in_group]


def _run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((16, 16), dtype=torch.float32, device=a.device)
    _fp32_mfma16x4_gemm_mapping_kernel[lambda: ((1, 1, 1), (64, 1, 1))](a, b, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("requires a CUDA/HIP GPU")
    torch.manual_seed(args.seed)
    a = torch.randn((16, 16), dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn((16, 16), dtype=torch.float32, device="cuda").contiguous()
    expected = a @ b

    actual = _run(a, b)
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        _run(a, b)
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        _run(a, b)
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1e3)

    diff = (actual - expected).abs()
    result = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
        "median_ms": statistics.median(samples_ms),
        "mapping": "C[row=4*(lane>>4)+acc_i, col=lane&15]",
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}={value}")
    if result["max_abs"] > 2e-5:
        raise AssertionError(result)


if __name__ == "__main__":
    main()

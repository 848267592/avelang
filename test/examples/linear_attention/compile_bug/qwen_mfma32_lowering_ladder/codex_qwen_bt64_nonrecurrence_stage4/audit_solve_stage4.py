#!/usr/bin/env python3
"""Same-input BT64 solve audit; no new solve kernel is introduced."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_kkt_bt64_mfma_v2_s0  # noqa: E402
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402


def time_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--out", type=Path, default=HERE / "solve_audit.csv")
    args = parser.parse_args()
    patch_rocm_autotune()
    rows = []
    for t in args.T:
        _, k, _, g, beta, _ = make_inputs(t, 20260715 + t, "random", False)
        g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
        a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
        ref_fp32 = qwen_gdn_solve_avelang_v18_bt64_layout(a)
        triton_fp32 = solve_tril(A=a, output_dtype=torch.float32)
        triton_bf16 = solve_tril(A=a, output_dtype=torch.bfloat16)
        error_fp32 = (ref_fp32 - triton_fp32.float()).abs()
        error_bf16 = (ref_fp32 - triton_bf16.float()).abs()
        row = {
            "T": t,
            "v18_fp32_ms": time_ms(lambda: qwen_gdn_solve_avelang_v18_bt64_layout(a), args.warmup, args.repeat),
            "vllm_fp32_ms": time_ms(lambda: solve_tril(A=a, output_dtype=torch.float32), args.warmup, args.repeat),
            "vllm_bf16_ms": time_ms(lambda: solve_tril(A=a, output_dtype=torch.bfloat16), args.warmup, args.repeat),
            "vllm_fp32_max_abs": error_fp32.max().item(),
            "vllm_bf16_max_abs": error_bf16.max().item(),
        }
        rows.append(row)
        print(row, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

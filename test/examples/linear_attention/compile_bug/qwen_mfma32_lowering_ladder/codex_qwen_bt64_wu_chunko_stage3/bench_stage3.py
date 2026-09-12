#!/usr/bin/env python3
"""Stage and full HIP-event benchmarks for the opt-in native-BT64 Stage 3 graph."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0_preallocated  # noqa: E402
from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1,
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import qwen_gdn_chunked_avelang_v24_kkt_mfma_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT, K, HV, V = 64, 128, 8, 128


def time_ms(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": samples[(len(samples) - 1) // 10],
        "p90_ms": samples[(len(samples) - 1) * 9 // 10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--session", default="session1")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--vllm-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for t in args.T:
        if args.verbose:
            print(f"prepare T={t}", flush=True)
        q, k, v, g, beta, h0 = make_inputs(t, 20260713 + t, "random", True)
        stages = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
        h = torch.empty((1, t // BT, HV, V, K), device="cuda", dtype=torch.bfloat16)
        vn = torch.empty_like(stages["u"])
        final_state = torch.empty((1, HV, V, K), device="cuda", dtype=torch.float32)
        stage_fns = {
            "preprocessing_cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT),
            "KKT": lambda: qwen_gdn_kkt_avelang_v6_standalone(k, stages["g_cumsum"], beta, chunk_size=BT),
            "solve": lambda: qwen_gdn_solve_avelang_v18_bt64_layout(stages["a"], chunk_size=BT),
            "w_u_native": lambda: qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, stages["g_cumsum"], beta, stages["a_solved"]),
            "asm_recurrence_preallocated": lambda: qwen_gdn_bt64_gfx942_asm_v0_preallocated(
                k, stages["w"], stages["u"], stages["g_cumsum"], stages["initial_state"], h, vn, final_state
            ),
            "chunk_o_native": lambda: qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(
                q, k, stages["v_new"], stages["h_bf16"], stages["g_cumsum"]
            ),
        }
        if not args.vllm_only and not args.full_only:
            for stage, fn in stage_fns.items():
                if args.verbose:
                    print(f"stage T={t} {stage}", flush=True)
                rows.append({"T": t, "implementation": "candidate_bt64_native_wu_o_v1", "scope": "stage", "stage": stage, **time_ms(fn, args.warmup, args.repeat)})
        calls = (
            {"vllm_full": lambda: vllm_full(q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5, initial_state=h0, output_final_state=True, head_first=False, use_qk_l2norm_in_kernel=False)}
            if args.vllm_only
            else {
                "candidate_bt64_native_wu_o_v1_cached_allocator": lambda: qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1(
                    q, k, v, g, beta, initial_state=h0, output_final_state=True
                ),
                "v24_bt16": lambda: qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(q, k, v, g, beta, initial_state=h0),
            }
        )
        for implementation, fn in calls.items():
            if args.verbose:
                print(f"full T={t} {implementation}", flush=True)
            rows.append({"T": t, "implementation": implementation, "scope": "full", "stage": "total", **time_ms(fn, args.warmup, args.repeat)})
        print(json.dumps([row for row in rows if row["T"] == t], indent=2), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["T", "implementation", "scope", "stage", "median_ms", "p10_ms", "p90_ms"]
    with (args.out / f"full_pipeline_benchmark_{args.session}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / f"stage_breakdown_{args.session}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in rows if row["scope"] == "stage")


if __name__ == "__main__":
    main()

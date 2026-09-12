#!/usr/bin/env python3
"""Incremental Stage 4 HIP-event stage/full benchmark."""

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

from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0_preallocated  # noqa: E402
from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (  # noqa: E402
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1,
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_full_bt64_stage4_all_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_s0,
    qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402


def time_ms(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
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
    parser.add_argument("--session", default="a")
    parser.add_argument("--out", type=Path, default=HERE / "full_pipeline")
    args = parser.parse_args()
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for t in args.T:
        q, k, v, g, beta, h0 = make_inputs(t, 20260715 + t, "random", True)
        stage4 = qwen_gdn_full_bt64_stage4_all_s0_stages(q, k, v, g, beta, initial_state=h0)
        h = torch.empty_like(stage4["h_bf16"])
        vn = torch.empty_like(stage4["v_new"])
        final_state = torch.empty_like(stage4["final_state"])
        stage_fns = {
            "cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64),
            "KKT": lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, stage4["g_cumsum"], beta),
            "solve": lambda: qwen_gdn_solve_avelang_v18_bt64_layout(stage4["a"]),
            "W_U": lambda: qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, stage4["g_cumsum"], beta, stage4["a_solved"]),
            "asm_recurrence_preallocated": lambda: qwen_gdn_bt64_gfx942_asm_v0_preallocated(
                k, stage4["w"], stage4["u"], stage4["g_cumsum"], stage4["initial_state"], h, vn, final_state
            ),
            "chunk_o": lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(
                q, k, stage4["v_new"], stage4["h_bf16"], stage4["g_cumsum"]
            ),
        }
        for stage, fn in stage_fns.items():
            rows.append({"T": t, "variant": "stage4_all_s0", "scope": "stage", "stage": stage, **time_ms(fn, args.warmup, args.repeat)})
        full_fns = {
            "stage3": lambda: qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1(
                q, k, v, g, beta, initial_state=h0, output_final_state=True
            ),
            "stage4_kkt": lambda: qwen_gdn_full_bt64_stage4_kkt_s0(
                q, k, v, g, beta, initial_state=h0, output_final_state=True
            ),
            "stage4_kkt_wu": lambda: qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages(
                q, k, v, g, beta, initial_state=h0
            ),
            "stage4_all": lambda: qwen_gdn_full_bt64_stage4_all_s0(
                q, k, v, g, beta, initial_state=h0, output_final_state=True
            ),
        }
        for variant, fn in full_fns.items():
            rows.append({"T": t, "variant": variant, "scope": "full", "stage": "total", **time_ms(fn, args.warmup, args.repeat)})
        for row in rows:
            if row["T"] == t:
                print(row, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"stage4_{args.session}.csv"
    fields = ["T", "variant", "scope", "stage", "median_ms", "p10_ms", "p90_ms"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

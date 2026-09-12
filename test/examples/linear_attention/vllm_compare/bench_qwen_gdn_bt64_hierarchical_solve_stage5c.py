#!/usr/bin/env python3
"""HIP-event Stage 5C A/B timing for the highest-level BT64 pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path.insert(0, str(STAGE2))

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    qwen_gdn_full_bt64_stage4_all_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1
from stage2_runner import make_inputs, patch_rocm_autotune


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
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for t in args.T:
        q, k, v, g, beta, h0 = make_inputs(t, 20260746 + t, "random", True)
        baseline = qwen_gdn_full_bt64_stage4_all_s0_stages(q, k, v, g, beta, initial_state=h0)
        candidate = qwen_gdn_full_bt64_stage4_all_s0_stages(
            q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
        )
        solve_error = float((candidate["a_solved"] - baseline["a_solved"]).abs().max().item())
        output_error = float((candidate["output"].float() - baseline["output"].float()).abs().max().item())
        state_error = float((candidate["final_state"] - baseline["final_state"]).abs().max().item())
        a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, baseline["g_cumsum"], beta)
        cases = {
            "solve_v18": lambda: qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=64),
            "solve_hierarchical_fp32_v1": lambda: qwen_gdn_solve_bt64_hierarchical_fp32_v1(a),
            "full_stage4_v18": lambda: qwen_gdn_full_bt64_stage4_all_s0(
                q, k, v, g, beta, initial_state=h0, output_final_state=True, solve_impl="v18"
            ),
            "full_stage4_hierarchical_fp32_v1": lambda: qwen_gdn_full_bt64_stage4_all_s0(
                q, k, v, g, beta, initial_state=h0, output_final_state=True, solve_impl="hierarchical_fp32_v1"
            ),
        }
        results = {name: time_ms(fn, args.warmup, args.repeat) for name, fn in cases.items()}
        solve_speedup = results["solve_v18"]["median_ms"] / results["solve_hierarchical_fp32_v1"]["median_ms"]
        full_speedup = results["full_stage4_v18"]["median_ms"] / results["full_stage4_hierarchical_fp32_v1"]["median_ms"]
        row = {
            "T": t,
            "solve_speedup": solve_speedup,
            "full_speedup": full_speedup,
            "solve_max_abs_vs_v18": solve_error,
            "output_max_abs_vs_v18": output_error,
            "state_max_abs_vs_v18": state_error,
            "timing": results,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

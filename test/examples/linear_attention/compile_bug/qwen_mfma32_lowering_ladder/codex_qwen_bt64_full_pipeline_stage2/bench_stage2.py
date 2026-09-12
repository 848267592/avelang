#!/usr/bin/env python3
"""Stage and end-to-end timings for the BT64 experimental pipeline."""

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
sys.path.insert(0, str(COMPARE))

from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0_preallocated  # noqa: E402
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import qwen_gdn_chunked_avelang_v24_kkt_mfma_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import (  # noqa: E402
    qwen_gdn_full_bt64_gfx942_asm_v0,
    qwen_gdn_full_bt64_gfx942_asm_v0_stages,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


BT, K, HV, V = 64, 128, 8, 128


def time_ms(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeat):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return {"median_ms": statistics.median(samples), "p10_ms": samples[(len(samples) - 1) // 10], "p90_ms": samples[(len(samples) - 1) * 9 // 10]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--session", type=str, default="session1")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--full-only", action="store_true", help="skip per-stage timing for an independent full-forward session")
    parser.add_argument("--vllm-only", action="store_true", help="run vLLM in an isolated process/session")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for t in args.T:
        if args.verbose:
            print(f"prepare T={t}", flush=True)
        q, k, v, g, beta, h0 = make_inputs(t, 20260900 + t, "random", True)
        # Materialize once before timings.  The asm recurrence gets its own
        # preallocated timing; generic historical wrappers still allocate their
        # public outputs, so full timing is explicitly labelled cached-allocator.
        st = qwen_gdn_full_bt64_gfx942_asm_v0_stages(q, k, v, g, beta, initial_state=h0)
        h = torch.empty((1, t // BT, HV, V, K), device="cuda", dtype=torch.bfloat16)
        vn = torch.empty_like(st["u"])
        ht = torch.empty((1, HV, V, K), device="cuda", dtype=torch.float32)
        stage_fns = {
            "preprocessing_cumsum": lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT),
            "KKT": lambda: qwen_gdn_kkt_avelang_v6_standalone(k, st["g_cumsum"], beta, chunk_size=BT),
            "solve": lambda: qwen_gdn_solve_avelang_v18_bt64_layout(st["a"], chunk_size=BT),
            "w_u": lambda: qwen_gdn_w_u_avelang_v6_standalone(k, v, st["g_cumsum"], beta, st["a_solved"], chunk_size=BT),
            "asm_recurrence_preallocated": lambda: qwen_gdn_bt64_gfx942_asm_v0_preallocated(k, st["w"], st["u"], st["g_cumsum"], st["initial_state"], h, vn, ht),
            "chunk_o": lambda: qwen_gdn_chunk_o_avelang_v6_standalone(q, k, st["v_new"], st["h_bf16"].float().contiguous(), st["g_cumsum"], chunk_size=BT),
        }
        if not args.vllm_only and not args.full_only:
            for stage, fn in stage_fns.items():
                if args.verbose:
                    print(f"stage T={t} {stage}", flush=True)
                rows.append({"T": t, "implementation": "candidate_bt64_asm_v0", "scope": "stage", "stage": stage, **time_ms(fn, args.warmup, args.repeat)})
        calls = {"vllm_full": lambda: vllm_full(q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5, initial_state=h0, output_final_state=True, head_first=False, use_qk_l2norm_in_kernel=False)} if args.vllm_only else {
            "candidate_bt64_asm_v0_cached_allocator": lambda: qwen_gdn_full_bt64_gfx942_asm_v0(q, k, v, g, beta, initial_state=h0, output_final_state=True),
            "v24_bt16": lambda: qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(q, k, v, g, beta, initial_state=h0),
        }
        for implementation, fn in calls.items():
            if args.verbose:
                print(f"full T={t} {implementation}", flush=True)
            rows.append({"T": t, "implementation": implementation, "scope": "full", "stage": "total", **time_ms(fn, args.warmup, args.repeat)})
        print(json.dumps([row for row in rows if row["T"] == t], indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"full_pipeline_benchmark_{args.session}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["T", "implementation", "scope", "stage", "median_ms", "p10_ms", "p90_ms"])
        writer.writeheader(); writer.writerows(rows)
    stage_rows = [row for row in rows if row["scope"] == "stage"]
    with (args.out / f"stage_breakdown_{args.session}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["T", "implementation", "scope", "stage", "median_ms", "p10_ms", "p90_ms"])
        writer.writeheader(); writer.writerows(stage_rows)
    (args.out / "benchmark_methodology.md").write_text(
        "# Benchmark Methodology\n\n"
        "Every row is a HIP-event median with configured warmup/repeat and a current stream. "
        "`asm_recurrence_preallocated` excludes output allocation/module loading. "
        "The generic historical Avelang wrappers allocate public output tensors internally; their full row is therefore labelled `cached_allocator`, not a production allocation-free claim.\n"
    )


if __name__ == "__main__":
    main()

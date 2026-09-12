#!/usr/bin/env python3
"""Diagnostic profiler entry point: each invocation uses one public full API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_fused_wu_eager_stage6t import (  # noqa: E402
    qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager,
    qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("stage6s", "f0", "f1", "vllm"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(args.T, 2026071700 + args.T, "random", True)
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    functions = {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f0": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }
    fn = functions[args.implementation]
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        result = fn()
    torch.cuda.synchronize()
    print(f"profiled_public_api={args.implementation} T={args.T} result_is_none={result is None}")


if __name__ == "__main__":
    main()

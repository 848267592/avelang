#!/usr/bin/env python3
"""Diagnostic profiler entry through complete Stage 6U eager public APIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager  # noqa: E402
from qwen_gdn_bt64_fused_wu_eager_stage6t import qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("f1", "u1", "vllm"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(args.T, 2026072900 + args.T, "random", True)
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    calls = {
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "u1": lambda: qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
            output_final_state=True, scale=128 ** -0.5, head_first=False,
            use_qk_l2norm_in_kernel=False,
        ),
    }
    fn = calls[args.implementation]
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        result = fn()
    torch.cuda.synchronize()
    print(f"profiled_public_api={args.implementation} T={args.T} result_is_none={result is None}")


if __name__ == "__main__":
    main()

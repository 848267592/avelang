#!/usr/bin/env python3
"""Direct-only Stage 6S trace driver; never used as a latency benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path[:0] = [
    str(REPO / "test/examples/linear_attention/vllm_compare"),
    str(REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"),
]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_current_asm,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=("a", "b", "c"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replay", type=int, default=3)
    args = parser.parse_args()
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(args.T, 20260717 + args.T, "random", True)
    if args.graph == "a":
        call = qwen_gdn_full_bt64_stage6s_current_asm
    elif args.graph == "b":
        call = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge
    else:
        call = lambda q, k, v, g, beta, initial_state, output_final_state: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
            output_final_state=output_final_state, scale=128 ** -0.5,
            head_first=False, use_qk_l2norm_in_kernel=False,
        )
    for _ in range(args.warmup):
        call(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()
    for _ in range(args.replay):
        call(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare each Stage 4 increment against the frozen vLLM golden."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (  # noqa: E402
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_full_bt64_stage4_all_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages,
)
from stage2_runner import make_inputs, metrics, patch_rocm_autotune, vllm_stages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--mode", default="random")
    parser.add_argument("--initial-state", action="store_true")
    args = parser.parse_args()
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(args.T, args.seed, args.mode, args.initial_state)
    golden = vllm_stages(q, k, v, g, beta, h0)
    variants = {
        "stage3": qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages,
        "kkt_s0": qwen_gdn_full_bt64_stage4_kkt_s0_stages,
        "kkt_wu_s0": qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages,
        "kkt_wu_s1": qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages,
        "all_s0": qwen_gdn_full_bt64_stage4_all_s0_stages,
    }
    for name, fn in variants.items():
        actual = fn(q, k, v, g, beta, initial_state=h0)
        print(name)
        for stage, golden_name in (
            ("a", "a"),
            ("a_solved", "a_solved"),
            ("w", "w"),
            ("u", "u"),
            ("h_bf16", "h_bf16"),
            ("v_new", "v_new"),
            ("final_state", "public_final_state"),
            ("output", "public_output"),
        ):
            result = metrics(actual.get(stage), golden.get(golden_name))
            print(f"  {stage:12s} max_abs={result['max_abs']:.9g} mean_abs={result['mean_abs']:.9g}")


if __name__ == "__main__":
    main()

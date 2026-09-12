#!/usr/bin/env python3
"""rocprof entry point for isolated Stage 6X KKT/KKT-to-solve bodies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_kkt_solve_handoff_stage6x import (  # noqa: E402
    qwen_gdn_kkt_bt64_one_cta_stage6x_x1,
    qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_kkt_bt64_mfma_v2_s0  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("current", "x1", "x2"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    _, k, _, g, beta, _ = make_inputs(args.T, 2026072600 + args.T, "random", True)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    implementations = {
        "current": qwen_gdn_kkt_bt64_mfma_v2_s0,
        "x1": qwen_gdn_kkt_bt64_one_cta_stage6x_x1,
        "x2": qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2,
    }
    fn = implementations[args.implementation]
    for _ in range(args.warmup):
        fn(k, g_cumsum, beta)
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        result = fn(k, g_cumsum, beta)
    torch.cuda.synchronize()
    print(f"profiled_kkt={args.implementation} T={args.T} shape={tuple(result.shape)}")


if __name__ == "__main__":
    main()

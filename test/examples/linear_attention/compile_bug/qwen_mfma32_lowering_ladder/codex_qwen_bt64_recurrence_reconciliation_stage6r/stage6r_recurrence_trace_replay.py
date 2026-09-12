#!/usr/bin/env python3
"""Replay one freshly compiled Stage 6R recurrence graph for rocprof only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
REPO = LADDER.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
sys.path[:0] = [str(STAGE6A), str(COMPARE)]

import stage6a_full_graph_audit as stage6a  # noqa: E402
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("vllm_actual_bf16", "asm_v0_fp32"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--replay", type=int, default=3)
    args = parser.parse_args()
    if args.T % 64:
        raise ValueError("T must be divisible by 64")
    patch_rocm_autotune()
    inputs = stage6a.fixed_inputs(args.T)
    values = stage6a.vllm_manual_stages(inputs)
    _, k, _, _, _, h0 = inputs
    if args.implementation == "vllm_actual_bf16":
        fn = lambda: chunk_gated_delta_rule_fwd_h(k, values["w"], values["u"], values["g_cumsum"], None, h0, True, 64, True, None)
    else:
        w = values["w"].float().contiguous()
        u = values["u"].float().contiguous()
        fn = lambda: qwen_gdn_bt64_gfx942_asm_v0(k, w, u, values["g_cumsum"], h0)
    graph = stage6a.CapturedGraph.create(args.implementation, fn)
    torch.cuda.synchronize()
    for _ in range(args.replay):
        graph.replay()
    torch.cuda.synchronize()
    print(json.dumps({"implementation": args.implementation, "T": args.T, "replay": args.replay,
                      "capture_output_ptrs": graph.capture_output_ptrs}))


if __name__ == "__main__":
    main()

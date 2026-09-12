#!/usr/bin/env python3
"""Write the V1 eager-public correctness comparison used by the Stage 6V report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager  # noqa: E402
from qwen_gdn_bt64_predicate_collapse_stage6v import qwen_gdn_full_bt64_stage6v_predicate_collapse_eager  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    patch_rocm_autotune()
    rows = []
    for t in args.T:
        q, k, v, g, beta, h0 = make_inputs(t, 2026073500 + t, "random", True)
        common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
        u1_output, u1_state = qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **common)
        v1_output, v1_state = qwen_gdn_full_bt64_stage6v_predicate_collapse_eager(q, k, v, g, beta, **common)
        ref_output, ref_state = vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        )
        torch.cuda.synchronize()
        assert u1_state is not None and v1_state is not None and ref_state is not None
        rows.append({
            "T": t,
            "v1_output_max_abs_vs_vllm": float((v1_output.float() - ref_output.float()).abs().max().item()),
            "v1_state_max_abs_vs_vllm": float((v1_state - ref_state).abs().max().item()),
            "v1_output_max_abs_vs_u1": float((v1_output.float() - u1_output.float()).abs().max().item()),
            "v1_state_max_abs_vs_u1": float((v1_state - u1_state).abs().max().item()),
            "v1_output_finite": bool(torch.isfinite(v1_output.float()).all().item()),
            "v1_state_finite": bool(torch.isfinite(v1_state).all().item()),
        })
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    if any(
        row["v1_output_max_abs_vs_vllm"] > 1.0 / 128.0
        or row["v1_state_max_abs_vs_vllm"] > 2.0e-2
        or not row["v1_output_finite"]
        or not row["v1_state_finite"]
        for row in rows
    ):
        raise AssertionError(rows)


if __name__ == "__main__":
    main()

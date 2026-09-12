#!/usr/bin/env python3
"""Expanded public-API correctness and seed stability for Stage 6T."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
DEFAULT_OUT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_fused_wu_eager_stage6t"
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


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def calls(inputs: tuple[torch.Tensor, ...]):
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f0": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        ),
    }


def case_inputs(t: int, seed: int, mode: str, with_initial_state: bool):
    base_mode = mode if mode in {"random", "neutral_gate", "high_dynamic", "cancellation", "small_values"} else "random"
    q, k, v, g, beta, h0 = make_inputs(t, seed, base_mode, with_initial_state)
    if mode == "zero_beta":
        beta.zero_()
    elif mode == "sparse_beta":
        beta.zero_()
        beta[:, ::7, :].fill_(1.0)
    return q, k, v, g, beta, h0


def evaluate(t: int, seed: int, mode: str, with_initial_state: bool, nondefault_stream: bool) -> list[dict[str, object]]:
    api = calls(case_inputs(t, seed, mode, with_initial_state))
    if nondefault_stream:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            values = {name: fn() for name, fn in api.items()}
        stream.synchronize()
    else:
        values = {name: fn() for name, fn in api.items()}
        torch.cuda.synchronize()
    ref_output, ref_state = values["vllm"]
    if ref_state is None:
        raise AssertionError("native vLLM public API did not produce final state")
    rows = []
    for name in ("stage6s", "f0", "f1"):
        output, state = values[name]
        if state is None:
            raise AssertionError(f"{name} public API did not produce final state")
        delta_out = (output.float() - ref_output.float()).abs()
        delta_state = (state - ref_state).abs()
        item = {
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "seed": seed, "mode": mode, "initial_state": with_initial_state,
            "nondefault_stream": nondefault_stream, "implementation": name,
            "output_max_abs": float(delta_out.max().item()), "output_mean_abs": float(delta_out.mean().item()),
            "final_state_max_abs": float(delta_state.max().item()), "final_state_mean_abs": float(delta_state.mean().item()),
            "output_finite": bool(torch.isfinite(output.float()).all().item()),
            "state_finite": bool(torch.isfinite(state).all().item()),
        }
        item["accepted"] = bool(
            item["output_max_abs"] <= OUTPUT_ATOL and item["final_state_max_abs"] <= STATE_ATOL
            and item["output_finite"] and item["state_finite"]
        )
        if not item["accepted"]:
            raise AssertionError(item)
        rows.append(item)
    f0_out, f0_state = values["f0"]
    f1_out, f1_state = values["f1"]
    assert f0_state is not None and f1_state is not None
    if float((f0_out.float() - f1_out.float()).abs().max().item()) != 0.0:
        raise AssertionError("F0/F1 public output diverged")
    if float((f0_state - f1_state).abs().max().item()) != 0.0:
        raise AssertionError("F0/F1 public final state diverged")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix = [
        (64, 2026071801, "random", True, False), (128, 2026071802, "neutral_gate", False, False),
        (512, 2026071803, "zero_beta", True, False), (1024, 2026071804, "small_values", False, False),
        (2048, 2026071805, "high_dynamic", True, False), (2048, 2026071806, "cancellation", True, False),
        (2048, 2026071807, "sparse_beta", True, True), (8192, 2026071808, "random", True, False),
    ]
    full_rows = [row for spec in matrix for row in evaluate(*spec)]
    stability_rows = []
    for seed in range(2026071900, 2026071920):
        stability_rows.extend(evaluate(2048, seed, "random", True, False))
    for seed in range(2026072000, 2026072005):
        stability_rows.extend(evaluate(8192, seed, "random", True, False))
    write_csv(args.out_dir / "eager_full_correctness.csv", full_rows)
    write_csv(args.out_dir / "eager_random_seed_stability.csv", stability_rows)
    all_rows = full_rows + stability_rows
    summary = {
        "timing_contract": "eager_public_api", "cuda_graph_used": False,
        "public_full_correct": all(bool(row["accepted"]) for row in all_rows),
        "full_cases": len(matrix), "expanded_seed_cases": 25,
        "rows": len(all_rows),
        "max_output_abs": max(float(row["output_max_abs"]) for row in all_rows),
        "max_final_state_abs": max(float(row["final_state_max_abs"]) for row in all_rows),
    }
    (args.out_dir / "correctness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

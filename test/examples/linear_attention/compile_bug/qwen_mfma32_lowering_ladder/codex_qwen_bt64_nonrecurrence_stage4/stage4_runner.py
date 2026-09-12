#!/usr/bin/env python3
"""Frozen Stage 2/3 correctness matrix for the Stage 4 BT64 graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
STAGE3 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_wu_chunko_stage3"
sys.path[:0] = [str(COMPARE), str(STAGE2), str(STAGE3)]

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_full_bt64_stage4_all_s0_stages  # noqa: E402
from stage2_runner import OUTPUT_ATOL, STATE_ATOL, make_inputs, metrics, patch_rocm_autotune, vllm_stages  # noqa: E402
from stage3_runner import stage3_case_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "full_pipeline")
    args = parser.parse_args()
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for case, (t, seed, mode, has_state) in enumerate(stage3_case_plan(args.random_cases, args.smoke)):
        q, k, v, g, beta, h0 = make_inputs(t, seed, mode, has_state)
        golden = vllm_stages(q, k, v, g, beta, h0)
        actual = qwen_gdn_full_bt64_stage4_all_s0_stages(q, k, v, g, beta, initial_state=h0)
        stages = {
            name: metrics(actual.get(name), golden.get(name))
            for name in ("g_cumsum", "a", "a_solved", "w", "u", "h_bf16", "v_new", "final_state", "output")
        }
        stages["public_output"] = metrics(actual["output"], golden["public_output"])
        stages["public_final_state"] = metrics(actual["final_state"], golden["public_final_state"])
        accepted = (
            stages["public_output"]["max_abs"] <= OUTPUT_ATOL
            and stages["public_final_state"]["max_abs"] <= STATE_ATOL
        )
        row = {
            "case": case,
            "T": t,
            "seed": seed,
            "mode": mode,
            "initial_state": has_state,
            "accepted": accepted,
            "stages": stages,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "case": case,
                    "T": t,
                    "mode": mode,
                    "accepted": accepted,
                    "output": stages["public_output"]["max_abs"],
                    "state": stages["public_final_state"]["max_abs"],
                }
            ),
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "correctness.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "case", "T", "seed", "mode", "initial_state", "accepted", "stage",
        "max_abs", "mean_abs", "max_rel", "first_mismatch",
    ]
    with (args.out / "correctness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for stage, result in row["stages"].items():
                writer.writerow({key: row[key] for key in fields[:6]} | {"stage": stage} | result)
    if not all(bool(row["accepted"]) for row in rows):
        raise SystemExit("Stage 4 full correctness exceeded the frozen thresholds")


if __name__ == "__main__":
    main()

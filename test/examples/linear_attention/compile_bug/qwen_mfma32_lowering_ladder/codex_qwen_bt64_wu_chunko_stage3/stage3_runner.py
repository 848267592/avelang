#!/usr/bin/env python3
"""Full correctness matrix for the opt-in native-BT64 W/U + chunk-o graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages  # noqa: E402
from stage2_runner import OUTPUT_ATOL, STATE_ATOL, make_inputs, metrics, patch_rocm_autotune, vllm_stages  # noqa: E402


def stage3_case_plan(random_cases: int, smoke: bool) -> list[tuple[int, int, str, bool]]:
    """Freeze the Stage 2 37-case contract inside Stage 3's own runner.

    Some Docker workspaces carry an older Stage 2 helper with only five special
    cases.  Keeping the plan local prevents that external revision skew from
    accidentally dropping the two long-sequence smoke cases.
    """
    if smoke:
        return [(64, 20260712, "random", True), (128, 20260713, "random", False)]
    lengths = (64, 128, 512, 2048)
    rows = [(lengths[index % len(lengths)], 20260712 + index, "random", index % 2 == 0) for index in range(random_cases)]
    rows.extend(
        [
            (64, 20260801, "neutral_gate", True),
            (128, 20260802, "neutral_gate", False),
            (512, 20260803, "high_dynamic", True),
            (2048, 20260804, "cancellation", True),
            (512, 20260805, "small_values", False),
            (8192, 20260806, "random", True),
            (8192, 20260807, "neutral_gate", False),
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-cases", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    patch_rocm_autotune()
    rows: list[dict[str, object]] = []
    for case, (t, seed, mode, has_state) in enumerate(stage3_case_plan(args.random_cases, args.smoke)):
        q, k, v, g, beta, h0 = make_inputs(t, seed, mode, has_state)
        golden = vllm_stages(q, k, v, g, beta, h0)
        native = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
        stages = {
            name: metrics(native.get(name), golden.get(name))
            for name in ("g_cumsum", "a", "a_solved", "w", "u", "h_bf16", "v_new", "final_state", "output")
        }
        stages["public_output"] = metrics(native["output"], golden["public_output"])
        stages["public_final_state"] = metrics(native["final_state"], golden["public_final_state"])
        accepted = stages["public_output"]["max_abs"] <= OUTPUT_ATOL and stages["public_final_state"]["max_abs"] <= STATE_ATOL
        row = {"case": case, "T": t, "seed": seed, "mode": mode, "initial_state": has_state, "accepted": accepted, "stages": stages}
        rows.append(row)
        print(json.dumps({"case": case, "T": t, "mode": mode, "accepted": accepted, "output": stages["public_output"]["max_abs"], "state": stages["public_final_state"]["max_abs"]}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "full_correctness.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out / "full_correctness.csv").open("w", newline="") as handle:
        fields = ["case", "T", "seed", "mode", "initial_state", "accepted", "stage", "max_abs", "mean_abs", "max_rel", "first_mismatch"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for stage, result in row["stages"].items():
                writer.writerow({key: row[key] for key in fields[:6]} | {"stage": stage} | result)
    if not all(bool(row["accepted"]) for row in rows):
        raise SystemExit("native Stage 3 full correctness exceeded the frozen Stage 2 threshold")


if __name__ == "__main__":
    main()

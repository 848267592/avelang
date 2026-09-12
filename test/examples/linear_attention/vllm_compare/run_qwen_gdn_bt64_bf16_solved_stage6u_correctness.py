#!/usr/bin/env python3
"""Authoritative complete eager public-API correctness matrix for Stage 6U."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
STAGE2 = ROOT / "codex_qwen_bt64_full_pipeline_stage2"
DEFAULT_OUT = ROOT / "codex_qwen_bt64_bf16_solved_boundary_stage6u"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (  # noqa: E402
    qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager,
    qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager,
)
from qwen_gdn_bt64_fused_wu_eager_stage6t import (  # noqa: E402
    qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
IMPLEMENTATIONS = ("stage6s", "f1", "u0", "u1")


def _calls(inputs: tuple[torch.Tensor, ...]):
    q, k, v, g, beta, h0 = inputs
    common = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    return {
        "stage6s": lambda: qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, **common),
        "f1": lambda: qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager(q, k, v, g, beta, **common),
        "u0": lambda: qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager(q, k, v, g, beta, **common),
        "u1": lambda: qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **common),
        "vllm": lambda: vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
            output_final_state=True, scale=128 ** -0.5, head_first=False,
            use_qk_l2norm_in_kernel=False,
        ),
    }


def _inputs(t: int, seed: int, mode: str, with_initial_state: bool):
    source_mode = mode if mode in {"random", "neutral_gate", "high_dynamic", "cancellation", "small_values"} else "random"
    values = list(make_inputs(t, seed, source_mode, with_initial_state))
    beta = values[4]
    if mode == "zero_beta":
        beta.zero_()
    elif mode == "sparse_beta":
        beta.zero_()
        beta[:, ::7, :].fill_(1.0)
    return tuple(values)


def _first_bad(delta: torch.Tensor) -> str:
    flat = int(delta.argmax().item())
    coord = []
    for size in reversed(delta.shape):
        coord.append(flat % size)
        flat //= size
    return str(tuple(reversed(coord)))


def evaluate(t: int, seed: int, mode: str, with_initial_state: bool, nondefault_stream: bool):
    calls = _calls(_inputs(t, seed, mode, with_initial_state))
    if nondefault_stream:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            results = {name: fn() for name, fn in calls.items()}
        stream.synchronize()
    else:
        results = {name: fn() for name, fn in calls.items()}
        torch.cuda.synchronize()
    ref_output, ref_state = results["vllm"]
    if ref_state is None:
        raise AssertionError("vLLM public API returned no final state")
    rows = []
    for name in IMPLEMENTATIONS:
        output, state = results[name]
        if state is None:
            raise AssertionError(f"{name} returned no final state")
        out_delta = (output.float() - ref_output.float()).abs()
        state_delta = (state - ref_state).abs()
        row = {
            "timing_contract": "eager_public_api", "cuda_graph_used": False,
            "T": t, "seed": seed, "mode": mode, "initial_state": with_initial_state,
            "nondefault_stream": nondefault_stream, "implementation": name,
            "output_max_abs": float(out_delta.max().item()),
            "output_mean_abs": float(out_delta.mean().item()),
            "final_state_max_abs": float(state_delta.max().item()),
            "final_state_mean_abs": float(state_delta.mean().item()),
            "first_output_bad": _first_bad(out_delta),
            "first_state_bad": _first_bad(state_delta),
            "output_finite": bool(torch.isfinite(output.float()).all().item()),
            "state_finite": bool(torch.isfinite(state).all().item()),
        }
        row["accepted"] = bool(
            row["output_max_abs"] <= OUTPUT_ATOL
            and row["final_state_max_abs"] <= STATE_ATOL
            and row["output_finite"] and row["state_finite"]
        )
        rows.append(row)
    u0_output, u0_state = results["u0"]
    u1_output, u1_state = results["u1"]
    assert u0_state is not None and u1_state is not None
    u0_u1 = {
        "T": t,
        "seed": seed,
        "output_max_abs": float((u0_output.float() - u1_output.float()).abs().max().item()),
        "state_max_abs": float((u0_state - u1_state).abs().max().item()),
    }
    del results, ref_output, ref_state, u0_output, u0_state, u1_output, u1_state
    torch.cuda.empty_cache()
    return rows, u0_u1


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--worker-spec", type=str)
    args = parser.parse_args()
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.worker_spec:
        spec = json.loads(args.worker_spec)
        rows, pair = evaluate(
            int(spec[0]), int(spec[1]), str(spec[2]), bool(spec[3]), bool(spec[4])
        )
        print("STAGE6U_WORKER_JSON=" + json.dumps({"rows": rows, "pair": pair}), flush=True)
        return

    def run_worker(spec):
        command = [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(args.out_dir), "--worker-spec", json.dumps(spec)]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        marker = "STAGE6U_WORKER_JSON="
        payload_lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
        if not payload_lines:
            raise RuntimeError(f"worker produced no result for {spec}:\n{result.stdout}")
        return json.loads(payload_lines[-1][len(marker):])

    matrix = [
        (64, 2026072201, "random", True, False),
        (128, 2026072202, "zero_beta", False, False),
        (512, 2026072203, "neutral_gate", True, False),
        (1024, 2026072204, "small_values", False, False),
        (2048, 2026072205, "high_dynamic", True, False),
        (2048, 2026072206, "cancellation", True, False),
        (8192, 2026072207, "sparse_beta", True, True),
    ]
    full_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for spec in matrix:
        rows, pair = evaluate(*spec)
        full_rows.extend(rows)
        pair_rows.append(pair)
        print(f"matrix complete: T={spec[0]} seed={spec[1]} mode={spec[2]}", flush=True)

    stability_specs = [
        *((2048, 2026072300 + i, "random", True, False) for i in range(30)),
        *((8192, 2026072400 + i, "random", True, False) for i in range(10)),
        *((16384, 2026072500 + i, "random", True, False) for i in range(3)),
    ]
    stability_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for spec in stability_specs:
        rows, pair = evaluate(*spec)
        stability_rows.extend(rows)
        pair_rows.append(pair)
        if spec[0] >= 8192:
            long_rows.extend(rows)
        print(f"stability complete: T={spec[0]} seed={spec[1]}", flush=True)

    _write_csv(args.out_dir / "eager_full_correctness.csv", full_rows)
    _write_csv(args.out_dir / "expanded_seed_stability.csv", stability_rows)
    _write_csv(args.out_dir / "long_sequence_correctness.csv", long_rows)
    _write_csv(args.out_dir / "u0_u1_equivalence.csv", pair_rows)
    all_rows = full_rows + stability_rows
    failed = [row for row in all_rows if not bool(row["accepted"])]
    summary = {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "public_full_correct": not failed,
        "full_case_count": len(matrix),
        "t2048_seed_count": 30,
        "t8192_seed_count": 10,
        "t16384_seed_count": 3,
        "max_output_abs": max(float(row["output_max_abs"]) for row in all_rows),
        "max_final_state_abs": max(float(row["final_state_max_abs"]) for row in all_rows),
        "u0_u1_output_max_abs": max(float(row["output_max_abs"]) for row in pair_rows),
        "u0_u1_state_max_abs": max(float(row["state_max_abs"]) for row in pair_rows),
        "failed_rows": len(failed),
    }
    (args.out_dir / "correctness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    first = failed[0] if failed else None
    (args.out_dir / "first_divergence.md").write_text(
        "# First Divergence\n\n" + (json.dumps(first, indent=2) if first else "No threshold violation.\n")
    )
    if failed:
        raise AssertionError(first)
    if summary["u0_u1_output_max_abs"] != 0.0 or summary["u0_u1_state_max_abs"] != 0.0:
        raise AssertionError("U0 and U1 diverged")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

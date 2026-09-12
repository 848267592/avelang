#!/usr/bin/env python3
"""Frozen correctness matrix for the opt-in Stage 6S full graph."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
STAGE6A = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_graph_gap_stage6a"
sys.path[:0] = [str(COMPARE), str(STAGE2), str(STAGE6A)]

from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (  # noqa: E402
    qwen_gdn_bt64_stage6s_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge,
    qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages,
    qwen_gdn_full_bt64_stage6s_current_asm,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
import stage6a_full_graph_audit as stage6a  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def err(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    delta = (lhs.float() - rhs.float()).abs()
    return float(delta.max().item()), float(delta.mean().item())


def mutate(case: str, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta, h0 = values
    if case == "random_nonzero" or case == "multichunk_feedback":
        return values
    if case == "zero_initial_state":
        return q, k, v, g, beta, torch.zeros_like(h0)
    if case == "high_dynamic":
        # The base inputs for this case use Stage 2's validated high-dynamic
        # mode. Do not apply a second ad-hoc transformation here.
        return values
    if case == "small_values":
        return q.mul(0.001), k.mul(0.001), v.mul(0.001), g.mul(0.01), beta, h0.mul(0.001)
    if case == "cancellation":
        sign = torch.where(torch.arange(q.shape[1], device=q.device) % 2 == 0, 1.0, -1.0).view(1, -1, 1, 1)
        return q, k, (v.float() * sign).to(torch.bfloat16), g, beta, h0
    if case == "neutral_gate":
        return q, k, v, torch.zeros_like(g), beta, h0
    raise ValueError(case)


def main() -> None:
    patch_rocm_autotune()
    if not torch.cuda.is_available():
        raise RuntimeError("requires HIP GPU")
    full_rows: list[dict[str, object]] = []
    boundary_rows: list[dict[str, object]] = []
    recurrence_rows: list[dict[str, object]] = []
    first: list[str] = ["# Stage 6S First Divergence\n"]
    matrix = [
        (64, "random_nonzero"), (128, "random_nonzero"), (512, "random_nonzero"),
        (1024, "random_nonzero"), (2048, "random_nonzero"), (8192, "random_nonzero"),
        (512, "zero_initial_state"), (512, "high_dynamic"), (512, "small_values"),
        (512, "cancellation"), (512, "neutral_gate"), (2048, "multichunk_feedback"),
    ]
    for index, (t, case) in enumerate(matrix):
        mode = "high_dynamic" if case == "high_dynamic" else "random"
        inputs = mutate(case, make_inputs(t, 2026071700 + index * 17 + t, mode, True))
        q, k, v, g, beta, h0 = inputs
        a_out, a_state = qwen_gdn_full_bt64_stage6s_current_asm(q, k, v, g, beta, initial_state=h0, output_final_state=True)
        b_out, b_state = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(q, k, v, g, beta, initial_state=h0, output_final_state=True)
        c_out, c_state = stage6a.vllm_full(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
            scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
        )
        b_stages = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages(q, k, v, g, beta, initial_state=h0)
        torch.cuda.synchronize()
        out_abs, out_mean = err(b_out, c_out)
        state_abs, state_mean = err(b_state, c_state)
        a_b_out, a_b_out_mean = err(a_out, b_out)
        a_b_state, a_b_state_mean = err(a_state, b_state)
        accepted = out_abs <= OUTPUT_ATOL and state_abs <= STATE_ATOL
        full_rows.append({
            "T": t, "case": case, "accepted": accepted,
            "b_vs_vllm_output_max_abs": out_abs, "b_vs_vllm_output_mean_abs": out_mean,
            "b_vs_vllm_final_state_max_abs": state_abs, "b_vs_vllm_final_state_mean_abs": state_mean,
            "a_vs_b_output_max_abs": a_b_out, "a_vs_b_output_mean_abs": a_b_out_mean,
            "a_vs_b_final_state_max_abs": a_b_state, "a_vs_b_final_state_mean_abs": a_b_state_mean,
        })
        boundary_rows.extend([
            {"T": t, "case": case, "boundary": "w_fp32_to_bf16_widened", **dict(zip(("max_abs", "mean_abs"), err(b_stages["w"], b_stages["w_bf16"].float())))},
            {"T": t, "case": case, "boundary": "u_fp32_to_bf16_widened", **dict(zip(("max_abs", "mean_abs"), err(b_stages["u"], b_stages["u_bf16"].float())))},
            {"T": t, "case": case, "boundary": "vnew_bf16_to_fp32_widened", **dict(zip(("max_abs", "mean_abs"), err(b_stages["v_new"], b_stages["v_new_bf16"].float())))},
        ])
        if not accepted:
            first.append(f"- T={t}, case={case}: first public divergence output={out_abs}, state={state_abs}\n")

    for t in (64, 512, 2048):
        q, k, v, g, beta, h0 = make_inputs(t, 2026071810 + t, "random", True)
        values = stage6a.vllm_manual_stages((q, k, v, g, beta, h0))
        for case, w, u in (("native_boundary", values["w"], values["u"]),
                           ("zero_w", torch.zeros_like(values["w"]), values["u"]),
                           ("zero_u", values["w"], torch.zeros_like(values["u"]))):
            native = chunk_gated_delta_rule_fwd_h(
                k=k, w=w, u=u, g=values["g_cumsum"], initial_state=h0,
                output_final_state=True, chunk_size=64, save_new_value=True, cu_seqlens=None,
            )
            bridged = qwen_gdn_bt64_stage6s_recurrence_bridge(k, w, u, values["g_cumsum"], h0)
            torch.cuda.synchronize()
            recurrence_rows.append({
                "T": t, "case": case,
                "h_bit_exact": bool(torch.equal(bridged[0], native[0])),
                "vnew_bit_exact": bool(torch.equal(bridged[1], native[1])),
                "final_state_bit_exact": bool(torch.equal(bridged[2], native[2])),
            })

    write_csv(HERE / "full_correctness.csv", full_rows)
    write_csv(HERE / "boundary_correctness.csv", boundary_rows)
    write_csv(HERE / "recurrence_correctness.csv", recurrence_rows)
    if len(first) == 1:
        first.append("- No full-graph threshold violation in the executed matrix.\n")
    (HERE / "first_divergence.md").write_text("".join(first))
    summary = {
        "full_cases": len(full_rows), "full_correct": all(bool(row["accepted"]) for row in full_rows),
        "recurrence_bit_exact": all(bool(row["h_bit_exact"] and row["vnew_bit_exact"] and row["final_state_bit_exact"]) for row in recurrence_rows),
        "public_output_max_abs": max(float(row["b_vs_vllm_output_max_abs"]) for row in full_rows),
        "final_state_max_abs": max(float(row["b_vs_vllm_final_state_max_abs"]) for row in full_rows),
    }
    (HERE / "correctness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

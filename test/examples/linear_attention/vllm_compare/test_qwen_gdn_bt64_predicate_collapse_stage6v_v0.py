"""V0 correctness gate for the Stage 6V predicate-collapse W/U probe."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u  # noqa: E402
from qwen_gdn_bt64_predicate_collapse_stage6v import qwen_gdn_w_u_bt64_predicate_collapse_v0  # noqa: E402
from run_qwen_gdn_stage6u_consumer_correctness import reference  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402


@pytest.mark.parametrize("t", [64, 512, 2048])
def test_predicate_collapse_v0_is_bit_exact_to_c0_and_reference(t: int) -> None:
    _, k, v, g, beta, _ = make_inputs(t, 2026073000 + t, "random", True)
    torch.manual_seed(2026073100 + t)
    a_solved = (torch.randn((1, t, 8, 64), device=k.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)

    baseline_w, baseline_u = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g, beta, a_solved)
    actual_w, actual_u = qwen_gdn_w_u_bt64_predicate_collapse_v0(k, v, g, beta, a_solved)
    torch.cuda.synchronize()

    # Wave-uniform MFMA changes only BF16 last-bit rounding relative to C0's
    # four separately predicated instructions. Keep a frozen 1e-3 W/U bound,
    # far tighter than the later full-graph public acceptance threshold.
    assert torch.isfinite(actual_w).all()
    assert torch.isfinite(actual_u).all()
    assert (actual_w.float() - baseline_w.float()).abs().max().item() <= 1.0e-3
    assert (actual_u.float() - baseline_u.float()).abs().max().item() <= 1.0e-3
    expected_w, expected_u = reference(k, v, g, beta, a_solved)
    assert (actual_w.float() - expected_w.float()).abs().max().item() <= 0.0078125
    assert (actual_u.float() - expected_u.float()).abs().max().item() <= 0.0078125

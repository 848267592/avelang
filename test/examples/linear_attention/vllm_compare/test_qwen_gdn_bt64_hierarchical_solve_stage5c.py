"""Stage 5C full-pipeline integration gate for the opt-in BT64 solve."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


AUDIT = Path(__file__).resolve().parents[1] / "compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path.insert(0, str(AUDIT))

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_full_bt64_stage4_all_s0_stages
from stage2_runner import OUTPUT_ATOL, STATE_ATOL, make_inputs, patch_rocm_autotune
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full


@pytest.fixture(autouse=True)
def _gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP GPU")
    patch_rocm_autotune()


@pytest.mark.parametrize(
    "t,mode,with_state",
    [(64, "random", True), (512, "high_dynamic", True), (512, "small_values", False)],
)
def test_stage5c_hierarchical_solve_preserves_stage4_semantics(t: int, mode: str, with_state: bool) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 20260716 + t, mode, with_state)
    baseline = qwen_gdn_full_bt64_stage4_all_s0_stages(q, k, v, g, beta, initial_state=h0)
    actual = qwen_gdn_full_bt64_stage4_all_s0_stages(
        q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
    )
    solve_error = (actual["a_solved"] - baseline["a_solved"]).abs()
    output_error = (actual["output"].float() - baseline["output"].float()).abs()
    state_error = (actual["final_state"] - baseline["final_state"]).abs()
    print(
        f"T={t} mode={mode} solve={solve_error.max().item():.8g} "
        f"output={output_error.max().item():.8g} state={state_error.max().item():.8g}"
    )
    assert solve_error.max().item() <= 1.0e-5
    assert output_error.max().item() <= 1.0e-3
    assert state_error.max().item() <= 1.0e-3


@pytest.mark.parametrize("t,mode,with_state", [(64, "random", True), (512, "high_dynamic", True)])
def test_stage5c_hierarchical_solve_preserves_frozen_vllm_contract(t: int, mode: str, with_state: bool) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 20260726 + t, mode, with_state)
    actual = qwen_gdn_full_bt64_stage4_all_s0_stages(
        q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
    )
    expected, expected_state = vllm_full(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=128**-0.5,
        initial_state=h0,
        output_final_state=True,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    output_error = (actual["output"].float() - expected.float()).abs()
    state_error = (actual["final_state"] - expected_state).abs()
    print(
        f"T={t} mode={mode} vLLM output={output_error.max().item():.8g} "
        f"state={state_error.max().item():.8g}"
    )
    assert output_error.max().item() <= OUTPUT_ATOL
    assert state_error.max().item() <= STATE_ATOL


def test_stage5c_rejects_unknown_solve_impl() -> None:
    q, k, v, g, beta, h0 = make_inputs(64, 20260736, "random", True)
    with pytest.raises(ValueError, match="solve_impl"):
        qwen_gdn_full_bt64_stage4_all_s0_stages(
            q, k, v, g, beta, initial_state=h0, solve_impl="not-a-solve"
        )

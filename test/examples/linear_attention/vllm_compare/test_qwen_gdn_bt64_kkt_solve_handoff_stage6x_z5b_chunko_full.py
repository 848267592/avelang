"""Full-graph correctness gate for the narrow X2-to-Z5B chunk-o substitution."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_kkt_solve_handoff_stage6x_full import qwen_gdn_full_bt64_stage6x_kkt_solve_eager  # noqa: E402
from qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full import (  # noqa: E402
    qwen_gdn_full_bt64_stage6x_z5b_chunko_eager,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")
OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


@pytest.mark.parametrize("t", (64, 512, 2048, 8192, 16384))
@pytest.mark.parametrize("mode", ("random", "high_dynamic", "cancellation", "neutral_gate"))
@pytest.mark.parametrize("with_initial_state", (False, True))
def test_stage6x_z5b_full_preserves_x2_state_and_output_contract(
    t: int, mode: str, with_initial_state: bool
) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 2026082900 + t, mode, True)
    kwargs = dict(
        initial_state=h0 if with_initial_state else None,
        output_final_state=True,
        scale=128 ** -0.5,
    )
    expected_output, expected_state = qwen_gdn_full_bt64_stage6x_kkt_solve_eager(q, k, v, g, beta, **kwargs)
    actual_output, actual_state = qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(q, k, v, g, beta, **kwargs)
    torch.cuda.synchronize()
    assert actual_state is not None and expected_state is not None
    assert torch.equal(actual_state, expected_state)
    mismatch = int((actual_output.view(torch.int16) != expected_output.view(torch.int16)).sum().item())
    output_max = float((actual_output.float() - expected_output.float()).abs().max().item())
    print(f"x2_z5b,T={t},mode={mode},initial={with_initial_state},mismatch={mismatch},max_abs={output_max:.9g}")
    assert output_max <= OUTPUT_ATOL


@pytest.mark.parametrize("t", (64, 8192, 16384))
def test_stage6x_z5b_full_zero_v_output_is_finite(t: int) -> None:
    q, k, v, g, beta, h0 = make_inputs(t, 2026083000 + t, "random", True)
    v.zero_()
    output, final_state = qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, scale=128 ** -0.5
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(output).all().item())
    assert final_state is not None and bool(torch.isfinite(final_state).all().item())


@pytest.mark.parametrize(
    ("t", "mode", "with_initial_state"),
    ((64, "random", True), (512, "high_dynamic", True), (2048, "neutral_gate", False), (8192, "cancellation", True)),
)
def test_stage6x_z5b_full_meets_native_public_contract(t: int, mode: str, with_initial_state: bool) -> None:
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, 2026083200 + t, mode, with_initial_state)
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    actual_output, actual_state = qwen_gdn_full_bt64_stage6x_z5b_chunko_eager(q, k, v, g, beta, **kwargs)
    reference_output, reference_state = vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    assert actual_state is not None and reference_state is not None
    output_max = float((actual_output.float() - reference_output.float()).abs().max().item())
    state_max = float((actual_state - reference_state).abs().max().item())
    print(f"x2_z5b-vllm,T={t},mode={mode},output={output_max:.9g},state={state_max:.9g}")
    assert output_max <= OUTPUT_ATOL
    assert state_max <= STATE_ATOL

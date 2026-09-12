"""Full eager-public correctness gate for Stage 6V V1."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_solved_boundary_stage6u import qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager  # noqa: E402
from qwen_gdn_bt64_predicate_collapse_stage6v import qwen_gdn_full_bt64_stage6v_predicate_collapse_eager  # noqa: E402
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full  # noqa: E402


OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2


@pytest.mark.parametrize("t", [64, 512, 2048])
def test_stage6v_v1_eager_public_contract(t: int) -> None:
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, 2026073400 + t, "random", True)
    kwargs = dict(initial_state=h0, output_final_state=True, scale=128 ** -0.5)
    u1_output, u1_state = qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(q, k, v, g, beta, **kwargs)
    v1_output, v1_state = qwen_gdn_full_bt64_stage6v_predicate_collapse_eager(q, k, v, g, beta, **kwargs)
    ref_output, ref_state = vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, initial_state=h0, output_final_state=True,
        scale=128 ** -0.5, head_first=False, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    assert u1_state is not None and v1_state is not None and ref_state is not None
    v1_output_error = float((v1_output.float() - ref_output.float()).abs().max().item())
    v1_state_error = float((v1_state - ref_state).abs().max().item())
    u1_v1_output = float((u1_output.float() - v1_output.float()).abs().max().item())
    u1_v1_state = float((u1_state - v1_state).abs().max().item())
    print(
        f"V1 T={t}: vs_vllm_output={v1_output_error:.9g}, vs_vllm_state={v1_state_error:.9g}, "
        f"vs_u1_output={u1_v1_output:.9g}, vs_u1_state={u1_v1_state:.9g}"
    )
    assert torch.isfinite(v1_output.float()).all() and torch.isfinite(v1_state).all()
    assert v1_output_error <= OUTPUT_ATOL
    assert v1_state_error <= STATE_ATOL

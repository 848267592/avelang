from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


AUDIT = Path(__file__).resolve().parents[1] / "compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path.insert(0, str(AUDIT))

from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import qwen_gdn_full_bt64_gfx942_asm_v0
from stage2_runner import OUTPUT_ATOL, STATE_ATOL, make_inputs, patch_rocm_autotune
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full


@pytest.fixture(autouse=True)
def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("requires HIP GPU")
    patch_rocm_autotune()


@pytest.mark.parametrize("t,with_state", [(64, True), (128, False), (512, True)])
def test_bt64_full_matches_vllm_contract(t: int, with_state: bool):
    q, k, v, g, beta, h0 = make_inputs(t, 20261000 + t, "random", with_state)
    actual, actual_state = qwen_gdn_full_bt64_gfx942_asm_v0(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    expected, expected_state = vllm_full(
        q=q, k=k, v=v, g=g, beta=beta, scale=128**-0.5, initial_state=h0,
        output_final_state=True, head_first=False, use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    assert actual_state is not None and expected_state is not None
    assert (actual.float() - expected.float()).abs().max().item() <= OUTPUT_ATOL
    assert (actual_state - expected_state).abs().max().item() <= STATE_ATOL


def test_bt64_full_rejects_tail():
    q, k, v, g, beta, h0 = make_inputs(64, 20261064, "random", True)
    with pytest.raises(ValueError, match="divisible by 64"):
        qwen_gdn_full_bt64_gfx942_asm_v0(q[:, :63], k[:, :63], v[:, :63], g[:, :63], beta[:, :63], initial_state=h0)

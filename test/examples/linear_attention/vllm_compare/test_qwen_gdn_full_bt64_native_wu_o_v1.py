"""Fast pytest coverage for the opt-in BT64 Stage 3 graph."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
STAGE2 = ROOT / "compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path.insert(0, str(STAGE2))

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages
from stage2_runner import OUTPUT_ATOL, STATE_ATOL, make_inputs, patch_rocm_autotune, vllm_stages


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")


@pytest.mark.parametrize("t,mode,with_state", [(64, "random", True), (128, "neutral_gate", False), (512, "high_dynamic", True)])
def test_native_bt64_full_matches_frozen_vllm_contract(t: int, mode: str, with_state: bool) -> None:
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, 20260750 + t, mode, with_state)
    golden = vllm_stages(q, k, v, g, beta, h0)
    native = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    output_error = (native["output"].float() - golden["public_output"].float()).abs().max().item()
    state_error = (native["final_state"] - golden["public_final_state"]).abs().max().item()
    print(f"T={t} mode={mode} output_max_abs={output_error:.8g} state_max_abs={state_error:.8g}")
    assert output_error <= OUTPUT_ATOL
    assert state_error <= STATE_ATOL

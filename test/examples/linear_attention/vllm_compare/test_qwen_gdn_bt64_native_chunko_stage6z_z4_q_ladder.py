"""Correctness gates for the independent Stage 6Z Z4 Q ladder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import _validate_stage6w_chunko_inputs  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
ARMS = {
    "z4a": (qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q, qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4a_vector_q_launch_into),
    "z4b": (qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency, qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4b_partial_q_residency_launch_into),
    "z4c": (qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion, qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z4c_full_k32_q_fusion_launch_into),
}


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


@pytest.mark.parametrize("t", (64, 512, 1024, 2048, 4096, 8192, 16384))
@pytest.mark.parametrize("arm", tuple(ARMS))
def test_z4_arm_is_bf16_byte_exact_to_fixed_z2(t: int, arm: str) -> None:
    q, k, vn, h, g = _inputs(t, 2026080700 + t)
    reference = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(q, k, vn, h, g)
    actual = ARMS[arm][0](q, k, vn, h, g)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert torch.equal(actual, reference), arm


@pytest.mark.parametrize("arm", tuple(ARMS))
@pytest.mark.parametrize("t", (64, 8192, 16384))
def test_z4_zero_v_nan_prefilled_caller_owned_output(arm: str, t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026090000 + t)
    vn.zero_()
    output = torch.full_like(vn, float("nan"))
    ARMS[arm][1](q, k, vn, h, g, output)
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert not bool(torch.isnan(output).any().item())
    assert torch.equal(output, expected), arm


def test_z4_shape_contract_is_fixed() -> None:
    assert _validate_stage6w_chunko_inputs is not None
    assert all(name in ARMS for name in ("z4a", "z4b", "z4c"))

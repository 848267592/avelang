"""Correctness gates for Stage 6Z Z2's phase-aware fragment materialization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z import qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
ATOL = 1.0 / 128.0


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


@pytest.mark.parametrize("t", (8192, 2048, 512, 64))
def test_stage6z_z2_is_bit_exact_to_z1_and_within_stage6w_contract(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026080600 + t)
    z1 = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, vn, h, g)
    z2 = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2(q, k, vn, h, g)
    frozen = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(z2).all().item())
    assert torch.equal(z2, z1)
    assert float((z2.float() - frozen.float()).abs().max().item()) <= ATOL


def test_stage6z_z2_zero_vnew_and_caller_owned_output_reuse() -> None:
    q, k, vn, h, g = _inputs(64, 2026080701)
    vn.zero_()
    output = torch.full_like(vn, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z2_launch_into(q, k, vn, h, g, output)
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert not bool(torch.isnan(output).any().item())
    assert torch.equal(output, expected)

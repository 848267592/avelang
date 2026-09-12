"""Correctness gates for the Z7AB-F fusion-only control."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache,
)
from qwen_gdn_bt64_native_chunko_stage6z_z7abf_fusion_only import (  # noqa: E402
    WORKGROUP,
    Z7ABF_WORKGROUP_CONTRACT,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 43)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q.contiguous(), k.contiguous(), vn.contiguous(), h.contiguous(), g.contiguous()


@pytest.mark.parametrize("t", (64, 512, 1024, 2048, 4096, 8192, 16384))
def test_z7abf_is_bf16_byte_exact_to_z5b(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026081500 + t)
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(q, k, vn, h, g)
    actual = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(actual).all().item())
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("t", (64, 8192, 16384))
def test_z7abf_caller_owned_zero_v_nan_prefilled_output(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026081600 + t)
    vn.zero_()
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(q, k, vn, h, g)
    output = torch.full_like(vn, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only_launch_into(
        q, k, vn, h, g, output
    )
    torch.cuda.synchronize()
    assert not bool(torch.isnan(output).any().item())
    assert torch.equal(output, expected)


def test_z7abf_shape_contract_is_fixed_wg256() -> None:
    assert WORKGROUP == 256 == Z7ABF_WORKGROUP_CONTRACT

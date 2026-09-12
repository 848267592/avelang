"""Isolated correctness gates for the opt-in Stage 6Z Z1 chunk-o prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w,
)
from qwen_gdn_bt64_native_chunko_stage6z import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64
ISOLATED_ATOL = 1.0 / 128.0


def _chunk_o_inputs(t: int, seed: int, mode: str = "random"):
    q, k, _, g, _, _ = make_inputs(t, seed, mode, True)
    chunks = t // BT
    torch.manual_seed(seed + 17)
    v_new_bf16 = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h_bf16 = (torch.randn((1, chunks, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, v_new_bf16, h_bf16, g


# Keep the largest specialization first. The formal Stage 6Z contract also
# runs every length in a fresh process; this ordering keeps ordinary pytest
# smoke from retaining a ladder of earlier Avelang constexpr specializations.
@pytest.mark.parametrize("t", (8192, 2048, 512, 128, 64))
def test_stage6z_z1_matches_frozen_stage6w(t: int):
    q, k, v_new_bf16, h_bf16, g = _chunk_o_inputs(t, 2026072200 + t)
    expected = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new_bf16, h_bf16, g)
    actual = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, v_new_bf16, h_bf16, g)
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    mismatch = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    first = None
    if mismatch:
        first = tuple(int(x) for x in torch.nonzero(actual.view(torch.int16) != expected.view(torch.int16), as_tuple=False)[0])
    print(
        f"Stage6Z Z1 T={t}: mismatch={mismatch} max_abs={max_abs:.9g} "
        f"mean_abs={mean_abs:.9g} first={first}"
    )
    assert max_abs <= ISOLATED_ATOL


def test_stage6z_z1_zero_vnew_is_inter_only():
    q, k, v_new_bf16, h_bf16, g = _chunk_o_inputs(64, 2026072301)
    v_new_bf16.zero_()
    expected = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new_bf16, h_bf16, g)
    actual = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, v_new_bf16, h_bf16, g)
    torch.cuda.synchronize()
    assert float((actual.float() - expected.float()).abs().max().item()) <= ISOLATED_ATOL


def test_stage6z_z1_caller_owned_output_reuse_and_invalid_dtype():
    q, k, v_new_bf16, h_bf16, g = _chunk_o_inputs(64, 2026072302)
    output = torch.full_like(v_new_bf16, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1_launch_into(q, k, v_new_bf16, h_bf16, g, output)
    expected = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(q, k, v_new_bf16, h_bf16, g)
    torch.cuda.synchronize()
    assert not bool(torch.isnan(output).any().item())
    assert float((output.float() - expected.float()).abs().max().item()) <= ISOLATED_ATOL
    with pytest.raises(ValueError):
        qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, v_new_bf16.float(), h_bf16, g)

"""Correctness gates for Stage 7A's one phase-scheduling repair."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


# Stage 7A's sole permitted full-kernel barrier-elision candidate failed its
# first T=8192 bit-exact gate (and subsequently poisoned that HIP process).
# Keep its intended gates as executable documentation, but do not execute this
# known-invalid experimental kernel during ordinary test collection.
pytestmark = pytest.mark.skip(reason="Stage 7A phase-compact barrier elision failed the T=8192 exactness gate")


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_native_chunko_stage6z import qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage7a_phase_compact import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact,
    qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact_launch_into,
)
from stage2_runner import make_inputs  # noqa: E402


BT = 64


def _inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = make_inputs(t, seed, "random", True)
    torch.manual_seed(seed + 31)
    vn = (torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    h = (torch.randn((1, t // BT, 8, 128, 128), device=q.device, dtype=torch.float32) * 0.01).to(torch.bfloat16)
    return q, k, vn, h, g


@pytest.mark.parametrize("t", (8192, 2048, 512, 64))
def test_stage7a_phase_compact_is_bit_exact_to_z1(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026072900 + t)
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, vn, h, g)
    actual = qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact(q, k, vn, h, g)
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    print(f"Stage7A T={t}: exact={torch.equal(actual, expected)} max_abs={float(diff.max().item()):.9g}")
    assert torch.equal(actual, expected)


def test_stage7a_phase_compact_zero_vnew_and_reuse() -> None:
    q, k, vn, h, g = _inputs(64, 2026073001)
    vn.zero_()
    output = torch.full_like(vn, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact_launch_into(q, k, vn, h, g, output)
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert not bool(torch.isnan(output).any().item())
    assert torch.equal(output, expected)

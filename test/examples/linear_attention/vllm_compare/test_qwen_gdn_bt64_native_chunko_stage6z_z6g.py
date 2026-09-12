"""Correctness gates for the independent Stage 6Z Z6G g-residency arms."""

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
from qwen_gdn_bt64_native_chunko_stage6z_z6g_g_residency import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable,
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into,
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
def test_z6g_arms_are_bf16_byte_exact_to_z5b(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026081000 + t)
    z5b = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(q, k, vn, h, g)
    stable = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable(q, k, vn, h, g)
    ideal = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal(q, k, vn, h, g)
    torch.cuda.synchronize()
    assert bool(torch.isfinite(stable).all().item())
    assert bool(torch.isfinite(ideal).all().item())
    assert torch.equal(stable, z5b)
    assert torch.equal(ideal, z5b)


@pytest.mark.parametrize("t", (64, 8192, 16384))
def test_z6g_caller_owned_zero_v_nan_prefill(t: int) -> None:
    q, k, vn, h, g = _inputs(t, 2026090000 + t)
    vn.zero_()
    expected = qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(q, k, vn, h, g)

    stable_output = torch.full_like(vn, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into(
        q, k, vn, h, g, stable_output
    )
    ideal_output = torch.full_like(vn, float("nan"))
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into(
        q, k, vn, h, g, ideal_output
    )
    torch.cuda.synchronize()
    assert not bool(torch.isnan(stable_output).any().item())
    assert not bool(torch.isnan(ideal_output).any().item())
    assert torch.equal(stable_output, expected)
    assert torch.equal(ideal_output, expected)


def test_z6g_shape_contract_is_fixed_wg256() -> None:
    from qwen_gdn_bt64_native_chunko_stage6z_z6g_g_residency import (
        WORKGROUP,
        Z6G_WORKGROUP_CONTRACT,
    )

    assert WORKGROUP == 256 == Z6G_WORKGROUP_CONTRACT

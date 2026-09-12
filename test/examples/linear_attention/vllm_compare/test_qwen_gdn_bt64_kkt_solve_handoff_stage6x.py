"""Correctness gates for Stage 6X-KS X1 one-CTA KKT ownership."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(HERE), str(STAGE2)]

from qwen_gdn_bt64_kkt_solve_handoff_stage6x import (  # noqa: E402
    _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into,
    qwen_gdn_kkt_bt64_one_cta_stage6x_x1,
    qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import qwen_gdn_kkt_bt64_mfma_v2_s0  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone  # noqa: E402
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (  # noqa: E402
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
)
from stage2_runner import make_inputs  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


def _inputs(t: int, seed: int, mode: str = "random"):
    _, k, _, g, beta, _ = make_inputs(t, seed, mode, True)
    return k, qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64), beta


def _summarize(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    mismatch = int((actual.view(torch.int32) != expected.view(torch.int32)).sum().item())
    first = torch.nonzero(actual.view(torch.int32) != expected.view(torch.int32))
    message = (
        f"{label}: mismatch={mismatch} max_abs={float(diff.max().item()):.9g} "
        f"mean_abs={float(diff.mean().item()):.9g}"
    )
    if first.numel():
        index = tuple(int(value) for value in first[0].cpu().tolist())
        message += f" first={index} actual={actual[index].item():.9g} expected={expected[index].item():.9g}"
    print(message)


@pytest.mark.parametrize("t", (64, 128, 512, 2048))
@pytest.mark.parametrize("mode", ("random", "high_dynamic"))
def test_stage6x_x1_kkt_is_bit_exact_to_current_kkt(t: int, mode: str) -> None:
    k, g_cumsum, beta = _inputs(t, 2026072100 + t, mode)
    expected = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    actual = qwen_gdn_kkt_bt64_one_cta_stage6x_x1(k, g_cumsum, beta)
    _summarize(f"x1_kkt,T={t},mode={mode}", actual, expected)
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.float32
    assert actual.stride() == expected.stride()


@pytest.mark.parametrize("t", (64, 512, 2048))
@pytest.mark.parametrize("mode", ("random", "cancellation"))
def test_stage6x_x1_preserves_stage6u_solve_input_layout(t: int, mode: str) -> None:
    k, g_cumsum, beta = _inputs(t, 2026072300 + t, mode)
    current_a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    x1_a = qwen_gdn_kkt_bt64_one_cta_stage6x_x1(k, g_cumsum, beta)
    current_solved = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(current_a)
    x1_solved = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(x1_a)
    torch.cuda.synchronize()
    mismatch = int((x1_solved.view(torch.int16) != current_solved.view(torch.int16)).sum().item())
    diff = (x1_solved.float() - current_solved.float()).abs()
    print(
        f"x1_solve_after_kkt,T={t},mode={mode},mismatch={mismatch},"
        f"max_abs={float(diff.max().item()):.9g},mean_abs={float(diff.mean().item()):.9g}"
    )
    assert mismatch == 0
    assert x1_solved.stride() == current_solved.stride()


@pytest.mark.parametrize("t", (64, 128, 512, 2048))
@pytest.mark.parametrize("mode", ("random", "high_dynamic", "cancellation"))
def test_stage6x_x2_fused_kkt_solve_is_bit_exact_to_current_chain(t: int, mode: str) -> None:
    k, g_cumsum, beta = _inputs(t, 2026072500 + t, mode)
    current_a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    expected = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(current_a)
    actual = qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(k, g_cumsum, beta)
    torch.cuda.synchronize()
    mismatch = int((actual.view(torch.int16) != expected.view(torch.int16)).sum().item())
    diff = (actual.float() - expected.float()).abs()
    print(
        f"x2_kkt_solve,T={t},mode={mode},mismatch={mismatch},"
        f"max_abs={float(diff.max().item()):.9g},mean_abs={float(diff.mean().item()):.9g}"
    )
    assert mismatch == 0
    assert actual.stride() == expected.stride()


@pytest.mark.parametrize("t", (64, 512, 2048))
def test_stage6x_x2_direct_out_overwrites_nan_prefill_and_reuses_storage(t: int) -> None:
    k, g_cumsum, beta = _inputs(t, 2026072900 + t, "random")
    expected = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(
        qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    )
    out = torch.full_like(expected, float("nan"))
    for iteration in range(2):
        _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into(k, g_cumsum, beta, out)
        torch.cuda.synchronize()
        assert torch.equal(out, expected), f"reuse iteration={iteration}"

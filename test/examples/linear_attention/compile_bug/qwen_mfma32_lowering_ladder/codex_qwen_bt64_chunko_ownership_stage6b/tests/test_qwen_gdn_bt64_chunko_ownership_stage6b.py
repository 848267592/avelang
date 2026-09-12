"""Correctness gates for the opt-in Stage 6B BT64 chunk-o ownership kernel."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_bt64_chunko_ownership_stage6b import (
    qwen_gdn_chunk_o_bt64_ownership_o0,
    qwen_gdn_full_bt64_stage6b_chunko_o0_stages,
)
from qwen_gdn_bt64_chunko_ownership_stage6b_o1 import (
    qwen_gdn_chunk_o_bt64_ownership_o1,
    qwen_gdn_full_bt64_stage6b_chunko_o1_stages,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
)
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")

OUTPUT_ATOL = 4.0e-3
PUBLIC_OUTPUT_ATOL = 1.0 / 128.0
STATE_ATOL = 2.0e-2
VARIANTS = {
    "o0": (qwen_gdn_chunk_o_bt64_ownership_o0, qwen_gdn_full_bt64_stage6b_chunko_o0_stages),
    "o1": (qwen_gdn_chunk_o_bt64_ownership_o1, qwen_gdn_full_bt64_stage6b_chunko_o1_stages),
}


def _inputs(t: int, seed: int = 20260716) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + t)
    q = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    k = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    v = (torch.randn((1, t, 8, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, 8), device="cuda") * 0.01).float().contiguous()
    beta = (0.5 + torch.rand((1, t, 8), device="cuda")).float().contiguous()
    h0 = (torch.randn((1, 8, 128, 128), device="cuda") * 0.01).float().contiguous()
    return q, k, v, g, beta, h0


def _chunko_inputs(t: int, mode: str = "random") -> tuple[torch.Tensor, ...]:
    q, k, _, g, _, _ = _inputs(t, seed=907)
    torch.manual_seed(917 + t)
    h = (torch.randn((1, t // 64, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
    v_new = (torch.randn((1, t, 8, 128), device="cuda") * 0.03).float().contiguous()
    if mode == "zero_h":
        h.zero_()
    elif mode == "zero_vn":
        v_new.zero_()
    elif mode == "inter":
        v_new.zero_()
    elif mode == "intra":
        h.zero_()
    elif mode.startswith("source"):
        h.zero_()
        v_new.zero_()
        source_tile = int(mode[-1])
        v_new[:, source_tile * 16 : (source_tile + 1) * 16].fill_(0.03125)
    elif mode == "small":
        h.mul_(1.0e-4)
        v_new.mul_(1.0e-4)
    elif mode == "high":
        h.mul_(8.0)
        v_new.mul_(8.0)
    elif mode == "cancellation":
        # Keep both terms nonzero while making their summed output sensitive
        # to tile ownership and BF16 staging order.
        h.mul_(0.25)
        signs = torch.where(
            (torch.arange(t, device="cuda") & 1).view(1, t, 1, 1) == 0,
            1.0,
            -1.0,
        )
        v_new.mul_(signs)
    return q, k, v_new, h, g


def _assert_close(current: torch.Tensor, actual: torch.Tensor, *, label: str) -> None:
    error = (actual - current).abs()
    print(f"{label}: max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= OUTPUT_ATOL


@pytest.mark.parametrize("t", [64, 128, 512, 2048])
@pytest.mark.parametrize("mode", ["random", "zero_h", "zero_vn", "inter", "intra", "small", "high", "cancellation"])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ownership_matches_current_fp32_staging(t: int, mode: str, variant: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(t, mode)
    current = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    actual = VARIANTS[variant][0](q, k, v_new, h, g)
    _assert_close(current, actual, label=f"{variant} T={t} {mode}")


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("mode", ["random", "cancellation", "high"])
def test_ownership_t8192_matches_current_fp32_staging(variant: str, mode: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(8192, mode)
    current = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    actual = VARIANTS[variant][0](q, k, v_new, h, g)
    _assert_close(current, actual, label=f"{variant} T=8192 {mode}")


@pytest.mark.parametrize("t", [64, 2048])
@pytest.mark.parametrize("mode", ["random", "inter", "intra", "cancellation"])
def test_stage4_fp32_staging_tracks_vllm_chunk_o(t: int, mode: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(t, mode)
    current = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    # vLLM stores H without the singleton batch dimension.  Its chunk-o
    # output dtype follows v_new (FP32 here), so this checks the body before
    # the public BF16 cast used by the full operator.
    vllm = chunk_fwd_o(q=q, k=k, v=v_new, h=h[0], g=g, scale=128 ** -0.5, chunk_size=64)
    error = (current - vllm).abs()
    print(f"stage4-vllm T={t} {mode}: max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= OUTPUT_ATOL


@pytest.mark.parametrize("source_tile", [0, 1, 2, 3])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ownership_cross_token16_source_coverage(source_tile: int, variant: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(64, f"source{source_tile}")
    current = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    actual = VARIANTS[variant][0](q, k, v_new, h, g)
    _assert_close(current, actual, label=f"{variant} source{source_tile}")
    if source_tile < 3:
        assert actual[:, (source_tile + 1) * 16 :].abs().max().item() > 0.0


@pytest.mark.parametrize("v_base", list(range(0, 128, 16)))
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ownership_v16_boundaries(v_base: int, variant: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(64, "intra")
    v_new.zero_()
    v_new[..., v_base : v_base + 16].fill_(0.03125)
    current = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    actual = VARIANTS[variant][0](q, k, v_new, h, g)
    _assert_close(current, actual, label=f"{variant} V{v_base}:V{v_base + 16}")


@pytest.mark.parametrize("t", [64, 512])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ownership_full_graph_matches_stage4(t: int, variant: str) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=1017)
    current = qwen_gdn_full_bt64_stage4_all_s0_stages(
        q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
    )
    actual = VARIANTS[variant][1](
        q, k, v, g, beta, initial_state=h0, solve_impl="hierarchical_fp32_v1"
    )
    staging_error = (actual["output_fp32"] - current["output_fp32"]).abs()
    output_error = (actual["output"].float() - current["output"].float()).abs()
    state_error = (actual["final_state"] - current["final_state"]).abs()
    print(
        f"{variant} T={t} staging={staging_error.max().item():.8g} output={output_error.max().item():.8g} "
        f"state={state_error.max().item():.8g}"
    )
    assert staging_error.max().item() <= OUTPUT_ATOL
    assert output_error.max().item() <= PUBLIC_OUTPUT_ATOL
    assert state_error.max().item() <= STATE_ATOL


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ownership_invalid_guard(variant: str) -> None:
    q, k, v_new, h, g = _chunko_inputs(64)
    with pytest.raises(ValueError, match="chunk_size=64"):
        VARIANTS[variant][0](q, k, v_new, h, g, chunk_size=16)

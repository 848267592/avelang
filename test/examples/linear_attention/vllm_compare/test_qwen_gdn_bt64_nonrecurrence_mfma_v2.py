"""Stage 4 standalone and incremental correctness gates."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_full_bt64_stage4_all_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages,
    qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages,
    qwen_gdn_full_bt64_stage4_kkt_s0_stages,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")


def _inputs(t: int, seed: int = 20260715) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + t)
    q = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    k = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    v = (torch.randn((1, t, 8, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, 8), device="cuda") * 0.01).float().contiguous()
    beta = (0.5 + torch.rand((1, t, 8), device="cuda")).float().contiguous()
    h0 = (torch.randn((1, 8, 128, 128), device="cuda") * 0.01).float().contiguous()
    return q, k, v, g, beta, h0


@pytest.mark.parametrize("t", [64, 128, 512])
def test_kkt_s0_matches_v6_and_preserves_mask(t: int) -> None:
    _, k, _, g, beta, _ = _inputs(t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    ref = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64)
    actual = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    error = (actual - ref).abs()
    local = torch.arange(64, device="cuda")
    upper_or_diag = local[None, :] >= local[:, None]
    masked_max = actual.view(1, t // 64, 64, 8, 64).permute(0, 1, 3, 2, 4)[..., upper_or_diag].abs().max()
    print(f"T={t} KKT max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g} mask={masked_max.item():.8g}")
    assert masked_max.item() == 0.0
    assert error.max().item() <= 1.0e-3


@pytest.mark.parametrize("t", [64, 512])
def test_kkt_s0_solve_consumer(t: int) -> None:
    _, k, _, g, beta, _ = _inputs(t, seed=41)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    ref_a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64)
    actual_a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    ref = qwen_gdn_solve_avelang_v18_bt64_layout(ref_a)
    actual = qwen_gdn_solve_avelang_v18_bt64_layout(actual_a)
    error = (actual - ref).abs()
    print(f"T={t} solve-after-KKT max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= 2.0e-3


@pytest.mark.parametrize("t", [64, 128])
def test_stage4_kkt_s0_full_matches_stage3(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=73)
    ref = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    actual = qwen_gdn_full_bt64_stage4_kkt_s0_stages(q, k, v, g, beta, initial_state=h0)
    output_error = (actual["output"] - ref["output"]).abs()
    state_error = (actual["final_state"] - ref["final_state"]).abs()
    print(
        f"T={t} full output max_abs={output_error.max().item():.8g} "
        f"state max_abs={state_error.max().item():.8g}"
    )
    assert output_error.max().item() <= 7.8125e-3
    assert state_error.max().item() <= 2.0e-2


@pytest.mark.parametrize("t", [64, 128, 512])
def test_wu_s0_matches_stage3(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=101)
    stages = qwen_gdn_full_bt64_stage4_kkt_s0_stages(q, k, v, g, beta, initial_state=h0)
    ref = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s0(k, v, stages["g_cumsum"], beta, stages["a_solved"])
    w_error = (w - ref["w"]).abs()
    u_error = (u - ref["u"]).abs()
    print(
        f"T={t} W-S0 max_abs={w_error.max().item():.8g} mean_abs={w_error.mean().item():.8g} "
        f"U-S0 max_abs={u_error.max().item():.8g} mean_abs={u_error.mean().item():.8g}"
    )
    assert w_error.max().item() <= 2.0e-3
    assert u_error.max().item() <= 2.0e-3


@pytest.mark.parametrize("t", [64, 128])
def test_stage4_kkt_wu_s0_full_matches_stage3(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=131)
    ref = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    actual = qwen_gdn_full_bt64_stage4_kkt_wu_s0_stages(q, k, v, g, beta, initial_state=h0)
    output_error = (actual["output"] - ref["output"]).abs()
    state_error = (actual["final_state"] - ref["final_state"]).abs()
    print(
        f"T={t} KKT+WU full output max_abs={output_error.max().item():.8g} "
        f"state max_abs={state_error.max().item():.8g}"
    )
    assert output_error.max().item() <= 7.8125e-3
    assert state_error.max().item() <= 2.0e-2


@pytest.mark.parametrize("t", [64, 128, 512])
def test_wu_s1_bf16_level_error(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=151)
    stages = qwen_gdn_full_bt64_stage4_kkt_s0_stages(q, k, v, g, beta, initial_state=h0)
    w_ref, u_ref = qwen_gdn_w_u_bt64_mfma_v2_s0(k, v, stages["g_cumsum"], beta, stages["a_solved"])
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, stages["g_cumsum"], beta, stages["a_solved"])
    w_error = (w - w_ref).abs()
    u_error = (u - u_ref).abs()
    print(
        f"T={t} W-S1 max_abs={w_error.max().item():.8g} mean_abs={w_error.mean().item():.8g} "
        f"U-S1 max_abs={u_error.max().item():.8g} mean_abs={u_error.mean().item():.8g}"
    )
    assert w_error.max().item() <= 2.0e-3
    assert u_error.max().item() <= 2.0e-3


@pytest.mark.parametrize("t", [64, 128, 512])
def test_stage4_kkt_wu_s1_full_contract(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=181)
    ref = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    actual = qwen_gdn_full_bt64_stage4_kkt_wu_s1_stages(q, k, v, g, beta, initial_state=h0)
    output_error = (actual["output"] - ref["output"]).abs()
    state_error = (actual["final_state"] - ref["final_state"]).abs()
    print(
        f"T={t} KKT+WU-S1 output max_abs={output_error.max().item():.8g} "
        f"state max_abs={state_error.max().item():.8g}"
    )
    assert output_error.max().item() <= 7.8125e-3
    assert state_error.max().item() <= 2.0e-2


@pytest.mark.parametrize("t", [64, 128, 512])
def test_chunko_s0_matches_stage3(t: int) -> None:
    q, k, _, g, _, _ = _inputs(t, seed=211)
    torch.manual_seed(211 + t)
    v_new = (torch.randn((1, t, 8, 128), device="cuda") * 0.03).float().contiguous()
    h = (torch.randn((1, t // 64, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
    ref = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h, g)
    actual = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    error = (actual - ref).abs()
    print(f"T={t} chunk-o-S0 max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= 4.0e-3


@pytest.mark.parametrize("mode", ["inter", "intra", "source0", "source1", "source2"])
def test_chunko_s0_component_and_cross_tile_modes(mode: str) -> None:
    t = 128
    q, k, _, g, _, _ = _inputs(t, seed=241)
    h = torch.zeros((1, 2, 8, 128, 128), dtype=torch.bfloat16, device="cuda")
    v_new = torch.zeros((1, t, 8, 128), dtype=torch.float32, device="cuda")
    if mode == "inter":
        h.normal_(mean=0.0, std=0.01)
    elif mode == "intra":
        v_new.normal_(mean=0.0, std=0.03)
    else:
        tile = {"source0": 0, "source1": 1, "source2": 2}[mode]
        v_new[:, tile * 16 : (tile + 1) * 16].fill_(0.03125)
    ref = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h, g)
    actual = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h, g)
    error = (actual - ref).abs()
    print(f"mode={mode} chunk-o-S0 max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= 4.0e-3


@pytest.mark.parametrize("t", [64, 128, 512])
def test_stage4_all_s0_full_contract(t: int) -> None:
    q, k, v, g, beta, h0 = _inputs(t, seed=271)
    ref = qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1_stages(q, k, v, g, beta, initial_state=h0)
    actual = qwen_gdn_full_bt64_stage4_all_s0_stages(q, k, v, g, beta, initial_state=h0)
    output_error = (actual["output"] - ref["output"]).abs()
    state_error = (actual["final_state"] - ref["final_state"]).abs()
    print(
        f"T={t} Stage4-all output max_abs={output_error.max().item():.8g} "
        f"state max_abs={state_error.max().item():.8g}"
    )
    assert output_error.max().item() <= 7.8125e-3
    assert state_error.max().item() <= 2.0e-2

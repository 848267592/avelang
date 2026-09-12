"""Standalone correctness gates for opt-in BT64 native MFMA W/U and chunk-o."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm/CUDA")


def _inputs(t: int, seed: int = 20260713) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed + t)
    device = "cuda"
    q = (torch.randn((1, t, 4, 128), device=device) * 0.05).to(torch.bfloat16).contiguous()
    k = (torch.randn((1, t, 4, 128), device=device) * 0.05).to(torch.bfloat16).contiguous()
    v = (torch.randn((1, t, 8, 128), device=device) * 0.05).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, 8), device=device) * 0.01).float().contiguous()
    beta = (0.5 + torch.rand((1, t, 8), device=device)).float().contiguous()
    return q, k, v, g, beta


@pytest.mark.parametrize("t", [64, 128])
def test_native_wu_matches_bt64_scalar_reference(t: int) -> None:
    q, k, v, g, beta = _inputs(t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a)
    w_ref, u_ref = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=64)
    w, u = qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)
    w_error = (w - w_ref).abs()
    u_error = (u - u_ref).abs()
    print(f"T={t} W max_abs={w_error.max().item():.8g} mean_abs={w_error.mean().item():.8g}")
    print(f"T={t} U max_abs={u_error.max().item():.8g} mean_abs={u_error.mean().item():.8g}")
    assert w_error.max().item() <= 2.0e-3
    assert u_error.max().item() <= 2.0e-3


@pytest.mark.parametrize("t", [64, 128])
def test_native_chunko_matches_bt64_scalar_reference(t: int) -> None:
    q, k, _, g, _ = _inputs(t, seed=17)
    torch.manual_seed(300 + t)
    num_chunks = t // 64
    v_new = (torch.randn((1, t, 8, 128), device="cuda") * 0.03).float().contiguous()
    h_bf16 = (torch.randn((1, num_chunks, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
    out_ref = qwen_gdn_chunk_o_avelang_v6_standalone(q, k, v_new, h_bf16.float().contiguous(), g, chunk_size=64)
    out = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g)
    error = (out - out_ref).abs()
    print(f"T={t} chunk_o max_abs={error.max().item():.8g} mean_abs={error.mean().item():.8g}")
    assert error.max().item() <= 4.0e-3


def test_native_chunko_carries_first_tile_into_later_tiles() -> None:
    t = 64
    q, k, _, g, _ = _inputs(t, seed=57)
    v_new = torch.zeros((1, t, 8, 128), dtype=torch.float32, device="cuda")
    v_new[:, :16].fill_(0.03125)
    h_bf16 = torch.zeros((1, 1, 8, 128, 128), dtype=torch.bfloat16, device="cuda")
    out_ref = qwen_gdn_chunk_o_avelang_v6_standalone(q, k, v_new, h_bf16.float().contiguous(), g, chunk_size=64)
    out = qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g)
    error = (out - out_ref).abs()
    later_energy = out[:, 16:].abs().max().item()
    print(f"cross-tile max_abs={error.max().item():.8g} later_energy={later_energy:.8g}")
    assert later_energy > 0.0
    assert error.max().item() <= 4.0e-3

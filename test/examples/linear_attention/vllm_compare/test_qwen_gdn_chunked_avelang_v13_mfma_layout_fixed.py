from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_gdr_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v12_mfma_layout_fixed import qwen_gdn_chunked_avelang_v12_mfma_layout
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v13_mfma_layout,
    qwen_gdn_chunked_avelang_v13_mfma_layout,
)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _make_inputs(t: int, with_initial_state: bool, seed: int):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def _build_chunk_gdr_inputs(k, v, g, beta, chunk_size: int):
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size, prefer_optimized=True)
    return g_cumsum, w, u


def _err(actual: torch.Tensor, expected: torch.Tensor):
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    return max_abs, max_rel


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    max_abs, max_rel = _err(actual, expected)
    print(f"{name}_max_abs={max_abs:.9g},{name}_max_rel={max_rel:.9g}")
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


@pytest.mark.parametrize("t", [32, 64, 512, 1024])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v13_chunk_gdr_bt32_matches_v6_chunk_gdr(t: int, with_initial_state: bool):
    chunk = 32
    _, k, v, g, beta, initial_state = _make_inputs(t, with_initial_state, seed=5000 + t + int(with_initial_state))
    g_cumsum, w, u = _build_chunk_gdr_inputs(k, v, g, beta, chunk)

    h_ref, vn_ref, fs_ref = qwen_gdn_chunk_gdr_avelang_v6_standalone(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk,
    )
    h_new, vn_new, fs_new = qwen_gdn_chunk_gdr_avelang_v13_mfma_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk,
    )

    _assert_close("chunk_gdr_h", h_new, h_ref, atol=2.5e-1, rtol=2.5e-1)
    _assert_close("chunk_gdr_vn", vn_new, vn_ref, atol=2.5e-1, rtol=2.5e-1)
    _assert_close("chunk_gdr_final_state", fs_new, fs_ref, atol=2.5e-1, rtol=2.5e-1)


@pytest.mark.parametrize("t", [32, 64, 512, 1024])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v13_full_forward_bt32_matches_v12_bt16(t: int, with_initial_state: bool):
    q, k, v, g, beta, initial_state = _make_inputs(t, with_initial_state, seed=6000 + t + int(with_initial_state))
    scale = 128 ** -0.5

    out_ref, fs_ref = qwen_gdn_chunked_avelang_v12_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )
    out_new, fs_new = qwen_gdn_chunked_avelang_v13_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=32,
    )

    _assert_close("full_output", out_new, out_ref, atol=2.5e-1, rtol=2.5e-1)
    _assert_close("final_state", fs_new, fs_ref, atol=2.5e-1, rtol=2.5e-1)

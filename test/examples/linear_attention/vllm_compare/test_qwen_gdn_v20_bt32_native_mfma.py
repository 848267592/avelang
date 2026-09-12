from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout
from qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_fixed import qwen_gdn_chunked_avelang_v19_bt32_mfma_layout
from qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout,
    qwen_gdn_kkt_avelang_v20_bt32_mfma_layout,
    qwen_gdn_mfma_32x32_smoke_v20,
    qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout,
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


def _build_wu_inputs(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int):
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    return g_cumsum, a, a_solved


def _err(actual: torch.Tensor, expected: torch.Tensor):
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    return max_abs, max_rel


def _print_first_mismatch(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    close = torch.isclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    bad = torch.nonzero(~close)
    if bad.numel() == 0:
        return
    idx = tuple(int(x) for x in bad[0].detach().cpu().tolist())
    print(f"{name}_first_mismatch_idx={idx},actual={actual[idx].item():.9g},expected={expected[idx].item():.9g}")


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-3, rtol: float = 1e-3):
    max_abs, max_rel = _err(actual, expected)
    print(f"{name}_max_abs={max_abs:.9g},{name}_max_rel={max_rel:.9g}")
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        _print_first_mismatch(name, actual, expected, atol, rtol)
        assert False, name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


def test_v20_mfma_32x32_smoke_matches_torch():
    torch.manual_seed(20000)
    a = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16).contiguous()
    b = torch.randn((32, 32), device="cuda", dtype=torch.bfloat16).contiguous()
    c = qwen_gdn_mfma_32x32_smoke_v20(a, b)
    ref = a.float() @ b.float().T
    _assert_close("smoke", c, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("t", [32, 64, 512, 1024])
def test_v20_bt32_kkt_matches_v6(t: int):
    chunk = 32
    _, k, _, g, beta, _ = _make_inputs(t, False, seed=20100 + t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a_ref = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_new = qwen_gdn_kkt_avelang_v20_bt32_mfma_layout(k, g_cumsum, beta, chunk_size=chunk)
    _assert_close("KKT", a_new, a_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("t", [32, 64, 512, 1024])
def test_v20_bt32_w_u_matches_v6(t: int):
    chunk = 32
    _, k, v, g, beta, _ = _make_inputs(t, False, seed=20200 + t)
    g_cumsum, _, a_solved = _build_wu_inputs(k, v, g, beta, chunk)
    w_ref, u_ref = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk,
        prefer_optimized=True,
    )
    w_new, u_new = qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    _assert_close("w", w_new, w_ref, atol=1e-3, rtol=1e-3)
    _assert_close("u", u_new, u_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("t", [32, 64, 512])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v20_bt32_full_matches_v19(t: int, with_initial_state: bool):
    chunk = 32
    q, k, v, g, beta, initial_state = _make_inputs(t, with_initial_state, seed=20300 + t + int(with_initial_state))
    scale = 128 ** -0.5
    out_ref, fs_ref = qwen_gdn_chunked_avelang_v19_bt32_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk,
    )
    out_new, fs_new = qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk,
    )
    _assert_close("output", out_new, out_ref, atol=1e-3, rtol=1e-3)
    _assert_close("final_state", fs_new, fs_ref, atol=1e-3, rtol=1e-3)

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import qwen_gdn_chunked_avelang_v13_mfma_layout
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (
    qwen_gdn_chunked_avelang_v18_bt32_layout,
    qwen_gdn_solve_avelang_v18_layout,
)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _make_a(t: int, chunk_size: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    b, hk, hv, kdim = 1, 4, 8, 128
    k = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)


def _make_full_inputs(t: int, with_initial_state: bool, seed: int):
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


def _err(actual: torch.Tensor, expected: torch.Tensor):
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    return max_abs, max_rel


def _print_first_mismatch(name: str, actual: torch.Tensor, expected: torch.Tensor, a: torch.Tensor, atol: float, rtol: float):
    close = torch.isclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    bad = torch.nonzero(~close)
    if bad.numel() == 0:
        return
    idx = tuple(int(x) for x in bad[0].detach().cpu().tolist())
    print(f"{name}_first_mismatch_idx={idx},actual={actual[idx].item():.9g},expected={expected[idx].item():.9g}")
    _, token_idx, head_idx, col_idx = idx
    chunk_size = a.shape[-1]
    chunk_start = (token_idx // chunk_size) * chunk_size
    row0 = max(chunk_start, token_idx - 2)
    row1 = min(a.shape[1], token_idx + 3)
    print(f"{name}_mismatch_chunk_start={chunk_start},head={head_idx},col={col_idx}")
    print("a_local_rows=", a[0, row0:row1, head_idx, :].detach().cpu())
    print("v6_rows=", expected[0, row0:row1, head_idx, :].detach().cpu())
    print("v18_rows=", actual[0, row0:row1, head_idx, :].detach().cpu())


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, a: torch.Tensor, atol: float, rtol: float):
    max_abs, max_rel = _err(actual, expected)
    print(f"{name}_max_abs={max_abs:.9g},{name}_max_rel={max_rel:.9g}")
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        _print_first_mismatch(name, actual, expected, a, atol, rtol)
        assert False, name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


@pytest.mark.parametrize("t", [32, 64, 512, 1024])
def test_v18_solve_bt32_matches_v6(t: int):
    chunk_size = 32
    a = _make_a(t, chunk_size, seed=18032 + t)
    solved_ref = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    solved_new = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    _assert_close("bt32_solve", solved_new, solved_ref, a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("t", [64, 128, 512, 1024])
def test_v18_solve_bt64_matches_v6(t: int):
    chunk_size = 64
    a = _make_a(t, chunk_size, seed=18064 + t)
    solved_ref = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    solved_new = qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk_size)
    _assert_close("bt64_solve", solved_new, solved_ref, a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("t", [32, 64])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v18_bt32_full_smoke_matches_v13(t: int, with_initial_state: bool):
    chunk_size = 32
    q, k, v, g, beta, initial_state = _make_full_inputs(t, with_initial_state, seed=18320 + t + int(with_initial_state))
    scale = 128 ** -0.5

    out_ref, fs_ref = qwen_gdn_chunked_avelang_v13_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    out_new, fs_new = qwen_gdn_chunked_avelang_v18_bt32_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )

    dummy_a = torch.empty((1, t, 8, chunk_size), device="cuda", dtype=torch.float32)
    _assert_close("bt32_full_output", out_new, out_ref, dummy_a, atol=1e-3, rtol=1e-3)
    _assert_close("bt32_full_final_state", fs_new, fs_ref, dummy_a, atol=1e-3, rtol=1e-3)

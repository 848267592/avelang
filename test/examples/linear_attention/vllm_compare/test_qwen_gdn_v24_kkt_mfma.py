from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed import (
    qwen_gdn_chunked_avelang_v23_gdr_distributed_layout,
)
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v24_kkt_mfma_layout,
    qwen_gdn_kkt_avelang_v24_bt16_mfma_layout,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


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


@pytest.mark.parametrize("t", [16, 32, 64, 512, 1024])
def test_v24_kkt_matches_v6(t: int):
    chunk = 16
    _, k, _, g, beta, _ = _make_inputs(t, False, seed=24100 + t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a_ref = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_new = qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(k, g_cumsum, beta, chunk_size=chunk)
    _assert_close("KKT", a_new, a_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("t", [16, 64, 512])
def test_v24_solve_after_kkt_matches_v6(t: int):
    chunk = 16
    _, k, _, g, beta, _ = _make_inputs(t, False, seed=24200 + t)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a_ref = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_new = qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(k, g_cumsum, beta, chunk_size=chunk)
    solved_ref = qwen_gdn_solve_avelang_v6_standalone(a_ref, chunk_size=chunk)
    solved_new = qwen_gdn_solve_avelang_v6_standalone(a_new, chunk_size=chunk)
    _assert_close("a_solved", solved_new, solved_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("t", [16, 32, 64, 512])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v24_full_matches_v23(t: int, with_initial_state: bool):
    chunk = 16
    q, k, v, g, beta, initial_state = _make_inputs(
        t,
        with_initial_state,
        seed=24300 + t + int(with_initial_state) * 101,
    )
    scale = 128 ** -0.5
    out_ref, fs_ref = qwen_gdn_chunked_avelang_v23_gdr_distributed_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk,
    )
    out_new, fs_new = qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(
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

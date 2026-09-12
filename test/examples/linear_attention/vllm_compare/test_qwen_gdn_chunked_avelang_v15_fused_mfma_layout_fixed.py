from __future__ import annotations

import pytest
import torch

import qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed as v15
from qwen_gdn_chunked_avelang_v14_mfma_layout_fixed import qwen_gdn_chunked_avelang_v14_mfma_layout
from qwen_gdn_chunked_avelang_v15_fused_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v15_fused_mfma_layout,
    qwen_gdn_chunked_avelang_v15_fused_mfma_layout_full,
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


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float):
    max_abs, max_rel = _err(actual, expected)
    print(f"{name}_max_abs={max_abs:.9g},{name}_max_rel={max_rel:.9g}")
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        _print_first_mismatch(name, actual, expected, atol, rtol)
        assert False, name


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


@pytest.mark.parametrize("t", [16, 32, 64, 512])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v15_fused_full_forward_matches_v14(t: int, with_initial_state: bool):
    chunk = 16
    q, k, v, g, beta, initial_state = _make_inputs(t, with_initial_state, seed=9000 + t + int(with_initial_state))
    scale = 128 ** -0.5

    out_ref, fs_ref = qwen_gdn_chunked_avelang_v14_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk,
    )
    out_new, fs_new = qwen_gdn_chunked_avelang_v15_fused_mfma_layout(
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


def test_v15_fused_full_does_not_call_non_fused_wrappers(monkeypatch):
    q, k, v, g, beta, initial_state = _make_inputs(16, False, seed=9916)

    def fail_non_fused(*args, **kwargs):
        raise AssertionError("v15 fused full path called a non-fused chunk_gdr/chunk_o wrapper")

    monkeypatch.setattr(v15, "qwen_gdn_chunk_gdr_avelang_v14_mfma_layout", fail_non_fused)
    monkeypatch.setattr(v15, "qwen_gdn_chunk_o_avelang_v14_mfma_layout", fail_non_fused)

    result = qwen_gdn_chunked_avelang_v15_fused_mfma_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=128 ** -0.5,
        chunk_size=16,
    )
    assert len(result) == 4
    _, output, _, final_state = result
    assert output.shape == (1, 16, 8, 128)
    assert final_state.shape == (1, 8, 128, 128)

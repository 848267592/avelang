from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v26_gdr_bt64_regstate_layout_fixed import (
    MODE_FULL_V26C,
    MODE_NO_DECAY,
    MODE_NO_H_STORE,
    MODE_NO_PRED_STATE_CONVERT,
    MODE_NO_VN_STORE,
    MODE_PRED_ONLY,
    MODE_UPDATE_ONLY,
    MODE_V26A_CORE,
    MODE_V26B_IO,
    qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout,
    qwen_gdn_chunk_gdr_torch_ref_bt64_regstate,
)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _make_inputs(t: int, with_initial_state: bool, seed: int):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    k = _l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    w = (torch.randn(b, t, hv, kdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    u = (torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.float32) * 0.05).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, w, u, g, initial_state


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


@pytest.mark.parametrize("t", [64, 128, 512])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_v26_bt64_bv32_regstate_full_matches_torch_ref(t: int, with_initial_state: bool):
    k, w, u, g, initial_state = _make_inputs(t, with_initial_state, seed=26000 + t + int(with_initial_state))

    h_ref, vn_ref, fs_ref = qwen_gdn_chunk_gdr_torch_ref_bt64_regstate(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        chunk_size=64,
    )
    h_new, vn_new, fs_new = qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        chunk_size=64,
        variant=MODE_FULL_V26C,
    )

    _assert_close("h", h_new, h_ref, atol=1e-3, rtol=1e-3)
    _assert_close("vn", vn_new, vn_ref, atol=2e-2, rtol=2e-2)
    _assert_close("final_state", fs_new, fs_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "variant",
    [
        MODE_PRED_ONLY,
        MODE_UPDATE_ONLY,
        MODE_NO_H_STORE,
        MODE_NO_VN_STORE,
        MODE_NO_DECAY,
        MODE_NO_PRED_STATE_CONVERT,
        MODE_V26A_CORE,
        MODE_V26B_IO,
    ],
)
def test_v26_ablation_variants_smoke(variant: str):
    k, w, u, g, initial_state = _make_inputs(64, True, seed=26123)
    h, vn, fs = qwen_gdn_chunk_gdr_avelang_v26_bt64_bv32_regstate_mfma_layout(
        k,
        w,
        u,
        g,
        initial_state=initial_state,
        chunk_size=64,
        variant=variant,
    )
    torch.cuda.synchronize()
    assert h.shape == (1, 1, 8, 128, 128)
    assert vn.shape == u.shape
    assert fs.shape == (1, 8, 128, 128)

"""Correctness tests for custom MFMA tile-dot prototypes."""

from __future__ import annotations

import pytest
import torch

from prototype_qwen_gdn_mfma_pred_custom import BT, BV, delta_custom, matmul16_custom, pred_custom

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


def check(actual: torch.Tensor, expected: torch.Tensor, *, name: str, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    assert torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol), (
        f"{name}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"
    )


def test_custom_mfma_pred_bk64_and_k128():
    gen = torch.Generator(device="cuda").manual_seed(13001)
    w64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    h64 = torch.randn((BV, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    check(pred_custom(w64, h64), w64.float() @ h64.float().T, name="pred_bk64")

    w128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    h128 = torch.randn((BV, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    check(pred_custom(w128, h128), w128.float() @ h128.float().T, name="pred_k128")


def test_custom_mfma_matmul16_and_delta():
    gen = torch.Generator(device="cuda").manual_seed(13003)
    a16 = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16, generator=gen)
    b16 = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16, generator=gen)
    check(matmul16_custom(a16, b16), a16.float() @ b16.float().T, name="matmul16")

    v_new = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16, generator=gen)
    k64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    check(delta_custom(v_new, k64), v_new.float().T @ k64.float(), name="delta_bk64")

    k128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    check(delta_custom(v_new, k128), v_new.float().T @ k128.float(), name="delta_k128")

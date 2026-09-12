from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only import (
    qwen_gdn_pred_only_avelang_v29_mfma32,
    qwen_gdn_pred_only_torch_reference,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


def _make_inputs(t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    w = (torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    return w, u, initial_state


@pytest.mark.parametrize("t", [64, 512])
def test_v29_pred_only_matches_torch_reference(t: int):
    w, u, initial_state = _make_inputs(t, seed=29000 + t)
    vn = qwen_gdn_pred_only_avelang_v29_mfma32(w, u, initial_state)
    ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)

    err = (vn.float() - ref.float()).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    max_rel = (err / ref.float().abs().clamp_min(1e-6)).max().item()
    print(f"T={t},max_abs={max_abs:.9g},mean_abs={mean_abs:.9g},max_rel={max_rel:.9g}")

    assert max_abs <= 5e-2
    assert mean_abs <= 1e-2


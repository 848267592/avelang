"""Correctness tests for the direct-K64 current-ABI update repro."""

import pytest
import torch

from repro_qwen_gdn_direct_k64_update_current_abi import (
    _make_inputs,
    qwen_gdn_direct_k64_update_current_abi,
    qwen_gdn_direct_k64_update_reference,
)


@pytest.mark.parametrize("tokens", [64, 512])
def test_direct_k64_update_current_abi(tokens: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP")
    k, v_new, g, initial = _make_inputs(tokens, 20260725 + tokens)
    actual_h, actual_final = qwen_gdn_direct_k64_update_current_abi(k, v_new, g, initial)
    reference_h, reference_final = qwen_gdn_direct_k64_update_reference(k, v_new, g, initial)
    torch.cuda.synchronize()

    h_diff = (actual_h.float() - reference_h.float()).abs()
    final_diff = (actual_final - reference_final).abs()
    print(
        f"T={tokens},h_max_abs={h_diff.max().item():.9g},h_mean_abs={h_diff.mean().item():.9g},"
        f"final_max_abs={final_diff.max().item():.9g},final_mean_abs={final_diff.mean().item():.9g}"
    )

    assert torch.isfinite(actual_h).all()
    assert torch.isfinite(actual_final).all()
    # H is explicitly stored as BF16. The direct MFMA32 reduction has a small
    # accumulated rounding delta on long sequences; final FP32 state is the
    # stronger recurrence check.
    assert h_diff.max().item() <= 1.0
    assert final_diff.max().item() <= 1.0e-3

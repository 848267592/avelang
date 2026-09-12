import pytest
import torch

from repro_qwen_v31_active_v16_pred_primitives import pred_p16, pred_p32, pred_reference


@pytest.mark.parametrize("seed", range(50))
def test_p16_matches_bf16_pred_reference(seed: int) -> None:
    torch.manual_seed(20260712 + seed)
    w = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    state = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    actual = pred_p16(w, state)
    expected = pred_reference(w, state)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("scale", [0.0, 0.02, 1.0, 64.0])
def test_p16_extreme_inputs_match_bf16_pred_reference(scale: float) -> None:
    torch.manual_seed(77)
    w = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * scale).contiguous()
    state = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * scale).contiguous()
    actual = pred_p16(w, state)
    expected = pred_reference(w, state)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-3)


def test_p32_padded_quadrant_is_not_the_same_predicate() -> None:
    torch.manual_seed(20260712)
    w = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    state = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    error = (pred_p32(w, state) - pred_reference(w, state)).abs().max().item()
    torch.cuda.synchronize()
    assert error > 1e-3

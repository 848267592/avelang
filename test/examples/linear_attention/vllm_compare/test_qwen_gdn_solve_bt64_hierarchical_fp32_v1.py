from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import qwen_gdn_solve_bt64_hierarchical_fp32_v1


BT = 64


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _make_kkt_a(t: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    k = _l2norm(torch.randn((1, t, 4, 128), dtype=torch.bfloat16, device="cuda"))
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, 8), device="cuda")) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn((1, t, 8), device="cuda")).contiguous()
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=BT, prefer_optimized=True)


def _make_kkt_consumer_inputs(t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    k = _l2norm(torch.randn((1, t, 4, 128), dtype=torch.bfloat16, device="cuda"))
    v = torch.randn((1, t, 8, 128), dtype=torch.bfloat16, device="cuda").contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, 8), device="cuda")) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn((1, t, 8), device="cuda")).contiguous()
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=BT, prefer_optimized=True)
    return k, v, g_cumsum, beta, a


def _make_random_lower_a(t: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    a = torch.randn((1, t, 8, BT), dtype=torch.float32, device="cuda") * 0.02
    lower = torch.tril(torch.ones((BT, BT), dtype=torch.float32, device="cuda"), diagonal=-1)
    lower = lower.repeat(t // BT, 1).view(1, t, 1, BT)
    return (a * lower).contiguous()


def _error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    return float(diff.max().item()), float(diff.mean().item()), float((diff / expected.float().abs().clamp_min(1e-6)).max().item())


def _print_first_mismatch(actual: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    bad = torch.nonzero((actual.float() - expected.float()).abs() > atol)
    if bad.numel() == 0:
        return
    index = tuple(int(v) for v in bad[0].cpu().tolist())
    print(f"first_mismatch={index},actual={actual[index].item():.9g},expected={expected[index].item():.9g}")


def _residual_inf(a: torch.Tensor, solved: torch.Tensor) -> float:
    chunks = a.shape[1] // BT
    identity = torch.eye(BT, device=a.device, dtype=torch.float32)
    worst = torch.zeros((), device=a.device, dtype=torch.float32)
    for chunk_idx in range(chunks):
        start = chunk_idx * BT
        lhs = a[0, start : start + BT].permute(1, 0, 2).float() + identity
        rhs = solved[0, start : start + BT].permute(1, 0, 2).float()
        worst = torch.maximum(worst, ((lhs @ rhs) - identity).abs().max())
    return float(worst.item())


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP GPU")


@pytest.mark.parametrize("t", [64, 128, 512])
def test_bt64_hierarchical_fp32_matches_v6_random_lower(t: int) -> None:
    a = _make_random_lower_a(t, seed=20260716 + t)
    expected = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=BT)
    actual = qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    max_abs, mean_abs, max_rel = _error(actual, expected)
    residual = _residual_inf(a, actual)
    print(f"random_lower,T={t},max_abs={max_abs:.9g},mean_abs={mean_abs:.9g},max_rel={max_rel:.9g},residual_inf={residual:.9g}")
    _print_first_mismatch(actual, expected, 1e-5)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    assert residual <= 1e-5


@pytest.mark.parametrize("t", [64, 512])
def test_bt64_hierarchical_fp32_matches_kkt_v6_and_v18(t: int) -> None:
    a = _make_kkt_a(t, seed=20260816 + t)
    expected_v6 = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=BT)
    expected_v18 = qwen_gdn_solve_avelang_v18_bt64_layout(a)
    actual = qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    max_abs, mean_abs, max_rel = _error(actual, expected_v6)
    v18_max_abs, _, _ = _error(actual, expected_v18)
    residual = _residual_inf(a, actual)
    print(
        f"kkt,T={t},max_abs_v6={max_abs:.9g},mean_abs_v6={mean_abs:.9g},"
        f"max_rel_v6={max_rel:.9g},max_abs_v18={v18_max_abs:.9g},residual_inf={residual:.9g}"
    )
    _print_first_mismatch(actual, expected_v6, 1e-5)
    assert torch.allclose(actual, expected_v6, atol=1e-5, rtol=1e-5)
    assert torch.allclose(actual, expected_v18, atol=1e-5, rtol=1e-5)
    assert residual <= 1e-5


@pytest.mark.parametrize("t", [64, 512])
def test_bt64_hierarchical_fp32_preserves_v6_w_u_consumer(t: int) -> None:
    k, v, g_cumsum, beta, a = _make_kkt_consumer_inputs(t, seed=20261016 + t)
    solved_v6 = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=BT)
    solved_v1 = qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    w_v6, u_v6 = qwen_gdn_w_u_avelang_v6_standalone(
        k, v, g_cumsum, beta, solved_v6, chunk_size=BT, prefer_optimized=True
    )
    w_v1, u_v1 = qwen_gdn_w_u_avelang_v6_standalone(
        k, v, g_cumsum, beta, solved_v1, chunk_size=BT, prefer_optimized=True
    )
    w_max, w_mean, _ = _error(w_v1, w_v6)
    u_max, u_mean, _ = _error(u_v1, u_v6)
    print(f"w_u_consumer,T={t},w_max_abs={w_max:.9g},w_mean_abs={w_mean:.9g},u_max_abs={u_max:.9g},u_mean_abs={u_mean:.9g}")
    assert torch.allclose(w_v1, w_v6, atol=1e-5, rtol=1e-5)
    assert torch.allclose(u_v1, u_v6, atol=1e-5, rtol=1e-5)

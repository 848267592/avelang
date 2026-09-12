"""Correctness smoke tests for the v7 vLLM-layout Qwen GDN wrapper."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunked_avelang_v6_standalone
from qwen_gdn_chunked_avelang_v7_vllm_layout_fixed import (
    qwen_gdn_chunked_avelang_v7_vllm_layout,
    qwen_gdn_chunked_avelang_v7_vllm_layout_full,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")

OUTPUT_NAMES = ("g_cumsum", "output", "A_solved", "chunk_states", "final_state")
FP32_ATOL = 8e-5
FP32_RTOL = 8e-5
BF16_ATOL = 5e-4
BF16_RTOL = 5e-4


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    y = x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)
    return y.to(x.dtype).contiguous()


def make_inputs(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    dtype: torch.dtype,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        batch_size,
        num_tokens,
        num_k_heads,
        head_dim_k,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        batch_size,
        num_tokens,
        num_k_heads,
        head_dim_k,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ).contiguous()
    q = l2norm(q)
    k = l2norm(k)
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    )
    g = (g / 16.0).contiguous()
    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_v_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    return q.contiguous(), k.contiguous(), v, g, beta


def make_initial_state(
    batch_size: int,
    num_v_heads: int,
    head_dim_v: int,
    head_dim_k: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        batch_size,
        num_v_heads,
        head_dim_v,
        head_dim_k,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()


def assert_forward_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    for name, actual_tensor, expected_tensor in zip(OUTPUT_NAMES, actual, expected, strict=True):
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=atol,
            rtol=rtol,
            msg=f"{name} mismatch",
        )


@pytest.mark.parametrize(
    (
        "case_name",
        "dtype",
        "batch_size",
        "num_tokens",
        "num_k_heads",
        "num_v_heads",
        "head_dim_k",
        "head_dim_v",
        "chunk_size",
        "use_initial_state",
    ),
    [
        ("fp32_grouped_initial", torch.float32, 1, 7, 2, 4, 4, 3, 4, True),
        ("fp32_multi_batch_partial", torch.float32, 2, 9, 1, 2, 4, 3, 4, False),
        ("bf16_grouped_initial", torch.bfloat16, 1, 7, 2, 4, 4, 3, 4, True),
        ("bf16_multi_batch_partial", torch.bfloat16, 2, 9, 1, 2, 4, 3, 4, False),
    ],
)
def test_qwen_gdn_chunked_v7_vllm_layout_matches_v6_baseline(
    case_name: str,
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    chunk_size: int,
    use_initial_state: bool,
):
    q, k, v, g, beta = make_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        dtype,
        seed=7300 + len(case_name),
    )
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(
            batch_size,
            num_v_heads,
            head_dim_v,
            head_dim_k,
            seed=9100 + len(case_name),
        )
    scale = head_dim_k**-0.5

    expected = qwen_gdn_chunked_avelang_v6_standalone(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    actual = qwen_gdn_chunked_avelang_v7_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )

    if dtype is torch.bfloat16:
        atol, rtol = BF16_ATOL, BF16_RTOL
    else:
        atol, rtol = FP32_ATOL, FP32_RTOL
    assert_forward_close(actual, expected, atol=atol, rtol=rtol)

    thin_output, thin_final_state = qwen_gdn_chunked_avelang_v7_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    torch.testing.assert_close(thin_output, actual[1], atol=atol, rtol=rtol)
    torch.testing.assert_close(thin_final_state, actual[4], atol=atol, rtol=rtol)


def test_qwen_gdn_chunked_v7_vllm_layout_value_tile2_matches_v6_baseline():
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v, chunk_size = 1, 7, 2, 4, 4, 3, 4
    q, k, v, g, beta = make_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        torch.bfloat16,
        seed=8102,
    )
    initial_state = make_initial_state(
        batch_size,
        num_v_heads,
        head_dim_v,
        head_dim_k,
        seed=9102,
    )
    scale = head_dim_k**-0.5

    expected = qwen_gdn_chunked_avelang_v6_standalone(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    actual = qwen_gdn_chunked_avelang_v7_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
        value_tile=2,
    )
    assert_forward_close(actual, expected, atol=BF16_ATOL, rtol=BF16_RTOL)

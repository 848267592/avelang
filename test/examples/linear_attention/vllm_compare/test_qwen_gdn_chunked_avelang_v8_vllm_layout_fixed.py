"""Correctness tests for v8 V-block parallel chunk_gdr."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_gdr_avelang_v6_standalone,
    qwen_gdn_chunked_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v7_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v7_vllm_layout,
    qwen_gdn_chunked_avelang_v7_vllm_layout_full,
)
from qwen_gdn_chunked_avelang_v8_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v8_vllm_layout,
    qwen_gdn_chunked_avelang_v8_vllm_layout,
    qwen_gdn_chunked_avelang_v8_vllm_layout_full,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")

BF16_ATOL = 5e-4
BF16_RTOL = 5e-4
OUTPUT_NAMES = ("g_cumsum", "output", "A_solved", "chunk_states", "final_state")


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
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    k = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
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
    return (
        torch.randn(
            batch_size,
            num_v_heads,
            head_dim_v,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
    ).contiguous()


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
        msg=f"{name} mismatch",
    )


def assert_forward_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    prefix: str,
) -> None:
    for name, actual_tensor, expected_tensor in zip(OUTPUT_NAMES, actual, expected, strict=True):
        assert_close(f"{prefix}:{name}", actual_tensor, expected_tensor)


@pytest.mark.parametrize(
    (
        "case_name",
        "batch_size",
        "num_tokens",
        "num_k_heads",
        "num_v_heads",
        "head_dim_k",
        "head_dim_v",
        "chunk_size",
        "use_initial_state",
        "parallel_mode",
        "block_v",
        "block_k",
    ),
    [
        ("grouped_heads_with_initial_vk", 1, 7, 2, 4, 8, 7, 4, True, "vk", 4, 4),
        ("partial_chunk_no_initial_vk", 2, 9, 1, 2, 8, 9, 4, False, "vk", 8, 4),
        ("vk_full_k_wave_no_initial", 1, 5, 1, 2, 32, 8, 4, False, "vk", 8, 32),
        ("vk_tail_with_initial", 1, 5, 1, 1, 16, 17, 4, True, "vk", 8, 8),
        ("vblock_tail_with_initial", 1, 5, 1, 1, 4, 65, 4, True, "vblock", 64, 8),
    ],
)
def test_qwen_gdn_chunked_v8_parallel_matches_v6_and_v7(
    case_name: str,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    chunk_size: int,
    use_initial_state: bool,
    parallel_mode: str,
    block_v: int,
    block_k: int,
):
    q, k, v, g, beta = make_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=8800 + len(case_name),
    )
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(
            batch_size,
            num_v_heads,
            head_dim_v,
            head_dim_k,
            seed=9900 + len(case_name),
        )
    scale = head_dim_k**-0.5

    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )

    h6, vn6, final6 = qwen_gdn_chunk_gdr_avelang_v6_standalone(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )
    h7, vn7, final7 = qwen_gdn_chunk_gdr_avelang_v7_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        prefer_optimized=True,
        value_tile=1,
    )
    h8, vn8, final8 = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )

    assert_close(f"{case_name}:h_vs_v6", h8, h6)
    assert_close(f"{case_name}:vn_vs_v6", vn8, vn6)
    assert_close(f"{case_name}:final_vs_v6", final8, final6)
    assert_close(f"{case_name}:h_vs_v7", h8, h7)
    assert_close(f"{case_name}:vn_vs_v7", vn8, vn7)
    assert_close(f"{case_name}:final_vs_v7", final8, final7)

    full6 = qwen_gdn_chunked_avelang_v6_standalone(
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
    full7 = qwen_gdn_chunked_avelang_v7_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
        value_tile=1,
    )
    full8 = qwen_gdn_chunked_avelang_v8_vllm_layout_full(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )
    assert_forward_close(full8, full6, prefix=f"{case_name}:full_vs_v6")
    assert_forward_close(full8, full7, prefix=f"{case_name}:full_vs_v7")

    thin_output, thin_final = qwen_gdn_chunked_avelang_v8_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        parallel_mode=parallel_mode,
        block_v=block_v,
        block_k=block_k,
    )
    assert_close(f"{case_name}:thin_output", thin_output, full8[1])
    assert_close(f"{case_name}:thin_final", thin_final, full8[4])

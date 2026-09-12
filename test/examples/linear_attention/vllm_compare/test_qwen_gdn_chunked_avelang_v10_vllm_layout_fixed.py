"""Correctness tests for v10 chunk-batched chunk_gdr on Qwen3Next TP4 shape."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v9_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v9_vllm_layout,
    qwen_gdn_chunked_avelang_v9_vllm_layout,
)
from qwen_gdn_chunked_avelang_v10_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v10_vllm_layout,
    qwen_gdn_chunked_avelang_v10_vllm_layout,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(seed: int = 10007):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = 1, 512, 4, 8, 128, 128
    q = l2norm(torch.randn(batch_size, num_tokens, num_k_heads, head_dim_k, device="cuda", dtype=torch.bfloat16, generator=generator))
    k = l2norm(torch.randn(batch_size, num_tokens, num_k_heads, head_dim_k, device="cuda", dtype=torch.bfloat16, generator=generator))
    v = torch.randn(batch_size, num_tokens, num_v_heads, head_dim_v, device="cuda", dtype=torch.bfloat16, generator=generator).contiguous()
    g = torch.nn.functional.logsigmoid(torch.randn(batch_size, num_tokens, num_v_heads, device="cuda", dtype=torch.float32, generator=generator))
    g = (g / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(batch_size, num_tokens, num_v_heads, device="cuda", dtype=torch.float32, generator=generator)).contiguous()
    initial_state = (torch.randn(batch_size, num_v_heads, head_dim_v, head_dim_k, device="cuda", dtype=torch.float32, generator=generator) * 0.01).contiguous()
    return q, k, v, g, beta, initial_state


def build_gdr_inputs(seed: int = 10007):
    q, k, v, g, beta, initial_state = make_inputs(seed=seed)
    chunk_size = 4
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk_size, prefer_optimized=True)
    return q, k, v, g, beta, initial_state, g_cumsum, w, u


def assert_close_named(actual: torch.Tensor, expected: torch.Tensor, *, name: str, atol: float = 8e-4, rtol: float = 8e-4) -> None:
    max_abs = (actual.float() - expected.float()).abs().max().item()
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = ((actual.float() - expected.float()).abs() / denom).max().item()
    assert max_abs <= atol and max_rel <= rtol, f"{name}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"


def test_chunk_gdr_v10_chunk_vk_matches_v9_vk_on_qwen3next_tp4():
    _q, k, _v, _g, _beta, initial_state, g_cumsum, w, u = build_gdr_inputs()
    expected_h, expected_vn, expected_final = qwen_gdn_chunk_gdr_avelang_v9_vllm_layout(
        k, w, u, g_cumsum, initial_state=initial_state, chunk_size=4, use_parallel_chunk_gdr=True, parallel_mode="vk", block_v=4, block_k=64
    )
    actual_h, actual_vn, actual_final = qwen_gdn_chunk_gdr_avelang_v10_vllm_layout(
        k, w, u, g_cumsum, initial_state=initial_state, chunk_size=4, use_parallel_chunk_gdr=True, parallel_mode="chunk_vk", block_v=4, block_k=64
    )
    torch.cuda.synchronize()
    assert_close_named(actual_h, expected_h, name="h")
    assert_close_named(actual_vn, expected_vn, name="vn")
    assert_close_named(actual_final, expected_final, name="final_state")


def test_full_v10_matches_v9_on_qwen3next_tp4():
    q, k, v, g, beta, initial_state = make_inputs(seed=10009)
    scale = 128**-0.5
    expected_output, expected_final = qwen_gdn_chunked_avelang_v9_vllm_layout(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=4,
        use_parallel_chunk_gdr=True, parallel_mode="vk", block_v=4, block_k=64,
        use_parallel_chunk_o=True, chunk_o_parallel_mode="vk", chunk_o_block_v=4, chunk_o_block_k=16,
    )
    actual_output, actual_final = qwen_gdn_chunked_avelang_v10_vllm_layout(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=4,
        use_parallel_chunk_gdr=True, parallel_mode="chunk_vk", block_v=4, block_k=64,
        use_parallel_chunk_o=True, chunk_o_parallel_mode="vk", chunk_o_block_v=4, chunk_o_block_k=16,
    )
    torch.cuda.synchronize()
    assert_close_named(actual_output, expected_output, name="output", atol=1e-3, rtol=5e-2)
    assert_close_named(actual_final, expected_final, name="final_state")

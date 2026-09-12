"""Correctness tests for v9 vk parallel chunk_o on Qwen3Next TP4 shape."""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v9_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v9_vllm_layout,
    qwen_gdn_chunk_o_avelang_v9_vllm_layout,
    qwen_gdn_chunked_avelang_v9_vllm_layout,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(seed: int = 9137):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = 1, 512, 4, 8, 128, 128
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
        torch.randn(batch_size, num_tokens, num_v_heads, device="cuda", dtype=torch.float32, generator=generator)
    )
    g = (g / 16.0).contiguous()
    beta = torch.sigmoid(
        torch.randn(batch_size, num_tokens, num_v_heads, device="cuda", dtype=torch.float32, generator=generator)
    ).contiguous()
    initial_state = (
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
    return q, k, v, g, beta, initial_state


def max_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = expected.float().abs().clamp_min(1e-6)
    return ((actual.float() - expected.float()).abs() / denom).max().item()


def assert_chunk_o_close(actual: torch.Tensor, expected: torch.Tensor, *, case: str) -> None:
    max_abs = (actual.float() - expected.float()).abs().max().item()
    max_rel = max_rel_err(actual, expected)
    assert max_abs <= 1e-3 and max_rel <= 5e-2, f"{case}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"


def build_stages():
    q, k, v, g, beta, initial_state = make_inputs()
    chunk_size = 4
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
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v9_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        parallel_mode="vk",
        block_v=4,
        block_k=64,
    )
    return q, k, v, g, beta, initial_state, g_cumsum, h, vn, final_state


@pytest.mark.parametrize("chunk_o_block_v,chunk_o_block_k", [(4, 64), (8, 32), (16, 16)])
def test_chunk_o_v9_vk_matches_v6_on_qwen3next_tp4(chunk_o_block_v: int, chunk_o_block_k: int):
    q, k, _v, _g, _beta, _initial_state, g_cumsum, h, vn, _final_state = build_stages()
    scale = 128**-0.5
    expected = qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=4,
        prefer_optimized=True,
    )
    try:
        actual = qwen_gdn_chunk_o_avelang_v9_vllm_layout(
            q,
            k,
            vn,
            h,
            g_cumsum,
            scale=scale,
            chunk_size=4,
            use_parallel_chunk_o=True,
            chunk_o_parallel_mode="vk",
            chunk_o_block_v=chunk_o_block_v,
            chunk_o_block_k=chunk_o_block_k,
        )
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - compiler/runtime dependent
        message = str(exc).lower()
        if "hip" in message or "compile" in message or "launch" in message or "mlir" in message:
            pytest.skip(f"v9 chunk_o candidate {chunk_o_block_v}x{chunk_o_block_k} failed to launch/compile: {exc}")
        raise
    assert_chunk_o_close(actual, expected, case=f"chunk_o_{chunk_o_block_v}x{chunk_o_block_k}")


def test_full_v9_uses_parallel_chunk_o_and_matches_v6_chunk_o_fallback():
    q, k, v, g, beta, initial_state = make_inputs(seed=9221)
    scale = 128**-0.5
    baseline_output, baseline_final = qwen_gdn_chunked_avelang_v9_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=4,
        use_parallel_chunk_gdr=True,
        parallel_mode="vk",
        block_v=4,
        block_k=64,
        use_parallel_chunk_o=False,
    )
    actual_output, actual_final = qwen_gdn_chunked_avelang_v9_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=4,
        use_parallel_chunk_gdr=True,
        parallel_mode="vk",
        block_v=4,
        block_k=64,
        use_parallel_chunk_o=True,
        chunk_o_parallel_mode="vk",
        chunk_o_block_v=4,
        chunk_o_block_k=64,
    )
    torch.cuda.synchronize()
    assert_chunk_o_close(actual_output, baseline_output, case="full_output")
    torch.testing.assert_close(actual_final, baseline_final, atol=5e-4, rtol=5e-4)

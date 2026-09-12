"""Qwen GDN v4 完整 chunked forward 测试。

当前测试分两层：先逐阶段对齐 PyTorch reference，方便定位错误；再做端到端
`qwen_gdn_forward_ref` 对齐。相比 v3，这里新增了 KKT、solve、w/u、chunk_gdr、
chunk_o 和完整 wrapper 的 correctness 测试。
当前测试仍然不覆盖 backward、cu_seqlens、bf16、raw_buffer、shared memory、MFMA、
向量化或性能优化。
"""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v4 import (
    qwen_gdn_chunk_cumsum_avelang_v4,
    qwen_gdn_chunk_gdr_avelang_v4,
    qwen_gdn_chunk_o_avelang_v4,
    qwen_gdn_chunked_avelang_v4,
    qwen_gdn_kkt_avelang_v4,
    qwen_gdn_solve_avelang_v4,
    qwen_gdn_w_u_avelang_v4,
)
from qwen_gdn_ref import (
    l2norm,
    qwen_gdn_forward_ref,
    torch_chunk_gdr_fwd,
    torch_chunk_o_fwd,
    torch_cumsum,
    torch_kkt_fwd,
    torch_solve,
    torch_w_u_fwd,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")

STAGE_ATOL = 3e-5
STAGE_RTOL = 3e-5
E2E_ATOL = 5e-5
E2E_RTOL = 5e-5


def make_inputs(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    *,
    seed: int = 123,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    k = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.float32,
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
    g = (g / 16).contiguous()
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
    return q, k, v, g, beta


def make_initial_state(
    batch_size: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        batch_size,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()


def test_qwen_gdn_chunked_v4_stage_by_stage():
    # 逐阶段测试使用 grouped-head + initial_state + partial chunk，尽量覆盖容易出错的索引路径。
    batch_size = 1
    num_tokens = 7
    num_k_heads = 2
    num_v_heads = 4
    head_dim_k = 4
    head_dim_v = 3
    chunk_size = 4
    q, k, v, g, beta = make_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=321,
    )
    initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=654)
    scale = head_dim_k**-0.5

    # cumsum：检查 chunk 内累加和边界重置。
    expected_g = torch_cumsum(g, chunk_size=chunk_size)
    actual_g = qwen_gdn_chunk_cumsum_avelang_v4(g, chunk_size=chunk_size)
    torch.testing.assert_close(actual_g, expected_g, atol=STAGE_ATOL, rtol=STAGE_RTOL)

    # kkt：检查严格过去 mask、head repeat 映射和 decay。
    expected_a = torch_kkt_fwd(k, expected_g, beta, chunk_size=chunk_size)
    actual_a = qwen_gdn_kkt_avelang_v4(k, expected_g, beta, chunk_size=chunk_size)
    torch.testing.assert_close(actual_a, expected_a, atol=STAGE_ATOL, rtol=STAGE_RTOL)

    # solve：检查每个 chunk/head 的小三角递推。
    expected_a_solved = torch_solve(expected_a)
    actual_a_solved = qwen_gdn_solve_avelang_v4(expected_a, chunk_size=chunk_size)
    torch.testing.assert_close(actual_a_solved, expected_a_solved, atol=STAGE_ATOL, rtol=STAGE_RTOL)

    # w/u：检查 solved A 乘回 k/v、beta 和 exp(g)。
    expected_w, expected_u = torch_w_u_fwd(k, v, expected_g, beta, expected_a_solved, chunk_size=chunk_size)
    actual_w, actual_u = qwen_gdn_w_u_avelang_v4(
        k,
        v,
        expected_g,
        beta,
        expected_a_solved,
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(actual_w, expected_w, atol=STAGE_ATOL, rtol=STAGE_RTOL)
    torch.testing.assert_close(actual_u, expected_u, atol=STAGE_ATOL, rtol=STAGE_RTOL)

    # chunk_gdr：检查 chunk 入口 state、vn 和 final_state。
    expected_h, expected_vn, expected_final = torch_chunk_gdr_fwd(
        k,
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    actual_h, actual_vn, actual_final = qwen_gdn_chunk_gdr_avelang_v4(
        k,
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(actual_h, expected_h, atol=STAGE_ATOL, rtol=STAGE_RTOL)
    torch.testing.assert_close(actual_vn, expected_vn, atol=STAGE_ATOL, rtol=STAGE_RTOL)
    torch.testing.assert_close(actual_final, expected_final, atol=STAGE_ATOL, rtol=STAGE_RTOL)

    # chunk_o：检查 chunk 入口 state 贡献和 chunk 内因果贡献。
    expected_o = torch_chunk_o_fwd(
        q,
        k,
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_o = qwen_gdn_chunk_o_avelang_v4(
        q,
        k,
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(actual_o, expected_o, atol=STAGE_ATOL, rtol=STAGE_RTOL)


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
    ),
    [
        # 基础 case：单 batch、单 head、单完整 chunk。
        ("small_base", 1, 4, 1, 1, 4, 4, 4, False),
        # 带 initial_state：检查初始 recurrent memory 会进入 chunk_gdr。
        ("with_initial_state", 1, 5, 2, 2, 4, 3, 4, True),
        # Hk < Hv：检查 q/k head repeat 映射。
        ("grouped_heads", 1, 7, 1, 2, 4, 3, 4, False),
        # Hk < Hv 且带 initial_state：组合覆盖 grouped head 和初始 state。
        ("grouped_heads_with_initial_state", 1, 7, 2, 4, 4, 3, 4, True),
        # B > 1：检查 batch 维 program 映射。
        ("multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
        # 较大 debug：chunk_size=8，检查 local solve 矩阵稍大时仍能通过。
        ("larger_debug", 1, 9, 2, 4, 8, 6, 8, False),
    ],
)
def test_qwen_gdn_chunked_v4_end_to_end(
    case_name: str,
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
        seed=123 + len(case_name),
    )
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=900 + len(case_name))
    scale = head_dim_k**-0.5

    expected_g, expected_o, expected_a, expected_h, expected_final = qwen_gdn_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_g, actual_o, actual_a, actual_h, actual_final = qwen_gdn_chunked_avelang_v4(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(actual_g, expected_g, atol=E2E_ATOL, rtol=E2E_RTOL)
    torch.testing.assert_close(actual_o, expected_o, atol=E2E_ATOL, rtol=E2E_RTOL)
    torch.testing.assert_close(actual_a, expected_a, atol=E2E_ATOL, rtol=E2E_RTOL)
    torch.testing.assert_close(actual_h, expected_h, atol=E2E_ATOL, rtol=E2E_RTOL)
    torch.testing.assert_close(actual_final, expected_final, atol=E2E_ATOL, rtol=E2E_RTOL)

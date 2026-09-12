"""Qwen GDN v5 最终半优化 chunked forward 测试。

当前版本测试完整前向五元组、各个 stage wrapper 和可切换的 v4 回退路径。
相比 v4 测试，本文件新增对真实优化的 KKT、融合 w/u、chunk_o 以及 v5 与 v4
逐张量一致性的检查，并覆盖指定的 partial chunk 与 grouped-head 组合。
当前测试仍然不覆盖 bf16、backward、cu_seqlens、raw_buffer、shared memory、
MFMA 或性能基准。
真正优化的三个阶段直接与 reference/v4 比较；复用的 cumsum、solve、chunk_gdr
同样通过 v5 wrapper 检查，以保证 correctness-first fallback 未破坏接口。
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
from qwen_gdn_chunked_avelang_v5 import (
    V5_STAGE_STRATEGY,
    qwen_gdn_chunk_cumsum_avelang_v5,
    qwen_gdn_chunk_gdr_avelang_v5,
    qwen_gdn_chunk_o_avelang_v5,
    qwen_gdn_chunked_avelang_v5,
    qwen_gdn_kkt_avelang_v5,
    qwen_gdn_solve_avelang_v5,
    qwen_gdn_w_u_avelang_v5,
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

STAGE_ATOL = 5e-5
STAGE_RTOL = 5e-5
E2E_ATOL = 8e-5
E2E_RTOL = 8e-5


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


def assert_stage_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=STAGE_ATOL, rtol=STAGE_RTOL)


def assert_forward_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=E2E_ATOL, rtol=E2E_RTOL)


def test_qwen_gdn_chunked_v5_strategy_marks_optimized_and_fallback_stages():
    # 策略摘要用于明确哪些 stage 真正优化，避免把 correctness-first 复用路径误报为优化。
    assert "优化" in V5_STAGE_STRATEGY["kkt"]
    assert "优化" in V5_STAGE_STRATEGY["w_u"]
    assert "优化" in V5_STAGE_STRATEGY["chunk_o"]
    assert "复用" in V5_STAGE_STRATEGY["g_cumsum"]
    assert "复用" in V5_STAGE_STRATEGY["solve"]
    assert "复用" in V5_STAGE_STRATEGY["chunk_gdr"]


def test_qwen_gdn_chunked_v5_stage_by_stage_against_reference_and_v4():
    # 逐阶段 case 同时覆盖 grouped-head、initial_state 和不完整末尾 chunk，便于定位优化偏差。
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

    # cumsum：复用路径必须保持 chunk 边界重置行为与 v4/reference 相同。
    expected_g = torch_cumsum(g, chunk_size=chunk_size)
    v4_g = qwen_gdn_chunk_cumsum_avelang_v4(g, chunk_size=chunk_size)
    actual_g = qwen_gdn_chunk_cumsum_avelang_v5(g, chunk_size=chunk_size)
    assert_stage_close(actual_g, expected_g)
    assert_stage_close(actual_g, v4_g)

    # KKT：优化后的整行 program 必须保持严格过去 mask、head repeat 与 decay。
    expected_a = torch_kkt_fwd(k, expected_g, beta, chunk_size=chunk_size)
    v4_a = qwen_gdn_kkt_avelang_v4(k, expected_g, beta, chunk_size=chunk_size)
    actual_a = qwen_gdn_kkt_avelang_v5(k, expected_g, beta, chunk_size=chunk_size)
    assert_stage_close(actual_a, expected_a)
    assert_stage_close(actual_a, v4_a)

    # solve：复用的三角递推仍需通过 v5 wrapper 对齐 reference 与 v4。
    expected_a_solved = torch_solve(expected_a)
    v4_a_solved = qwen_gdn_solve_avelang_v4(expected_a, chunk_size=chunk_size)
    actual_a_solved = qwen_gdn_solve_avelang_v5(expected_a, chunk_size=chunk_size)
    assert_stage_close(actual_a_solved, expected_a_solved)
    assert_stage_close(actual_a_solved, v4_a_solved)

    # w/u：融合 kernel 应同时正确生成 key 校正项和 value 校正项。
    expected_w, expected_u = torch_w_u_fwd(k, v, expected_g, beta, expected_a_solved, chunk_size=chunk_size)
    v4_w, v4_u = qwen_gdn_w_u_avelang_v4(k, v, expected_g, beta, expected_a_solved, chunk_size=chunk_size)
    actual_w, actual_u = qwen_gdn_w_u_avelang_v5(k, v, expected_g, beta, expected_a_solved, chunk_size=chunk_size)
    assert_stage_close(actual_w, expected_w)
    assert_stage_close(actual_u, expected_u)
    assert_stage_close(actual_w, v4_w)
    assert_stage_close(actual_u, v4_u)

    # chunk_gdr：复用路径必须保持 chunk 入口状态、vn 与最终状态的递推顺序。
    expected_h, expected_vn, expected_final = torch_chunk_gdr_fwd(
        k,
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    v4_h, v4_vn, v4_final = qwen_gdn_chunk_gdr_avelang_v4(
        k,
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    actual_h, actual_vn, actual_final = qwen_gdn_chunk_gdr_avelang_v5(
        k,
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    assert_stage_close(actual_h, expected_h)
    assert_stage_close(actual_vn, expected_vn)
    assert_stage_close(actual_final, expected_final)
    assert_stage_close(actual_h, v4_h)
    assert_stage_close(actual_vn, v4_vn)
    assert_stage_close(actual_final, v4_final)

    # chunk_o：整行 value 输出优化必须保持 inter 与含对角线 intra 的总和。
    expected_o = torch_chunk_o_fwd(
        q,
        k,
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    v4_o = qwen_gdn_chunk_o_avelang_v4(
        q,
        k,
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_o = qwen_gdn_chunk_o_avelang_v5(
        q,
        k,
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    assert_stage_close(actual_o, expected_o)
    assert_stage_close(actual_o, v4_o)


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
        # 基础 case：单 batch、单 head、单个完整 chunk。
        ("small_base", 1, 4, 1, 1, 4, 4, 4, False),
        # initial_state case：同时验证 V 与 K 不同以及第二个 partial chunk。
        ("with_initial_state", 1, 5, 2, 2, 4, 3, 4, True),
        # grouped-head case：验证 Hk 小于 Hv 的重复映射。
        ("grouped_heads", 1, 7, 1, 2, 4, 3, 4, False),
        # grouped-head 与 initial_state 组合：验证多 value head 状态初始化。
        ("grouped_heads_with_initial_state", 1, 7, 2, 4, 4, 3, 4, True),
        # multi-batch case：验证 batch 维 program 映射。
        ("multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
        # 非整除 case：显式覆盖 T=9 与 chunk_size=4 的三个 chunk。
        ("partial_third_chunk", 1, 9, 2, 4, 4, 3, 4, True),
        # 较大 debug case：验证 chunk_size=8 与更宽 K/V local buffer。
        ("larger_debug", 1, 9, 2, 4, 8, 6, 8, False),
    ],
)
def test_qwen_gdn_chunked_v5_end_to_end_against_reference_and_v4(
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
    # 端到端 case 对五个返回张量同时检查 reference 与 v4，性能优化不能牺牲中间结果。
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

    expected = qwen_gdn_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    v4_actual = qwen_gdn_chunked_avelang_v4(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual = qwen_gdn_chunked_avelang_v5(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    assert_forward_close(actual, expected)
    assert_forward_close(actual, v4_actual)


def test_qwen_gdn_chunked_v5_fallback_switch_matches_v4():
    # 回退 case 验证关闭优化后仍能通过统一 v5 入口精确复用 v4 stage 组合。
    q, k, v, g, beta = make_inputs(1, 5, 1, 2, 4, 3, seed=777)
    initial_state = make_initial_state(1, 2, 4, 3, seed=778)
    expected = qwen_gdn_chunked_avelang_v4(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        chunk_size=4,
    )
    actual = qwen_gdn_chunked_avelang_v5(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        chunk_size=4,
        prefer_optimized=False,
    )
    assert_forward_close(actual, expected)

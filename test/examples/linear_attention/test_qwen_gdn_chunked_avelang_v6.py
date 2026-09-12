"""Qwen GDN v6：强化 fixed-length forward 正确性测试。

当前测试覆盖 v6 支持范围内的 normal batch、forward-only、无 cu_seqlens 输入：
`q/k/v` 同为 FP32 或同为 BF16，`g/beta/initial_state/output/final_state`
为 FP32。
相比上一版测试，本文件显式加入 required FP32 与 BF16 case，并对
`prefer_optimized=True/False` 都检查完整五元组。
当前测试仍然不覆盖性能、backward、cu_seqlens、variable-length packed input
或高级 intrinsic。
"""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v5 import qwen_gdn_chunked_avelang_v5
from qwen_gdn_chunked_avelang_v6 import (
    V6_STAGE_STRATEGY,
    qwen_gdn_chunk_cumsum_avelang_v6,
    qwen_gdn_chunk_gdr_avelang_v6,
    qwen_gdn_chunk_o_avelang_v6,
    qwen_gdn_chunked_avelang_v6,
    qwen_gdn_kkt_avelang_v6,
    qwen_gdn_solve_avelang_v6,
    qwen_gdn_w_u_avelang_v6,
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

FP32_ATOL = 8e-5
FP32_RTOL = 8e-5
BF16_ATOL = 5e-4
BF16_RTOL = 5e-4

OUTPUT_NAMES = ("g_cumsum", "output", "A_solved", "chunk_states", "final_state")


def make_fp32_inputs(
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


def make_noncontiguous_like(tensor: torch.Tensor) -> torch.Tensor:
    # 构造形状相同但最后一维 stride 不为 1 的输入，用于验证 wrapper 的 contiguous 检查。
    expanded_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    storage = torch.empty(expanded_shape, dtype=tensor.dtype, device=tensor.device)
    return storage[..., ::2]


def to_bf16_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous()


def tensor_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    max_abs = float(diff.max().item())
    denom = expected.abs().clamp_min(1e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def assert_tensor_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    atol: float,
    rtol: float,
) -> None:
    max_abs, max_rel = tensor_errors(actual, expected)
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=f"{name} failed: max_abs={max_abs:.8g}, max_rel={max_rel:.8g}, atol={atol}, rtol={rtol}",
    )


def assert_forward_close(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    case_name: str,
    atol: float,
    rtol: float,
) -> None:
    for name, actual_tensor, expected_tensor in zip(OUTPUT_NAMES, actual, expected, strict=True):
        assert_tensor_close(
            actual_tensor,
            expected_tensor,
            name=f"{case_name}:{name}",
            atol=atol,
            rtol=rtol,
        )


def test_qwen_gdn_chunked_v6_strategy_identifies_correctness_scope():
    # 策略摘要必须明确 v6 是 correctness baseline，而不是新的性能优化结论。
    assert "BF16 correctness" in V6_STAGE_STRATEGY["kkt"]
    assert "BF16 correctness" in V6_STAGE_STRATEGY["w_u"]
    assert "BF16 correctness" in V6_STAGE_STRATEGY["chunk_gdr"]
    assert "BF16 correctness" in V6_STAGE_STRATEGY["chunk_o"]


def test_qwen_gdn_chunked_v6_bf16_stage_by_stage_against_reference():
    # 逐阶段 case 覆盖 grouped heads、initial_state 和 partial chunk，便于定位具体 stage。
    batch_size = 1
    num_tokens = 7
    num_k_heads = 2
    num_v_heads = 4
    head_dim_k = 8
    head_dim_v = 6
    chunk_size = 4
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=321,
    )
    q, k, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=654)
    scale = head_dim_k**-0.5

    # cumsum：g 保持 FP32，确认 chunk 边界累加仍对齐 reference。
    expected_g = torch_cumsum(g, chunk_size=chunk_size)
    actual_g = qwen_gdn_chunk_cumsum_avelang_v6(g, chunk_size=chunk_size)
    assert_tensor_close(actual_g, expected_g, name="stage:g_cumsum", atol=FP32_ATOL, rtol=FP32_RTOL)

    # KKT：使用 BF16 量化后的 key 转回 FP32 作为 reference 输入。
    expected_a = torch_kkt_fwd(k.float(), expected_g, beta, chunk_size=chunk_size)
    actual_a = qwen_gdn_kkt_avelang_v6(k, expected_g, beta, chunk_size=chunk_size)
    assert_tensor_close(actual_a, expected_a, name="stage:KKT", atol=BF16_ATOL, rtol=BF16_RTOL)

    # solve：A 为 FP32，检查小三角递推未被 dtype 路径影响。
    expected_a_solved = torch_solve(expected_a)
    actual_a_solved = qwen_gdn_solve_avelang_v6(actual_a, chunk_size=chunk_size)
    assert_tensor_close(actual_a_solved, expected_a_solved, name="stage:solve", atol=BF16_ATOL, rtol=BF16_RTOL)

    # w/u：检查 BF16 k/v 读取后以 FP32 累加。
    expected_w, expected_u = torch_w_u_fwd(k.float(), v.float(), expected_g, beta, expected_a_solved, chunk_size=4)
    actual_w, actual_u = qwen_gdn_w_u_avelang_v6(k, v, expected_g, beta, actual_a_solved, chunk_size=4)
    assert_tensor_close(actual_w, expected_w, name="stage:w", atol=BF16_ATOL, rtol=BF16_RTOL)
    assert_tensor_close(actual_u, expected_u, name="stage:u", atol=BF16_ATOL, rtol=BF16_RTOL)

    # chunk_gdr：检查 BF16 key 推进的 state、vn 与 final_state。
    expected_h, expected_vn, expected_final = torch_chunk_gdr_fwd(
        k.float(),
        expected_w,
        expected_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    actual_h, actual_vn, actual_final = qwen_gdn_chunk_gdr_avelang_v6(
        k,
        actual_w,
        actual_u,
        expected_g,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
    assert_tensor_close(actual_h, expected_h, name="stage:chunk_states", atol=BF16_ATOL, rtol=BF16_RTOL)
    assert_tensor_close(actual_vn, expected_vn, name="stage:vn", atol=BF16_ATOL, rtol=BF16_RTOL)
    assert_tensor_close(actual_final, expected_final, name="stage:final_state", atol=BF16_ATOL, rtol=BF16_RTOL)

    # chunk_o：检查 BF16 q/k 与 FP32 中间状态合成后的输出。
    expected_o = torch_chunk_o_fwd(
        q.float(),
        k.float(),
        expected_vn,
        expected_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_o = qwen_gdn_chunk_o_avelang_v6(
        q,
        k,
        actual_vn,
        actual_h,
        expected_g,
        scale=scale,
        chunk_size=chunk_size,
    )
    assert_tensor_close(actual_o, expected_o, name="stage:output", atol=BF16_ATOL, rtol=BF16_RTOL)


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
        # FP32，Hk == Hv，无 initial_state。
        ("fp32_same_heads_no_initial", 1, 4, 1, 1, 4, 4, 4, False),
        # FP32，Hk < Hv，带 initial_state。
        ("fp32_grouped_heads_with_initial", 1, 7, 2, 4, 4, 3, 4, True),
        # FP32，B > 1。
        ("fp32_multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
    ],
)
def test_qwen_gdn_chunked_v6_fp32_required_cases_against_reference_and_v5(
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
    # FP32 case 必须同时对齐 PyTorch reference 和 v5，且两个 prefer_optimized 分支都检查。
    q, k, v, g, beta = make_fp32_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=1000 + len(case_name),
    )
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=2000 + len(case_name))
    scale = head_dim_k**-0.5

    expected_ref = qwen_gdn_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    expected_v5 = qwen_gdn_chunked_avelang_v5(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    for prefer_optimized in (True, False):
        actual = qwen_gdn_chunked_avelang_v6(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=prefer_optimized,
        )
        assert_forward_close(
            actual,
            expected_ref,
            case_name=f"{case_name}:ref:prefer={prefer_optimized}",
            atol=FP32_ATOL,
            rtol=FP32_RTOL,
        )
        assert_forward_close(
            actual,
            expected_v5,
            case_name=f"{case_name}:v5:prefer={prefer_optimized}",
            atol=FP32_ATOL,
            rtol=FP32_RTOL,
        )


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
        # BF16，Hk == Hv，无 initial_state。
        ("bf16_same_heads_no_initial", 1, 5, 2, 2, 4, 3, 4, False),
        # BF16，Hk < Hv，带 initial_state。
        ("bf16_grouped_heads_with_initial", 1, 7, 2, 4, 4, 3, 4, True),
        # BF16，T 不能被 chunk_size 整除。
        ("bf16_partial_chunk", 1, 9, 2, 4, 4, 3, 4, False),
        # BF16，B > 1。
        ("bf16_multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
        # 可选较大 debug case，验证 chunk_size=8 和更宽 K/V。
        ("bf16_larger_debug", 1, 9, 2, 4, 8, 6, 8, False),
    ],
)
def test_qwen_gdn_chunked_v6_bf16_required_cases_against_quantized_reference(
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
    # BF16 case 使用量化后的 q/k/v 转回 FP32 作为 reference，两个 prefer_optimized 分支都检查五元组。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=3000 + len(case_name),
    )
    q, k, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=4000 + len(case_name))
    scale = head_dim_k**-0.5

    expected = qwen_gdn_forward_ref(
        q.float(),
        k.float(),
        v.float(),
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_by_preference = []
    for prefer_optimized in (True, False):
        actual = qwen_gdn_chunked_avelang_v6(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            chunk_size=chunk_size,
            prefer_optimized=prefer_optimized,
        )
        actual_by_preference.append(actual)
        assert actual[1].dtype == torch.float32
        assert actual[4].dtype == torch.float32
        assert_forward_close(
            actual,
            expected,
            case_name=f"{case_name}:bf16_ref:prefer={prefer_optimized}",
            atol=BF16_ATOL,
            rtol=BF16_RTOL,
        )

    assert_forward_close(
        actual_by_preference[0],
        actual_by_preference[1],
        case_name=f"{case_name}:prefer_true_vs_false",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )


def test_qwen_gdn_chunked_v6_rejects_mixed_qkv_dtype():
    # q/k/v 必须同为 FP32 或同为 BF16，混合 dtype 应在 wrapper 阶段报错。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(1, 5, 2, 2, 4, 3, seed=5050)
    q, _, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    with pytest.raises(ValueError):
        qwen_gdn_chunked_avelang_v6(q, k_fp32.contiguous(), v, g, beta, chunk_size=4)


@pytest.mark.parametrize("bad_name", ["q", "k", "v"])
def test_qwen_gdn_chunked_v6_rejects_non_contiguous_qkv(bad_name: str):
    # 非连续 q/k/v 容易导致 stride 假设失效，因此支持范围内必须拒绝。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(1, 5, 2, 2, 4, 3, seed=6060)
    q, k, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    if bad_name == "q":
        q = make_noncontiguous_like(q)
    elif bad_name == "k":
        k = make_noncontiguous_like(k)
    else:
        v = make_noncontiguous_like(v)
    with pytest.raises(ValueError, match="contiguous"):
        qwen_gdn_chunked_avelang_v6(q, k, v, g, beta, chunk_size=4)

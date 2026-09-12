"""Qwen GDN v7：端到端正确性测试。

本文件验证 v7 在当前支持范围内同时对齐 PyTorch reference 和 standalone v6：
normal batch、forward-only、无 cu_seqlens，q/k/v 同为 FP32 或同为 BF16。
"""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunked_avelang_v6_standalone import qwen_gdn_chunked_avelang_v6_standalone
from qwen_gdn_chunked_avelang_v7 import qwen_gdn_chunked_avelang_v7
from qwen_gdn_ref import l2norm, qwen_gdn_forward_ref


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


def to_bf16_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous()


def make_noncontiguous_like(tensor: torch.Tensor) -> torch.Tensor:
    # 构造形状相同但 stride 不同的张量，专门触发 wrapper 的 contiguous 校验。
    expanded_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
    storage = torch.empty(expanded_shape, dtype=tensor.dtype, device=tensor.device)
    return storage[..., ::2]


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
        ("fp32_same_heads_no_initial", 1, 4, 1, 1, 4, 4, 4, False),
        ("fp32_grouped_heads_with_initial", 1, 7, 2, 4, 4, 3, 4, True),
        ("fp32_multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
    ],
)
@pytest.mark.parametrize("prefer_optimized", [True, False])
def test_qwen_gdn_chunked_v7_fp32_matches_ref_and_standalone_v6(
    case_name: str,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    chunk_size: int,
    use_initial_state: bool,
    prefer_optimized: bool,
):
    # FP32 路径同时对齐 reference 和 standalone v6，且比较完整五元组。
    q, k, v, g, beta = make_fp32_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=1100 + len(case_name),
    )
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=2100 + len(case_name))
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
    expected_standalone = qwen_gdn_chunked_avelang_v6_standalone(
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
    actual = qwen_gdn_chunked_avelang_v7(
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
        case_name=f"{case_name}:fp32_ref:prefer={prefer_optimized}",
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    )
    assert_forward_close(
        actual,
        expected_standalone,
        case_name=f"{case_name}:fp32_standalone:prefer={prefer_optimized}",
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
        ("bf16_same_heads_no_initial", 1, 5, 2, 2, 4, 3, 4, False),
        ("bf16_grouped_heads_with_initial", 1, 7, 2, 4, 4, 3, 4, True),
        ("bf16_partial_chunk", 1, 9, 2, 4, 4, 3, 4, False),
        ("bf16_multi_batch", 2, 6, 1, 2, 4, 3, 4, False),
    ],
)
@pytest.mark.parametrize("prefer_optimized", [True, False])
def test_qwen_gdn_chunked_v7_bf16_matches_quantized_ref_and_standalone_v6(
    case_name: str,
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    chunk_size: int,
    use_initial_state: bool,
    prefer_optimized: bool,
):
    # BF16 reference 使用量化后的 q/k/v 转回 FP32，避免和未量化 FP32 输入比较。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        seed=3100 + len(case_name),
    )
    q, k, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    initial_state = None
    if use_initial_state:
        initial_state = make_initial_state(batch_size, num_v_heads, head_dim_k, head_dim_v, seed=4100 + len(case_name))
    scale = head_dim_k**-0.5

    expected_ref = qwen_gdn_forward_ref(
        q.float(),
        k.float(),
        v.float(),
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    expected_standalone = qwen_gdn_chunked_avelang_v6_standalone(
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
    actual = qwen_gdn_chunked_avelang_v7(
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

    assert actual[1].dtype == torch.float32
    assert actual[2].dtype == torch.float32
    assert actual[3].dtype == torch.float32
    assert actual[4].dtype == torch.float32
    assert_forward_close(
        actual,
        expected_ref,
        case_name=f"{case_name}:bf16_ref:prefer={prefer_optimized}",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )
    assert_forward_close(
        actual,
        expected_standalone,
        case_name=f"{case_name}:bf16_standalone:prefer={prefer_optimized}",
        atol=BF16_ATOL,
        rtol=BF16_RTOL,
    )


def test_qwen_gdn_chunked_v7_rejects_mixed_qkv_dtype():
    # q/k/v 必须同为 FP32 或同为 BF16。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(1, 5, 2, 2, 4, 3, seed=5050)
    q, _, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    with pytest.raises(ValueError):
        qwen_gdn_chunked_avelang_v7(q, k_fp32.contiguous(), v, g, beta, chunk_size=4)


@pytest.mark.parametrize("bad_name", ["q", "k", "v"])
def test_qwen_gdn_chunked_v7_rejects_non_contiguous_qkv(bad_name: str):
    # 非连续 q/k/v 会破坏当前 layout 假设，v7 wrapper 应直接拒绝。
    q_fp32, k_fp32, v_fp32, g, beta = make_fp32_inputs(1, 5, 2, 2, 4, 3, seed=6060)
    q, k, v = to_bf16_qkv(q_fp32, k_fp32, v_fp32)
    if bad_name == "q":
        q = make_noncontiguous_like(q)
    elif bad_name == "k":
        k = make_noncontiguous_like(k)
    else:
        v = make_noncontiguous_like(v)
    with pytest.raises(ValueError, match="contiguous"):
        qwen_gdn_chunked_avelang_v7(q, k, v, g, beta, chunk_size=4)

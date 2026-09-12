"""Qwen GDN v3 chunk-local cumsum 测试。

当前测试只校验 g 在 chunk 内的局部累加，直接对齐 `qwen_gdn_ref.torch_cumsum`。
相比前面 v0/v1/v2 的 recurrent forward 测试，这里新增的是 chunk 边界重置语义。
当前测试仍然不覆盖完整 GDN forward、cu_seqlens、kkt/solve/w/u、chunk_gdr、
chunk_o、backward 或任何优化路径。
"""

from __future__ import annotations

import pytest
import torch

from qwen_gdn_chunk_cumsum_avelang_v3 import qwen_gdn_chunk_cumsum_avelang_v3
from qwen_gdn_ref import torch_cumsum


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


def make_g(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    *,
    seed: int = 123,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        batch_size,
        num_tokens,
        num_heads,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).contiguous()


def assert_matches_torch_cumsum(g: torch.Tensor, *, chunk_size: int) -> None:
    expected = torch_cumsum(g, chunk_size=chunk_size)
    actual = qwen_gdn_chunk_cumsum_avelang_v3(g, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_qwen_gdn_chunk_cumsum_v3_manual_small_case():
    # 手写小例子专门检查 chunk_size=4 时，第 5 个 token 从新 chunk 重新累加。
    g = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]],
        device="cuda",
        dtype=torch.float32,
    )
    expected_manual = torch.tensor(
        [[[1.0], [3.0], [6.0], [10.0], [5.0], [11.0]]],
        device="cuda",
        dtype=torch.float32,
    )

    expected_ref = torch_cumsum(g, chunk_size=4)
    actual = qwen_gdn_chunk_cumsum_avelang_v3(g, chunk_size=4)

    torch.testing.assert_close(expected_ref, expected_manual, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual, expected_manual, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("batch_size", "num_tokens", "num_heads", "chunk_size"),
    [
        # T 小于 chunk_size，整段序列只在一个 chunk 内累加。
        (1, 7, 2, 64),
        # T 等于 chunk_size，正好覆盖完整单 chunk。
        (2, 64, 3, 64),
        # T 大于 chunk_size，检查第二个 chunk 从 0 重新开始。
        (1, 65, 4, 64),
        # 小 chunk debug case，多个 batch/head/chunk 同时覆盖 program 映射。
        (2, 11, 3, 4),
    ],
)
def test_qwen_gdn_chunk_cumsum_v3_random_cases(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    chunk_size: int,
):
    g = make_g(batch_size, num_tokens, num_heads)
    assert_matches_torch_cumsum(g, chunk_size=chunk_size)


def test_qwen_gdn_chunk_cumsum_v3_uses_preallocated_out():
    # 检查 wrapper 会写入调用方传入的 out，并返回同一个 tensor 对象。
    g = make_g(1, 7, 2)
    out = torch.empty_like(g)

    actual = qwen_gdn_chunk_cumsum_avelang_v3(g, chunk_size=4, out=out)
    expected = torch_cumsum(g, chunk_size=4)

    assert actual is out
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


def test_qwen_gdn_chunk_cumsum_v3_rejects_non_fp32():
    # 检查 g 必须是 fp32，避免误把 bf16/fp16 的误差带入这个阶段。
    g = make_g(1, 7, 2).to(torch.float16).contiguous()

    with pytest.raises(ValueError, match="torch.float32"):
        qwen_gdn_chunk_cumsum_avelang_v3(g)


def test_qwen_gdn_chunk_cumsum_v3_rejects_non_contiguous():
    # 检查 wrapper 对 contiguous 的要求，保证 make_layout 的 stride 假设成立。
    g = torch.randn((1, 2, 7), device="cuda", dtype=torch.float32).transpose(1, 2)

    with pytest.raises(ValueError, match="contiguous"):
        qwen_gdn_chunk_cumsum_avelang_v3(g)

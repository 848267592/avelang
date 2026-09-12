from __future__ import annotations

import pytest
import torch

from qwen_gdn_naive_avelang import qwen_gdn_naive_avelang
from qwen_gdn_recurrent_ref import l2norm, qwen_gdn_recurrent_forward_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


def make_inputs(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
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
            num_heads,
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
            num_heads,
            head_dim_k,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    v = torch.randn(
        batch_size,
        num_tokens,
        num_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_heads,
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
            num_heads,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
    ).contiguous()
    return q, k, v.contiguous(), g, beta


@pytest.mark.parametrize(
    ("batch_size", "num_tokens", "num_heads", "head_dim_k", "head_dim_v"),
    [
        (1, 4, 1, 4, 4),
        (1, 5, 2, 4, 3),
        (2, 4, 1, 4, 3),
    ],
)
def test_qwen_gdn_naive_avelang_matches_recurrent_ref(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_dim_k: int,
    head_dim_v: int,
):
    q, k, v, g, beta = make_inputs(
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
    )
    scale = head_dim_k**-0.5

    expected_out, expected_state = qwen_gdn_recurrent_forward_ref(q, k, v, g, beta, scale=scale)
    actual_out, actual_state = qwen_gdn_naive_avelang(q, k, v, g, beta, scale=scale)

    torch.testing.assert_close(actual_out, expected_out, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)

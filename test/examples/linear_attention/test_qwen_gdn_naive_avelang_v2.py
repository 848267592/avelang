from __future__ import annotations

import pytest
import torch

from qwen_gdn_naive_avelang_v2 import qwen_gdn_naive_avelang_v2
from qwen_gdn_ref import l2norm, qwen_gdn_forward_ref


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP GPU")


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
    )
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
    )
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
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
    return q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous(), g, beta


@pytest.mark.parametrize(
    (
        "name",
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
        ("same_heads_no_initial_state", 1, 4, 1, 1, 4, 4, 4, False),
        ("same_heads_with_initial_state", 1, 5, 2, 2, 4, 3, 4, True),
        ("grouped_heads_no_initial_state", 1, 5, 1, 2, 4, 3, 4, False),
        ("grouped_heads_with_initial_state", 1, 7, 2, 4, 8, 6, 4, True),
        ("multi_batch", 2, 4, 1, 2, 4, 3, 4, False),
        ("chunk_boundary", 1, 65, 2, 4, 8, 6, 64, False),
    ],
)
def test_qwen_gdn_naive_avelang_v2_bf16_matches_chunked_ref(
    name: str,
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
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
    )
    initial_state = None
    if use_initial_state:
        generator = torch.Generator(device="cuda").manual_seed(2000 + len(name))
        initial_state = torch.randn(
            batch_size,
            num_v_heads,
            head_dim_k,
            head_dim_v,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        ).contiguous()

    scale = head_dim_k**-0.5
    _, expected_out, _, _, expected_state = qwen_gdn_forward_ref(
        q.float(),
        k.float(),
        v.float(),
        g.float(),
        beta.float(),
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual_out, actual_state = qwen_gdn_naive_avelang_v2(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
    )

    assert actual_out.dtype == torch.float32
    assert actual_state.dtype == torch.float32
    torch.testing.assert_close(actual_out, expected_out, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(actual_state, expected_state, atol=3e-4, rtol=3e-4)

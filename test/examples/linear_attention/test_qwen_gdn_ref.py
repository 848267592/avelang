import math

import pytest
import torch

from qwen_gdn_recurrent_ref import qwen_gdn_recurrent_forward_ref
from qwen_gdn_ref import l2norm, qwen_gdn_forward_ref


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def make_inputs(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device=device,
            generator=generator,
        )
    )
    k = l2norm(
        torch.randn(
            batch_size,
            num_tokens,
            num_k_heads,
            head_dim_k,
            device=device,
            generator=generator,
        )
    )
    v = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        head_dim_v,
        device=device,
        generator=generator,
    )
    g = torch.nn.functional.logsigmoid(
        torch.randn(
            batch_size,
            num_tokens,
            num_v_heads,
            device=device,
            generator=generator,
        )
    )
    g = g / 16
    beta = torch.randn(
        batch_size,
        num_tokens,
        num_v_heads,
        device=device,
        generator=generator,
    ).sigmoid()
    return q, k, v, g, beta


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize(
    ("num_tokens", "chunk_size"),
    [
        (7, 64),
        (64, 64),
        (65, 64),
        (11, 4),
    ],
)
@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads", "head_dim_k", "head_dim_v"),
    [
        (2, 2, 8, 8),
        (2, 4, 8, 8),
        (1, 2, 4, 6),
    ],
)
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_qwen_gdn_chunked_forward_matches_recurrent_reference(
    device: str,
    num_tokens: int,
    chunk_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    use_initial_state: bool,
):
    q, k, v, g, beta = make_inputs(
        batch_size=1,
        num_tokens=num_tokens,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_dim_k=head_dim_k,
        head_dim_v=head_dim_v,
        device=device,
    )
    initial_state = None
    if use_initial_state:
        initial_state = torch.randn(
            q.shape[0],
            v.shape[2],
            q.shape[-1],
            v.shape[-1],
            device=device,
        )

    scale = q.shape[-1] ** -0.5
    _, output, _, chunk_states, final_state = qwen_gdn_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    recurrent_output, recurrent_state = qwen_gdn_recurrent_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
    )

    assert output.shape == v.shape
    assert chunk_states.shape == (
        q.shape[0],
        math.ceil(num_tokens / chunk_size),
        v.shape[2],
        q.shape[-1],
        v.shape[-1],
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, recurrent_output, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(final_state, recurrent_state, atol=2e-5, rtol=2e-5)


def test_qwen_gdn_chunked_forward_preserves_scale_zero():
    q, k, v, g, beta = make_inputs(
        batch_size=1,
        num_tokens=9,
        num_k_heads=1,
        num_v_heads=1,
        head_dim_k=4,
        head_dim_v=5,
        device="cpu",
    )

    _, output, _, _, final_state = qwen_gdn_forward_ref(
        q,
        k,
        v,
        g,
        beta,
        scale=0.0,
        chunk_size=4,
    )

    assert torch.count_nonzero(output).item() == 0
    assert torch.count_nonzero(final_state).item() > 0


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda q, k, v, g, beta: (q[0], k, v, g, beta), "q must have shape"),
        (lambda q, k, v, g, beta: (q, k[:, :-1], v, g, beta), "q and k must have identical shapes"),
        (lambda q, k, v, g, beta: (q, k, v[:, :-1], g, beta), "v sequence length must match"),
        (lambda q, k, v, g, beta: (q, k, v, g[:, :, :-1], beta), "v/g/beta Hv dimensions must match"),
        (lambda q, k, v, g, beta: (q, k, v, g, beta[:, :, :-1]), "v/g/beta Hv dimensions must match"),
    ],
)
def test_qwen_gdn_chunked_forward_rejects_bad_shapes(mutate, match: str):
    q, k, v, g, beta = make_inputs(
        batch_size=1,
        num_tokens=7,
        num_k_heads=2,
        num_v_heads=2,
        head_dim_k=4,
        head_dim_v=4,
        device="cpu",
    )
    args = mutate(q, k, v, g, beta)

    with pytest.raises(AssertionError, match=match):
        qwen_gdn_forward_ref(*args)


def test_qwen_gdn_chunked_forward_rejects_bad_cu_seqlens():
    q, k, v, g, beta = make_inputs(
        batch_size=1,
        num_tokens=7,
        num_k_heads=1,
        num_v_heads=1,
        head_dim_k=4,
        head_dim_v=4,
        device="cpu",
    )
    bad_cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

    with pytest.raises(AssertionError, match="cu_seqlens must end at T=7"):
        qwen_gdn_forward_ref(q, k, v, g, beta, cu_seqlens=bad_cu_seqlens)

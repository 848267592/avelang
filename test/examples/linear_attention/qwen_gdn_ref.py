"""Chunked PyTorch reference for Qwen FlashQLA GDN forward.

This module intentionally avoids importing ``flash_qla`` because that package
initializes TileLang/Hopper-only kernels.  The math below is adapted from
``FlashQLA/tests/ref_gdr.py`` and kept slow/readable for correctness tests.

This is the Qwen-style chunked reference.  The minimal token-by-token formula
lives in ``qwen_gdn_recurrent_ref.py``.
"""

from __future__ import annotations

import torch


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return (x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


def _assert_shape(condition: bool, message: str) -> None:
    assert condition, message


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, packed_batch: int, packed_tokens: int) -> None:
    _assert_shape(cu_seqlens.ndim == 1, f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}")
    _assert_shape(cu_seqlens.numel() >= 2, "cu_seqlens must contain at least start and end offsets")
    _assert_shape(cu_seqlens[0].item() == 0, f"cu_seqlens must start at 0, got {cu_seqlens[0].item()}")
    _assert_shape(
        cu_seqlens[-1].item() == packed_tokens,
        f"cu_seqlens must end at T={packed_tokens}, got {cu_seqlens[-1].item()}",
    )
    _assert_shape(packed_batch == 1, f"packed cu_seqlens input must have B=1, got B={packed_batch}")
    _assert_shape(
        bool((cu_seqlens[1:] >= cu_seqlens[:-1]).all().item()),
        "cu_seqlens must be monotonically nondecreasing",
    )


def _validate_qwen_gdn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int,
) -> None:
    _assert_shape(q.ndim == 4, f"q must have shape [B, T, Hk, K], got {tuple(q.shape)}")
    _assert_shape(k.ndim == 4, f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}")
    _assert_shape(v.ndim == 4, f"v must have shape [B, T, Hv, V], got {tuple(v.shape)}")
    _assert_shape(g.ndim == 3, f"g must have shape [B, T, Hv], got {tuple(g.shape)}")
    _assert_shape(beta.ndim == 3, f"beta must have shape [B, T, Hv], got {tuple(beta.shape)}")
    _assert_shape(q.shape == k.shape, f"q and k must have identical shapes, got {tuple(q.shape)} and {tuple(k.shape)}")

    batch_size, num_tokens, num_k_heads, head_dim_k = q.shape
    v_batch, v_tokens, num_v_heads, head_dim_v = v.shape

    _assert_shape(v_batch == batch_size, f"v batch size must match q/k: {v_batch} != {batch_size}")
    _assert_shape(g.shape[0] == batch_size, f"g batch size must match q/k: {g.shape[0]} != {batch_size}")
    _assert_shape(beta.shape[0] == batch_size, f"beta batch size must match q/k: {beta.shape[0]} != {batch_size}")
    _assert_shape(v_tokens == num_tokens, f"v sequence length must match q/k: {v_tokens} != {num_tokens}")
    _assert_shape(g.shape[1] == num_tokens, f"g sequence length must match q/k: {g.shape[1]} != {num_tokens}")
    _assert_shape(beta.shape[1] == num_tokens, f"beta sequence length must match q/k: {beta.shape[1]} != {num_tokens}")
    _assert_shape(
        v.shape[2] == g.shape[2] == beta.shape[2],
        f"v/g/beta Hv dimensions must match, got {v.shape[2]}, {g.shape[2]}, {beta.shape[2]}",
    )
    _assert_shape(num_v_heads % num_k_heads == 0, f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}")
    _assert_shape(chunk_size > 0, f"chunk_size must be > 0, got {chunk_size}")

    if cu_seqlens is not None:
        _validate_cu_seqlens(cu_seqlens, batch_size, num_tokens)
        expected_state_batch = cu_seqlens.numel() - 1
    else:
        expected_state_batch = batch_size

    if initial_state is not None:
        expected = (expected_state_batch, num_v_heads, head_dim_k, head_dim_v)
        _assert_shape(
            tuple(initial_state.shape) == expected,
            f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}",
        )


def unpack(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    assert x.shape[0] == 1
    max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    batch_size = cu_seqlens.numel() - 1
    y = torch.zeros((batch_size, max_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[i, : end - start] = x[0, start:end]
    return y


def pack(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    sum_len = cu_seqlens[-1].item()
    batch_size = cu_seqlens.numel() - 1
    y = torch.empty((1, sum_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for i in range(batch_size):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        y[0, start:end] = x[i, : end - start]
    return y


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = torch.div(seqlens + chunk_size - 1, chunk_size, rounding_mode="floor")
    return torch.nn.functional.pad(torch.cumsum(num_chunks, dim=0), (1, 0))


def pad_and_reshape(x: torch.Tensor, dim: int, chunk_size: int = 64) -> torch.Tensor:
    sequence_length = x.shape[dim]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    zeros = [0] * (2 * (x.ndim - 1 - dim))
    padded = torch.nn.functional.pad(x, (*zeros, 0, pad_size))
    return padded.reshape((*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1 :]))


def fill_last_chunk_of_g(
    g: torch.Tensor,
    num_tokens: int,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if cu_seqlens is None:
        last_chunk_size = num_tokens % chunk_size
        if last_chunk_size > 0:
            g[:, -1, last_chunk_size:] = g[:, -1, last_chunk_size - 1 : last_chunk_size]
        return g

    for i in range(cu_seqlens.numel() - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        last_chunk_idx = (end - start) // chunk_size
        last_chunk_size = (end - start) % chunk_size
        if last_chunk_size > 0:
            g[i, last_chunk_idx, last_chunk_size:] = g[
                i, last_chunk_idx, last_chunk_size - 1 : last_chunk_size
            ]
    return g


def torch_cumsum(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    batch_size, num_tokens, num_heads = x.shape
    x = pad_and_reshape(x, dim=1, chunk_size=chunk_size).cumsum(dim=2)
    x = x.reshape(batch_size, -1, num_heads)[:, :num_tokens]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def torch_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        g = unpack(g, cu_seqlens)
        beta = unpack(beta, cu_seqlens)

    batch_size, num_tokens, num_k_heads, _ = k.shape
    num_v_heads = g.shape[-1]
    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    beta = pad_and_reshape(beta, dim=1, chunk_size=chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device))
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)
    attn = torch.einsum("bnchk,bndhk->bnchd", k * beta.unsqueeze(-1), k)
    attn = attn * decay_mask.swapaxes(-2, -1)
    attn = attn.reshape(batch_size, -1, num_v_heads, chunk_size)[:, :num_tokens]

    if cu_seqlens is not None:
        attn = pack(attn, cu_seqlens)
    return attn


def torch_solve(x: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    if cu_seqlens is not None:
        x = unpack(x, cu_seqlens)

    batch_size, num_tokens, num_heads, chunk_size = x.shape
    x = -pad_and_reshape(x, dim=1, chunk_size=chunk_size).swapaxes(2, 3)

    for i in range(1, chunk_size):
        row = x[..., i, :i].clone()
        sub = x[..., :i, :i].clone()
        x[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    x += torch.eye(chunk_size, dtype=x.dtype, device=x.device)
    x = x.swapaxes(2, 3).reshape((batch_size, -1, num_heads, chunk_size))[:, :num_tokens]

    if cu_seqlens is not None:
        x = pack(x, cu_seqlens)
    return x


def torch_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        g = unpack(g, cu_seqlens)
        beta = unpack(beta, cu_seqlens)
        a = unpack(a, cu_seqlens)

    batch_size, num_tokens, _, solved_chunk_size = a.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, head_dim_v = v.shape
    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k_beta = pad_and_reshape(
        k * beta.unsqueeze(-1) * g.exp().unsqueeze(-1),
        dim=1,
        chunk_size=chunk_size,
    )
    v_beta = pad_and_reshape(v * beta.unsqueeze(-1), dim=1, chunk_size=chunk_size)
    a = pad_and_reshape(a, dim=1, chunk_size=solved_chunk_size)

    w = torch.einsum("bnchd,bndhk->bnchk", a, k_beta)
    w = w.reshape((batch_size, -1, num_v_heads, k.shape[-1]))[:, :num_tokens]
    u = torch.einsum("bnchd,bndhk->bnchk", a, v_beta)
    u = u.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]

    if cu_seqlens is not None:
        w = pack(w, cu_seqlens)
        u = pack(u, cu_seqlens)
    return w, u


def torch_chunk_gdr_fwd(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None:
        k = unpack(k, cu_seqlens)
        w = unpack(w, cu_seqlens)
        u = unpack(u, cu_seqlens)
        g = unpack(g, cu_seqlens)

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = u.shape
    if num_k_heads != num_v_heads:
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    w = pad_and_reshape(w, dim=1, chunk_size=chunk_size)
    u = pad_and_reshape(u, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)
    g = fill_last_chunk_of_g(g, num_tokens, cu_seqlens, chunk_size=chunk_size)

    if initial_state is None:
        last_state = torch.zeros((batch_size, num_v_heads, head_dim_k, head_dim_v), dtype=g.dtype, device=g.device)
    else:
        last_state = initial_state.to(g.dtype, copy=True)

    h, vn = [], []
    for i in range(k.shape[1]):
        h.append(last_state)
        v_new = u[:, i] - torch.einsum("bchk,bhkv->bchv", w[:, i], last_state)
        vn.append(v_new)
        last_state = last_state * g[:, i, -1, :, None, None].exp()
        last_state = last_state + torch.einsum(
            "bchk,bchv->bhkv",
            k[:, i] * (g[:, i, -1:, :, None] - g[:, i, :, :, None]).exp(),
            v_new,
        )

    h = torch.stack(h, dim=1).contiguous()
    vn = torch.stack(vn, dim=1).reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens].contiguous()

    if cu_seqlens is not None:
        vn = pack(vn, cu_seqlens)
        h = pack(h, prepare_chunk_offsets(cu_seqlens, chunk_size))
    return h, vn, last_state


def torch_chunk_o_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    if cu_seqlens is not None:
        q = unpack(q, cu_seqlens)
        k = unpack(k, cu_seqlens)
        v = unpack(v, cu_seqlens)
        g = unpack(g, cu_seqlens)
        h = unpack(h, prepare_chunk_offsets(cu_seqlens, chunk_size))

    batch_size, num_tokens, num_k_heads, head_dim_k = k.shape
    _, _, num_v_heads, head_dim_v = v.shape
    if num_k_heads != num_v_heads:
        q = q.repeat_interleave(num_v_heads // num_k_heads, dim=2)
        k = k.repeat_interleave(num_v_heads // num_k_heads, dim=2)

    if scale is None:
        scale = head_dim_k**-0.5
    q = pad_and_reshape(q, dim=1, chunk_size=chunk_size) * scale
    k = pad_and_reshape(k, dim=1, chunk_size=chunk_size)
    v = pad_and_reshape(v, dim=1, chunk_size=chunk_size)
    g = pad_and_reshape(g, dim=1, chunk_size=chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device), diagonal=1)
    decay_mask = torch.exp(g[:, :, :, None, :] - g[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(mask[None, None, :, :, None], 0.0)

    attn = torch.einsum("bnchk,bndhk->bncdh", q, k) * decay_mask
    attn_inter = torch.einsum("bnchk,bnhkv->bnchv", q * g.exp().unsqueeze(-1), h)
    o = attn_inter + torch.einsum("bncdh,bndhv->bnchv", attn, v)
    o = o.reshape((batch_size, -1, num_v_heads, head_dim_v))[:, :num_tokens]

    if cu_seqlens is not None:
        o = pack(o, cu_seqlens)
    return o


def qwen_gdn_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen GDN chunked forward reference.

    Shapes:
      q, k:          [B, T, Hk, K]
      v:             [B, T, Hv, V]
      g, beta:       [B, T, Hv]
      initial_state: [B, Hv, K, V], or real-batch equivalent for packed input

    Returns ``(g_cumsum, output, A, chunk_states, final_state)``.
    """
    _validate_qwen_gdn_inputs(q, k, v, g, beta, initial_state, cu_seqlens, chunk_size)

    if scale is None:
        scale = q.shape[-1] ** -0.5

    # FlashQLA first turns per-token log-decay into chunk-local cumulative log-decay.
    g = torch_cumsum(x=g, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    a = torch_kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    a = torch_solve(x=a, cu_seqlens=cu_seqlens)
    w, u = torch_w_u_fwd(k=k, v=v, beta=beta, a=a, g=g, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    h, vn, final_state = torch_chunk_gdr_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = torch_chunk_o_fwd(q=q, k=k, v=vn, h=h, g=g, cu_seqlens=cu_seqlens, scale=scale, chunk_size=chunk_size)
    return g, o, a, h, final_state

"""Minimal token-by-token PyTorch reference for Qwen GDN forward.

这个文件是给“看懂公式”和“校验未来 DSL kernel”用的 reference。
它故意不用 chunk、不做优化、不做 backward，只保留最直接的逐 token
递推逻辑。后面写 Avelang naive kernel 时，最应该先对齐这里的语义。

核心状态:
  state[b, h, k, v] 表示第 b 个 batch、第 h 个 value head 上的
  K x V recurrent memory。

每个 token 会做四件事:
  1. 用 gate g 衰减旧 state。
  2. 用当前 k 从旧 state 读出 prediction。
  3. 用 beta 控制 v 和 prediction 的差值写回 state。
  4. 用当前 q 从更新后的 state 读出 output。
"""

from __future__ import annotations

import torch


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Normalize q/k along the last dimension, matching FlashQLA test input setup."""
    return (x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)).to(x.dtype)


def _assert_shape(condition: bool, message: str) -> None:
    """Tiny helper so shape checks keep readable error messages."""
    assert condition, message


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int, int]:
    """Validate the tensor contract used by Qwen GDN forward.

    Naming convention:
      B  = batch size
      T  = sequence length / number of tokens
      Hk = number of q/k heads
      Hv = number of value/state heads
      K  = q/k head dimension
      V  = value head dimension

    Qwen GDN allows grouped q/k heads: Hk can be smaller than Hv.  In that
    case each q/k head is shared by ``Hv // Hk`` value heads.
    """
    # Rank checks first: these make later shape unpacking safe and give a clear
    # error when a caller passes [B, H, T, D] or another layout by accident.
    _assert_shape(q.ndim == 4, f"q must have shape [B, T, Hk, K], got {tuple(q.shape)}")
    _assert_shape(k.ndim == 4, f"k must have shape [B, T, Hk, K], got {tuple(k.shape)}")
    _assert_shape(v.ndim == 4, f"v must have shape [B, T, Hv, V], got {tuple(v.shape)}")
    _assert_shape(g.ndim == 3, f"g must have shape [B, T, Hv], got {tuple(g.shape)}")
    _assert_shape(beta.ndim == 3, f"beta must have shape [B, T, Hv], got {tuple(beta.shape)}")

    # q and k are used in the same dot products, so every dimension must match.
    _assert_shape(q.shape == k.shape, f"q and k must have identical shapes, got {tuple(q.shape)} and {tuple(k.shape)}")

    batch_size, num_tokens, num_k_heads, head_dim_k = q.shape
    v_batch, v_tokens, num_v_heads, head_dim_v = v.shape

    # All inputs describe the same batch and sequence positions.
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

    # Hk < Hv is allowed, but only when each q/k head can be repeated an
    # integer number of times to cover all value heads.
    _assert_shape(num_v_heads % num_k_heads == 0, f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}")

    if initial_state is not None:
        # state is a per-(batch, value-head) K x V memory matrix.
        expected = (batch_size, num_v_heads, head_dim_k, head_dim_v)
        _assert_shape(
            tuple(initial_state.shape) == expected,
            f"initial_state must have shape {expected}, got {tuple(initial_state.shape)}",
        )

    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


def qwen_gdn_recurrent_forward_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    compute_dtype: torch.dtype = torch.float32,
    return_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Naive recurrent GDN forward reference.

    Shapes:
      q, k:          [B, T, Hk, K]
      v:             [B, T, Hv, V]
      g, beta:       [B, T, Hv]
      initial_state: [B, Hv, K, V]
      output:        [B, T, Hv, V]
      final_state:   [B, Hv, K, V]

    ``g`` is expected to be log-decay values.  The actual decay multiplier is
    ``exp(g[:, t])`` at token t.  FlashQLA tests generate it as
    ``logsigmoid(randn) / 16``, so the multiplier is close to but below 1.

    ``beta`` controls how much of the delta correction is written into state.
    A beta near 0 barely updates memory; a beta near 1 writes the full
    residual ``v_t - prediction``.
    """
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
    )

    if scale is None:
        scale = head_dim_k**-0.5

    # Compute in float32 by default.  This makes the reference stable and keeps
    # it independent of whatever dtype the optimized kernel will eventually use.
    q = q.to(compute_dtype)
    k = k.to(compute_dtype)
    v = v.to(compute_dtype)
    g = g.to(compute_dtype)
    beta = beta.to(compute_dtype)

    if num_k_heads != num_v_heads:
        # FlashQLA supports grouped heads: q/k may have fewer heads than v.
        # For the simple reference, expand q/k so every value head has its own
        # q/k slice.  Example: Hk=2, Hv=8 means each q/k head is reused 4 times.
        repeat = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)

    if initial_state is None:
        # state[b, h] is a [K, V] matrix.  Reading uses k_t @ state, and
        # writing uses outer(k_t, v_new).
        state = torch.zeros(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            dtype=compute_dtype,
            device=q.device,
        )
    else:
        state = initial_state.to(compute_dtype, copy=True)

    outputs = []
    for t in range(num_tokens):
        # Current token slices:
        #   q_t, k_t: [B, Hv, K] after optional head expansion
        #   v_t:      [B, Hv, V]
        #   g/beta:   [B, Hv] 
        #state:   [B, Hv, K, V]
        #decay_t: [B, Hv]
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        decay_t = torch.exp(g[:, t])
        beta_t = beta[:, t]

        # Step 1: forget part of the old memory.  Since g is a log-decay,
        # exp(g) is the multiplicative decay applied to the whole K x V state
        # matrix for each value head.
        state = state * decay_t[:, :, None, None]

        # Step 2: read the current key from the decayed state.
        # For every batch/head: prediction[b,h,v] = sum_k k_t[b,h,k] * state[b,h,k,v].
        prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)

        # Step 3: delta update.  If prediction already matches v_t, the update
        # is small.  beta gates how strongly this residual is written.
        v_new = beta_t[:, :, None] * (v_t - prediction)

        # Step 4: write the delta back as a rank-1 outer product.
        # state[b,h,k,v] += k_t[b,h,k] * v_new[b,h,v].
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, v_new)

        # Step 5: read with q from the updated state to produce output token t.
        output_t = scale * torch.einsum("bhk,bhkv->bhv", q_t, state)
        outputs.append(output_t)

    # Stack token outputs back into [B, T, Hv, V].
    output = torch.stack(outputs, dim=1)
    if return_final_state:
        return output, state
    return output

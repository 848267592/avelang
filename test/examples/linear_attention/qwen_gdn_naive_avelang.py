from __future__ import annotations

import torch

import avelang
import avelang.language as al


@avelang.jit
def _qwen_gdn_naive_forward_kernel(
    q_ptr: al.Pointer(al.f32),
    k_ptr: al.Pointer(al.f32),
    v_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
):
    q = al.make_tensor(
        q_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_k),
            (num_tokens * num_heads * head_dim_k, num_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    k = al.make_tensor(
        k_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_k),
            (num_tokens * num_heads * head_dim_k, num_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    v = al.make_tensor(
        v_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_v),
            (num_tokens * num_heads * head_dim_v, num_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_heads), (num_tokens * num_heads, num_heads, 1)),
    )
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_heads), (num_tokens * num_heads, num_heads, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_heads, head_dim_v),
            (num_tokens * num_heads * head_dim_v, num_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_heads, head_dim_k, head_dim_v),
            (num_heads * head_dim_k * head_dim_v, head_dim_k * head_dim_v, head_dim_v, 1),
        ),
    )

    program_id = al.block_id(0)
    total_programs = batch_size * num_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            head_idx = (program_id // head_dim_v) % num_heads
            batch_idx = program_id // (num_heads * head_dim_v)

            state = al.make_local((head_dim_k,), al.f32)
            scale_f32 = al.convert(scale, al.f32)
            for kk in al.range(head_dim_k):
                state[kk] = al.convert(0.0, al.f32)

            for token_idx in al.range(num_tokens):
                # Step 1: decay the old recurrent memory by exp(g).
                decay = al.exp(g[batch_idx, token_idx, head_idx])
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] * decay

                # Step 2: read prediction = k_t @ state for this value column.
                pred = al.convert(0.0, al.f32)
                for kk in al.range(head_dim_k):
                    pred = pred + k[batch_idx, token_idx, head_idx, kk] * state[kk]

                # Step 3: compute beta-gated delta value.
                v_new = beta[batch_idx, token_idx, head_idx] * (
                    v[batch_idx, token_idx, head_idx, value_idx] - pred
                )

                # Step 4: write the rank-1 update column state += k_t * v_new.
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] + k[batch_idx, token_idx, head_idx, kk] * v_new

                # Step 5: read output = scale * q_t @ updated_state.
                acc = al.convert(0.0, al.f32)
                for kk in al.range(head_dim_k):
                    acc = acc + q[batch_idx, token_idx, head_idx, kk] * state[kk]
                out[batch_idx, token_idx, head_idx, value_idx] = scale_f32 * acc

            for kk in al.range(head_dim_k):
                final_state[batch_idx, head_idx, kk, value_idx] = state[kk]


def _require_fp32_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def qwen_gdn_naive_avelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the correctness-first Qwen GDN recurrent forward kernel.

    This v0 wrapper intentionally supports only fp32, contiguous tensors,
    zero initial state, and Hk == Hv.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")

    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}.")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}.")
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B, T, H, V], got {tuple(v.shape)}.")
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, H], got {tuple(g.shape)}.")
    if beta.shape != g.shape:
        raise ValueError(f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}.")

    batch_size, num_tokens, num_heads, head_dim_k = q.shape
    if v.shape[:3] != (batch_size, num_tokens, num_heads):
        raise ValueError(f"v must share q's [B, T, H], got {tuple(v.shape[:3])}.")
    if g.shape != (batch_size, num_tokens, num_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_heads)}, got {tuple(g.shape)}.")

    head_dim_v = v.shape[-1]
    if scale is None:
        scale = head_dim_k**-0.5

    out_shape = (batch_size, num_tokens, num_heads, head_dim_v)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.float32, device=q.device)
    else:
        _require_fp32_cuda_contiguous("out", out)
        if out.shape != out_shape:
            raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}.")
        if out.device != q.device:
            raise ValueError(f"out must be on device {q.device}, got {out.device}.")

    state_shape = (batch_size, num_heads, head_dim_k, head_dim_v)
    if final_state is None:
        final_state = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    else:
        _require_fp32_cuda_contiguous("final_state", final_state)
        if final_state.shape != state_shape:
            raise ValueError(f"final_state must have shape {state_shape}, got {tuple(final_state.shape)}.")
        if final_state.device != q.device:
            raise ValueError(f"final_state must be on device {q.device}, got {final_state.device}.")

    grid_size = batch_size * num_heads * head_dim_v
    _qwen_gdn_naive_forward_kernel[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        q,
        k,
        v,
        g,
        beta,
        out,
        final_state,
        float(scale),
        batch_size,
        num_tokens,
        num_heads,
        head_dim_k,
        head_dim_v,
    )
    return out, final_state

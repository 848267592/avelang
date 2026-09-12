from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_naive_avelang_v1 import qwen_gdn_naive_avelang_v1


@avelang.jit
def _qwen_gdn_naive_bf16_forward_kernel_v2(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    batch_size: al.constexpr,
    num_tokens: al.constexpr,
    num_k_heads: al.constexpr,
    num_v_heads: al.constexpr,
    head_dim_k: al.constexpr,
    head_dim_v: al.constexpr,
    has_initial_state: al.constexpr,
):
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_k_heads, head_dim_k),
            (num_tokens * num_k_heads * head_dim_k, num_k_heads * head_dim_k, head_dim_k, 1),
        ),
    )
    v = al.make_tensor(
        v_ptr,
        al.bf16,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((batch_size, num_tokens, num_v_heads), (num_tokens * num_v_heads, num_v_heads, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            (num_v_heads * head_dim_k * head_dim_v, head_dim_k * head_dim_v, head_dim_v, 1),
        ),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_tokens, num_v_heads, head_dim_v),
            (num_tokens * num_v_heads * head_dim_v, num_v_heads * head_dim_v, head_dim_v, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout(
            (batch_size, num_v_heads, head_dim_k, head_dim_v),
            (num_v_heads * head_dim_k * head_dim_v, head_dim_k * head_dim_v, head_dim_v, 1),
        )
    )
    program_id= al.block_id(0)
    total_programs = batch_size * num_v_heads * head_dim_v

    if al.thread_id(0) == 0:
        if program_id < total_programs:
            value_idx = program_id % head_dim_v
            value_head_idx = (program_id // head_dim_v) % num_v_heads
            batch_idx = program_id // (num_v_heads * head_dim_v)
            repeat = num_v_heads // num_k_heads
            key_head_idx = value_head_idx // repeat

            state = al.make_local((head_dim_k,), al.f32)
            scale_f32 = al.convert(scale, al.f32)
            for kk in al.range(head_dim_k):
                if has_initial_state:
                    state[kk] = initial_state[batch_idx, value_head_idx, kk, value_idx]
                else:
                    state[kk] = al.convert(0.0, al.f32)

            for token_idx in al.range(num_tokens):
                # Step 1: decay state.
                decay = al.exp(g[batch_idx, token_idx, value_head_idx])
                for kk in al.range(head_dim_k):
                    state[kk] = state[kk] * decay

                # Step 2: prediction.
                pred = al.convert(0.0, al.f32)
                for kk in al.range(head_dim_k):
                    k_val = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                    pred = pred + k_val * state[kk]

                # Step 3: v_new.
                v_val = al.convert(v[batch_idx, token_idx, value_head_idx, value_idx], al.f32)
                v_new = beta[batch_idx, token_idx, value_head_idx] * (v_val - pred)

                # Step 4: state update.
                for kk in al.range(head_dim_k):
                    k_val = al.convert(k[batch_idx, token_idx, key_head_idx, kk], al.f32)
                    state[kk] = state[kk] + k_val * v_new

                # Step 5: output.
                acc = al.convert(0.0, al.f32)
                for kk in al.range(head_dim_k):
                    q_val = al.convert(q[batch_idx, token_idx, key_head_idx, kk], al.f32)
                    acc = acc + q_val * state[kk]
                out[batch_idx, token_idx, value_head_idx, value_idx] = scale_f32 * acc

            for kk in al.range(head_dim_k):
                final_state[batch_idx, value_head_idx, kk, value_idx] = state[kk]


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _require_fp32_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}.")
    _require_cuda_contiguous(name, tensor)


def _validate_common_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, Hk, K], got {tuple(q.shape)}.")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}.")
    if v.ndim != 4:
        raise ValueError(f"v must have shape [B, T, Hv, V], got {tuple(v.shape)}.")
    if g.ndim != 3:
        raise ValueError(f"g must have shape [B, T, Hv], got {tuple(g.shape)}.")
    if beta.shape != g.shape:
        raise ValueError(f"beta must have shape {tuple(g.shape)}, got {tuple(beta.shape)}.")

    batch_size, num_tokens, num_k_heads, head_dim_k = q.shape
    v_batch, v_tokens, num_v_heads, head_dim_v = v.shape
    if (v_batch, v_tokens) != (batch_size, num_tokens):
        raise ValueError(f"v must share q's [B, T], got {tuple(v.shape[:2])}.")
    if g.shape != (batch_size, num_tokens, num_v_heads):
        raise ValueError(f"g must have shape {(batch_size, num_tokens, num_v_heads)}, got {tuple(g.shape)}.")
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hv={num_v_heads}, Hk={num_k_heads}.")

    return batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v


def qwen_gdn_naive_avelang_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen GDN naive forward with bf16 q/k/v and fp32 accumulation.

    The fp32 q/k/v case is delegated to the already validated v1 kernel.
    """
    if q.dtype == k.dtype == v.dtype == torch.float32:
        return qwen_gdn_naive_avelang_v1(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            scale=scale,
            out=out,
            final_state=final_state,
        )

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must have dtype torch.bfloat16 for v2 bf16 kernel, got {tensor.dtype}.")
        _require_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")

    for name, tensor in (("g", g), ("beta", beta)):
        _require_fp32_cuda_contiguous(name, tensor)
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on device {q.device}, got {tensor.device}.")

    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_common_shapes(
        q,
        k,
        v,
        g,
        beta,
    )

    state_shape = (batch_size, num_v_heads, head_dim_k, head_dim_v)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state_arg = None
    else:
        _require_fp32_cuda_contiguous("initial_state", initial_state)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}, got {tuple(initial_state.shape)}.")
        if initial_state.device != q.device:
            raise ValueError(f"initial_state must be on device {q.device}, got {initial_state.device}.")
        initial_state_arg = initial_state

    if scale is None:
        scale = head_dim_k**-0.5

    out_shape = (batch_size, num_tokens, num_v_heads, head_dim_v)
    if out is None:
        out = torch.empty(out_shape, dtype=torch.float32, device=q.device)
    else:
        _require_fp32_cuda_contiguous("out", out)
        if out.shape != out_shape:
            raise ValueError(f"out must have shape {out_shape}, got {tuple(out.shape)}.")
        if out.device != q.device:
            raise ValueError(f"out must be on device {q.device}, got {out.device}.")

    if final_state is None:
        final_state = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    else:
        _require_fp32_cuda_contiguous("final_state", final_state)
        if final_state.shape != state_shape:
            raise ValueError(f"final_state must have shape {state_shape}, got {tuple(final_state.shape)}.")
        if final_state.device != q.device:
            raise ValueError(f"final_state must be on device {q.device}, got {final_state.device}.")

    if initial_state_arg is None:
        initial_state_arg = final_state

    grid_size = batch_size * num_v_heads * head_dim_v
    _qwen_gdn_naive_bf16_forward_kernel_v2[lambda: ((grid_size, 1, 1), (1, 1, 1))](
        q,
        k,
        v,
        g,
        beta,
        initial_state_arg,
        out,
        final_state,
        float(scale),
        batch_size,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_dim_k,
        head_dim_v,
        has_initial_state,
    )
    return out, final_state

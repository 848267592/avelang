"""Opt-in BT64 full Qwen GDN experiment around the frozen gfx942 asm v0.

This module deliberately keeps v24 and all production dispatch untouched.  It
uses the existing generic Avelang stages for the BT64 contract, the fixed
external-HSACO recurrence, and the generic (not v24 BT16) Avelang chunk-o
stage.  It never imports or invokes the vLLM full-forward wrapper.

The public result has the vLLM layout and dtype: ``[B, T, Hv, V]`` BF16 output
plus an optional FP32 final state.  The path is experimental because its
upstream stages use the historical generic BT64 Avelang implementations rather
than a native BT64 lowering.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from qwen_gdn_bt64_gfx942_asm_v0_experimental import (
    BT as ASM_BT,
    contract as asm_contract,
    qwen_gdn_bt64_gfx942_asm_v0,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_chunk_o_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)


BT = 64
BATCH = 1
H_K = 4
H_V = 8
K_DIM = 128
V_DIM = 128


def _require_target(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    """Validate the intentionally narrow BT64 ABI and create a zero state."""
    if ASM_BT != BT:
        raise RuntimeError(f"asm v0 contract unexpectedly changed BT={ASM_BT}")
    expected_qk = (BATCH, q.shape[1], H_K, K_DIM)
    expected_v = (BATCH, q.shape[1], H_V, V_DIM)
    expected_gate = (BATCH, q.shape[1], H_V)
    if q.ndim != 4 or tuple(q.shape) != expected_qk:
        raise ValueError(f"q must have shape {expected_qk}, got {tuple(q.shape)}")
    if tuple(k.shape) != expected_qk:
        raise ValueError(f"k must have shape {expected_qk}, got {tuple(k.shape)}")
    if tuple(v.shape) != expected_v:
        raise ValueError(f"v must have shape {expected_v}, got {tuple(v.shape)}")
    if tuple(g.shape) != expected_gate or tuple(beta.shape) != expected_gate:
        raise ValueError("g and beta must have shape [1,T,8]")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("the BT64 experiment requires BF16 q/k/v")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("the BT64 experiment requires FP32 g/beta")
    tensors = (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    for name, tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA/HIP tensor")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
    t = q.shape[1]
    if t < BT or t % BT:
        raise ValueError("the BT64 experiment only supports T divisible by 64")
    if initial_state is None:
        return torch.zeros((BATCH, H_V, V_DIM, K_DIM), dtype=torch.float32, device=q.device)
    if tuple(initial_state.shape) != (BATCH, H_V, V_DIM, K_DIM):
        raise ValueError("initial_state must have shape [1,8,128,128]")
    if initial_state.dtype != torch.float32 or not initial_state.is_cuda or not initial_state.is_contiguous():
        raise ValueError("initial_state must be a contiguous CUDA/HIP FP32 tensor")
    if initial_state.device != q.device:
        raise ValueError(f"initial_state must be on {q.device}, got {initial_state.device}")
    return initial_state


def qwen_gdn_chunk_o_bt64_avelang_experimental(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_bf16: torch.Tensor,
    g_cumsum: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """BT64 output stage consuming the asm's BF16 state snapshots.

    The generic v6 BF16 output primitive is parameterized by ``chunk_size``;
    unlike v24's MFMA output kernel it has no BT16 layout assumption.  The
    widening is explicit and preserves every BF16 element emitted by asm v0.
    """
    if h_bf16.dtype != torch.bfloat16:
        raise ValueError("BT64 chunk_o requires BF16 h emitted by asm v0")
    return qwen_gdn_chunk_o_avelang_v6_standalone(
        q,
        k,
        v_new,
        h_bf16.float().contiguous(),
        g_cumsum,
        scale=scale,
        chunk_size=BT,
        prefer_optimized=True,
    )


def qwen_gdn_full_bt64_gfx942_asm_v0_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Run the candidate without vLLM and return every materialized stage."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=BT, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=BT)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k, v, g_cumsum, beta, a_solved, chunk_size=BT, prefer_optimized=True
    )
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_avelang_experimental(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_gfx942_asm_v0(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    fallback: Callable[[], tuple[torch.Tensor, torch.Tensor | None]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Experimental vLLM-layout forward using Avelang stages plus frozen asm.

    A caller can provide an explicit fallback for guard failures.  There is no
    implicit fallback and this function never imports the vLLM full wrapper.
    """
    try:
        stages = qwen_gdn_full_bt64_gfx942_asm_v0_stages(
            q, k, v, g, beta, initial_state=initial_state, scale=scale
        )
    except ValueError:
        if fallback is not None:
            return fallback()
        raise
    return stages["output"], stages["final_state"] if output_final_state else None


def contract() -> dict[str, object]:
    """The explicit experimental public and recurrence contract."""
    return {
        "experimental": True,
        "candidate_calls_vllm_full_wrapper": False,
        "shape": "B=1,T%64=0,Hk=4,Hv=8,K=V=128,layout=[B,T,H,D]",
        "inputs": {"q": "bf16", "k": "bf16", "v": "bf16", "g": "fp32", "beta": "fp32", "initial_state": "fp32|None"},
        "outputs": {"output": "bf16", "final_state": "fp32|None"},
        "recurrence": asm_contract(),
        "upstream": "Avelang v6 generic BT64 cumsum/KKT/w_u + v18 BT64 solve",
        "downstream": "Avelang v6 generic BF16 chunk_o with explicit BF16->FP32 h widening",
    }


__all__ = [
    "BT",
    "contract",
    "qwen_gdn_chunk_o_bt64_avelang_experimental",
    "qwen_gdn_full_bt64_gfx942_asm_v0",
    "qwen_gdn_full_bt64_gfx942_asm_v0_stages",
]

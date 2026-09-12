"""Audit-only caller-provided output wrappers for the two BT64 solves.

These wrappers deliberately launch the existing kernels unchanged.  They are
not imported by the production Stage 4/5C path and never allocate, initialize,
or copy the solve output.
"""

from __future__ import annotations

import torch

from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (
    _qwen_gdn_solve_kernel_v18_parallel,
)
from qwen_gdn_solve_bt64_hierarchical_fp32_v1 import (
    _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1,
)


BT = 64
HEADS = 8


def _validate_direct_out(a: torch.Tensor, out: torch.Tensor) -> tuple[int, int]:
    for name, tensor in (("a", a), ("out", out)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32.")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a CUDA/HIP device.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[2:] != (HEADS, BT):
            raise ValueError(f"{name} must have shape [1,T,8,64], got {tuple(tensor.shape)}.")

    if out.shape != a.shape:
        raise ValueError(f"out shape {tuple(out.shape)} must equal input shape {tuple(a.shape)}.")
    if out.device != a.device:
        raise ValueError(f"out must be on {a.device}, got {out.device}.")

    num_tokens = int(a.shape[1])
    if num_tokens <= 0 or num_tokens % BT != 0:
        raise ValueError("Stage 5E direct-out requires T > 0 and divisible by 64.")

    # Both kernels assume a distinct output.  The hierarchical kernel clears
    # out before consuming all input blocks, so any shared storage is unsafe.
    if a.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
        raise ValueError("a and out must not alias or share storage.")

    pointer = int(out.data_ptr())
    if pointer % 16:
        raise ValueError("out must be at least 16-byte aligned.")
    return num_tokens, num_tokens // BT


def qwen_gdn_solve_v18_bt64_direct_out_audit(a: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Launch the unmodified v18 solve directly into ``out``."""
    num_tokens, num_chunks = _validate_direct_out(a, out)
    grid = num_chunks * HEADS
    _qwen_gdn_solve_kernel_v18_parallel[lambda: ((grid, 1, 1), (128, 1, 1))](
        a,
        out,
        num_tokens,
        BT,
        num_chunks,
    )
    return out


def qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(
    a: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch the unmodified hierarchical FP32 solve directly into ``out``."""
    num_tokens, num_chunks = _validate_direct_out(a, out)
    grid = num_chunks * HEADS
    _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1[
        lambda: ((grid, 1, 1), (256, 1, 1))
    ](a, out, num_tokens, num_chunks)
    return out


__all__ = [
    "qwen_gdn_solve_v18_bt64_direct_out_audit",
    "qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit",
]

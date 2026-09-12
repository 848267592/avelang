#!/usr/bin/env python3
"""Standalone MFMA feasibility smoke for tiled Qwen GDN chunk_gdr math.

This is deliberately not a production chunk_gdr kernel. It reuses the existing
Avelang AMDGPU BF16 GEMM kernel, which currently requires M/N/K multiples of
128, so small Qwen tiles are zero-padded before calling the MFMA path.

Validated operations:
  pred[BT,BV]      = W_chunk[BT,K] @ H_tile[BV,K].T
  split K=128 pred = W1@H1.T + W2@H2.T, each K tile is 64
  delta_H[BV,K]    = v_new[BT,BV].T @ K_chunk[BT,K]
"""

from __future__ import annotations

import torch
from avelang_kernels.amdgpu_gemm import gemm_pipeline_transposed_b


def pad_rows_cols(x: torch.Tensor, rows: int = 128, cols: int = 128) -> torch.Tensor:
    out = torch.zeros((rows, cols), device=x.device, dtype=torch.bfloat16)
    out[: x.shape[0], : x.shape[1]] = x.to(torch.bfloat16)
    return out


def mfma_dot_transposed_b(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Return A @ B.T through the existing padded Avelang MFMA GEMM path."""
    if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"expected A[M,K], B[N,K], got {tuple(A.shape)} and {tuple(B.shape)}")
    if A.shape[0] > 128 or B.shape[0] > 128 or A.shape[1] > 128:
        raise ValueError("this smoke only pads into one 128x128 GEMM tile")
    Ap = pad_rows_cols(A)
    Bp = pad_rows_cols(B)
    Cp = gemm_pipeline_transposed_b(Ap, Bp)
    torch.cuda.synchronize()
    return Cp[: A.shape[0], : B.shape[0]].float()


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 2.5e-1, rtol: float = 2.5e-1) -> None:
    max_abs = (actual.float() - expected.float()).abs().max().item()
    max_rel = ((actual.float() - expected.float()).abs() / expected.float().abs().clamp_min(1e-6)).max().item()
    ok = bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))
    print(f"{name},ok={ok},max_abs={max_abs:.9g},max_rel={max_rel:.9g},actual_shape={tuple(actual.shape)}")
    if not ok:
        raise AssertionError(f"{name} failed: max_abs={max_abs}, max_rel={max_rel}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    torch.manual_seed(20260614)
    BT, BV = 16, 16

    # Step 3: pred[BT,BV] = W[BT,64] @ H[BV,64].T.
    K64 = 64
    W64 = torch.randn((BT, K64), device="cuda", dtype=torch.bfloat16)
    H64 = torch.randn((BV, K64), device="cuda", dtype=torch.bfloat16)
    pred64 = mfma_dot_transposed_b(W64, H64)
    expected64 = (W64.float() @ H64.float().T).to(torch.bfloat16).float()
    check("pred_bt16_bv16_k64", pred64, expected64)

    # Step 4: Qwen target K=128 as two 64-wide tiles.
    W128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16)
    H128 = torch.randn((BV, 128), device="cuda", dtype=torch.bfloat16)
    pred128 = mfma_dot_transposed_b(W128[:, :64], H128[:, :64]) + mfma_dot_transposed_b(W128[:, 64:], H128[:, 64:])
    expected128 = (W128.float() @ H128.float().T).to(torch.bfloat16).float()
    check("pred_bt16_bv16_k128_split64", pred128.to(torch.bfloat16).float(), expected128, atol=3.5e-1, rtol=3.5e-1)

    # Step 5: delta_H[BV,K] = v_new[BT,BV].T @ K_chunk[BT,K].
    v_new = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16)
    Kchunk64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16)
    delta64 = mfma_dot_transposed_b(v_new.T.contiguous(), Kchunk64.T.contiguous())
    expected_delta64 = (v_new.float().T @ Kchunk64.float()).to(torch.bfloat16).float()
    check("delta_h_bv16_k64", delta64, expected_delta64)

    Kchunk128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16)
    delta128 = mfma_dot_transposed_b(v_new.T.contiguous(), Kchunk128[:, :64].T.contiguous())
    delta128 = torch.cat([delta128, mfma_dot_transposed_b(v_new.T.contiguous(), Kchunk128[:, 64:].T.contiguous())], dim=1)
    expected_delta128 = (v_new.float().T @ Kchunk128.float()).to(torch.bfloat16).float()
    check("delta_h_bv16_k128_split64", delta128.to(torch.bfloat16).float(), expected_delta128, atol=3.5e-1, rtol=3.5e-1)

    print("prototype_status,ok")


if __name__ == "__main__":
    main()

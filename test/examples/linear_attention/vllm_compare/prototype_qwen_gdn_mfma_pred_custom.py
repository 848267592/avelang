#!/usr/bin/env python3
"""Custom Avelang MFMA prototypes for Qwen GDN tiled chunk_gdr.

This file intentionally does not call or pad into the existing 128x128 GEMM
kernel.  It validates the first small tiles needed by a vLLM-style chunk_gdr:

  pred[BT,BV]   = W[BT,K] @ H[BV,K].T
  delta[BV,K]   = V_new[BT,BV].T @ K_chunk[BT,K]

First fixed targets:
  BT=16, BV=16, BK=64, and K=128 as two/split BK=64 blocks.
"""

from __future__ import annotations

import argparse

import torch

import avelang
import avelang.language as al

BT = 16
BV = 16
BF16_BYTES = 2
F32_BYTES = 4


@avelang.jit
def _mfma_pred_16x16_kernel(
    w_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
):
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    h = al.make_tensor(h_ptr, al.bf16, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BT, BV), (BV, 1)))

    packed_row_stride = total_k >> 1
    k_vecs = total_k >> 3
    w_vec = al.view(w, al.i32, al.make_layout((BT, k_vecs, 4), (packed_row_stride, 4, 1)))
    h_vec = al.view(h, al.i32, al.make_layout((BV, k_vecs, 4), (packed_row_stride, 4, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    acc = al.full((4,), 0.0, al.f32)

    # Match the Avelang GEMM fragment pattern: each packed row fragment plus
    # two static MFMA calls covers a 32-wide logical K slice. BK=64 uses two
    # batches, K=128 uses four batches.
    for batch in al.range(total_k >> 5):
        k_vec = lane_group + batch * 4
        a_words = w_vec[lane_col, k_vec]
        b_words = h_vec[lane_col, k_vec]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

    for r in al.range(4):
        out_row = lane_group * 4 + r
        out_col = lane_col
        out[out_row, out_col] = acc[r]


@avelang.jit
def _mfma_matmul_16x16x16_kernel(
    a_ptr: al.Pointer(al.bf16),
    b_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
):
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((16, 16), (16, 1)))
    b = al.make_tensor(b_ptr, al.bf16, al.make_layout((16, 16), (16, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((16, 16), (16, 1)))
    a_vec = al.view(a, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))
    b_vec = al.view(b, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    acc = al.full((4,), 0.0, al.f32)

    if lane_group == 0:
        a_words = a_vec[lane_col, 0]
        b_words = b_vec[lane_col, 0]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
    if lane_group == 1:
        a_words = a_vec[lane_col, 0]
        b_words = b_vec[lane_col, 0]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
    if lane_group == 2:
        a_words = a_vec[lane_col, 1]
        b_words = b_vec[lane_col, 1]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
    if lane_group == 3:
        a_words = a_vec[lane_col, 1]
        b_words = b_vec[lane_col, 1]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

    for r in al.range(4):
        out_row = lane_group * 4 + r
        out_col = lane_col
        out[out_row, out_col] = acc[r]


def pred_custom(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    if w.shape[0] != BT or h.shape[0] != BV or w.shape[1] != h.shape[1]:
        raise ValueError(f"expected W[16,K], H[16,K], got {tuple(w.shape)} and {tuple(h.shape)}")
    if w.dtype != torch.bfloat16 or h.dtype != torch.bfloat16:
        raise ValueError("W/H must be BF16")
    if w.shape[1] not in (64, 128):
        raise ValueError("prototype supports K=64 or K=128")
    out = torch.empty((BT, BV), device=w.device, dtype=torch.float32)
    _mfma_pred_16x16_kernel[lambda: ((1, 1, 1), (64, 1, 1))](w.contiguous(), h.contiguous(), out, w.shape[1])
    return out


def matmul16_custom(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != (16, 16) or b.shape != (16, 16):
        raise ValueError(f"expected A/B [16,16], got {tuple(a.shape)} and {tuple(b.shape)}")
    out = torch.empty((16, 16), device=a.device, dtype=torch.float32)
    _mfma_matmul_16x16x16_kernel[lambda: ((1, 1, 1), (64, 1, 1))](a.contiguous(), b.contiguous(), out)
    return out


def delta_custom(v_new: torch.Tensor, k_chunk: torch.Tensor) -> torch.Tensor:
    if v_new.shape != (BT, BV):
        raise ValueError(f"expected v_new[16,16], got {tuple(v_new.shape)}")
    if k_chunk.shape[0] != BT or k_chunk.shape[1] not in (64, 128):
        raise ValueError(f"expected K_chunk[16,64|128], got {tuple(k_chunk.shape)}")
    a = v_new.T.contiguous()
    tiles = []
    for base in range(0, k_chunk.shape[1], 16):
        b = k_chunk[:, base : base + 16].T.contiguous()
        tiles.append(matmul16_custom(a, b))
    torch.cuda.synchronize()
    return torch.cat(tiles, dim=1)


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-1, rtol: float = 1e-1) -> None:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    print(f"{name},ok={bool(ok)},max_abs={max_abs:.9g},max_rel={max_rel:.9g},shape={tuple(actual.shape)}")
    if not ok:
        raise AssertionError(f"{name} failed: max_abs={max_abs}, max_rel={max_rel}")


def run(seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    gen = torch.Generator(device="cuda").manual_seed(seed)

    w64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    h64 = torch.randn((BV, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    check("pred_bt16_bv16_bk64_custom", pred_custom(w64, h64), w64.float() @ h64.float().T, atol=2e-1, rtol=2e-1)

    w128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    h128 = torch.randn((BV, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    check("pred_bt16_bv16_k128_custom", pred_custom(w128, h128), w128.float() @ h128.float().T, atol=3e-1, rtol=3e-1)

    a16 = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16, generator=gen)
    b16 = torch.randn((16, 16), device="cuda", dtype=torch.bfloat16, generator=gen)
    check("matmul16x16x16_custom", matmul16_custom(a16, b16), a16.float() @ b16.float().T, atol=1e-1, rtol=1e-1)

    v_new = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16, generator=gen)
    k64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    check("delta_bv16_bk64_custom", delta_custom(v_new, k64), v_new.float().T @ k64.float(), atol=2e-1, rtol=2e-1)

    k128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16, generator=gen)
    check("delta_bv16_k128_custom", delta_custom(v_new, k128), v_new.float().T @ k128.float(), atol=3e-1, rtol=3e-1)

    print("custom_mfma_prototype_status,ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()
    run(args.seed)

#!/usr/bin/env python3
"""Standalone internal-staging MFMA delta prototype for Qwen GDN v11.

Computes:
    delta[BV, K] = v_decay[BT, BV].T @ k_chunk[BT, K]

Fixed staged-prototype shape:
    BT=16, BV=16, K in {64,128}, subtile=16, BF16 inputs, FP32 accumulation.

Unlike prototype_qwen_gdn_mfma_pred_custom.py, this kernel performs the
transpose/staging inside Avelang, matching the failing integrated v11 update.
"""

from __future__ import annotations

import argparse

import torch

import avelang
import avelang.language as al

BT = 16
BV = 16


@avelang.jit
def _mfma_delta_staged_16x16_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    tile = al.block_id(0)
    k_base = tile * 16

    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_tile_t = al.make_shared((16, BT), al.bf16)
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    ktile_vec = al.view(k_tile_t, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))

    # Stage v_decay_source[BT,BV] as v_decay_t[BV,BT].
    # Stage one K output tile as k_tile_t[16,BT].
    for rep in al.range(4):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        v_decay_t[row, tok] = v[tok, row]
        k_tile_t[row, tok] = k[tok, k_base + row]

    al.syncthreads()

    acc = al.full((4,), 0.0, al.f32)
    if lane_group == 0:
        a_words = vdecay_vec[lane_col, 0]
        b_words = ktile_vec[lane_col, 0]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
    if lane_group == 1:
        a_words = vdecay_vec[lane_col, 0]
        b_words = ktile_vec[lane_col, 0]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
    if lane_group == 2:
        a_words = vdecay_vec[lane_col, 1]
        b_words = ktile_vec[lane_col, 1]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
    if lane_group == 3:
        a_words = vdecay_vec[lane_col, 1]
        b_words = ktile_vec[lane_col, 1]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

    for r in al.range(4):
        out_row = lane_group * 4 + r
        out_col = k_base + lane_col
        out[out_row, out_col] = acc[r]


def delta_staged(v_decay: torch.Tensor, k_chunk: torch.Tensor) -> torch.Tensor:
    if v_decay.shape != (BT, BV):
        raise ValueError(f"v_decay must have shape {(BT, BV)}, got {tuple(v_decay.shape)}")
    if k_chunk.shape[0] != BT or k_chunk.shape[1] not in (64, 128):
        raise ValueError(f"k_chunk must have shape (16,64|128), got {tuple(k_chunk.shape)}")
    if v_decay.dtype != torch.bfloat16 or k_chunk.dtype != torch.bfloat16:
        raise ValueError("v_decay and k_chunk must be BF16")
    out = torch.empty((BV, k_chunk.shape[1]), device=v_decay.device, dtype=torch.float32)
    grid = (k_chunk.shape[1] // 16, 1, 1)
    _mfma_delta_staged_16x16_kernel[lambda: (grid, (64, 1, 1))](
        v_decay.contiguous(),
        k_chunk.contiguous(),
        out,
        k_chunk.shape[1],
    )
    return out


def max_rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).abs() / expected.float().abs().clamp_min(1e-6)).max().item()


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-4, rtol: float = 1e-4) -> None:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = max_rel_err(actual, expected)
    print(f"{name},ok={bool(torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol))},max_abs={max_abs:.9g},max_rel={max_rel:.9g}")
    for base in range(0, actual.shape[1], 16):
        tile_abs = diff[:, base : base + 16].max().item()
        print(f"{name}_tile,{base}:{base+16},max_abs={tile_abs:.9g}")
    if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
        raise AssertionError(f"{name} failed: max_abs={max_abs}, max_rel={max_rel}")


def run(seed: int = 20260614) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    v_decay = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16, generator=gen)
    k64 = torch.randn((BT, 64), device="cuda", dtype=torch.bfloat16, generator=gen)
    k128 = torch.randn((BT, 128), device="cuda", dtype=torch.bfloat16, generator=gen)

    actual64 = delta_staged(v_decay, k64)
    expected64 = v_decay.float().T @ k64.float()
    check("delta_staged_k64", actual64, expected64)

    actual128 = delta_staged(v_decay, k128)
    expected128 = v_decay.float().T @ k128.float()
    check("delta_staged_k128", actual128, expected128)
    print("delta_staged_status,ok")


# Main entry point is kept at the end of this file so later isolation kernels
# are defined before argparse runs.

@avelang.jit
def _mfma_delta_state_staged_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t[row, tok] = v[tok, row]

    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16
        acc = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 1:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
        if lane_group == 2:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 3:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc[r]

        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


def delta_state_staged(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out = torch.empty_like(init, dtype=torch.float32)
    _mfma_delta_state_staged_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, k_chunk.shape[1], float(scale)
    )
    return out


@avelang.jit
def _mfma_delta_state_staged_with_dummy_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    # Match the integrated v11 shared-memory footprint before update.  These
    # arrays are deliberately touched so compiler/LDS allocation behaviour is
    # close to the full kernel while the math stays standalone.
    h0_bf16 = al.make_shared((BV, 64), al.bf16)
    h1_bf16 = al.make_shared((BV, 64), al.bf16)
    w0_bf16 = al.make_shared((BT, 64), al.bf16)
    w1_bf16 = al.make_shared((BT, 64), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // 64
        kk = idx - vv * 64
        h0_bf16[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1_bf16[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        w0_bf16[vv, kk] = v[vv, kk & 15]
        w1_bf16[vv, kk] = v[vv, kk & 15]

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t[row, tok] = v[tok, row]

    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16
        acc = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 1:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
        if lane_group == 2:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 3:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc[r]

        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


def delta_state_staged_with_dummy(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out = torch.empty_like(init, dtype=torch.float32)
    _mfma_delta_state_staged_with_dummy_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, k_chunk.shape[1], float(scale)
    )
    return out


@avelang.jit
def _mfma_delta_state_staged_with_pred_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    h0_bf16 = al.make_shared((BV, 64), al.bf16)
    h1_bf16 = al.make_shared((BV, 64), al.bf16)
    w0_bf16 = al.make_shared((BT, 64), al.bf16)
    w1_bf16 = al.make_shared((BT, 64), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // 64
        kk = idx - vv * 64
        h0_bf16[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1_bf16[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        # Fill W tiles with deterministic BF16 data from K so the pred MFMA is real work.
        tok = vv
        w0_bf16[tok, kk] = k[tok, kk]
        w1_bf16[tok, kk] = k[tok, kk + 64]

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t[row, tok] = v[tok, row]

    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        k_vec = lane_group + batch * 4
        a_words = w0_vec[lane_col, k_vec]
        b_words = h0_vec[lane_col, k_vec]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)
        a_words = w1_vec[lane_col, k_vec]
        b_words = h1_vec[lane_col, k_vec]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

    # Consume pred_acc in a harmless way so it cannot be trivially dropped, then
    # restore the staged v_decay tile before the update.
    for r in al.range(4):
        row = lane_group * 4 + r
        v_decay_t[row, lane_col] = al.convert(pred_acc[r] * 0.0 + v[lane_col, row], al.bf16)

    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16
        acc = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 1:
            a_words = vdecay_vec[lane_col, 0]
            b_words = kall_vec[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
        if lane_group == 2:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        if lane_group == 3:
            a_words = vdecay_vec[lane_col, 1]
            b_words = kall_vec[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc[r]

        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


def delta_state_staged_with_pred(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out = torch.empty_like(init, dtype=torch.float32)
    _mfma_delta_state_staged_with_pred_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, k_chunk.shape[1], float(scale)
    )
    return out


@avelang.jit
def _mfma_delta_state_staged_with_pred_restaged_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    total_k: al.constexpr,
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, total_k), (total_k, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, total_k), (total_k, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    # pred-stage shared buffers (will be used and then dropped)
    h0_bf16 = al.make_shared((BV, 64), al.bf16)
    h1_bf16 = al.make_shared((BV, 64), al.bf16)
    w0_bf16 = al.make_shared((BT, 64), al.bf16)
    w1_bf16 = al.make_shared((BT, 64), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    # views for pred stage
    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))

    # staging for pred: fill helper buffers to create real pred MFMA work
    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // 64
        kk = idx - vv * 64
        h0_bf16[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1_bf16[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        # Fill W tiles with deterministic BF16 data from K so the pred MFMA is real work.
        tok = vv
        w0_bf16[tok, kk] = k[tok, kk]
        w1_bf16[tok, kk] = k[tok, kk + 64]

    # initial state scaling/staging (same as other kernels)
    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    # also stage authors' original k/v (but these pred buffers will not be reused later)
    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t[row, tok] = v[tok, row]

    al.syncthreads()

    # pred MFMA: perform useful MFMA work but discard result. Use distinct local acc.
    pred_acc_local = al.full((4,), 0.0, al.f32)
    for batch in al.range(2):
        k_vec = lane_group + batch * 4
        a_words = w0_vec[lane_col, k_vec]
        b_words = h0_vec[lane_col, k_vec]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc_local = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc_local)
        pred_acc_local = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc_local)
        a_words = w1_vec[lane_col, k_vec]
        b_words = h1_vec[lane_col, k_vec]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc_local = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc_local)
        pred_acc_local = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc_local)

    # pred-stage shared buffers are no longer needed; ensure pred uses are finished
    al.syncthreads()

    # RESTAGE: reload v and k from global into brand-new shared buffers
    v_decay_t2 = al.make_shared((BV, BT), al.bf16)
    k_all_t2 = al.make_shared((128, BT), al.bf16)
    vdecay_vec2 = al.view(v_decay_t2, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec2 = al.view(k_all_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    # Before writing over these buffers ensure threads are synchronized
    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t2[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t2[row, tok] = v[tok, row]

    al.syncthreads()

    # UPDATE stage: use fresh accumulators/fragments and fresh shared views
    for tile in al.range(8):
        base = tile * 16
        acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 1:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)
        if lane_group == 2:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 3:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)

        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc2[r]

        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


def delta_state_staged_with_pred_restaged(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    out = torch.empty_like(init, dtype=torch.float32)
    _mfma_delta_state_staged_with_pred_restaged_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, k_chunk.shape[1], float(scale)
    )
    return out


# -----------------------------------------------------------------------------
# Isolation experiments for pred-MFMA -> update-MFMA correctness.
# These kernels are intentionally standalone and do not touch v9/v10/v11 paths.
# -----------------------------------------------------------------------------

@avelang.jit
def _mfma_delta_state_isolation_unpadded_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    debug_ptr: al.Pointer(al.f32),
    pred_mode: al.constexpr,  # 0 none, 1 noop no MFMA, 2 one MFMA, 3 two MFMA, 4 full MFMA discard, 5 full MFMA consume
    extra_barriers: al.constexpr,
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, 128), (128, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, 128), (128, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, 128), (128, 1)))
    debug = al.make_tensor(debug_ptr, al.f32, al.make_layout((256,), (1,)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    # All shared buffers are declared at top. Pred and update use distinct buffers.
    state = al.make_shared((BV, 128), al.f32)
    h0_bf16 = al.make_shared((BV, 64), al.bf16)
    h1_bf16 = al.make_shared((BV, 64), al.bf16)
    w0_bf16 = al.make_shared((BT, 64), al.bf16)
    w1_bf16 = al.make_shared((BT, 64), al.bf16)
    v_decay_t2 = al.make_shared((BV, BT), al.bf16)
    k_all_t2 = al.make_shared((128, BT), al.bf16)

    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (32, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (32, 4, 1)))
    vdecay_vec2 = al.view(v_decay_t2, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec2 = al.view(k_all_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    # Pred staging only. No dead stores to update buffers before pred.
    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // 64
        kk = idx - vv * 64
        h0_bf16[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1_bf16[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        tok = vv
        w0_bf16[tok, kk] = k[tok, kk]
        w1_bf16[tok, kk] = k[tok, kk + 64]

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    dummy = al.convert(0.0, al.f32)
    if pred_mode == 1:
        for batch in al.range(2):
            k_vec = lane_group + batch * 4
            dummy = dummy + al.convert(h0_bf16[lane_col, 0], al.f32) * al.convert(0.0, al.f32)
            dummy = dummy + al.convert(w0_bf16[lane_col, 0], al.f32) * al.convert(0.0, al.f32)
    if pred_mode == 2:
        a_words = w0_vec[lane_col, lane_group]
        b_words = h0_vec[lane_col, lane_group]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
    if pred_mode == 3:
        a_words = w0_vec[lane_col, lane_group]
        b_words = h0_vec[lane_col, lane_group]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)
    if pred_mode == 4 or pred_mode == 5:
        for batch in al.range(2):
            k_vec = lane_group + batch * 4
            a_words = w0_vec[lane_col, k_vec]
            b_words = h0_vec[lane_col, k_vec]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)
            a_words = w1_vec[lane_col, k_vec]
            b_words = h1_vec[lane_col, k_vec]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

    if pred_mode == 5:
        for r in al.range(4):
            debug[lane_group * 64 + lane_col * 4 + r] = pred_acc[r]
    else:
        debug[lane] = dummy

    if extra_barriers:
        al.syncthreads()
        # Force LDS reads from pred buffers between barriers.
        probe0 = al.convert(h0_bf16[lane_col, 0], al.f32)
        probe1 = al.convert(w0_bf16[lane_col, 0], al.f32)
        debug[128 + lane] = probe0 + probe1 + dummy
        al.syncthreads()
        debug[192 + lane] = al.convert(h1_bf16[lane_col, 0], al.f32) + al.convert(w1_bf16[lane_col, 0], al.f32)

    al.syncthreads()

    # Stage update buffers after pred. This is a full restage from global.
    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t2[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t2[row, tok] = v[tok, row]

    al.syncthreads()

    for tile in al.range(8):
        base = tile * 16
        acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 1:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)
        if lane_group == 2:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 3:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)
        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


@avelang.jit
def _mfma_delta_state_isolation_padded_canary_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    debug_ptr: al.Pointer(al.f32),
    pred_mode: al.constexpr,  # 4 full MFMA, 1 no-op
    scale: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, 128), (128, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, 128), (128, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, 128), (128, 1)))
    debug = al.make_tensor(debug_ptr, al.f32, al.make_layout((256,), (1,)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 160), al.f32)
    h0_bf16 = al.make_shared((BV, 80), al.bf16)
    h1_bf16 = al.make_shared((BV, 80), al.bf16)
    w0_bf16 = al.make_shared((BT, 80), al.bf16)
    w1_bf16 = al.make_shared((BT, 80), al.bf16)
    v_decay_t2 = al.make_shared((BV, 32), al.bf16)
    k_all_t2 = al.make_shared((160, BT), al.bf16)

    h0_vec = al.view(h0_bf16, al.i32, al.make_layout((BV, 8, 4), (40, 4, 1)))
    h1_vec = al.view(h1_bf16, al.i32, al.make_layout((BV, 8, 4), (40, 4, 1)))
    w0_vec = al.view(w0_bf16, al.i32, al.make_layout((BT, 8, 4), (40, 4, 1)))
    w1_vec = al.view(w1_bf16, al.i32, al.make_layout((BT, 8, 4), (40, 4, 1)))
    vdecay_vec2 = al.view(v_decay_t2, al.i32, al.make_layout((BV, 2, 4), (16, 4, 1)))
    kall_vec2 = al.view(k_all_t2, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    # Canary in padding-only slots.
    if lane < 16:
        h0_bf16[lane, 79] = al.convert(1.25, al.bf16)
        h1_bf16[lane, 79] = al.convert(2.25, al.bf16)
        w0_bf16[lane, 79] = al.convert(3.25, al.bf16)
        w1_bf16[lane, 79] = al.convert(4.25, al.bf16)
        v_decay_t2[lane, 31] = al.convert(5.25, al.bf16)
        k_all_t2[128 + lane, 0] = al.convert(6.25, al.bf16)

    for rep in al.range(16):
        idx = lane + rep * 64
        vv = idx // 64
        kk = idx - vv * 64
        h0_bf16[vv, kk] = al.convert(init[vv, kk], al.bf16)
        h1_bf16[vv, kk] = al.convert(init[vv, kk + 64], al.bf16)
        tok = vv
        w0_bf16[tok, kk] = k[tok, kk]
        w1_bf16[tok, kk] = k[tok, kk + 64]

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    al.syncthreads()

    pred_acc = al.full((4,), 0.0, al.f32)
    dummy = al.convert(0.0, al.f32)
    if pred_mode == 1:
        dummy = dummy + al.convert(h0_bf16[lane_col, 0], al.f32) * al.convert(0.0, al.f32)
    if pred_mode == 4:
        for batch in al.range(2):
            k_vec = lane_group + batch * 4
            a_words = w0_vec[lane_col, k_vec]
            b_words = h0_vec[lane_col, k_vec]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)
            a_words = w1_vec[lane_col, k_vec]
            b_words = h1_vec[lane_col, k_vec]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

    al.syncthreads()

    if lane < 16:
        debug[lane * 6 + 0] = al.convert(h0_bf16[lane, 79], al.f32)
        debug[lane * 6 + 1] = al.convert(h1_bf16[lane, 79], al.f32)
        debug[lane * 6 + 2] = al.convert(w0_bf16[lane, 79], al.f32)
        debug[lane * 6 + 3] = al.convert(w1_bf16[lane, 79], al.f32)
        debug[lane * 6 + 4] = al.convert(v_decay_t2[lane, 31], al.f32)
        debug[lane * 6 + 5] = al.convert(k_all_t2[128 + lane, 0], al.f32)

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t2[row, tok] = k[tok, row]
        if row < 16:
            v_decay_t2[row, tok] = v[tok, row]

    al.syncthreads()

    if lane < 16:
        debug[128 + lane * 6 + 0] = al.convert(h0_bf16[lane, 79], al.f32)
        debug[128 + lane * 6 + 1] = al.convert(h1_bf16[lane, 79], al.f32)
        debug[128 + lane * 6 + 2] = al.convert(w0_bf16[lane, 79], al.f32)
        debug[128 + lane * 6 + 3] = al.convert(w1_bf16[lane, 79], al.f32)
        debug[128 + lane * 6 + 4] = al.convert(v_decay_t2[lane, 31], al.f32)
        debug[128 + lane * 6 + 5] = al.convert(k_all_t2[128 + lane, 0], al.f32)

    for tile in al.range(8):
        base = tile * 16
        acc2 = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 1:
            a_words = vdecay_vec2[lane_col, 0]
            b_words = kall_vec2[base + lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)
        if lane_group == 2:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc2)
        if lane_group == 3:
            a_words = vdecay_vec2[lane_col, 1]
            b_words = kall_vec2[base + lane_col, 1]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            acc2 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc2)
        for r in al.range(4):
            out_row = lane_group * 4 + r
            out_col = base + lane_col
            state[out_row, out_col] = state[out_row, out_col] + acc2[r]
        al.syncthreads()

    for rep in al.range(32):
        idx = lane + rep * 64
        row = idx // 128
        col = idx - row * 128
        out[row, col] = state[row, col]


def _run_isolation_kernel(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, *, pred_mode: int, extra_barriers: bool = False, padded: bool = False, scale: float = 1.0):
    out = torch.empty_like(init, dtype=torch.float32)
    debug = torch.empty((256,), device=v_decay.device, dtype=torch.float32)
    debug.fill_(float('nan'))
    if padded:
        _mfma_delta_state_isolation_padded_canary_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
            v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, debug, int(pred_mode), float(scale)
        )
    else:
        _mfma_delta_state_isolation_unpadded_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
            v_decay.contiguous(), k_chunk.contiguous(), init.contiguous(), out, debug, int(pred_mode), bool(extra_barriers), float(scale)
        )
    return out, debug


def delta_state_isolation_all_shared_top(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=4, scale=scale)[0]


def delta_state_isolation_no_dead_store(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=4, scale=scale)[0]


def delta_state_isolation_noop_no_mfma(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=1, scale=scale)[0]


def delta_state_isolation_consume_acc(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=5, scale=scale)


def delta_state_isolation_one_mfma(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=2, scale=scale)[0]


def delta_state_isolation_two_mfma(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=3, scale=scale)[0]


def delta_state_isolation_extra_barriers(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=4, extra_barriers=True, scale=scale)[0]


def delta_state_isolation_padded_shared(v_decay, k_chunk, init, scale: float = 1.0):
    return _run_isolation_kernel(v_decay, k_chunk, init, pred_mode=4, padded=True, scale=scale)


def run_isolation_experiments(seed: int = 20260616) -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        raise RuntimeError('requires CUDA/HIP GPU')
    gen = torch.Generator(device='cuda').manual_seed(seed)
    v = torch.randn((BT, BV), device='cuda', dtype=torch.bfloat16, generator=gen)
    k = torch.randn((BT, 128), device='cuda', dtype=torch.bfloat16, generator=gen)
    init = torch.randn((BV, 128), device='cuda', dtype=torch.float32, generator=gen) * 0.01
    scale = 0.73
    expected = init * scale + v.float().T @ k.float()

    cases = []
    cases.append(('update_only_baseline', 'pass', delta_state_staged(v, k, init, scale), None, 'baseline update-only MFMA'))
    cases.append(('dummy_footprint', 'pass', delta_state_staged_with_dummy(v, k, init, scale), None, 'rules out static shared footprint alone'))
    cases.append(('pred_restaged_existing', 'fail', delta_state_staged_with_pred_restaged(v, k, init, scale), None, 'existing pred MFMA + restage + update reproducer'))
    cases.append(('all_shared_declared_at_top', 'pass-if-shared-lifetime-bug', delta_state_isolation_all_shared_top(v, k, init, scale), None, 'all shared declared at top, separate update buffers'))
    cases.append(('no_dead_store_before_pred', 'pass-if-dead-store-bug', delta_state_isolation_no_dead_store(v, k, init, scale), None, 'no pred-before update-buffer dead stores'))
    cases.append(('no_op_pred_no_mfma', 'pass-if-MFMA-trigger', delta_state_isolation_noop_no_mfma(v, k, init, scale), None, 'pred staging and loop without pred MFMA'))
    out, dbg = delta_state_isolation_consume_acc(v, k, init, scale)
    cases.append(('pred_mfma_consume_acc', 'pass-if-dead-acc-bug', out, dbg, 'pred accumulator written to debug global'))
    cases.append(('pred_one_mfma_only', 'diagnostic', delta_state_isolation_one_mfma(v, k, init, scale), None, 'only one pred MFMA before update'))
    cases.append(('pred_two_mfma_only', 'diagnostic', delta_state_isolation_two_mfma(v, k, init, scale), None, 'only two pred MFMAs before update'))
    cases.append(('extra_barriers_dummy_lds_reads', 'pass-if-waitcnt-barrier-bug', delta_state_isolation_extra_barriers(v, k, init, scale), None, 'extra barriers and dummy LDS reads'))
    out, dbg = delta_state_isolation_padded_shared(v, k, init, scale)
    cases.append(('padded_unique_shared_buffers_canary', 'pass-if-layout-overlap-bug', out, dbg, 'padded shared buffers and canary snapshots'))

    results = []
    for name, expected_status, actual, debug, conclusion_hint in cases:
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
        passed = bool(torch.allclose(actual.float(), expected.float(), atol=1e-4, rtol=1e-4))
        print(f'test,{name}')
        print(f'expected,{expected_status}')
        print(f'actual_max_abs,{max_abs:.9g}')
        print(f'actual_max_rel,{max_rel:.9g}')
        tile_values = []
        for base in range(0, 128, 16):
            tile_abs = diff[:, base:base+16].max().item()
            tile_values.append(tile_abs)
            print(f'per_tile,{base}:{base+16},{tile_abs:.9g}')
        if debug is not None and name == 'padded_unique_shared_buffers_canary':
            before = debug[:96].detach().cpu()
            after = debug[128:224].detach().cpu()
            expected_canary = torch.tensor([1.25, 2.25, 3.25, 4.25, 5.25, 6.25] * 16, dtype=torch.float32)
            before_err = (before - expected_canary).abs().max().item()
            after_err = (after - expected_canary).abs().max().item()
            print(f'canary_before_max_abs,{before_err:.9g}')
            print(f'canary_after_max_abs,{after_err:.9g}')
        elif debug is not None:
            dbg_max = debug.detach().float().abs().nan_to_num().max().item()
            print(f'debug_max_abs,{dbg_max:.9g}')
        print(f'status,{"pass" if passed else "fail"}')
        print(f'conclusion,{conclusion_hint}')
        results.append({
            'name': name,
            'expected': expected_status,
            'max_abs': max_abs,
            'max_rel': max_rel,
            'tiles': tile_values,
            'passed': passed,
            'conclusion': conclusion_hint,
        })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260616)
    parser.add_argument('--isolation-experiments', action='store_true')
    args = parser.parse_args()
    if args.isolation_experiments:
        run_isolation_experiments(args.seed)
    else:
        run(args.seed)

#!/usr/bin/env python3
"""Minimal repro for the old AveLang FullOp/Pure sequential-MFMA bug.

This file is intentionally Qwen-free.  It isolates the compiler issue where two
independent `al.full((4,), 0.0, al.f32)` accumulator initializers were CSE-merged
when `FullOp` was incorrectly marked `Pure`.

Expected behavior:
    update_only: pass
    no_op_pred_no_mfma: pass
    pred_one_mfma_then_update: pass after the FullOp fix, fail before the fix

To reproduce the bug on an old compiler, run this file on a revision where:

    def FullOp : AveLang_Op<"full", [Pure]>

If you only rebuild into `build-vllm-rocm722`, run with that binding first in
`PYTHONPATH`, otherwise Python may keep loading `python/_avelang_bindings*.so`:

    PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python \
    HIP_LAUNCH_BLOCKING=1 python test/examples/linear_attention/compile_bug/repro_mfma_fullop_pure_cse.py

On the fixed compiler, where `FullOp` has no `Pure` trait, all cases should pass.
"""

from __future__ import annotations

import argparse

import torch

import avelang
import avelang.language as al

BT = 16
BV = 16
KDIM = 128

MODE_UPDATE_ONLY = 0
MODE_NO_OP_PRED_NO_MFMA = 1
MODE_PRED_ONE_MFMA_THEN_UPDATE = 2


@avelang.jit
def _mfma_fullop_pure_cse_repro_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    init_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    pred_sink_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    pred_mode: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((BT, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((BT, KDIM), (KDIM, 1)))
    init = al.make_tensor(init_ptr, al.f32, al.make_layout((BV, KDIM), (KDIM, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((BV, KDIM), (KDIM, 1)))
    pred_sink = al.make_tensor(pred_sink_ptr, al.f32, al.make_layout((4,), (1,)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, KDIM), al.f32)
    pred_a = al.make_shared((BT, 16), al.bf16)
    pred_b = al.make_shared((BV, 16), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((KDIM, BT), al.bf16)

    pred_a_vec = al.view(pred_a, al.i32, al.make_layout((BT, 2, 4), (8, 4, 1)))
    pred_b_vec = al.view(pred_b, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((KDIM, 2, 4), (8, 4, 1)))

    if lane < 4:
        pred_sink[lane] = 0.0

    for rep_state in al.range(32):
        idx = lane + rep_state * 64
        row = idx // KDIM
        col = idx - row * KDIM
        state[row, col] = al.convert(init[row, col] * scale, al.f32)

    for rep_stage in al.range(32):
        idx = lane + rep_stage * 64
        row = idx // 16
        tok = idx - row * 16
        k_all_t[row, tok] = k[tok, row]
        if row < BV:
            v_decay_t[row, tok] = v[tok, row]
        if row < BT:
            pred_a[row, tok] = k[row, tok]
            pred_b[row, tok] = v[row, tok]

    al.syncthreads()

    if pred_mode == MODE_NO_OP_PRED_NO_MFMA:
        # Keep the pred staging/control-flow shape, but intentionally do not
        # emit a pred MFMA.  This passed both before and after the fix.
        pass

    if pred_mode == MODE_PRED_ONE_MFMA_THEN_UPDATE:
        pred_acc = al.full((4,), 0.0, al.f32)
        if lane_group == 0:
            a_words = pred_a_vec[lane_col, 0]
            b_words = pred_b_vec[lane_col, 0]
            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
            pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        if lane == 0:
            for r in al.range(4):
                pred_sink[r] = pred_acc[r]

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

    for rep_out in al.range(32):
        idx = lane + rep_out * 64
        row = idx // KDIM
        col = idx - row * KDIM
        out[row, col] = state[row, col]


def _run_kernel(v_decay: torch.Tensor, k_chunk: torch.Tensor, init: torch.Tensor, scale: float, pred_mode: int) -> torch.Tensor:
    out = torch.empty((BV, KDIM), device=v_decay.device, dtype=torch.float32)
    pred_sink = torch.empty((4,), device=v_decay.device, dtype=torch.float32)
    _mfma_fullop_pure_cse_repro_kernel[lambda: ((1, 1, 1), (64, 1, 1))](
        v_decay.contiguous(),
        k_chunk.contiguous(),
        init.contiguous(),
        out,
        pred_sink,
        float(scale),
        pred_mode,
    )
    return out


def _case_name(pred_mode: int) -> str:
    if pred_mode == MODE_UPDATE_ONLY:
        return "update_only"
    if pred_mode == MODE_NO_OP_PRED_NO_MFMA:
        return "no_op_pred_no_mfma"
    if pred_mode == MODE_PRED_ONE_MFMA_THEN_UPDATE:
        return "pred_one_mfma_then_update"
    return f"unknown_{pred_mode}"


def run(seed: int, scale: float, atol: float, rtol: float) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")

    torch.manual_seed(seed)
    v_decay = torch.randn((BT, BV), device="cuda", dtype=torch.bfloat16)
    k_chunk = torch.randn((BT, KDIM), device="cuda", dtype=torch.bfloat16)
    init = torch.randn((BV, KDIM), device="cuda", dtype=torch.float32) * 0.01
    expected = init * scale + v_decay.float().T @ k_chunk.float()

    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("expected_before_fix=pred_one_mfma_then_update fails when FullOp has Pure")
    print("expected_after_fix=all cases pass when FullOp has no Pure trait")

    all_ok = True
    for pred_mode in [MODE_UPDATE_ONLY, MODE_NO_OP_PRED_NO_MFMA, MODE_PRED_ONE_MFMA_THEN_UPDATE]:
        actual = _run_kernel(v_decay, k_chunk, init, scale, pred_mode)
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
        ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
        all_ok = all_ok and bool(ok)
        print(f"case={_case_name(pred_mode)},ok={bool(ok)},max_abs={max_abs:.9g},max_rel={max_rel:.9g}")

    if not all_ok:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--scale", type=float, default=0.73)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    run(args.seed, args.scale, args.atol, args.rtol)


if __name__ == "__main__":
    main()

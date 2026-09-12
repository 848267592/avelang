"""Isolated BT64-hierarchical active-V16 pred primitives.

P32 reproduces D1's padded MFMA32 pred path. P16 computes the same logical
``W[16,128] @ state[16,128].T`` directly with MFMA16 and two K64 partials.
Neither path contains recurrence/update logic.
"""

from __future__ import annotations

import argparse
import statistics

import torch

import avelang
import avelang.language as al


WG = 128


@avelang.jit
def _qwen_v31_pred_p32_kernel(
    w_ptr: al.Pointer(al.f32),
    state_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
):
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((16, 128), (128, 1)))
    state = al.make_tensor(state_ptr, al.f32, al.make_layout((16, 128), (128, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((16, 16), (16, 1)))
    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_mod32 = lane & 31

    # Duplicate each real row only to satisfy the 32x32 MFMA shape. Only the
    # logical [0:16,0:16] quadrant is consumed below.
    state_bf16 = al.make_shared((2, 32, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    serial = al.make_shared((2, 64, 16), al.f32)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))

    for rep in al.range(32):
        linear = tid + rep * WG
        khalf = linear // (32 * 64)
        rem = linear - khalf * (32 * 64)
        row = rem // 64
        col = rem - row * 64
        source_row = row & 15
        global_k = khalf * 64 + col
        state_bf16[khalf, row, col] = al.convert(state[source_row, global_k], al.bf16)
        w_bf16[khalf, row, col] = al.convert(w[source_row, global_k], al.bf16)

    al.syncthreads()
    acc = al.full((16,), 0.0, al.f32)
    for kpack in al.range(4):
        a_words = w_vec[wave_id, lane_mod32, kpack]
        b_words = state_vec[wave_id, lane_mod32, kpack]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], acc)
    for acc_i in al.range(16):
        serial[wave_id, lane, acc_i] = acc[acc_i]

    al.syncthreads()
    for rep_out in al.range(2):
        linear = tid + rep_out * WG
        token = linear // 16
        value = linear - token * 16
        token_hi = token // 8
        value_hi = value // 8
        value_low = value - value_hi * 8
        lane_for_pred = (value_low & 3) * 16 + ((value_low // 4) * 8) + (token & 7)
        acc_for_pred = value_hi * 4 + token_hi
        out[token, value] = serial[0, lane_for_pred, acc_for_pred] + serial[1, lane_for_pred, acc_for_pred]


@avelang.jit
def _qwen_v31_pred_p16_kernel(
    w_ptr: al.Pointer(al.f32),
    state_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
):
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((16, 128), (128, 1)))
    state = al.make_tensor(state_ptr, al.f32, al.make_layout((16, 128), (128, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((16, 16), (16, 1)))
    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4

    state_bf16 = al.make_shared((2, 16, 64), al.bf16)
    w_bf16 = al.make_shared((2, 16, 64), al.bf16)
    partial = al.make_shared((2, 16, 16), al.f32)
    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, 16, 8, 4), (16 * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 16, 8, 4), (16 * 8 * 4, 8 * 4, 4, 1)))

    for rep in al.range(16):
        linear = lane + rep * 64
        row = linear // 64
        col = linear - row * 64
        global_k = wave_id * 64 + col
        state_bf16[wave_id, row, col] = al.convert(state[row, global_k], al.bf16)
        w_bf16[wave_id, row, col] = al.convert(w[row, global_k], al.bf16)

    al.syncthreads()
    acc = al.full((4,), 0.0, al.f32)
    for seg32 in al.range(2):
        vec_idx = lane_group + seg32 * 4
        a_words = w_vec[wave_id, lane_col, vec_idx]
        b_words = state_vec[wave_id, lane_col, vec_idx]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
        acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)
    for r in al.range(4):
        partial[wave_id, lane_group * 4 + r, lane_col] = acc[r]

    al.syncthreads()
    for rep_out in al.range(2):
        linear = tid + rep_out * WG
        token = linear // 16
        value = linear - token * 16
        out[token, value] = partial[0, token, value] + partial[1, token, value]


def _validate(w: torch.Tensor, state: torch.Tensor) -> None:
    if w.shape != (16, 128) or state.shape != (16, 128):
        raise ValueError("P32/P16 require w/state shapes [16,128].")
    if w.dtype != torch.float32 or state.dtype != torch.float32:
        raise ValueError("P32/P16 require FP32 inputs before BF16 staging.")
    if not w.is_cuda or not state.is_cuda or not w.is_contiguous() or not state.is_contiguous():
        raise ValueError("P32/P16 require contiguous CUDA/HIP tensors.")


def pred_p32(w: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    _validate(w, state)
    out = torch.empty((16, 16), dtype=torch.float32, device=w.device)
    _qwen_v31_pred_p32_kernel[lambda: ((1, 1, 1), (WG, 1, 1))](w, state, out, num_warps=2)
    return out


def pred_p16(w: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    _validate(w, state)
    out = torch.empty((16, 16), dtype=torch.float32, device=w.device)
    _qwen_v31_pred_p16_kernel[lambda: ((1, 1, 1), (WG, 1, 1))](w, state, out, num_warps=2)
    return out


def pred_reference(w: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return w.to(torch.bfloat16).float() @ state.to(torch.bfloat16).float().t()


def _median_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--only", choices=("both", "p32", "p16"), default="both")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    w = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    state = (torch.randn((16, 128), device="cuda", dtype=torch.float32) * 0.02).contiguous()
    ref = pred_reference(w, state)
    p32 = pred_p32(w, state) if args.only in ("both", "p32") else None
    p16 = pred_p16(w, state) if args.only in ("both", "p16") else None
    torch.cuda.synchronize()
    for name, value in (("p32", p32), ("p16", p16)):
        if value is None:
            continue
        error = (value - ref).abs()
        first = (error > 1e-3).nonzero()
        print(f"{name}_max_abs={error.max().item():.8e} {name}_mean_abs={error.mean().item():.8e} first={first[0].tolist() if first.numel() else None}")
    if args.only in ("both", "p32"):
        print(f"p32_ms={_median_ms(lambda: pred_p32(w, state), args.warmup, args.repeat):.6f}")
    if args.only in ("both", "p16"):
        print(f"p16_ms={_median_ms(lambda: pred_p16(w, state), args.warmup, args.repeat):.6f}")


if __name__ == "__main__":
    main()

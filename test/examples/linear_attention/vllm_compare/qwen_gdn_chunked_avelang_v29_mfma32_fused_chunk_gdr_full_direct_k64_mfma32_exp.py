"""Experimental v29 direct-K64/MFMA32 recurrence with current-vLLM BF16 ABI.

This file is intentionally isolated from the production v23/v24 baselines.

The kernel preserves the current-ABI v29 pred/V-new path and replaces only
the compact-K MFMA16 update with direct K64 MFMA32 update operands:

* BT=64, BV=32, workgroup=128 threads / 2 waves.
* Pred uses the existing v29 K-split 32x32 MFMA schedule.
* Corrected values are staged as BF16 V-major data in LDS.
* Full global vn materialization is removed except for the required BF16
  current-vLLM ``v_new`` boundary.
* The update loads K directly as logical K[0:64, 64-token] and
  K[64:128, 64-token] blocks; it does not materialize ``k_all_t[128,64]``.

This is a diagnostic experiment, not a production recurrence. It retains the
same pred MFMA32 schedule, corrected-value math, launch, and current-vLLM
storage boundary as the overflowing compact-K current-ABI candidate. The only
algorithmic change is the update K operand/geometry.

* ``k/w/u`` are BF16 inputs;
* ``gdr_decay/gdr_g_last_exp`` and ``initial_state`` remain FP32;
* ``h`` and ``v_new`` use BF16 at the boundary;
* ``final_state`` remains FP32.

The public wrapper returns ``h``, ``v_new``, and ``final_state``.  The new
``v_new`` store is the one required ABI materialization; the compact schedule
continues to stage the same corrected values directly for its state update.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 128
NUM_VALUE_HEADS = 8
NUM_KEY_HEADS = 4
WORKGROUP = 128
GRID_SIZE = 32  # 4 V-blocks * 8 value heads
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)),
    )
    w = al.make_tensor(
        w_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u_flat = al.make_tensor(
        u_ptr,
        al.bf16,
        al.make_layout((num_tokens * 8 * 128,), (1,)),
    )
    gdr_decay = al.make_tensor(
        gdr_decay_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8, 64), (num_chunks * 8 * 64, 8 * 64, 64, 1)),
    )
    gdr_g_last_exp = al.make_tensor(
        gdr_g_last_exp_ptr,
        al.f32,
        al.make_layout((1, num_chunks, 8), (num_chunks * 8, 8, 1)),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    v_new_flat = al.make_tensor(
        v_new_ptr,
        al.bf16,
        al.make_layout((num_tokens * 8 * 128,), (1,)),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_mod32 = lane & 31
    lane_col = lane & 15
    lane_group = lane >> 4
    mfma32_lane_group = lane >> 5

    program_id = al.block_id(0)
    v_block_idx = program_id % 4
    value_head_idx = program_id // 4
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    v_decay_t_bf16 = al.make_shared((BV, BT), al.bf16)
    # Immediate direct-K operands only. These replace the broad
    # k_all_t[128,64] producer and its MFMA16 consumer view.
    # Keep the exact packed view rank used by the verified direct-K64 probe.
    # The A values are intentionally duplicated for the two K-owning waves.
    update_a_stage = al.make_shared((2, BV, 32), al.bf16)
    update_b_stage = al.make_shared((2, 32, 32), al.bf16)

    state_vec = al.view(
        state_bf16,
        al.i32,
        al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)),
    )
    w_vec = al.view(
        w_bf16,
        al.i32,
        al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)),
    )
    update_a_vec = al.view(
        update_a_stage,
        al.i32,
        al.make_layout((2, BV, 4, 4), (BV * 4 * 4, 4 * 4, 4, 1)),
    )
    update_b_vec = al.view(
        update_b_stage,
        al.i32,
        al.make_layout((2, 32, 4, 4), (32 * 4 * 4, 4 * 4, 4, 1)),
    )

    for rep_init in al.range(32):
        linear_init = tid + rep_init * WORKGROUP
        vv_init = linear_init // 128
        kk_init = linear_init - vv_init * 128
        global_v_init = value_base + vv_init
        if has_initial_state:
            state[vv_init, kk_init] = initial_state[0, value_head_idx, global_v_init, kk_init]
        else:
            state[vv_init, kk_init] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last_exp = gdr_g_last_exp[0, chunk_idx, value_head_idx]

        # h stores the chunk-start state, matching the chunk_gdr contract.
        for rep_h in al.range(32):
            linear_h = tid + rep_h * WORKGROUP
            vv_h = linear_h // 128
            kk_h = linear_h - vv_h * 128
            global_v_h = value_base + vv_h
            h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = al.convert(state[vv_h, kk_h], al.bf16)

        # Stage the old chunk-start state once. Both token tiles use it for
        # pred; state is only updated after all 64 tokens are corrected.
        for rep_state in al.range(32):
            linear_s = tid + rep_state * WORKGROUP
            kb_s = linear_s // (BV * 64)
            rem_s = linear_s - kb_s * (BV * 64)
            row_s = rem_s // 64
            col_s = rem_s - row_s * 64
            state_bf16[kb_s, row_s, col_s] = al.convert(state[row_s, kb_s * 64 + col_s], al.bf16)

        al.syncthreads()

        # Two 32-token pred tiles fill V-major corrected values for BT64.
        for token_tile in al.range(2):
            token_base = token_tile * 32

            for rep_w in al.range(32):
                linear_w = tid + rep_w * WORKGROUP
                kb_w = linear_w // (32 * 64)
                rem_w = linear_w - kb_w * (32 * 64)
                token_off_w = rem_w // 64
                col_w = rem_w - token_off_w * 64
                token_idx_w = chunk_start + token_base + token_off_w
                global_k_w = kb_w * 64 + col_w
                w_bf16[kb_w, token_off_w, col_w] = w[0, token_idx_w, value_head_idx, global_k_w]

            al.syncthreads()

            pred_acc = al.full((16,), 0.0, al.f32)
            a_row = lane_mod32
            b_row = lane_mod32

            for kpack in al.range(4):
                a_words = w_vec[wave_id, a_row, kpack]
                b_words = state_vec[wave_id, b_row, kpack]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

            row_base = lane_col & 7
            col_base = ((lane_col >> 3) * 4) + lane_group
            for acc_i in al.range(16):
                out_row = ((acc_i & 3) * 8) + row_base
                out_col = ((acc_i >> 2) * 8) + col_base
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]

            al.syncthreads()

            for rep_corr in al.range(8):
                linear_corr = tid + rep_corr * WORKGROUP
                token_off = linear_corr // BV
                local_v = linear_corr - token_off * BV
                token_idx = chunk_start + token_base + token_off
                out_offset = token_idx * (8 * 128) + value_head_idx * 128 + value_base + local_v
                pred_value = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
                corrected = al.convert(u_flat[out_offset], al.f32) - pred_value
                v_new_flat[out_offset] = al.convert(corrected, al.bf16)
                decay = gdr_decay[0, chunk_idx, value_head_idx, token_base + token_off]
                v_decay_t_bf16[local_v, token_base + token_off] = al.convert(corrected * decay, al.bf16)

            al.syncthreads()

        # One wave owns K[0:64], the other K[64:128]. For each direct K64
        # block, two K32 output tiles and two token32 reductions reconstruct
        # the same V32 x K128 state update as the compact MFMA16 path.
        for col_half in al.range(2):
            update_acc32 = al.full((16,), 0.0, al.f32)
            for token_half in al.range(2):
                for rep_a in al.range(16):
                    linear_a = tid + rep_a * WORKGROUP
                    a_wave = linear_a // (BV * 32)
                    a_rem = linear_a - a_wave * (BV * 32)
                    a_row = a_rem // 32
                    a_token = a_rem - a_row * 32
                    update_a_stage[a_wave, a_row, a_token] = v_decay_t_bf16[
                        a_row, token_half * 32 + a_token
                    ]

                for rep_b in al.range(16):
                    linear_b = tid + rep_b * WORKGROUP
                    b_wave = linear_b // (32 * 32)
                    b_rem = linear_b - b_wave * (32 * 32)
                    b_row = b_rem // 32
                    b_token = b_rem - b_row * 32
                    update_b_stage[b_wave, b_row, b_token] = k[
                        0,
                        chunk_start + token_half * 32 + b_token,
                        key_head_idx,
                        b_wave * 64 + col_half * 32 + b_row,
                    ]

                al.syncthreads()

                for k_pack in al.range(2):
                    word = k_pack * 2 + mfma32_lane_group
                    a_words = update_a_vec[wave_id, lane_mod32, word]
                    b_words = update_b_vec[wave_id, lane_mod32, word]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                    update_acc32 = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], update_acc32)
                    update_acc32 = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], update_acc32)

                al.syncthreads()

            for acc_i in al.range(16):
                out_v = lane_mod32
                out_k = ((acc_i >> 2) * 8) + mfma32_lane_group * 4 + (acc_i & 3)
                state_k = wave_id * 64 + col_half * 32 + out_k
                state[out_v, state_k] = state[out_v, state_k] * g_last_exp + update_acc32[acc_i]

        # The next chunk snapshots and reuses state across both waves.
        al.syncthreads()

    for rep_final in al.range(32):
        linear_final = tid + rep_final * WORKGROUP
        vv_final = linear_final // 128
        kk_final = linear_final - vv_final * 128
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be on CUDA/HIP device.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("v29 fused chunk_gdr only supports chunk_size=64.")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"k must be torch.bfloat16, got {k.dtype}.")
    if w.dtype != torch.bfloat16 or u.dtype != torch.bfloat16:
        raise ValueError("current-ABI direct-K64 MFMA32 requires BF16 w/u.")
    if gdr_decay.dtype != torch.float32 or gdr_g_last_exp.dtype != torch.float32:
        raise ValueError("gdr_decay/gdr_g_last_exp must be torch.float32.")
    for name, tensor in (("k", k), ("w", w), ("u", u), ("gdr_decay", gdr_decay), ("gdr_g_last_exp", gdr_g_last_exp)):
        _require_cuda_contiguous(name, tensor)
    if tuple(k.shape) != (1, k.shape[1], 4, 128):
        raise ValueError(f"k must have shape [1,T,4,128], got {tuple(k.shape)}.")
    num_tokens = k.shape[1]
    if tuple(w.shape) != (1, num_tokens, 8, 128):
        raise ValueError(f"w must have shape [1,{num_tokens},8,128], got {tuple(w.shape)}.")
    if tuple(u.shape) != (1, num_tokens, 8, 128):
        raise ValueError(f"u must have shape [1,{num_tokens},8,128], got {tuple(u.shape)}.")
    if num_tokens % BT != 0:
        raise ValueError("v29 fused chunk_gdr requires T divisible by 64.")
    num_chunks = num_tokens // BT
    if tuple(gdr_decay.shape) != (1, num_chunks, 8, 64):
        raise ValueError(f"gdr_decay must have shape [1,{num_chunks},8,64], got {tuple(gdr_decay.shape)}.")
    if tuple(gdr_g_last_exp.shape) != (1, num_chunks, 8):
        raise ValueError(f"gdr_g_last_exp must have shape [1,{num_chunks},8], got {tuple(gdr_g_last_exp.shape)}.")
    if initial_state is not None:
        _require_cuda_contiguous("initial_state", initial_state)
        if initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state must be torch.float32, got {initial_state.dtype}.")
        if tuple(initial_state.shape) != (1, 8, 128, 128):
            raise ValueError(f"initial_state must have shape [1,8,128,128], got {tuple(initial_state.shape)}.")
        if initial_state.device != k.device:
            raise ValueError("initial_state must be on the same device as k.")
    return num_tokens, num_chunks


def qwen_gdn_gdr_decay_bt64_reference(g: torch.Tensor, *, chunk_size: int = BT) -> tuple[torch.Tensor, torch.Tensor]:
    if chunk_size != BT:
        raise ValueError("BT64 reference decay only supports chunk_size=64.")
    if tuple(g.shape) != (1, g.shape[1], 8):
        raise ValueError(f"g must have shape [1,T,8], got {tuple(g.shape)}.")
    if g.dtype != torch.float32:
        raise ValueError(f"g must be torch.float32, got {g.dtype}.")
    if g.shape[1] % BT != 0:
        raise ValueError("T must be divisible by 64.")
    num_chunks = g.shape[1] // BT
    g_chunks = g.view(1, num_chunks, BT, 8).transpose(2, 3).contiguous()
    g_last = g_chunks[:, :, :, -1]
    return torch.exp(g_last.unsqueeze(-1) - g_chunks).contiguous(), torch.exp(g_last).contiguous()


def qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_avelang_v29(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_inputs(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, chunk_size)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32[
            lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))
        ](
            k,
            w,
            u,
            gdr_decay,
            gdr_g_last_exp,
            initial_state,
            h,
            v_new,
            final_state,
            num_tokens,
            num_chunks,
            has_initial_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch, "_qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32")
    launch()
    return h, v_new, final_state


def qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_inputs(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, chunk_size)
    state = torch.zeros((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state is not None:
        state.copy_(initial_state)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BT
        h[:, chunk_idx].copy_(state.to(torch.bfloat16))
        for vh in range(8):
            kh = vh // 2
            state_bf16 = state[0, vh].to(torch.bfloat16).float()
            w_bf16 = w[0, start : start + BT, vh].float()
            pred = w_bf16 @ state_bf16.t()
            corrected = u[0, start : start + BT, vh].float() - pred
            v_new[0, start : start + BT, vh] = corrected.to(torch.bfloat16)
            v_decay = (corrected * gdr_decay[0, chunk_idx, vh].unsqueeze(-1)).t().to(torch.bfloat16).float()
            k_chunk = k[0, start : start + BT, kh].float()
            state[0, vh] = state[0, vh] * gdr_g_last_exp[0, chunk_idx, vh] + v_decay @ k_chunk
    return h, v_new, state


def _benchmark_cuda(fn: Callable[[], None], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_substr: str) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    existing = list(dump_dir.glob(f"{kernel_substr}*.hsaco"))
    if existing:
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if not dumped and kernel_substr in kernel_name:
            path = dump_dir / f"{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped = True
            print(f"dumped_hsaco: {path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no compiled kernel matched {kernel_substr!r}")


def _make_inputs(T: int, *, device: str = "cuda", seed: int = 0):
    torch.manual_seed(seed)
    k = torch.randn((1, T, 4, 128), device=device, dtype=torch.bfloat16).contiguous()
    w = (torch.randn((1, T, 8, 128), device=device, dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    u = torch.randn((1, T, 8, 128), device=device, dtype=torch.float32).to(torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn((1, T, 8), device=device, dtype=torch.float32)) / 16.0).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.02).contiguous()
    gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_bt64_reference(g)
    return k, w, u, gdr_decay, gdr_g_last_exp, initial_state


def _smoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check-ref", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()

    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir

    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = _make_inputs(args.T)
    h, v_new, final_state = qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_avelang_v29(
        k,
        w,
        u,
        gdr_decay,
        gdr_g_last_exp,
        initial_state,
    )
    torch.cuda.synchronize()
    if not args.no_check_ref:
        h_ref, v_new_ref, final_ref = qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_reference(
            k, w, u, gdr_decay, gdr_g_last_exp, initial_state
        )
        h_err = (h - h_ref).abs()
        v_new_err = (v_new - v_new_ref).abs()
        fs_err = (final_state - final_ref).abs()
        print(f"h_max_abs={h_err.max().item():.8e}")
        print(f"h_mean_abs={h_err.mean().item():.8e}")
        print(f"v_new_max_abs={v_new_err.max().item():.8e}")
        print(f"v_new_mean_abs={v_new_err.mean().item():.8e}")
        print(f"final_state_max_abs={fs_err.max().item():.8e}")
        print(f"final_state_mean_abs={fs_err.mean().item():.8e}")
    latency = _benchmark_cuda(
        lambda: qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_avelang_v29(
            k, w, u, gdr_decay, gdr_g_last_exp, initial_state
        ),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"T={args.T} v29_direct_k64_mfma32_chunk_gdr_ms={latency:.6f}")


if __name__ == "__main__":
    _smoke_main()


__all__ = [
    "BT",
    "BV",
    "KDIM",
    "WORKGROUP",
    "GRID_SIZE",
    "_qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_exp_bf16_kernel_v29_mfma32",
    "qwen_gdn_gdr_decay_bt64_reference",
    "qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_avelang_v29",
    "qwen_gdn_fused_chunk_gdr_full_direct_k64_mfma32_reference",
]

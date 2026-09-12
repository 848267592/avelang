"""Qwen GDN v30 hierarchical BT64/BV32 chunk_gdr experiment.

This file is intentionally isolated from the production v23/v24 baselines.

The kernel is a scheduling experiment for the v30 MFMA32 path:

* BT=64, BV=32, workgroup=128 threads / 2 waves.
* Pred uses the existing v30 K-split 32x32 MFMA schedule.
* Corrected values are staged as BF16 V-major data in LDS.
* Full global vn materialization is removed.
* State update is computed inside the same kernel.

The public wrapper returns ``h`` and ``final_state``.  It intentionally does
not return full ``vn`` because avoiding that global buffer is the point of the
candidate.

The w=0 update recurrence is validated. The nonzero-W pred microtile mapping
is intentionally retained as a visible experimental failure; this file is not
a production or reference-correct implementation.
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
def _qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v30_bt64_bv32_hierarchical(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
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
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
    )
    u_flat = al.make_tensor(
        u_ptr,
        al.f32,
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
        al.f32,
        al.make_layout(
            (1, num_chunks, 8, 128, 128),
            (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1),
        ),
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

    program_id = al.block_id(0)
    v_block_idx = program_id % 4
    value_head_idx = program_id // 4
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    # The macro recurrence remains BT64/BV32.  These buffers deliberately
    # describe only the active V16/token16/K16 microtile.  MFMA32 requires
    # 32 rows, so pred operands duplicate the active 16 rows; correction
    # consumes only the first 16x16 result quadrant.
    state_bf16 = al.make_shared((2, 32, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_acc_serial = al.make_shared((2, 64, 16), al.f32)
    v_decay_t_bf16 = al.make_shared((16, 16), al.bf16)
    k_tile_t = al.make_shared((16, 16), al.bf16)

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
    vdecay_vec = al.view(v_decay_t_bf16, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))
    ktile_vec = al.view(k_tile_t, al.i32, al.make_layout((16, 2, 4), (8, 4, 1)))

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
            h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]

        # Complete one V16 block before beginning the other V16 block. The
        # duplicated rows are an MFMA32 shape adapter, not a second live V
        # tile: only output rows/columns [0:16, 0:16] are consumed.
        for v_tile in al.range(2):
            for rep_state in al.range(32):
                linear_s = tid + rep_state * WORKGROUP
                kb_s = linear_s // (32 * 64)
                rem_s = linear_s - kb_s * (32 * 64)
                row_s = rem_s // 64
                col_s = rem_s - row_s * 64
                source_v_s = v_tile * 16 + (row_s & 15)
                state_bf16[kb_s, row_s, col_s] = al.convert(state[source_v_s, kb_s * 64 + col_s], al.bf16)

            al.syncthreads()

            # Each BT64 recurrence chunk is still one recurrence. Token16 is
            # only the active pred/update microtile, and state_bf16 retains the
            # chunk-start V16 state until all four token tiles have consumed it.
            for token_tile in al.range(4):
                token_base = token_tile * 16

                for rep_w in al.range(32):
                    linear_w = tid + rep_w * WORKGROUP
                    kb_w = linear_w // (32 * 64)
                    rem_w = linear_w - kb_w * (32 * 64)
                    row_w = rem_w // 64
                    col_w = rem_w - row_w * 64
                    token_idx_w = chunk_start + token_base + (row_w & 15)
                    global_k_w = kb_w * 64 + col_w
                    w_bf16[kb_w, row_w, col_w] = al.convert(w[0, token_idx_w, value_head_idx, global_k_w], al.bf16)

                al.syncthreads()

                pred_acc = al.full((16,), 0.0, al.f32)
                for kpack in al.range(4):
                    a_words = w_vec[wave_id, lane_mod32, kpack]
                    b_words = state_vec[wave_id, lane_mod32, kpack]
                    a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                    b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
                    pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

                # Lane-major serialization avoids a permuted [32,32] pred
                # tile. The inverse mapping below reloads only the active
                # token16/V16 quadrant after pred_acc has been stored.
                for acc_i in al.range(16):
                    pred_acc_serial[wave_id, lane, acc_i] = pred_acc[acc_i]

                al.syncthreads()

                for rep_corr in al.range(2):
                    linear_corr = tid + rep_corr * WORKGROUP
                    token_off = linear_corr // 16
                    local_v = linear_corr - token_off * 16
                    row_group = token_off // 8
                    col_group = local_v // 8
                    acc_i = col_group * 4 + row_group
                    source_lane_col = (token_off - row_group * 8) + ((local_v - col_group * 8) // 4) * 8
                    source_lane_group = local_v - (local_v // 4) * 4
                    source_lane = source_lane_group * 16 + source_lane_col
                    token_idx = chunk_start + token_base + token_off
                    out_offset = token_idx * (8 * 128) + value_head_idx * 128 + value_base + v_tile * 16 + local_v
                    pred_value = pred_acc_serial[0, source_lane, acc_i] + pred_acc_serial[1, source_lane, acc_i]
                    corrected = u_flat[out_offset] - pred_value
                    decay = gdr_decay[0, chunk_idx, value_head_idx, token_base + token_off]
                    v_decay_t_bf16[local_v, token_off] = al.convert(corrected * decay, al.bf16)

                al.syncthreads()

                # K16 is staged at its single immediate consumer. Only wave0
                # owns the V16xK16 update accumulator; wave1 participates in
                # pred and synchronization but carries no second update tile.
                for k_tile in al.range(8):
                    for rep_k in al.range(2):
                        linear_k = tid + rep_k * WORKGROUP
                        local_k = linear_k // 16
                        local_t = linear_k - local_k * 16
                        k_tile_t[local_k, local_t] = k[0, chunk_start + token_base + local_t, key_head_idx, k_tile * 16 + local_k]

                    al.syncthreads()

                    update_acc16 = al.full((4,), 0.0, al.f32)
                    if wave_id == 0:
                        if lane_group == 0:
                            a_words_u16 = vdecay_vec[lane_col, 0]
                            b_words_u16 = ktile_vec[lane_col, 0]
                            a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[0], b_frag_u16[0], update_acc16)
                        if lane_group == 1:
                            a_words_u16 = vdecay_vec[lane_col, 0]
                            b_words_u16 = ktile_vec[lane_col, 0]
                            a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[1], b_frag_u16[1], update_acc16)
                        if lane_group == 2:
                            a_words_u16 = vdecay_vec[lane_col, 1]
                            b_words_u16 = ktile_vec[lane_col, 1]
                            a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[0], b_frag_u16[0], update_acc16)
                        if lane_group == 3:
                            a_words_u16 = vdecay_vec[lane_col, 1]
                            b_words_u16 = ktile_vec[lane_col, 1]
                            a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                            update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[1], b_frag_u16[1], update_acc16)

                        out_col_u = k_tile * 16 + lane_col
                        for r_up in al.range(4):
                            out_v_u = v_tile * 16 + lane_group * 4 + r_up
                            if token_tile == 0:
                                state[out_v_u, out_col_u] = state[out_v_u, out_col_u] * g_last_exp + update_acc16[r_up]
                            else:
                                state[out_v_u, out_col_u] = state[out_v_u, out_col_u] + update_acc16[r_up]

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
        raise ValueError("v30 hierarchical chunk_gdr only supports chunk_size=64.")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"k must be torch.bfloat16, got {k.dtype}.")
    if w.dtype != torch.float32 or u.dtype != torch.float32:
        raise ValueError("w/u must be torch.float32.")
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
        raise ValueError("v30 hierarchical chunk_gdr requires T divisible by 64.")
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


def qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_inputs(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, chunk_size)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v30_bt64_bv32_hierarchical[
            lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))
        ](
            k,
            w,
            u,
            gdr_decay,
            gdr_g_last_exp,
            initial_state,
            h,
            final_state,
            num_tokens,
            num_chunks,
            has_initial_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch, "_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v30_bt64_bv32_hierarchical")
    launch()
    return h, final_state


def qwen_gdn_fused_chunk_gdr_full_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_chunks = _validate_inputs(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, chunk_size)
    state = torch.zeros((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state is not None:
        state.copy_(initial_state)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    for chunk_idx in range(num_chunks):
        start = chunk_idx * BT
        h[:, chunk_idx].copy_(state)
        for vh in range(8):
            kh = vh // 2
            state_bf16 = state[0, vh].to(torch.bfloat16).float()
            w_bf16 = w[0, start : start + BT, vh].to(torch.bfloat16).float()
            pred = w_bf16 @ state_bf16.t()
            corrected = u[0, start : start + BT, vh].float() - pred
            v_decay = (corrected * gdr_decay[0, chunk_idx, vh].unsqueeze(-1)).t().to(torch.bfloat16).float()
            k_chunk = k[0, start : start + BT, kh].float()
            state[0, vh] = state[0, vh] * gdr_g_last_exp[0, chunk_idx, vh] + v_decay @ k_chunk
    return h, state


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
    w = (torch.randn((1, T, 8, 128), device=device, dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, T, 8, 128), device=device, dtype=torch.float32).contiguous()
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
    h, final_state = qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(
        k,
        w,
        u,
        gdr_decay,
        gdr_g_last_exp,
        initial_state,
    )
    torch.cuda.synchronize()
    if not args.no_check_ref:
        h_ref, final_ref = qwen_gdn_fused_chunk_gdr_full_reference(k, w, u, gdr_decay, gdr_g_last_exp, initial_state)
        h_err = (h - h_ref).abs()
        fs_err = (final_state - final_ref).abs()
        print(f"h_max_abs={h_err.max().item():.8e}")
        print(f"h_mean_abs={h_err.mean().item():.8e}")
        print(f"final_state_max_abs={fs_err.max().item():.8e}")
        print(f"final_state_mean_abs={fs_err.mean().item():.8e}")
    latency = _benchmark_cuda(
        lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical(k, w, u, gdr_decay, gdr_g_last_exp, initial_state),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"T={args.T} v30_bt64_bv32_hierarchical_ms={latency:.6f}")


if __name__ == "__main__":
    _smoke_main()


__all__ = [
    "BT",
    "BV",
    "KDIM",
    "WORKGROUP",
    "GRID_SIZE",
    "_qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v30_bt64_bv32_hierarchical",
    "qwen_gdn_gdr_decay_bt64_reference",
    "qwen_gdn_fused_chunk_gdr_full_avelang_v30_bt64_bv32_hierarchical",
    "qwen_gdn_fused_chunk_gdr_full_reference",
]

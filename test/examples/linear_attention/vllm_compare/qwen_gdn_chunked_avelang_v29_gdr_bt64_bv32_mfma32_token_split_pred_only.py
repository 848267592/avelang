"""Qwen GDN v29 pred-only token-split MFMA32 experiment.

This is a performance-oriented pred-only variant.

Main idea
---------
The previous v29_pred_only was k-split:

    wave0 computes K 0:64 partial
    wave1 computes K 64:128 partial
    pred = partial0 + partial1 through LDS

That version was correct and generated 32x32 MFMA, but it was slow because it
had expensive accumulator unpack + LDS partial reduction + high AccVGPR.

This variant is token-split instead:

    wave0 computes token 0:32, full K=128
    wave1 computes token 32:64, full K=128

Each wave computes a complete pred tile by accumulating both K halves into the
same 32x32 accumulator.  Therefore no cross-wave pred reduction is needed.

This is not the final full-v29 recurrence schedule.  It is a pred-only
performance upper-bound test for source-level 32x32 MFMA in Avelang.

Expected ISA
------------
    v_mfma_f32_32x32x8_bf16

Expected absent:
    v_mfma_f32_16x16x16_bf16
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 128
NUM_VALUE_HEADS = 8
WORKGROUP = 128
GRID_SIZE = 32  # 4 V-blocks * 8 value heads


@avelang.jit
def _qwen_gdn_pred_only_bf16_kernel_v29_token_split_mfma32(
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Token-split pred-only kernel.

    Shapes:
        w:             [1, T, 8, 128]   fp32
        u:             [1, T, 8, 128]   fp32
        initial_state: [1, 8, 128,128]  fp32, [B,H,V,K]
        vn:            [1, T, 8, 128]   fp32

    CTA mapping:
        program_id = value_head * 4 + v_block

    Wave mapping:
        wave0 -> token tile 0:32
        wave1 -> token tile 32:64

    Each wave computes:
        pred[32,32] = W[token_tile, 0:128] @ state[V-block, 0:128]^T
    """

    w = al.make_tensor(
        w_ptr,
        al.f32,
        al.make_layout(
            (1, num_tokens, 8, 128),
            (num_tokens * 8 * 128, 8 * 128, 128, 1),
        ),
    )
    u = al.make_tensor(
        u_ptr,
        al.f32,
        al.make_layout(
            (1, num_tokens, 8, 128),
            (num_tokens * 8 * 128, 8 * 128, 128, 1),
        ),
    )
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout(
            (1, 8, 128, 128),
            (8 * 128 * 128, 128 * 128, 128, 1),
        ),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout(
            (1, num_tokens, 8, 128),
            (num_tokens * 8 * 128, 8 * 128, 128, 1),
        ),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64        # 0 or 1
    lane = tid - wave_id * 64  # 0..63

    lane_mod32 = lane & 31
    lane_col = lane & 15
    lane_group = lane >> 4

    program_id = al.block_id(0)
    v_block_idx = program_id % 4
    value_head_idx = program_id // 4

    value_base = v_block_idx * BV

    # Shared state staging:
    #   state_bf16[khalf, local_v, local_k]
    # khalf=0 -> K 0:64
    # khalf=1 -> K 64:128
    #
    # This is still staging, not persistent recurrent state storage.
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)

    # Shared W staging:
    #   w_bf16[wave_token_tile, token_offset, local_k]
    #
    # This buffer is reused for khalf=0 and khalf=1.
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)

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

    # ------------------------------------------------------------
    # Stage nonzero initial_state once.
    # All 128 threads cooperatively load both K halves:
    #   total = 2 * 32 * 64 = 4096 bf16 elements
    #   128 threads * 32 reps = 4096
    # ------------------------------------------------------------
    for rep_state in al.range(32):
        linear_s = tid + rep_state * WORKGROUP
        khalf_s = linear_s // (BV * 64)
        rem_s = linear_s - khalf_s * (BV * 64)
        row_s = rem_s // 64
        col_s = rem_s - row_s * 64

        global_v_s = value_base + row_s
        global_k_s = khalf_s * 64 + col_s

        state_bf16[khalf_s, row_s, col_s] = al.convert(
            initial_state[0, value_head_idx, global_v_s, global_k_s],
            al.bf16,
        )

    al.syncthreads()

    # ------------------------------------------------------------
    # Main chunk loop.
    # ------------------------------------------------------------
    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT

        # wave0 owns token tile 0:32
        # wave1 owns token tile 32:64
        token_base_for_wave = wave_id * 32

        # One complete pred accumulator per wave:
        #   C[32 token rows, 32 value cols]
        pred_acc = al.full((16,), 0.0, al.f32)

        # Accumulate both K halves into the same pred_acc.
        for khalf in al.range(2):
            # ----------------------------------------------------
            # Stage W for both wave token tiles for this K half:
            # total = 2 * 32 * 64 = 4096 bf16 elements
            # ----------------------------------------------------
            for rep_w in al.range(32):
                linear_w = tid + rep_w * WORKGROUP
                tile_w = linear_w // (32 * 64)  # 0 or 1, token tile owner
                rem_w = linear_w - tile_w * (32 * 64)
                token_off_w = rem_w // 64
                col_w = rem_w - token_off_w * 64

                token_idx_w = chunk_start + tile_w * 32 + token_off_w
                global_k_w = khalf * 64 + col_w

                w_bf16[tile_w, token_off_w, col_w] = al.convert(
                    w[0, token_idx_w, value_head_idx, global_k_w],
                    al.bf16,
                )

            al.syncthreads()

            # ----------------------------------------------------
            # 32x32x8 MFMA for this K half.
            #
            # The mapping below is the same candidate mapping that already
            # passed correctness in the previous v29 pred-only experiment.
            # ----------------------------------------------------
            a_row = lane_mod32
            b_row = lane_mod32

            for kpack in al.range(4):
                a_words = w_vec[wave_id, a_row, kpack]
                b_words = state_vec[khalf, b_row, kpack]

                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    b_frag[0],
                    a_frag[0],
                    pred_acc,
                )
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(
                    b_frag[1],
                    a_frag[1],
                    pred_acc,
                )

            # Make sure no thread overwrites w_bf16 for next khalf before
            # all lanes have consumed the current staged W.
            al.syncthreads()

        # --------------------------------------------------------
        # Store vn directly.
        #
        # No pred_partial LDS.
        # No cross-wave reduction.
        # Each wave writes its own 32-token tile.
        # --------------------------------------------------------
        for acc_i in al.range(16):
            out_row = ((acc_i & 3) * 8) + (lane_col & 7)
            out_col = ((acc_i >> 2) * 8) + ((lane_col >> 3) * 4) + lane_group

            token_idx_o = chunk_start + token_base_for_wave + out_row
            global_v_o = value_base + out_col

            pred_value = pred_acc[acc_i]
            vn[0, token_idx_o, value_head_idx, global_v_o] = (
                u[0, token_idx_o, value_head_idx, global_v_o] - pred_value
            )


def _validate_inputs(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("token_split v29 pred-only only supports chunk_size=64.")

    if w.ndim != 4:
        raise ValueError(f"expected w shape [1,T,8,128], got {tuple(w.shape)}")
    if u.ndim != 4:
        raise ValueError(f"expected u shape [1,T,8,128], got {tuple(u.shape)}")

    if w.shape != u.shape:
        raise ValueError(f"w and u must have the same shape, got {tuple(w.shape)} vs {tuple(u.shape)}")

    batch, num_tokens, num_heads, head_dim = w.shape
    if (batch, num_heads, head_dim) != (1, NUM_VALUE_HEADS, KDIM):
        raise ValueError("only B=1,H=8,D=128 is supported.")

    if num_tokens % BT != 0:
        raise ValueError("T must be divisible by 64.")

    if w.dtype != torch.float32:
        raise ValueError(f"expected w float32, got {w.dtype}")
    if u.dtype != torch.float32:
        raise ValueError(f"expected u float32, got {u.dtype}")

    if initial_state is not None:
        if tuple(initial_state.shape) != (1, NUM_VALUE_HEADS, 128, 128):
            raise ValueError(
                "expected initial_state shape [1,8,128,128], "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise ValueError(f"expected initial_state float32, got {initial_state.dtype}")

    return num_tokens, num_tokens // BT


def qwen_gdn_pred_only_avelang_v29_token_split_mfma32(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    num_tokens, num_chunks = _validate_inputs(w, u, initial_state, chunk_size)

    if initial_state is None:
        initial_state = torch.zeros((1, NUM_VALUE_HEADS, 128, 128), device=w.device, dtype=torch.float32)

    vn = torch.empty_like(u)

    _qwen_gdn_pred_only_bf16_kernel_v29_token_split_mfma32[
        lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))
    ](
        w,
        u,
        initial_state,
        vn,
        num_tokens,
        num_chunks,
        num_warps=2,
    )

    return vn


def qwen_gdn_pred_only_torch_reference(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    pred = torch.einsum("bthk,bhvk->bthv", w.float(), initial_state.float())
    return u.float() - pred


def _make_inputs(
    T: int,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    # Keep magnitudes moderate because kernel converts W/state to BF16.
    w = (torch.randn((1, T, 8, 128), device=device, dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, T, 8, 128), device=device, dtype=torch.float32).contiguous()
    initial_state = (
        torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.02
    ).contiguous()

    return w, u, initial_state


def _benchmark_cuda(fn: Callable[[], None], *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    return statistics.median(times)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check-ref", action="store_true")
    args = parser.parse_args()

    w, u, initial_state = _make_inputs(args.T)

    vn = qwen_gdn_pred_only_avelang_v29_token_split_mfma32(w, u, initial_state)
    torch.cuda.synchronize()

    if not args.no_check_ref:
        ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)
        err = (vn - ref).abs()
        print(f"T={args.T}")
        print(f"max_abs={err.max().item():.8e}")
        print(f"mean_abs={err.mean().item():.8e}")

    latency = _benchmark_cuda(
        lambda: qwen_gdn_pred_only_avelang_v29_token_split_mfma32(w, u, initial_state),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"T={args.T} token_split_v29_pred_only_ms={latency:.6f}")


if __name__ == "__main__":
    _main()


__all__ = [
    "BT",
    "BV",
    "KDIM",
    "WORKGROUP",
    "GRID_SIZE",
    "_qwen_gdn_pred_only_bf16_kernel_v29_token_split_mfma32",
    "qwen_gdn_pred_only_avelang_v29_token_split_mfma32",
    "qwen_gdn_pred_only_torch_reference",
]

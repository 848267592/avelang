"""Qwen GDN v29 pred-only BT64/BV32 MFMA32 ISA-tuning experiment.

This is an experimental source-level schedule probe, not a production kernel.

Goal
----
Replace the v28 pred path's 4-wave / 16x16 MFMA structure with a Triton-like
2-wave / 32x32 MFMA structure:

    BT = 64
    BV = 32
    WORKGROUP = 128
    wave0 handles K 0:64
    wave1 handles K 64:128

For each token tile of 32 tokens, each wave computes one partial:

    partial0 = W[32, 64] @ state[V=32, K=0:64]^T
    partial1 = W[32, 64] @ state[V=32, K=64:128]^T

Then the two partials are reduced through LDS:

    pred = partial0 + partial1
    vn = u - pred

Important
---------
This file starts from the original k-split v29 pred-only kernel and adds
source-level ablation modes.  The goal is to isolate which part of the source
schedule causes high AccVGPR / VALU / SALU cost around 32x32 MFMA.

Expected ISA
------------
The kernel should contain:

    v_mfma_f32_32x32x8_bf16

and should not fall back to:

    v_mfma_f32_16x16x16_bf16

"""

from __future__ import annotations

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

MODE_BASELINE = "baseline"
MODE_NO_ACC_UNPACK = "no_acc_unpack"
MODE_UNPACK_ONLY_NO_REDUCE = "unpack_only_no_reduce"
MODE_REDUCE_ONLY_NO_VN_STORE = "reduce_only_no_vn_store"
MODE_CONSTANT_STATE = "constant_state"
MODE_PRECOMPUTED_MAPPING = "precomputed_mapping"
MODE_OPTIMIZED = "optimized"

_VARIANTS = {
    MODE_BASELINE: dict(
        do_acc_unpack=True,
        do_reduce=True,
        store_full_vn=True,
        constant_state=False,
        precomputed_mapping=False,
    ),
    MODE_NO_ACC_UNPACK: dict(
        do_acc_unpack=False,
        do_reduce=False,
        store_full_vn=False,
        constant_state=False,
        precomputed_mapping=False,
    ),
    MODE_UNPACK_ONLY_NO_REDUCE: dict(
        do_acc_unpack=True,
        do_reduce=False,
        store_full_vn=False,
        constant_state=False,
        precomputed_mapping=False,
    ),
    MODE_REDUCE_ONLY_NO_VN_STORE: dict(
        do_acc_unpack=True,
        do_reduce=True,
        store_full_vn=False,
        constant_state=False,
        precomputed_mapping=False,
    ),
    MODE_CONSTANT_STATE: dict(
        do_acc_unpack=True,
        do_reduce=True,
        store_full_vn=True,
        constant_state=True,
        precomputed_mapping=False,
    ),
    MODE_PRECOMPUTED_MAPPING: dict(
        do_acc_unpack=True,
        do_reduce=True,
        store_full_vn=True,
        constant_state=False,
        precomputed_mapping=True,
    ),
    # The first optimized candidate intentionally uses the lowest-risk source
    # rewrite: keep the k-split schedule and make the unpack mapping constant.
    MODE_OPTIMIZED: dict(
        do_acc_unpack=True,
        do_reduce=True,
        store_full_vn=True,
        constant_state=False,
        precomputed_mapping=True,
    ),
}


@avelang.jit
def _qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune(
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    do_acc_unpack: al.constexpr,
    do_reduce: al.constexpr,
    store_full_vn: al.constexpr,
    constant_state: al.constexpr,
    precomputed_mapping: al.constexpr,
):
    """Pred-only v29 MFMA32 kernel.

    Shapes:
        w:             [1, T, 8, 128]      fp32
        u:             [1, T, 8, 128]      fp32
        initial_state: [1, 8, 128, 128]    fp32, layout [B,H,V,K]
        vn:            [1, T, 8, 128]      fp32

    One CTA handles one (value_head, V-block) and loops over chunks.
    """

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
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.f32,
        al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)),
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
    k_block = wave_id
    k_base = k_block * 64

    # Shared staging.
    #
    # state_bf16[k_block, local_v, local_k]
    # w_bf16[k_block, token, local_k]
    # pred_partial[k_block, token, local_v]
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)

    # Packed i32 views.
    #
    # For a [32,64] BF16 tile:
    #   64 BF16 = 128 bytes = 32 i32 words
    #   represent as 8 chunks of 4 i32 words.
    #
    # However, for mfma32 K=64 we use 4 packed groups, each group provides
    # two 4-BF16 fragments. This gives 8 MFMA32 calls per K=64 half.
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

    # ------------------------------------------------------------------
    # Load initial_state once into BF16 shared state.
    #
    # This pred-only experiment uses a fixed non-updated state for all chunks.
    # Full v29 will later replace this with persistent accumulator state.
    # ------------------------------------------------------------------
    for rep_state in al.range(32):
        linear_s = tid + rep_state * WORKGROUP
        kb_s = linear_s // (BV * 64)
        rem_s = linear_s - kb_s * (BV * 64)
        row_s = rem_s // 64
        col_s = rem_s - row_s * 64

        global_v_s = value_base + row_s
        global_k_s = kb_s * 64 + col_s

        if constant_state:
            # Deterministic nonzero state, independent of global memory.
            state_bf16[kb_s, row_s, col_s] = al.convert(0.03125, al.bf16)
        else:
            state_bf16[kb_s, row_s, col_s] = al.convert(
                initial_state[0, value_head_idx, global_v_s, global_k_s],
                al.bf16,
            )

    al.syncthreads()

    # ------------------------------------------------------------------
    # Main chunk loop.
    # ------------------------------------------------------------------
    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT

        # Two 32-token tiles per BT64 chunk.
        for token_tile in al.range(2):
            token_base = token_tile * 32

            # ----------------------------------------------------------
            # Stage W[2 K-blocks, 32 tokens, 64 K] into BF16 shared.
            # All 128 threads cooperatively load both K halves.
            # ----------------------------------------------------------
            for rep_w in al.range(32):
                linear_w = tid + rep_w * WORKGROUP
                kb_w = linear_w // (32 * 64)
                rem_w = linear_w - kb_w * (32 * 64)
                token_off_w = rem_w // 64
                col_w = rem_w - token_off_w * 64

                token_idx_w = chunk_start + token_base + token_off_w
                global_k_w = kb_w * 64 + col_w

                w_bf16[kb_w, token_off_w, col_w] = al.convert(
                    w[0, token_idx_w, value_head_idx, global_k_w],
                    al.bf16,
                )

            al.syncthreads()

            # ----------------------------------------------------------
            # Each wave computes one 32x32 partial pred.
            #
            # partial[wave_id] = W_tile[:, K_half] @ state[:, K_half]^T
            #
            # Candidate operand row mapping:
            #   A row uses lane_mod32       -> token row 0..31
            #   B row uses lane_mod32       -> value row 0..31
            #
            # Candidate accumulator mapping:
            #   acc_i -> (row, col)
            #
            # This mapping must be verified by Codex using one-hot layout
            # tests. If correctness fails, fix mapping here first.
            # ----------------------------------------------------------
            pred_acc = al.full((16,), 0.0, al.f32)

            a_row = lane_mod32
            b_row = lane_mod32

            # K=64. Use 4 packed groups, each group gives two 4-BF16
            # fragments, for 8 MFMA32 calls.
            for kpack in al.range(4):
                a_words = w_vec[wave_id, a_row, kpack]
                b_words = state_vec[wave_id, b_row, kpack]

                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))

                # Keep operand order consistent with the working MFMA32 probe.
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

            if do_acc_unpack:
                row_base = lane_col & 7
                col_base = ((lane_col >> 3) * 4) + lane_group
                if precomputed_mapping:
                    pred_partial[wave_id, row_base, col_base] = pred_acc[0]
                    pred_partial[wave_id, row_base + 8, col_base] = pred_acc[1]
                    pred_partial[wave_id, row_base + 16, col_base] = pred_acc[2]
                    pred_partial[wave_id, row_base + 24, col_base] = pred_acc[3]
                    pred_partial[wave_id, row_base, col_base + 8] = pred_acc[4]
                    pred_partial[wave_id, row_base + 8, col_base + 8] = pred_acc[5]
                    pred_partial[wave_id, row_base + 16, col_base + 8] = pred_acc[6]
                    pred_partial[wave_id, row_base + 24, col_base + 8] = pred_acc[7]
                    pred_partial[wave_id, row_base, col_base + 16] = pred_acc[8]
                    pred_partial[wave_id, row_base + 8, col_base + 16] = pred_acc[9]
                    pred_partial[wave_id, row_base + 16, col_base + 16] = pred_acc[10]
                    pred_partial[wave_id, row_base + 24, col_base + 16] = pred_acc[11]
                    pred_partial[wave_id, row_base, col_base + 24] = pred_acc[12]
                    pred_partial[wave_id, row_base + 8, col_base + 24] = pred_acc[13]
                    pred_partial[wave_id, row_base + 16, col_base + 24] = pred_acc[14]
                    pred_partial[wave_id, row_base + 24, col_base + 24] = pred_acc[15]
                else:
                    # Original mapping from the seed v29 kernel.
                    for acc_i in al.range(16):
                        out_row = ((acc_i & 3) * 8) + row_base
                        out_col = ((acc_i >> 2) * 8) + col_base
                        pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]
            else:
                # Tiny observable side effect so the MFMA chain stays live.
                if tid == 0:
                    vn[0, chunk_start + token_base, value_head_idx, value_base] = pred_acc[0]

            al.syncthreads()

            if do_reduce:
                if store_full_vn:
                    # 32 tokens * 32 V values = 1024 elements.
                    # 128 threads * 8 reps = 1024 stores.
                    for rep_out in al.range(8):
                        linear_o = tid + rep_out * WORKGROUP
                        token_off_o = linear_o // BV
                        local_v_o = linear_o - token_off_o * BV

                        token_idx_o = chunk_start + token_base + token_off_o
                        global_v_o = value_base + local_v_o

                        pred_o = pred_partial[0, token_off_o, local_v_o] + pred_partial[1, token_off_o, local_v_o]
                        vn[0, token_idx_o, value_head_idx, global_v_o] = (
                            u[0, token_idx_o, value_head_idx, global_v_o] - pred_o
                        )
                else:
                    # Profiling-only reduction side effect without the full VN store.
                    if tid == 0:
                        pred_o_small = pred_partial[0, 0, 0] + pred_partial[1, 0, 0]
                        vn[0, chunk_start + token_base, value_head_idx, value_base] = pred_o_small
            else:
                if do_acc_unpack:
                    # Profiling-only unpack side effect.
                    if tid == 0:
                        vn[0, chunk_start + token_base, value_head_idx, value_base] = pred_partial[0, 0, 0]

            al.syncthreads()


def _validate_v29_pred_only_inputs(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("v29_pred_only only supports chunk_size=64.")

    if w.ndim != 4 or tuple(w.shape[0:1]) != (1,):
        raise ValueError(f"expected w shape [1,T,8,128], got {tuple(w.shape)}")
    if u.ndim != 4 or tuple(u.shape[0:1]) != (1,):
        raise ValueError(f"expected u shape [1,T,8,128], got {tuple(u.shape)}")

    if w.shape != u.shape:
        raise ValueError(f"w and u must have same shape, got {tuple(w.shape)} vs {tuple(u.shape)}")

    batch, num_tokens, num_heads, head_dim = w.shape
    if (batch, num_heads, head_dim) != (1, NUM_VALUE_HEADS, KDIM):
        raise ValueError("v29_pred_only only supports B=1,Hv=8,K/V=128.")

    if num_tokens % BT != 0:
        raise ValueError("v29_pred_only requires T divisible by 64.")

    if w.dtype != torch.float32:
        raise ValueError(f"expected w dtype torch.float32, got {w.dtype}")
    if u.dtype != torch.float32:
        raise ValueError(f"expected u dtype torch.float32, got {u.dtype}")

    if initial_state is not None:
        if tuple(initial_state.shape) != (1, NUM_VALUE_HEADS, 128, 128):
            raise ValueError(
                "expected initial_state shape [1,8,128,128], "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise ValueError(f"expected initial_state dtype torch.float32, got {initial_state.dtype}")

    num_chunks = num_tokens // BT
    return num_tokens, num_chunks


def _variant_flags(variant: str) -> dict[str, bool]:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown v29 isa_tune variant {variant!r}; expected one of {sorted(_VARIANTS)}")
    return _VARIANTS[variant]


def qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
    variant: str = MODE_BASELINE,
) -> torch.Tensor:
    """Run v29 pred-only MFMA32 experiment.

    Args:
        w: [1,T,8,128] fp32. This is the precomputed W operand used by GDN.
        u: [1,T,8,128] fp32. This is the uncorrected value/update operand.
        initial_state: [1,8,128,128] fp32, layout [B,H,V,K].
            If None, a zero state is used. For correctness tests, pass a
            nonzero state so pred is actually tested.

    Returns:
        vn: [1,T,8,128] fp32, expected to be u - W @ state.T.
    """
    num_tokens, num_chunks = _validate_v29_pred_only_inputs(w, u, initial_state, chunk_size)

    if initial_state is None:
        initial_state = torch.zeros((1, NUM_VALUE_HEADS, 128, 128), device=w.device, dtype=torch.float32)

    flags = _variant_flags(variant)
    vn = torch.empty_like(u)

    def launch() -> None:
        _qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune[
            lambda: ((GRID_SIZE, 1, 1), (WORKGROUP, 1, 1))
        ](
            w,
            u,
            initial_state,
            vn,
            num_tokens,
            num_chunks,
            flags["do_acc_unpack"],
            flags["do_reduce"],
            flags["store_full_vn"],
            flags["constant_state"],
            flags["precomputed_mapping"],
            num_warps=2,
        )

    _maybe_dump_hsaco(launch, "_qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune")
    launch()

    return vn


# Backward-compatible alias inside the copied experiment file.
def qwen_gdn_pred_only_avelang_v29_mfma32(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    return qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune(
        w,
        u,
        initial_state,
        chunk_size=chunk_size,
        variant=MODE_BASELINE,
    )


def qwen_gdn_pred_only_torch_reference(
    w: torch.Tensor,
    u: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for v29 pred-only.

    w:             [1,T,8,128]
    u:             [1,T,8,128]
    initial_state: [1,8,128,128], [B,H,V,K]

    pred[b,t,h,v] = sum_k w[b,t,h,k] * initial_state[b,h,v,k]
    vn = u - pred
    """
    pred = torch.einsum(
        "bthk,bhvk->bthv",
        w.float(),
        initial_state.float(),
    )
    return u.float() - pred


def qwen_gdn_pred_only_constant_state_reference(w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    initial_state = torch.full((1, 8, 128, 128), 0.03125, device=w.device, dtype=torch.float32)
    return qwen_gdn_pred_only_torch_reference(w, u, initial_state)


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
        raise RuntimeError(f"no compiled kernel matched {kernel_substr!r} for hsaco dump")


def _make_inputs(T: int, *, device: str = "cuda", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    # Keep magnitude moderate to avoid huge BF16 accumulation error.
    w = (torch.randn((1, T, 8, 128), device=device, dtype=torch.float32) * 0.02).contiguous()
    u = torch.randn((1, T, 8, 128), device=device, dtype=torch.float32).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), device=device, dtype=torch.float32) * 0.02).contiguous()

    return w, u, initial_state


def _smoke_main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--variant", choices=sorted(_VARIANTS), default=MODE_BASELINE)
    parser.add_argument("--no-check-ref", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()

    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir

    w, u, initial_state = _make_inputs(args.T)

    vn = qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune(w, u, initial_state, variant=args.variant)
    torch.cuda.synchronize()

    flags = _variant_flags(args.variant)
    correctness_meaningful = flags["do_reduce"] and flags["store_full_vn"]
    if not args.no_check_ref and correctness_meaningful:
        if flags["constant_state"]:
            ref = qwen_gdn_pred_only_constant_state_reference(w, u)
        else:
            ref = qwen_gdn_pred_only_torch_reference(w, u, initial_state)
        err = (vn - ref).abs()
        print(f"max_abs={err.max().item():.8e}")
        print(f"mean_abs={err.mean().item():.8e}")
    elif not args.no_check_ref:
        print(f"correctness_skipped=profiling_only_variant:{args.variant}")

    latency = _benchmark_cuda(
        lambda: qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune(w, u, initial_state, variant=args.variant),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    print(f"T={args.T} variant={args.variant} v29_isa_tune_ms={latency:.6f}")


if __name__ == "__main__":
    _smoke_main()


__all__ = [
    "BT",
    "BV",
    "KDIM",
    "WORKGROUP",
    "GRID_SIZE",
    "MODE_BASELINE",
    "MODE_NO_ACC_UNPACK",
    "MODE_UNPACK_ONLY_NO_REDUCE",
    "MODE_REDUCE_ONLY_NO_VN_STORE",
    "MODE_CONSTANT_STATE",
    "MODE_PRECOMPUTED_MAPPING",
    "MODE_OPTIMIZED",
    "_qwen_gdn_pred_only_bf16_kernel_v29_mfma32_isa_tune",
    "qwen_gdn_pred_only_avelang_v29_mfma32_isa_tune",
    "qwen_gdn_pred_only_avelang_v29_mfma32",
    "qwen_gdn_pred_only_torch_reference",
    "qwen_gdn_pred_only_constant_state_reference",
]

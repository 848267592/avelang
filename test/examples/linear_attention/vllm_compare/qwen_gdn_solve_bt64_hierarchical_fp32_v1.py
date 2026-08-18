"""Opt-in BT64 FP32 hierarchical triangular solve for the Qwen GDN layout.

This is a standalone Stage 5B S0 experiment.  It preserves the v6/v18 solve
contract exactly: for each `[chunk, value_head]`, it returns
`X = (I + A)^-1` in the original `[1, T, 8, 64]` layout.  There is no v18
fallback and it is intentionally not wired into the full BT64 pipeline.

The four diagonal 16x16 inverses use the existing row recurrence.  The six
strict-lower 16x16 blocks are formed by the 4x16 block triangular inverse DAG
and the source-level FP32 `mfma_16x16x4_f32_f32` intrinsic.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al


BT = 64
HEADS = 8
BLOCK = 16
WORKGROUP = 256


@avelang.jit
def _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1(
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout((1, num_tokens, HEADS, BT), (num_tokens * HEADS * BT, HEADS * BT, BT, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, HEADS, BT), (num_tokens * HEADS * BT, HEADS * BT, BT, 1)),
    )

    # The trailing unit dimension supplies a vector<1xf32> operand to MFMA.
    a_frag = al.view(
        a,
        al.f32,
        al.make_layout((1, num_tokens, HEADS, BT, 1), (num_tokens * HEADS * BT, HEADS * BT, BT, 1, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    head_idx = program_id % HEADS
    chunk_idx = program_id // HEADS
    chunk_start = chunk_idx * BT

    # Seven retained lower/diagonal blocks: X11/X22/X33/X44/X21/X32/X43.
    # The final two dependency levels reuse X33, X43, and X32 slots only
    # after their values have been written to global output.
    x = al.make_shared((7, BLOCK, BLOCK), al.f32)
    # Used as four 16-element diagonal row snapshots first, then as the
    # single 16x16 FP32 product workspace.  Total LDS is exactly 8 KiB.
    work = al.make_shared((BLOCK, BLOCK), al.f32)
    x_frag = al.view(x, al.f32, al.make_layout((7, BLOCK, BLOCK, 1), (256, 16, 1, 1)))
    work_frag = al.view(work, al.f32, al.make_layout((BLOCK, BLOCK, 1), (16, 1, 1)))

    # Clear all output blocks, including strict-upper cross blocks that the
    # lower-block DAG never explicitly writes.
    for rep_zero in al.range(16):
        linear = tid + rep_zero * WORKGROUP
        row = linear // BT
        col = linear - row * BT
        out[0, chunk_start + row, head_idx, col] = al.convert(0.0, al.f32)
    al.syncthreads()

    # Initialize four diagonal blocks to I - strictly_lower(A).
    for rep_diag in al.range(4):
        linear = tid + rep_diag * WORKGROUP
        diag_block = linear // 256
        local = linear - diag_block * 256
        row = local // BLOCK
        col = local - row * BLOCK
        value = al.convert(0.0, al.f32)
        if col < row:
            value = al.convert(0.0, al.f32) - a[0, chunk_start + diag_block * BLOCK + row, head_idx, diag_block * BLOCK + col]
        x[diag_block, row, col] = value
    al.syncthreads()

    # Independent diagonal inverses.  Each wave owns one 16x16 block.
    for row_solve in al.range(1, BLOCK):
        if lane < BLOCK:
            work[wave_id, lane] = x[wave_id, row_solve, lane]
        al.syncthreads()
        if lane < row_solve:
            value = work[wave_id, lane]
            for inner in al.range(BLOCK):
                if inner < row_solve:
                    value = value + work[wave_id, inner] * x[wave_id, inner, lane]
            x[wave_id, row_solve, lane] = value
        al.syncthreads()

    # The recurrence operates on M=-A, exactly like v6/v18.  Add I only
    # after all strict-lower rows have been solved; adding it earlier would
    # double the first sub-diagonal contribution.
    if lane < BLOCK:
        x[wave_id, lane, lane] = x[wave_id, lane, lane] + al.convert(1.0, al.f32)
    al.syncthreads()

    # Write diagonal output blocks while keeping their LDS copies for the
    # following block-DAG products.
    for rep_diag_out in al.range(4):
        linear = tid + rep_diag_out * WORKGROUP
        diag_block = linear // 256
        local = linear - diag_block * 256
        row = local // BLOCK
        col = local - row * BLOCK
        out[0, chunk_start + diag_block * BLOCK + row, head_idx, diag_block * BLOCK + col] = x[diag_block, row, col]
    al.syncthreads()

    # Level 1: X21=-X22*A21*X11, X32=-X33*A32*X22, X43=-X44*A43*X33.
    # Waves 0/1/2 use slots 4/5/6 as their temporary and final blocks.
    if wave_id < 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = x_frag[wave_id + 1, lane_col, k_idx]
            rhs = a_frag[0, chunk_start + (wave_id + 1) * BLOCK + k_idx, head_idx, wave_id * BLOCK + lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            x[wave_id + 4, lane_group * 4 + r, lane_col] = acc[r]
    if wave_id == 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                x_frag[3, lane_col, k_idx],
                a_frag[0, chunk_start + 3 * BLOCK + k_idx, head_idx, 2 * BLOCK + lane_col],
                acc,
            )
        for r in al.range(4):
            x[6, lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id < 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = x_frag[wave_id + 4, lane_col, k_idx]
            rhs = x_frag[wave_id, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            x[wave_id + 4, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    if wave_id == 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(x_frag[6, lane_col, k_idx], x_frag[2, k_idx, lane_col], acc)
        for r in al.range(4):
            x[6, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id < 2:
        for r in al.range(4):
            out[0, chunk_start + (wave_id + 1) * BLOCK + lane_group * 4 + r, head_idx, wave_id * BLOCK + lane_col] = x[
                wave_id + 4, lane_group * 4 + r, lane_col
            ]
    if wave_id == 2:
        for r in al.range(4):
            out[0, chunk_start + 3 * BLOCK + lane_group * 4 + r, head_idx, 2 * BLOCK + lane_col] = x[
                6, lane_group * 4 + r, lane_col
            ]
    al.syncthreads()

    # Level 2a: X31=-X33*(A31*X11 + A32*X21).  Wave 0 owns the shared
    # product workspace, then reuses the no-longer-needed X33 slot for X31.
    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 2 * BLOCK + lane_col, head_idx, k_idx]
            rhs = x_frag[0, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 2 * BLOCK + lane_col, head_idx, BLOCK + k_idx]
            rhs = x_frag[4, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = x_frag[2, lane_col, k_idx]
            rhs = work_frag[k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            x[2, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 0:
        for r in al.range(4):
            out[0, chunk_start + 2 * BLOCK + lane_group * 4 + r, head_idx, lane_col] = x[2, lane_group * 4 + r, lane_col]
    al.syncthreads()

    # Level 2b: X42=-X44*(A42*X22 + A43*X32).  X43 is no longer needed,
    # so its slot becomes X42 after the product is consumed.
    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 3 * BLOCK + lane_col, head_idx, BLOCK + k_idx]
            rhs = x_frag[1, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 3 * BLOCK + lane_col, head_idx, 2 * BLOCK + k_idx]
            rhs = x_frag[5, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = x_frag[3, lane_col, k_idx]
            rhs = work_frag[k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            x[6, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 1:
        for r in al.range(4):
            out[0, chunk_start + 3 * BLOCK + lane_group * 4 + r, head_idx, BLOCK + lane_col] = x[6, lane_group * 4 + r, lane_col]
    al.syncthreads()

    # Level 3: X41=-X44*(A41*X11 + A42*X21 + A43*X31).  X32 is no longer
    # needed after X42, so slot 5 becomes X41.
    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 3 * BLOCK + lane_col, head_idx, k_idx]
            rhs = x_frag[0, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 3 * BLOCK + lane_col, head_idx, BLOCK + k_idx]
            rhs = x_frag[4, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = a_frag[0, chunk_start + 3 * BLOCK + lane_col, head_idx, 2 * BLOCK + k_idx]
            rhs = x_frag[2, k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            lhs = x_frag[3, lane_col, k_idx]
            rhs = work_frag[k_idx, lane_col]
            acc = al.amdgpu.mfma_16x16x4_f32_f32(lhs, rhs, acc)
        for r in al.range(4):
            x[5, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 0:
        for r in al.range(4):
            out[0, chunk_start + 3 * BLOCK + lane_group * 4 + r, head_idx, lane_col] = x[5, lane_group * 4 + r, lane_col]


def _validate_input(a: torch.Tensor) -> int:
    if a.dtype != torch.float32:
        raise ValueError("Stage 5B requires torch.float32 input.")
    if not a.is_cuda:
        raise ValueError("Stage 5B requires a CUDA/HIP tensor.")
    if not a.is_contiguous():
        raise ValueError("Stage 5B requires a contiguous tensor.")
    if a.ndim != 4 or a.shape[0] != 1 or a.shape[2] != HEADS or a.shape[3] != BT:
        raise ValueError("Stage 5B requires shape [1,T,8,64].")
    num_tokens = int(a.shape[1])
    if num_tokens == 0 or num_tokens % BT != 0:
        raise ValueError("Stage 5B requires T > 0 and T divisible by 64.")
    return num_tokens


def qwen_gdn_solve_bt64_hierarchical_fp32_v1(a: torch.Tensor) -> torch.Tensor:
    """Solve fixed BT64 lower-triangular chunk matrices without fallback."""
    num_tokens = _validate_input(a)
    out = torch.empty_like(a)
    num_chunks = num_tokens // BT
    _qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1[lambda: ((num_chunks * HEADS, 1, 1), (WORKGROUP, 1, 1))](
        a,
        out,
        num_tokens,
        num_chunks,
    )
    return out


__all__ = [
    "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1",
    "qwen_gdn_solve_bt64_hierarchical_fp32_v1",
]

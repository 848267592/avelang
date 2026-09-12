"""Audit-only v18 solve diagnostics.

These kernels are deliberately not mathematical replacements for the v18
solve.  They are imported only by ``solve_rootcause_harness.py --mode
ablation`` to separate launch, input/output mapping, and final-store costs.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al


BT = 64
HEADS = 8


@avelang.jit
def _solve_launch_floor_kernel(
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, HEADS, BT), (num_tokens * HEADS * BT, HEADS * BT, BT, 1)),
    )
    tid = al.thread_id(0)
    program_id = al.block_id(0)
    head_idx = program_id % HEADS
    chunk_idx = program_id // HEADS
    if tid == 0:
        out[0, chunk_idx * BT, head_idx, 0] = al.convert(0.0, al.f32)


@avelang.jit
def _solve_load_store_only_kernel(
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
    tid = al.thread_id(0)
    program_id = al.block_id(0)
    head_idx = program_id % HEADS
    chunk_idx = program_id // HEADS
    chunk_start = chunk_idx * BT
    for rep in al.range(32):
        flat = tid + rep * 128
        if flat < BT * BT:
            row = flat // BT
            col = flat - row * BT
            out[0, chunk_start + row, head_idx, col] = a[0, chunk_start + row, head_idx, col]


@avelang.jit
def _solve_no_store_checksum_kernel(
    a_ptr: al.Pointer(al.f32),
    checksum_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    a = al.make_tensor(
        a_ptr,
        al.f32,
        al.make_layout((1, num_tokens, HEADS, BT), (num_tokens * HEADS * BT, HEADS * BT, BT, 1)),
    )
    checksum = al.make_tensor(checksum_ptr, al.f32, al.make_layout((num_chunks * HEADS,), (1,)))
    tid = al.thread_id(0)
    program_id = al.block_id(0)
    head_idx = program_id % HEADS
    chunk_idx = program_id // HEADS
    chunk_start = chunk_idx * BT
    groups_per_col = 2
    group_idx = tid // BT
    col_idx_thread = tid - group_idx * BT

    mat = al.make_shared((BT, BT), al.f32)
    row_buf = al.make_shared((BT,), al.f32)
    partial = al.make_shared((4, BT), al.f32)

    for rep_load in al.range(32):
        flat_idx = tid + rep_load * 128
        if flat_idx < BT * BT:
            row_idx = flat_idx // BT
            col_idx = flat_idx - row_idx * BT
            mat[row_idx, col_idx] = al.convert(0.0, al.f32) - a[0, chunk_start + row_idx, head_idx, col_idx]

    al.syncthreads()

    for row_idx_solve in al.range(1, BT):
        if tid < BT:
            row_buf[tid] = mat[row_idx_solve, tid]
        al.syncthreads()

        if group_idx < groups_per_col:
            part = al.convert(0.0, al.f32)
            if col_idx_thread < row_idx_solve:
                for inner_rep in al.range(32):
                    inner_idx = group_idx + inner_rep * groups_per_col
                    if inner_idx < row_idx_solve:
                        part = part + row_buf[inner_idx] * mat[inner_idx, col_idx_thread]
            partial[group_idx, col_idx_thread] = part

        al.syncthreads()

        if group_idx == 0:
            if col_idx_thread < row_idx_solve:
                acc = row_buf[col_idx_thread]
                for reduce_group in al.range(4):
                    if reduce_group < groups_per_col:
                        acc = acc + partial[reduce_group, col_idx_thread]
                mat[row_idx_solve, col_idx_thread] = acc

        al.syncthreads()

    if tid == 0:
        # The last row's first/last entries keep the triangular dependency live.
        checksum[program_id] = mat[BT - 1, 0] + mat[BT - 1, BT - 1]


def _grid(num_tokens: int):
    if num_tokens % BT:
        raise ValueError("audit diagnostics require T divisible by 64")
    return ((num_tokens // BT * HEADS, 1, 1), (128, 1, 1))


def avelang_launch_floor(out: torch.Tensor, num_tokens: int) -> None:
    _solve_launch_floor_kernel[lambda: _grid(num_tokens)](out, num_tokens, num_tokens // BT)


def avelang_load_store_only(a: torch.Tensor, out: torch.Tensor, num_tokens: int) -> None:
    _solve_load_store_only_kernel[lambda: _grid(num_tokens)](a, out, num_tokens, num_tokens // BT)


def avelang_no_store_checksum(a: torch.Tensor, checksum: torch.Tensor, num_tokens: int) -> None:
    if checksum.numel() != num_tokens // BT * HEADS:
        raise ValueError("checksum needs one FP32 element per chunk/head program")
    _solve_no_store_checksum_kernel[lambda: _grid(num_tokens)](a, checksum, num_tokens, num_tokens // BT)

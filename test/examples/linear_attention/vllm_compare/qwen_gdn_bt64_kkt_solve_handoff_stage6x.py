"""Stage 6X-KS X1: one-CTA BT64 KKT ownership experiment.

X1 intentionally retains the current global FP32 ``a`` output.  It changes
only KKT ownership from sixteen WG64 tile CTAs to one WG256 CTA per
``(chunk, value_head)``.  This is the prerequisite resource/parallelism gate
for a later, separate CTA-local KKT-to-solve handoff experiment.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    BT,
    BT_SUB,
    H_K,
    H_V,
    K_DIM,
    _num_chunks,
    _require_kkt_bt64_inputs,
)


WORKGROUP = 256
SOLVE_BLOCK = 16


@avelang.jit
def _qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Compute one complete strict-lower 64x64 KKT matrix per CTA.

    Each wave owns one token16 row tile.  K is staged once for the whole
    chunk/head, then each wave walks four compile-time column tiles.  The
    current FP32 row-major output ABI and causal writeback rule are unchanged.
    """
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * H_K * K_DIM, H_K * K_DIM, K_DIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    value_head_idx = program_id % H_V
    chunk_idx = program_id // H_V
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2
    row_base = wave_id * BT_SUB

    # The only X1 shared allocation: 64 * 128 BF16 = 16 KiB.  X2 may later
    # reuse this storage after the KKT phase, but X1 deliberately does not.
    k_all_bf16 = al.make_shared((BT, K_DIM), al.bf16)
    k_all_vec = al.view(k_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))

    for rep_stage in al.range(32):
        linear = tid + rep_stage * WORKGROUP
        row = linear // K_DIM
        col = linear - row * K_DIM
        k_all_bf16[row, col] = k[0, chunk_start + row, key_head_idx, col]
    al.syncthreads()

    # Four waves each own a row tile.  The condition is wave-uniform; it
    # skips only strict-upper dot products while preserving zero writeback.
    for col_tile in al.range(4):
        col_base = col_tile * BT_SUB
        acc = al.full((4,), 0.0, al.f32)
        if wave_id >= col_tile:
            for batch128 in al.range(4):
                vec_idx = lane_group + batch128 * 4
                a_words = k_all_vec[row_base + lane_col, vec_idx]
                b_words = k_all_vec[col_base + lane_col, vec_idx]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            source_offset = col_base + lane_col
            token_idx = chunk_start + token_offset
            source_idx = chunk_start + source_offset
            value = al.convert(0.0, al.f32)
            if source_offset < token_offset:
                decay = al.exp(g[0, token_idx, value_head_idx] - g[0, source_idx, value_head_idx])
                value = beta[0, token_idx, value_head_idx] * acc[r] * decay
            out[0, token_idx, value_head_idx, source_offset] = value


def qwen_gdn_kkt_bt64_one_cta_stage6x_x1(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    """X1 KKT public diagnostic API; no fallback to the 16-CTA KKT exists."""
    num_tokens, num_chunks = _require_kkt_bt64_inputs(k, g, beta, chunk_size)
    out = torch.empty((1, num_tokens, H_V, BT), dtype=torch.float32, device=k.device)
    _qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1[
        lambda: ((num_chunks * H_V, 1, 1), (WORKGROUP, 1, 1))
    ](k, g, beta, out, num_tokens, num_chunks)
    return out


@avelang.jit
def _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2(
    k_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    """Fuse X1 KKT and the existing four-wave hierarchical solve per CTA.

    This is deliberately the conservative X2 memory plan.  K staging, the
    FP32 ``a`` matrix, and the existing solve workspace occupy distinct LDS
    regions (16 + 16 + 8 KiB).  It establishes the global handoff-elimination
    contract before any optional source-level LDS lifetime/alias experiment.
    """
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * H_K * K_DIM, H_K * K_DIM, K_DIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    beta = al.make_tensor(
        beta_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    value_head_idx = program_id % H_V
    chunk_idx = program_id // H_V
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2
    row_base = wave_id * BT_SUB

    # KKT phase: stage K once and retain the FP32 strict-lower matrix only in
    # CTA-local LDS.  No FP32 a global allocation, store, or later reload.
    k_all_bf16 = al.make_shared((BT, K_DIM), al.bf16)
    # Tile the first A dimension as [4,16,64].  Besides matching the solve
    # ownership, this gives the shared view a non-degenerate leading tile
    # dimension, which Avelang preserves as vector<1xf32> MFMA operands.
    a_lds = al.make_shared((4, SOLVE_BLOCK, BT), al.f32)
    k_all_vec = al.view(k_all_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    # As in Stage 6U's x_frag, consumers below deliberately index only the
    # first three dimensions. The trailing unit dimension remains the
    # required vector<1xf32> MFMA fragment.
    a_lds_frag = al.view(
        a_lds,
        al.f32,
        al.make_layout((4, SOLVE_BLOCK, BT, 1), (SOLVE_BLOCK * BT, BT, 1, 1)),
    )

    for rep_stage in al.range(32):
        linear = tid + rep_stage * WORKGROUP
        row = linear // K_DIM
        col = linear - row * K_DIM
        k_all_bf16[row, col] = k[0, chunk_start + row, key_head_idx, col]
    al.syncthreads()

    for col_tile in al.range(4):
        col_base = col_tile * BT_SUB
        acc = al.full((4,), 0.0, al.f32)
        if wave_id >= col_tile:
            for batch128 in al.range(4):
                vec_idx = lane_group + batch128 * 4
                a_words = k_all_vec[row_base + lane_col, vec_idx]
                b_words = k_all_vec[col_base + lane_col, vec_idx]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], acc)
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], acc)

        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            source_offset = col_base + lane_col
            value = al.convert(0.0, al.f32)
            if source_offset < token_offset:
                decay = al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
                value = beta[0, chunk_start + token_offset, value_head_idx] * acc[r] * decay
            a_lds[wave_id, lane_group * 4 + r, source_offset] = value
    al.syncthreads()

    # Solve phase: this is the Stage 6U FP32 hierarchical solve, changing
    # only A's source from global a_ptr to the KKT-owned local a_lds tile.
    x = al.make_shared((7, SOLVE_BLOCK, SOLVE_BLOCK), al.f32)
    work = al.make_shared((SOLVE_BLOCK, SOLVE_BLOCK), al.f32)
    x_frag = al.view(x, al.f32, al.make_layout((7, SOLVE_BLOCK, SOLVE_BLOCK, 1), (256, 16, 1, 1)))
    work_frag = al.view(work, al.f32, al.make_layout((SOLVE_BLOCK, SOLVE_BLOCK, 1), (16, 1, 1)))

    for rep_zero in al.range(16):
        linear = tid + rep_zero * WORKGROUP
        row = linear // BT
        col = linear - row * BT
        out[0, chunk_start + row, value_head_idx, col] = al.convert(0.0, al.bf16)
    al.syncthreads()

    for rep_diag in al.range(4):
        linear = tid + rep_diag * WORKGROUP
        diag_block = linear // 256
        local = linear - diag_block * 256
        row = local // SOLVE_BLOCK
        col = local - row * SOLVE_BLOCK
        value = al.convert(0.0, al.f32)
        if col < row:
            value = al.convert(0.0, al.f32) - a_lds[diag_block, row, diag_block * SOLVE_BLOCK + col]
        x[diag_block, row, col] = value
    al.syncthreads()

    for row_solve in al.range(1, SOLVE_BLOCK):
        if lane < SOLVE_BLOCK:
            work[wave_id, lane] = x[wave_id, row_solve, lane]
        al.syncthreads()
        if lane < row_solve:
            value = work[wave_id, lane]
            for inner in al.range(SOLVE_BLOCK):
                if inner < row_solve:
                    value = value + work[wave_id, inner] * x[wave_id, inner, lane]
            x[wave_id, row_solve, lane] = value
        al.syncthreads()

    if lane < SOLVE_BLOCK:
        x[wave_id, lane, lane] = x[wave_id, lane, lane] + al.convert(1.0, al.f32)
    al.syncthreads()

    for rep_diag_out in al.range(4):
        linear = tid + rep_diag_out * WORKGROUP
        diag_block = linear // 256
        local = linear - diag_block * 256
        row = local // SOLVE_BLOCK
        col = local - row * SOLVE_BLOCK
        out[0, chunk_start + diag_block * SOLVE_BLOCK + row, value_head_idx, diag_block * SOLVE_BLOCK + col] = al.convert(
            x[diag_block, row, col], al.bf16
        )
    al.syncthreads()

    # Level 1: X21, X32, X43.
    if wave_id < 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                x_frag[wave_id + 1, lane_col, k_idx],
                a_lds_frag[wave_id + 1, k_idx, wave_id * SOLVE_BLOCK + lane_col],
                acc,
            )
        for r in al.range(4):
            x[wave_id + 4, lane_group * 4 + r, lane_col] = acc[r]
    if wave_id == 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                x_frag[3, lane_col, k_idx],
                a_lds_frag[3, k_idx, 2 * SOLVE_BLOCK + lane_col],
                acc,
            )
        for r in al.range(4):
            x[6, lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id < 2:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(x_frag[wave_id + 4, lane_col, k_idx], x_frag[wave_id, k_idx, lane_col], acc)
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
            out[0, chunk_start + (wave_id + 1) * SOLVE_BLOCK + lane_group * 4 + r, value_head_idx, wave_id * SOLVE_BLOCK + lane_col] = al.convert(
                x[wave_id + 4, lane_group * 4 + r, lane_col], al.bf16
            )
    if wave_id == 2:
        for r in al.range(4):
            out[0, chunk_start + 3 * SOLVE_BLOCK + lane_group * 4 + r, value_head_idx, 2 * SOLVE_BLOCK + lane_col] = al.convert(
                x[6, lane_group * 4 + r, lane_col], al.bf16
            )
    al.syncthreads()

    # Level 2a: X31=-X33*(A31*X11 + A32*X21).
    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[2, lane_col, k_idx], x_frag[0, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[2, lane_col, SOLVE_BLOCK + k_idx], x_frag[4, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(x_frag[2, lane_col, k_idx], work_frag[k_idx, lane_col], acc)
        for r in al.range(4):
            x[2, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 0:
        for r in al.range(4):
            out[0, chunk_start + 2 * SOLVE_BLOCK + lane_group * 4 + r, value_head_idx, lane_col] = al.convert(
                x[2, lane_group * 4 + r, lane_col], al.bf16
            )
    al.syncthreads()

    # Level 2b: X42=-X44*(A42*X22 + A43*X32).
    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[3, lane_col, SOLVE_BLOCK + k_idx], x_frag[1, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[3, lane_col, 2 * SOLVE_BLOCK + k_idx], x_frag[5, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 1:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(x_frag[3, lane_col, k_idx], work_frag[k_idx, lane_col], acc)
        for r in al.range(4):
            x[6, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 1:
        for r in al.range(4):
            out[0, chunk_start + 3 * SOLVE_BLOCK + lane_group * 4 + r, value_head_idx, SOLVE_BLOCK + lane_col] = al.convert(
                x[6, lane_group * 4 + r, lane_col], al.bf16
            )
    al.syncthreads()

    # Level 3: X41=-X44*(A41*X11 + A42*X21 + A43*X31).
    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[3, lane_col, k_idx], x_frag[0, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[3, lane_col, SOLVE_BLOCK + k_idx], x_frag[4, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for r in al.range(4):
            acc[r] = work[lane_group * 4 + r, lane_col]
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(
                a_lds_frag[3, lane_col, 2 * SOLVE_BLOCK + k_idx], x_frag[2, k_idx, lane_col], acc
            )
        for r in al.range(4):
            work[lane_group * 4 + r, lane_col] = acc[r]
    al.syncthreads()

    if wave_id == 0:
        acc = al.full((4,), 0.0, al.f32)
        for piece in al.range(4):
            k_idx = lane_group + piece * 4
            acc = al.amdgpu.mfma_16x16x4_f32_f32(x_frag[3, lane_col, k_idx], work_frag[k_idx, lane_col], acc)
        for r in al.range(4):
            x[5, lane_group * 4 + r, lane_col] = al.convert(0.0, al.f32) - acc[r]
    al.syncthreads()

    if wave_id == 0:
        for r in al.range(4):
            out[0, chunk_start + 3 * SOLVE_BLOCK + lane_group * 4 + r, value_head_idx, lane_col] = al.convert(
                x[5, lane_group * 4 + r, lane_col], al.bf16
            )


def qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    """X2 fused KKT-to-solve API; emits only the BF16 solved matrix."""
    num_tokens, num_chunks = _require_kkt_bt64_inputs(k, g, beta, chunk_size)
    out = torch.empty((1, num_tokens, H_V, BT), dtype=torch.bfloat16, device=k.device)
    return _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into(k, g, beta, out, chunk_size=chunk_size)


def _qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Direct-out X2 launcher for preallocated body and full-graph harnesses."""
    num_tokens, num_chunks = _require_kkt_bt64_inputs(k, g, beta, chunk_size)
    expected_shape = (1, num_tokens, H_V, BT)
    if (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != expected_shape
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != k.device
    ):
        raise ValueError("Stage 6X X2 output must be contiguous BF16 [1,T,8,64] on the input device.")
    _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2[
        lambda: ((num_chunks * H_V, 1, 1), (WORKGROUP, 1, 1))
    ](k, g, beta, out, num_tokens, num_chunks)
    return out


__all__ = [
    "_qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1",
    "_qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2",
    "qwen_gdn_kkt_bt64_one_cta_stage6x_x1",
    "qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2",
    "_qwen_gdn_kkt_solve_bt64_one_cta_stage6x_x2_launch_into",
]

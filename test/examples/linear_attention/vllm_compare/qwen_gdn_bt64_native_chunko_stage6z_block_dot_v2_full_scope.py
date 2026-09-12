"""Stage 6Z BDV2: reusable full-scope logical BF16 block-dot.

This is an experimental-only source fork of the Z5B/Z7B chunk-o body.  The
Q cache, accumulator ordering, MFMA geometry, and public BF16 contract are
unchanged.  The H and K logical blocks are no longer staged by handwritten
source loops: both are presented through the same target-independent
``block_dot_bf16_f32_logical`` operation.  The block-dot lowering owns the
global packet producer, shared placement, and MFMA-B consumer.

``AVELANG_BLOCK_DOT_LOWERING=generic`` and ``specialized`` select the two
lowering arms.  ``AVELANG_BLOCK_DOT_LAYOUT_PLANNER=bdv2_p1_affine`` selects
the experimental common affine-map planner; ``legacy`` is the frozen BDV2
baseline.  They compile this same source and create the same
``ave.gpu.amdgpu_block_dot_bf16_f32`` operation; only the late producer
materialization differs.

``AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p2_first_class``,
``p3_packed_reuse`` and ``p4_accumulator_reuse`` are compiler-only selectors
for the same source. They keep
the planned MFMA operand consumer as an internal representation until
GPU-module lowering; P3 groups two adjacent fragments into one packed LDS
consumer without changing the source schedule. P4 additionally forwards a
same-memref FP32 accumulator SSA value across reset-free full-scope block-dot
stages; reset-bearing source-half loops remain materialized.
"""

from __future__ import annotations

import os

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_bf16_chunko_boundary_stage6w import (
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    _validate_stage6w_chunko_inputs,
)


BT = 64
BV = 64
BK = 32
WORKGROUP = 256
BDV2_WORKGROUP_CONTRACT = 256
Q_CACHE_ROWS = 4 * BT


def set_block_dot_lowering(lowering: str) -> None:
    if lowering not in {"generic", "specialized"}:
        raise ValueError("lowering must be generic or specialized")
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = lowering


def set_block_dot_planner(planner: str) -> None:
    if planner not in {"legacy", "bdv2_p1_affine"}:
        raise ValueError("planner must be legacy or bdv2_p1_affine")
    os.environ["AVELANG_BLOCK_DOT_LAYOUT_PLANNER"] = planner


def set_block_dot_operand_preservation(preservation: str) -> None:
    if preservation not in {
        "none",
        "p2_first_class",
        "p3_packed_reuse",
        "p4_accumulator_reuse",
    }:
        raise ValueError(
            "preservation must be none, p2_first_class, p3_packed_reuse, "
            "or p4_accumulator_reuse"
        )
    if preservation == "none":
        os.environ.pop("AVELANG_BLOCK_DOT_OPERAND_PRESERVATION", None)
    else:
        os.environ["AVELANG_BLOCK_DOT_OPERAND_PRESERVATION"] = preservation


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    q = al.make_tensor(
        q_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)),
    )
    vn = al.make_tensor(
        vn_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, V_DIM, K_DIM),
            (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1),
        ),
    )
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)),
    )
    out = al.make_tensor(
        out_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)),
    )

    tid = al.thread_id(0)
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5
    wave_id = tid >> 6
    row_half = wave_id >> 1
    value_half = wave_id & 1
    program_id = al.block_id(0)
    v_block_idx = program_id % 2
    value_head_idx = (program_id // 2) % H_V
    chunk_idx = program_id // (2 * H_V)
    value_base = v_block_idx * BV
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx >> 1

    # This is the unchanged Z5B dedicated full-Q cache.  BDV2 only changes
    # the ownership of H/K logical block production.
    shared = al.make_shared((2 * Q_CACHE_ROWS, BK), al.bf16)
    q_cache = shared
    phase = shared
    phase_vec = al.view(phase, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))

    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            q_cache[k_stage * BT + row, col] = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32)
                * scale,
                al.bf16,
            )
        al.syncthreads()
    al.syncthreads()

    inter_acc = al.full((16,), 0.0, al.f32)
    zero_f32 = al.convert(0.0, al.f32)

    # H is a logical [V,K] block.  The transpose metadata is part of the
    # generic block-dot contract, not a separate H-specific intrinsic.
    for k_stage in al.range(4):
        inter_pair = al.amdgpu.block_dot_bf16_f32_logical_transposed(
            q_cache,
            phase,
            h,
            vn,
            g,
            tid,
            chunk_start,
            value_head_idx,
            key_head_idx,
            value_base,
            k_stage,
            zero_f32,
            inter_acc,
            inter_acc,
        )
        for acc_i in al.range(16):
            inter_acc[acc_i] = inter_pair[acc_i]
        al.syncthreads()

    # The logical K operation is called by the whole CTA so its lowering can
    # place a uniform producer/consumer barrier.  Only the original value
    # half (the first two waves) consumes and commits the score accumulator.
    for source_half in al.range(2):
        score_acc = al.full((16,), 0.0, al.f32)
        for k_stage in al.range(4):
            score_pair = al.amdgpu.block_dot_bf16_f32_logical(
                q_cache,
                phase,
                k,
                vn,
                g,
                tid,
                chunk_start,
                value_head_idx,
                key_head_idx,
                source_half,
                k_stage,
                zero_f32,
                score_acc,
                score_acc,
            )
            if value_half == 0:
                for acc_i in al.range(16):
                    score_acc[acc_i] = score_pair[acc_i]
            al.syncthreads()

        if value_half == 0:
            for r in al.range(16):
                out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
                token_offset = row_half * 32 + lane_col
                source_offset = source_half * 32 + out_col
                score = al.convert(0.0, al.f32)
                if source_offset <= token_offset:
                    score = score_acc[r] * al.exp(
                        g[0, chunk_start + token_offset, value_head_idx]
                        - g[0, chunk_start + source_offset, value_head_idx]
                    )
                phase[Q_CACHE_ROWS + token_offset * 2 + source_half, out_col] = al.convert(
                    score, al.bf16
                )
        al.syncthreads()

    # Phase C and output are byte-for-byte the Z5B path.
    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[
            Q_CACHE_ROWS + 128 + value_offset * 2 + token_offset // 32,
            token_offset - (token_offset // 32) * 32,
        ] = vn[0, chunk_start + token_offset, value_head_idx, value_base + value_offset]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[
                Q_CACHE_ROWS + (row_half * 32 + lane_col) * 2 + source_half,
                word,
            ]
            v_words = phase_vec[
                Q_CACHE_ROWS + 128 + (value_half * 32 + lane_col) * 2 + source_half,
                word,
            ]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_offset = row_half * 32 + lane_col
        result = inter_acc[r] * al.exp(g[0, chunk_start + token_offset, value_head_idx]) + intra_acc[r]
        out[
            0,
            chunk_start + token_offset,
            value_head_idx,
            value_base + value_half * 32 + out_col,
        ] = al.convert(result, al.bf16)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
    lowering: str = "generic",
    planner: str = "legacy",
    preservation: str = "none",
) -> None:
    if WORKGROUP != BDV2_WORKGROUP_CONTRACT:
        raise RuntimeError("BDV2 shape contract violation: WG must remain 256")
    set_block_dot_lowering(lowering)
    set_block_dot_planner(planner)
    set_block_dot_operand_preservation(preservation)
    t, num_chunks = _validate_stage6w_chunko_inputs(
        q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size
    )
    if (
        output_bf16.dtype != torch.bfloat16
        or tuple(output_bf16.shape) != tuple(v_new_bf16.shape)
        or not output_bf16.is_cuda
        or not output_bf16.is_contiguous()
        or output_bf16.device != q.device
    ):
        raise ValueError("BDV2 output must be contiguous BF16 [1,T,8,128]")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
    lowering: str = "generic",
    planner: str = "legacy",
    preservation: str = "none",
) -> torch.Tensor:
    output_bf16 = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into(
        q, k, v_new_bf16, h_bf16, g, output_bf16,
        scale=scale, chunk_size=chunk_size, lowering=lowering,
        planner=planner, preservation=preservation,
    )
    return output_bf16


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_bdv2_full_scope_launch_into",
]

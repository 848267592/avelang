"""Stage 6Z C21 selected-native WG256 pipeline reconstruction.

This is an experimental-only chunk-o candidate.  It keeps the C19 logical
Q/H/K/V ownership and the established BF16 output contract, but presents the
Q@H and both Q@K consumers in one K32 source superloop.  The C21 compiler
mode owns the two Q physical slots and the plan annotations; it is not a
production selector.
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
from qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region import (
    set_block_dot_lowering,
    set_block_dot_operand_preservation,
    set_block_dot_planner,
)


BT = 64
BV = 64
BK = 32
WORKGROUP = 256
C21_SHARED_ROWS = 3 * BT * 2


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline(
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

    # C21 reserves the same complete arena as C19.  The compiler maps Q to
    # two rotating K32 slots; H/K stay in the frozen C19 phase band and V is
    # unchanged after Q/H/K last use.
    shared = al.make_shared((C21_SHARED_ROWS, BK), al.bf16)
    phase = shared
    zero_f32 = al.convert(0.0, al.f32)
    scale_f32 = al.convert(scale, al.f32)

    inter_acc = al.full((16,), 0.0, al.f32)
    score_acc0 = al.full((16,), 0.0, al.f32)
    score_acc1 = al.full((16,), 0.0, al.f32)

    # This is deliberately one source K32 superloop.  Each logical Q stage
    # is an owner-only operation so the C21 planner can assign it to the
    # current rotating Q slot.  The immediately following H/K consumers use
    # the same K32 source before the next iteration may overwrite that slot.
    for k_stage in al.range(4):
        q_owner_acc = al.full((16,), 0.0, al.f32)
        _q_stage_owner = al.amdgpu.block_dot_bf16_f32_logical(
            shared,
            shared,
            q,
            q,
            g,
            tid,
            chunk_start,
            value_head_idx,
            key_head_idx,
            0,
            k_stage,
            scale_f32,
            q_owner_acc,
            q_owner_acc,
        )

        inter_pair = al.amdgpu.block_dot_bf16_f32_logical_transposed(
            shared,
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

        score_pair0 = al.amdgpu.block_dot_bf16_f32_logical(
            shared,
            phase,
            k,
            vn,
            g,
            tid,
            chunk_start,
            value_head_idx,
            key_head_idx,
            0,
            k_stage,
            zero_f32,
            score_acc0,
            score_acc0,
        )
        if value_half == 0:
            for acc_i in al.range(16):
                score_acc0[acc_i] = score_pair0[acc_i]

        score_pair1 = al.amdgpu.block_dot_bf16_f32_logical(
            shared,
            phase,
            k,
            vn,
            g,
            tid,
            chunk_start,
            value_head_idx,
            key_head_idx,
            1,
            k_stage,
            zero_f32,
            score_acc1,
            score_acc1,
        )
        if value_half == 0:
            for acc_i in al.range(16):
                score_acc1[acc_i] = score_pair1[acc_i]

    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score_acc0[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[token_offset * 2, out_col] = al.convert(score, al.bf16)
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = 32 + out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score_acc1[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[token_offset * 2 + 1, out_col] = al.convert(score, al.bf16)
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            v_pair = al.amdgpu.block_dot_bf16_f32_logical(
                phase,
                phase,
                vn,
                vn,
                g,
                tid,
                chunk_start,
                value_head_idx,
                key_head_idx,
                value_base,
                source_half * 2 + kt,
                zero_f32,
                intra_acc,
                intra_acc,
            )
            for acc_i in al.range(16):
                intra_acc[acc_i] = v_pair[acc_i]

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


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
    lowering: str = "specialized",
    planner: str = "bdv2_p1_affine",
    preservation: str = "p2_first_class",
) -> None:
    if WORKGROUP != 256:
        raise RuntimeError("C21 selected-native contract requires WG256")
    set_block_dot_lowering(lowering)
    set_block_dot_planner(planner)
    set_block_dot_operand_preservation(preservation)
    os.environ["AVELANG_STAGE6Z_FULL_PHYSICAL_REGION"] = "c21"
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
        raise ValueError("C21 output must be contiguous BF16 [1,T,8,128]")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
    lowering: str = "specialized",
    planner: str = "bdv2_p1_affine",
    preservation: str = "p2_first_class",
) -> torch.Tensor:
    output_bf16 = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into(
        q,
        k,
        v_new_bf16,
        h_bf16,
        g,
        output_bf16,
        scale=scale,
        chunk_size=chunk_size,
        lowering=lowering,
        planner=planner,
        preservation=preservation,
    )
    return output_bf16


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_c21_selected_native_pipeline_launch_into",
]

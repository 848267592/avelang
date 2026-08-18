"""Opt-in Stage 6W: keep the BT64 chunk-o boundary in BF16.

The established BT64 chunk-o body already consumes BF16 LDS operands and
accumulates in FP32.  Stage 6W changes only its global boundary: it directly
loads BF16 recurrence V-new and converts its final FP32 result to a BF16
public-output store.  No recurrence, MFMA geometry, LDS tile, or layout is
changed.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (
    _require_target,
    qwen_gdn_bt64_stage6s_recurrence_bridge,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import (
    qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u,
)
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    BT,
    BT_SUB,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
)
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
)


WORKGROUP = 256


def _validate_stage6w_chunko_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("Stage 6W chunk-o only supports chunk_size=64.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Stage 6W chunk-o requires BF16 q/k.")
    if v_new_bf16.dtype != torch.bfloat16 or h_bf16.dtype != torch.bfloat16:
        raise ValueError("Stage 6W chunk-o requires BF16 v_new/h.")
    if g.dtype != torch.float32:
        raise ValueError("Stage 6W chunk-o requires FP32 g.")
    values = (q, k, v_new_bf16, h_bf16, g)
    if any(not value.is_cuda or not value.is_contiguous() or value.device != q.device for value in values):
        raise ValueError("Stage 6W chunk-o requires contiguous tensors on one HIP device.")
    t = int(q.shape[1]) if q.ndim == 4 else -1
    num_chunks = _num_chunks(t, BT) if t >= BT and t % BT == 0 else -1
    if tuple(q.shape) != (1, t, H_K, K_DIM) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Stage 6W chunk-o requires q/k=[1,T,4,128].")
    if tuple(v_new_bf16.shape) != (1, t, H_V, V_DIM) or tuple(g.shape) != (1, t, H_V):
        raise ValueError("Stage 6W chunk-o requires v_new=[1,T,8,128] and g=[1,T,8].")
    if t < BT or t % BT or tuple(h_bf16.shape) != (1, num_chunks, H_V, V_DIM, K_DIM):
        raise ValueError("Stage 6W chunk-o requires h=[1,T/64,8,128,128] and T divisible by 64.")
    return t, num_chunks


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w(
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
    q = al.make_tensor(q_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout((1, num_chunks, H_V, V_DIM, K_DIM), (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    out = al.make_tensor(out_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % H_V
    chunk_idx = program_id // 64
    row_base = wave_id * BT_SUB
    value_base = v_block_idx * BT_SUB
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2

    q_scaled_bf16 = al.make_shared((BT, K_DIM), al.bf16)
    h_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    k_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    score_decay_bf16 = al.make_shared((4, BT_SUB, BT_SUB), al.bf16)
    vn_t_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    q_vec = al.view(q_scaled_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    h_vec = al.view(h_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    score_vec = al.view(score_decay_bf16, al.i32, al.make_layout((4, BT_SUB, 2, 4), (128, 8, 4, 1)))
    vn_vec = al.view(vn_t_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    for rep in al.range(32):
        idx = tid + rep * WORKGROUP
        row = idx // K_DIM
        col = idx - row * K_DIM
        q_scaled_bf16[row, col] = al.convert(
            al.convert(q[0, chunk_start + row, key_head_idx, col], al.f32) * scale,
            al.bf16,
        )
    for rep in al.range(8):
        idx = tid + rep * WORKGROUP
        row = idx // K_DIM
        col = idx - row * K_DIM
        h_bf16[row, col] = h[0, chunk_idx, value_head_idx, value_base + row, col]
    al.syncthreads()

    inter_acc = al.full((4,), 0.0, al.f32)
    for batch128 in al.range(4):
        vec_idx = lane_group + batch128 * 4
        a_frag = al.view(q_vec[row_base + lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(h_vec[lane_col, vec_idx], al.Tensor((2, 4, 1), al.bf16))
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], inter_acc)
        inter_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], inter_acc)

    intra_acc = al.full((4,), 0.0, al.f32)
    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // K_DIM
            col = idx - row * K_DIM
            k_bf16[row, col] = k[0, chunk_start + source_base + row, key_head_idx, col]
        value_offset = tid // BT_SUB
        source_offset = tid - value_offset * BT_SUB
        # Direct BF16 global load: the baseline expanded recurrence output to
        # FP32 here only to immediately truncate it back to this LDS tile.
        vn_t_bf16[value_offset, source_offset] = vn[
            0, chunk_start + source_base + source_offset, value_head_idx, value_base + value_offset
        ]
        al.syncthreads()

        score_acc = al.full((4,), 0.0, al.f32)
        for batch128_score in al.range(4):
            vec_idx_score = lane_group + batch128_score * 4
            a_frag_score = al.view(q_vec[row_base + lane_col, vec_idx_score], al.Tensor((2, 4, 1), al.bf16))
            b_frag_score = al.view(k_vec[lane_col, vec_idx_score], al.Tensor((2, 4, 1), al.bf16))
            score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_score[0], b_frag_score[0], score_acc)
            score_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_score[1], b_frag_score[1], score_acc)

        for r_score in al.range(4):
            token_offset = lane_group * 4 + r_score
            output_abs = row_base + token_offset
            source_abs = source_base + lane_col
            score_value = al.convert(0.0, al.f32)
            if source_abs <= output_abs:
                score_value = score_acc[r_score] * al.exp(
                    g[0, chunk_start + output_abs, value_head_idx]
                    - g[0, chunk_start + source_abs, value_head_idx]
                )
            score_decay_bf16[wave_id, token_offset, lane_col] = al.convert(score_value, al.bf16)
        al.syncthreads()

        if lane_group == 0:
            a_frag_intra = al.view(score_vec[wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
        if lane_group == 1:
            a_frag_intra = al.view(score_vec[wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
        if lane_group == 2:
            a_frag_intra = al.view(score_vec[wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
        if lane_group == 3:
            a_frag_intra = al.view(score_vec[wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
        al.syncthreads()

    for r_out in al.range(4):
        token_offset_out = row_base + lane_group * 4 + r_out
        token_idx_out = chunk_start + token_offset_out
        result = inter_acc[r_out] * al.exp(g[0, token_idx_out, value_head_idx]) + intra_acc[r_out]
        out[0, token_idx_out, value_head_idx, value_base + lane_col] = al.convert(result, al.bf16)


def qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    """Stage 6W chunk-o with a BF16 recurrence input and public BF16 output."""
    t, _ = _validate_stage6w_chunko_inputs(q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size)
    output = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into(
        q, k, v_new_bf16, h_bf16, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


def qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> None:
    """Launch Stage 6W chunk-o into caller-owned BF16 output storage."""
    t, num_chunks = _validate_stage6w_chunko_inputs(q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size)
    if (
        output_bf16.dtype != torch.bfloat16
        or tuple(output_bf16.shape) != tuple(v_new_bf16.shape)
        or not output_bf16.is_cuda
        or not output_bf16.is_contiguous()
        or output_bf16.device != q.device
    ):
        raise ValueError("Stage 6W chunk-o output must be contiguous BF16 [1,T,8,128] on the input device.")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w[
        lambda: ((num_chunks * H_V * 8, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """U1 graph with only the BF16 chunk-o storage boundaries replaced."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved_bf16 = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a)
    w_bf16, u_bf16 = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g_cumsum, beta, a_solved_bf16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    output_bf16 = qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(
        q, k, v_new_bf16, h_bf16, g_cumsum, scale=scale
    )
    return output_bf16, final_state if output_final_state else None


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w",
    "qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w",
    "qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w_launch_into",
    "qwen_gdn_full_bt64_stage6w_bf16_chunko_eager",
]

"""Stage 6V experimental C0 predicate collapse.

This module changes one C0 scheduling detail only.  The four lane-group
fragment selections remain semantically identical, but their MFMA call is
moved out of the divergent regions.  P0 solve, BF16 boundary, CTA/WG, C0 math,
recurrence, chunk-o, layouts and public dtype contracts remain unchanged.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (
    _require_target,
    qwen_gdn_bt64_stage6s_recurrence_bridge,
)
from qwen_gdn_bt64_bf16_solved_boundary_stage6u import _validate_c0_inputs
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    BT,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u


BT_SUB = 16
WORKGROUP = 256


@avelang.jit
def _qwen_gdn_wu_kernel_bt64_predicate_collapse_v0(
    k_ptr: al.Pointer(al.bf16),
    v_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    beta_ptr: al.Pointer(al.f32),
    a_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.bf16),
    u_ptr: al.Pointer(al.bf16),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    beta = al.make_tensor(beta_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    a = al.make_tensor(a_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, BT), (num_tokens * H_V * BT, H_V * BT, BT, 1)))
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, K_DIM), (num_tokens * 1024, 1024, 128, 1)))
    u = al.make_tensor(u_ptr, al.bf16, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    value_head_idx = program_id % H_V
    chunk_idx = program_id // H_V
    key_head_idx = value_head_idx // 2
    chunk_start = chunk_idx * BT
    row_base = wave_id * BT_SUB

    coeff_bf16 = al.make_shared((BT, BT_SUB), al.bf16)
    operand0_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    operand1_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    coeff_vec = al.view(coeff_bf16, al.i32, al.make_layout((BT, 2, 4), (8, 4, 1)))
    operand0_vec = al.view(operand0_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))
    operand1_vec = al.view(operand1_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    # W main only. The four operand choices are selected first; MFMA is
    # intentionally wave-uniform and occurs once below the selection.
    for col_pair in al.range(4):
        col_base = col_pair * 32
        w_acc0 = al.full((4,), 0.0, al.f32)
        w_acc1 = al.full((4,), 0.0, al.f32)
        for source_tile in al.range(4):
            source_base = source_tile * BT_SUB
            for rep in al.range(4):
                idx = tid + rep * WORKGROUP
                row = idx // BT_SUB
                source_offset = idx - row * BT_SUB
                token_idx = chunk_start + row
                source_idx = chunk_start + source_base + source_offset
                coeff = al.convert(a[0, token_idx, value_head_idx, source_base + source_offset], al.f32)
                coeff = coeff * beta[0, source_idx, value_head_idx]
                coeff = coeff * al.exp(g[0, source_idx, value_head_idx])
                coeff_bf16[row, source_offset] = al.convert(coeff, al.bf16)
            operand_row = tid // BT_SUB
            operand_col = tid - operand_row * BT_SUB
            source_idx = chunk_start + source_base + operand_col
            operand0_bf16[operand_row, operand_col] = k[0, source_idx, key_head_idx, col_base + operand_row]
            operand1_bf16[operand_row, operand_col] = k[0, source_idx, key_head_idx, col_base + BT_SUB + operand_row]
            al.syncthreads()

            a_frag0 = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            a_frag1 = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b0_frag0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b0_frag1 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b1_frag0 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b1_frag1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            # Conditional expressions lower to arith.select. Statement-level
            # if blocks have isolated scopes and cannot carry these fragments.
            a_operand = a_frag0[0] if lane_group == 0 else (a_frag0[1] if lane_group == 1 else (a_frag1[0] if lane_group == 2 else a_frag1[1]))
            b0_operand = b0_frag0[0] if lane_group == 0 else (b0_frag0[1] if lane_group == 1 else (b0_frag1[0] if lane_group == 2 else b0_frag1[1]))
            b1_operand = b1_frag0[0] if lane_group == 0 else (b1_frag0[1] if lane_group == 1 else (b1_frag1[0] if lane_group == 2 else b1_frag1[1]))
            w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b0_operand, w_acc0)
            w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b1_operand, w_acc1)
            al.syncthreads()
        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            token_idx = chunk_start + token_offset
            w[0, token_idx, value_head_idx, col_base + lane_col] = al.convert(w_acc0[r], al.bf16)
            w[0, token_idx, value_head_idx, col_base + BT_SUB + lane_col] = al.convert(w_acc1[r], al.bf16)
        al.syncthreads()

    # U main only. Keep W/U accumulator lifetime ordering unchanged.
    for col_pair in al.range(4):
        col_base = col_pair * 32
        u_acc0 = al.full((4,), 0.0, al.f32)
        u_acc1 = al.full((4,), 0.0, al.f32)
        for source_tile in al.range(4):
            source_base = source_tile * BT_SUB
            for rep in al.range(4):
                idx = tid + rep * WORKGROUP
                row = idx // BT_SUB
                source_offset = idx - row * BT_SUB
                token_idx = chunk_start + row
                source_idx = chunk_start + source_base + source_offset
                coeff = al.convert(a[0, token_idx, value_head_idx, source_base + source_offset], al.f32)
                coeff = coeff * beta[0, source_idx, value_head_idx]
                coeff_bf16[row, source_offset] = al.convert(coeff, al.bf16)
            operand_row = tid // BT_SUB
            operand_col = tid - operand_row * BT_SUB
            source_idx = chunk_start + source_base + operand_col
            operand0_bf16[operand_row, operand_col] = v[0, source_idx, value_head_idx, col_base + operand_row]
            operand1_bf16[operand_row, operand_col] = v[0, source_idx, value_head_idx, col_base + BT_SUB + operand_row]
            al.syncthreads()

            a_frag0 = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            a_frag1 = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b0_frag0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b0_frag1 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            b1_frag0 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
            b1_frag1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
            a_operand = a_frag0[0] if lane_group == 0 else (a_frag0[1] if lane_group == 1 else (a_frag1[0] if lane_group == 2 else a_frag1[1]))
            b0_operand = b0_frag0[0] if lane_group == 0 else (b0_frag0[1] if lane_group == 1 else (b0_frag1[0] if lane_group == 2 else b0_frag1[1]))
            b1_operand = b1_frag0[0] if lane_group == 0 else (b1_frag0[1] if lane_group == 1 else (b1_frag1[0] if lane_group == 2 else b1_frag1[1]))
            u_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b0_operand, u_acc0)
            u_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b1_operand, u_acc1)
            al.syncthreads()
        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            token_idx = chunk_start + token_offset
            u[0, token_idx, value_head_idx, col_base + lane_col] = al.convert(u_acc0[r], al.bf16)
            u[0, token_idx, value_head_idx, col_base + BT_SUB + lane_col] = al.convert(u_acc1[r], al.bf16)
        al.syncthreads()


def qwen_gdn_w_u_bt64_predicate_collapse_v0(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V0 isolated C0 replacement; diagnostic only, no fallback."""
    t, num_chunks = _validate_c0_inputs(k, v, g, beta, a_solved, chunk_size=chunk_size)
    w = torch.empty((1, t, H_V, K_DIM), dtype=torch.bfloat16, device=k.device)
    u = torch.empty_like(w)
    _qwen_gdn_wu_kernel_bt64_predicate_collapse_v0[
        lambda: ((num_chunks * H_V, 1, 1), (WORKGROUP, 1, 1))
    ](k, v, g, beta, a_solved, w, u, t, num_chunks)
    return w, u


def qwen_gdn_full_bt64_stage6v_predicate_collapse_eager(
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
    """V1 eager public path: U1 stages with V0 as the sole changed body."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved_bf16 = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a)
    w_bf16, u_bf16 = qwen_gdn_w_u_bt64_predicate_collapse_v0(k, v, g_cumsum, beta, a_solved_bf16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    output_fp32 = qwen_gdn_chunk_o_bt64_mfma_v2_s0(
        q, k, v_new_bf16.to(torch.float32), h_bf16, g_cumsum, scale=scale
    )
    return output_fp32.to(q.dtype), final_state if output_final_state else None


__all__ = [
    "_qwen_gdn_wu_kernel_bt64_predicate_collapse_v0",
    "qwen_gdn_w_u_bt64_predicate_collapse_v0",
    "qwen_gdn_full_bt64_stage6v_predicate_collapse_eager",
]

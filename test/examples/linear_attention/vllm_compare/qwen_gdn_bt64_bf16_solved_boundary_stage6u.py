"""Opt-in Stage 6U BT64 BF16-solved producer/consumer full paths.

U0 uses the frozen FP32 solve followed by an explicit BF16 cast. U1 uses the
native Stage 6U BF16 solve writeback. Both feed the same main-only fused W/U
consumer and the unchanged Stage 6S recurrence/chunk-o tail.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    BT,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
)
from qwen_gdn_bt64_bf16_recurrence_full_stage6s import (
    _require_target,
    qwen_gdn_bt64_stage6s_recurrence_bridge,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    _num_chunks,
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
)
from qwen_gdn_solve_bt64_hierarchical_bf16_stage6u import (
    qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u,
    qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u,
)


BT_SUB = 16
WORKGROUP = 256


def _validate_c0_inputs(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("Stage 6U C0 only supports chunk_size=64.")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or a_solved.dtype != torch.bfloat16:
        raise ValueError("Stage 6U C0 requires BF16 k/v/a_solved.")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("Stage 6U C0 requires FP32 g/beta.")
    values = (k, v, g, beta, a_solved)
    if any(not value.is_cuda or not value.is_contiguous() or value.device != k.device for value in values):
        raise ValueError("Stage 6U C0 requires contiguous tensors on one HIP device.")
    t = int(k.shape[1]) if k.ndim == 4 else -1
    if t < BT or t % BT:
        raise ValueError("Stage 6U C0 requires T >= 64 and T divisible by 64.")
    if tuple(k.shape) != (1, t, H_K, K_DIM) or tuple(v.shape) != (1, t, H_V, V_DIM):
        raise ValueError("Stage 6U C0 requires k=[1,T,4,128], v=[1,T,8,128].")
    if tuple(g.shape) != (1, t, H_V) or tuple(beta.shape) != (1, t, H_V):
        raise ValueError("Stage 6U C0 requires g/beta=[1,T,8].")
    if tuple(a_solved.shape) != (1, t, H_V, BT):
        raise ValueError("Stage 6U C0 requires a_solved=[1,T,8,64].")
    return t, _num_chunks(t, BT)


@avelang.jit
def _qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u(
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

    # W main only. The Stage 6T low-part residual phase is intentionally absent.
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
            if lane_group == 0:
                af = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b0[0], w_acc0); w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b1[0], w_acc1)
            if lane_group == 1:
                af = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b0[1], w_acc0); w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b1[1], w_acc1)
            if lane_group == 2:
                af = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b0[0], w_acc0); w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b1[0], w_acc1)
            if lane_group == 3:
                af = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b0[1], w_acc0); w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b1[1], w_acc1)
            al.syncthreads()
        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            token_idx = chunk_start + token_offset
            w[0, token_idx, value_head_idx, col_base + lane_col] = al.convert(w_acc0[r], al.bf16)
            w[0, token_idx, value_head_idx, col_base + BT_SUB + lane_col] = al.convert(w_acc1[r], al.bf16)
        al.syncthreads()

    # U main only. W has already been stored, preserving the F1 lifetime order.
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
            if lane_group == 0:
                af = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); u_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b0[0], u_acc0); u_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b1[0], u_acc1)
            if lane_group == 1:
                af = al.view(coeff_vec[row_base + lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16)); u_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b0[1], u_acc0); u_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b1[1], u_acc1)
            if lane_group == 2:
                af = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); u_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b0[0], u_acc0); u_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[0], b1[0], u_acc1)
            if lane_group == 3:
                af = al.view(coeff_vec[row_base + lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b0 = al.view(operand0_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); b1 = al.view(operand1_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16)); u_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b0[1], u_acc0); u_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(af[1], b1[1], u_acc1)
            al.syncthreads()
        for r in al.range(4):
            token_offset = row_base + lane_group * 4 + r
            token_idx = chunk_start + token_offset
            u[0, token_idx, value_head_idx, col_base + lane_col] = al.convert(u_acc0[r], al.bf16)
            u[0, token_idx, value_head_idx, col_base + BT_SUB + lane_col] = al.convert(u_acc1[r], al.bf16)
        al.syncthreads()


def qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_solved: torch.Tensor,
    *,
    chunk_size: int = BT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C0 diagnostic API; outputs BF16 W/U and never falls back to F1."""
    t, num_chunks = _validate_c0_inputs(k, v, g, beta, a_solved, chunk_size=chunk_size)
    w = torch.empty((1, t, H_V, K_DIM), dtype=torch.bfloat16, device=k.device)
    u = torch.empty_like(w)
    _qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u[
        lambda: ((num_chunks * H_V, 1, 1), (WORKGROUP, 1, 1))
    ](k, v, g, beta, a_solved, w, u, t, num_chunks)
    return w, u


def _full_stage6u_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    scale: float | None,
    native_solve_store: bool,
) -> dict[str, torch.Tensor]:
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    if native_solve_store:
        a_solved_bf16 = qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u(a)
    else:
        a_solved_bf16 = qwen_gdn_solve_hierarchical_bt64_casted_bf16_solved_stage6u(a)
    w_bf16, u_bf16 = qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u(k, v, g_cumsum, beta, a_solved_bf16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    v_new = v_new_bf16.to(torch.float32)
    output_fp32 = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved_bf16": a_solved_bf16,
        "w_bf16": w_bf16,
        "u_bf16": u_bf16,
        "h_bf16": h_bf16,
        "v_new_bf16": v_new_bf16,
        "v_new": v_new,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
    }


def qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager(
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
    """U0 complete eager public API with the explicit solved cast control."""
    stages = _full_stage6u_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, native_solve_store=False
    )
    return stages["output"], stages["final_state"] if output_final_state else None


def qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager(
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
    """U1 complete eager public API with native BF16 solved writeback."""
    stages = _full_stage6u_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, native_solve_store=True
    )
    return stages["output"], stages["final_state"] if output_final_state else None


__all__ = [
    "_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u",
    "qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u",
    "qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager",
    "qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager",
]

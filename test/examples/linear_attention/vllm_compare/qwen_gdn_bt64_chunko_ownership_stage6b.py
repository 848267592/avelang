"""Opt-in Stage 6B BT64 ``chunk_o`` ownership experiment for gfx942.

The Stage 4 kernel assigns one 256-thread CTA to a ``[token=64, value=16]``
output tile.  It consequently computes the same QK score tiles once per V16
CTA.  O0 assigns a CTA to ``[token=64, value=64]`` instead.  The four waves
continue to own token16 rows; they cooperatively build the ten causal score
tiles once, retain them in LDS, and reuse them across four V16 subtiles.

This module deliberately leaves output staging FP32 and keeps the standalone
FP32-to-BF16 cast in the caller.  It is opt-in and never changes Stage 4,
v24, recurrence asm, or a production selector.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0
from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    BT,
    BT_SUB,
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    _require_kkt_bt64_inputs,
    _solve_bt64_stage5c,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import _num_chunks, qwen_gdn_chunk_cumsum_avelang_v6_standalone
from qwen_gdn_full_bt64_gfx942_asm_v0_experimental import _require_target


V_TILE = 64
V_SUBTILES = V_TILE // BT_SUB
V_TILES = V_DIM // V_TILE
WORKGROUP = 256


@avelang.jit
def _qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.f32),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    q = al.make_tensor(q_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, H_K, K_DIM), (num_tokens * 512, 512, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.f32, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout((1, num_chunks, H_V, V_DIM, K_DIM), (num_chunks * H_V * 16384, H_V * 16384, 16384, 128, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((1, num_tokens, H_V, V_DIM), (num_tokens * 1024, 1024, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane = tid & 63
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_tile_idx = program_id % V_TILES
    value_head_idx = (program_id // V_TILES) % H_V
    chunk_idx = program_id // (V_TILES * H_V)
    row_base = wave_id * BT_SUB
    value_base = v_tile_idx * V_TILE
    chunk_start = chunk_idx * BT
    key_head_idx = value_head_idx // 2

    # Shared score storage has one 16x16 tile for each (source16, output16)
    # pair.  Only the lower triangular ten tiles are written and subsequently
    # consumed.  The 8 KiB allocation removes the V-tile score recomputation.
    q_scaled_bf16 = al.make_shared((BT, K_DIM), al.bf16)
    k_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    h_bf16 = al.make_shared((BT_SUB, K_DIM), al.bf16)
    score_decay_bf16 = al.make_shared((4, 4, BT_SUB, BT_SUB), al.bf16)
    vn_t_bf16 = al.make_shared((BT_SUB, BT_SUB), al.bf16)
    q_vec = al.view(q_scaled_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    k_vec = al.view(k_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    h_vec = al.view(h_bf16, al.i32, al.make_layout((BT_SUB, 16, 4), (64, 4, 1)))
    score_vec = al.view(score_decay_bf16, al.i32, al.make_layout((4, 4, BT_SUB, 2, 4), (512, 128, 8, 4, 1)))
    vn_vec = al.view(vn_t_bf16, al.i32, al.make_layout((BT_SUB, 2, 4), (8, 4, 1)))

    # Q is invariant across all four V16 subtiles handled by this CTA.
    for rep in al.range(32):
        idx = tid + rep * WORKGROUP
        row = idx // K_DIM
        col = idx - row * K_DIM
        q_scaled_bf16[row, col] = al.convert(
            al.convert(q[0, chunk_start + row, key_head_idx, col], al.f32) * scale,
            al.bf16,
        )
    al.syncthreads()

    # Build each QK/decay tile once.  The old V16 ownership runs this loop in
    # each of four CTAs for the same V64 range and also executes upper blocks
    # that are later masked to zero.
    for source_tile in al.range(4):
        source_base = source_tile * BT_SUB
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // K_DIM
            col = idx - row * K_DIM
            k_bf16[row, col] = k[0, chunk_start + source_base + row, key_head_idx, col]
        al.syncthreads()

        if source_tile <= wave_id:
            score_acc = al.full((4,), 0.0, al.f32)
            for batch128_score in al.range(4):
                vec_idx_score = lane_group + batch128_score * 4
                a_frag_score = al.view(
                    q_vec[row_base + lane_col, vec_idx_score], al.Tensor((2, 4, 1), al.bf16)
                )
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
                score_decay_bf16[source_tile, wave_id, token_offset, lane_col] = al.convert(score_value, al.bf16)
        al.syncthreads()

    # Each pass owns one V16 subrange.  It reuses Q and all causal score tiles
    # without materializing partial output globally.  Its accumulator only
    # lives for this one V16 output tile.
    for v_subtile in al.range(V_SUBTILES):
        value_sub_base = value_base + v_subtile * BT_SUB
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // K_DIM
            col = idx - row * K_DIM
            h_bf16[row, col] = h[0, chunk_idx, value_head_idx, value_sub_base + row, col]
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
            value_offset = tid // BT_SUB
            source_offset = tid - value_offset * BT_SUB
            vn_t_bf16[value_offset, source_offset] = al.convert(
                vn[0, chunk_start + source_base + source_offset, value_head_idx, value_sub_base + value_offset],
                al.bf16,
            )
            al.syncthreads()

            if source_tile <= wave_id:
                if lane_group == 0:
                    a_frag_intra = al.view(score_vec[source_tile, wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                    b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                    intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
                if lane_group == 1:
                    a_frag_intra = al.view(score_vec[source_tile, wave_id, lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                    b_frag_intra = al.view(vn_vec[lane_col, 0], al.Tensor((2, 4, 1), al.bf16))
                    intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
                if lane_group == 2:
                    a_frag_intra = al.view(score_vec[source_tile, wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                    b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                    intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[0], b_frag_intra[0], intra_acc)
                if lane_group == 3:
                    a_frag_intra = al.view(score_vec[source_tile, wave_id, lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                    b_frag_intra = al.view(vn_vec[lane_col, 1], al.Tensor((2, 4, 1), al.bf16))
                    intra_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_intra[1], b_frag_intra[1], intra_acc)
            al.syncthreads()

        for r_out in al.range(4):
            token_offset_out = row_base + lane_group * 4 + r_out
            token_idx_out = chunk_start + token_offset_out
            out[0, token_idx_out, value_head_idx, value_sub_base + lane_col] = (
                inter_acc[r_out] * al.exp(g[0, token_idx_out, value_head_idx]) + intra_acc[r_out]
            )
        al.syncthreads()


def _require_chunko_o0_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
    if chunk_size != BT:
        raise ValueError("Stage 6B chunk-o-O0 only supports chunk_size=64.")
    num_tokens = q.shape[1]
    num_chunks = _num_chunks(num_tokens, BT)
    for name, tensor, dtype in (
        ("q", q, torch.bfloat16),
        ("k", k, torch.bfloat16),
        ("v_new", v_new, torch.float32),
        ("h_bf16", h_bf16, torch.bfloat16),
        ("g", g, torch.float32),
    ):
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous() or tensor.device != q.device:
            raise ValueError(f"{name} must be contiguous CUDA/HIP {dtype} on {q.device}.")
    if num_tokens % BT:
        raise ValueError("Stage 6B chunk-o-O0 requires T divisible by 64.")
    if tuple(q.shape) != (1, num_tokens, H_K, K_DIM) or tuple(k.shape) != tuple(q.shape):
        raise ValueError("Stage 6B chunk-o-O0 requires q/k=[1,T,4,128].")
    if tuple(v_new.shape) != (1, num_tokens, H_V, V_DIM) or tuple(g.shape) != (1, num_tokens, H_V):
        raise ValueError("Stage 6B chunk-o-O0 requires v_new=[1,T,8,128] and g=[1,T,8].")
    if tuple(h_bf16.shape) != (1, num_chunks, H_V, V_DIM, K_DIM):
        raise ValueError("Stage 6B chunk-o-O0 requires h_bf16=[1,T/64,8,128,128].")
    return num_tokens, num_chunks


def qwen_gdn_chunk_o_bt64_ownership_o0(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    """FP32-staging O0 chunk-o with one CTA per ``[chunk, head, V64]``."""
    num_tokens, num_chunks = _require_chunko_o0_inputs(q, k, v_new, h_bf16, g, chunk_size)
    if scale is None:
        scale = K_DIM ** -0.5
    output = torch.empty_like(v_new)
    grid_size = num_chunks * H_V * V_TILES
    _qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0[lambda: ((grid_size, 1, 1), (WORKGROUP, 1, 1))](
        q, k, v_new, h_bf16, g, output, float(scale), num_tokens, num_chunks
    )
    return output


def qwen_gdn_full_bt64_stage6b_chunko_o0_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    solve_impl: str = "hierarchical_fp32_v1",
) -> dict[str, torch.Tensor]:
    """Opt-in Stage 6B graph; every stage except chunk-o is Stage 4."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = _solve_bt64_stage5c(a, solve_impl)
    w, u = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, h0)
    output_fp32 = qwen_gdn_chunk_o_bt64_ownership_o0(q, k, v_new, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w,
        "u": u,
        "h_bf16": h_bf16,
        "v_new": v_new,
        "output_fp32": output_fp32,
        "output": output_fp32.to(torch.bfloat16),
        "final_state": final_state,
    }


def qwen_gdn_full_bt64_stage6b_chunko_o0(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    solve_impl: str = "hierarchical_fp32_v1",
) -> tuple[torch.Tensor, torch.Tensor]:
    stages = qwen_gdn_full_bt64_stage6b_chunko_o0_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, solve_impl=solve_impl
    )
    return stages["output"], stages["final_state"]


__all__ = [
    "V_TILE",
    "WORKGROUP",
    "_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0",
    "qwen_gdn_chunk_o_bt64_ownership_o0",
    "qwen_gdn_full_bt64_stage6b_chunko_o0_stages",
    "qwen_gdn_full_bt64_stage6b_chunko_o0",
]

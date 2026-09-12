"""Stage 7A's one permitted local synchronization-scheduling experiment.

This is a copy of Stage 6Z Z1 with only two changes: lane-private
``frag_words[tid]`` pack/reuse barriers are removed, and the score-half store
barrier is deferred to the existing V-new producer->consumer barrier.  The
tile, CTA mapping, MFMA geometry, data types, and math are unchanged.
"""

from __future__ import annotations

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


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage7a_phase_compact(
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

    phase = al.make_shared((256, 32), al.bf16)
    phase_vec = al.view(phase, al.Tensor((256, 4, 4), al.i32))
    frag_words = al.make_shared((768, 4), al.i32)

    inter_acc = al.full((16,), 0.0, al.f32)
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[row, col] = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                al.bf16,
            )
            phase[64 + row, col] = h[0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col]
        # Required: Q/H values are distributed across the whole CTA.
        al.syncthreads()
        for kt in al.range(2):
            word = kt * 2 + lane_group
            frag_words[tid] = phase_vec[row_half * 32 + lane_col, word]
            frag_words[256 + tid] = phase_vec[64 + value_half * 32 + lane_col, word]
            # No barrier: this lane alone writes and reads its two fragment rows.
            q_words = frag_words[tid]
            h_words = frag_words[256 + tid]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
            # No barrier: the next kt overwrites only frag_words for this lane.

    for source_half in al.range(2):
        score_stage_base = source_half * 128
        score_acc = al.full((16,), 0.0, al.f32)
        for k_stage in al.range(4):
            for rep in al.range(8):
                idx = tid + rep * WORKGROUP
                row = idx // BK
                col = idx - row * BK
                phase[score_stage_base + row, col] = al.convert(
                    al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                    al.bf16,
                )
                phase[score_stage_base + 64 + row, col] = k[
                    0, chunk_start + source_half * 32 + row, key_head_idx, k_stage * BK + col
                ]
            # Required: owner waves consume Q/K values produced across CTA.
            al.syncthreads()
            for kt in al.range(2):
                word = kt * 2 + lane_group
                frag_words[tid] = phase_vec[score_stage_base + row_half * 32 + lane_col, word]
                frag_words[512 + tid] = phase_vec[score_stage_base + 64 + lane_col, word]
                # Lane-private fragment pack/reuse has no cross-lane consumer.
                if value_half == 0:
                    q_words = frag_words[tid]
                    k_words = frag_words[512 + tid]
                    q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                    k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc)
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc)

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
                phase[token_offset * 2 + source_half, out_col] = al.convert(score, al.bf16)
        # No barrier here: score lives in phase[0:128], while the next source
        # half and later V transpose use phase[128:256].  The barrier after
        # V staging below orders both score halves before their first consumer.

    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[128 + value_offset * 2 + token_offset // 32, token_offset - (token_offset // 32) * 32] = vn[
            0, chunk_start + token_offset, value_head_idx, value_base + value_offset
        ]
    # Required: all score/V producer stores are complete before intra MFMA.
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            frag_words[tid] = phase_vec[(row_half * 32 + lane_col) * 2 + source_half, word]
            frag_words[512 + tid] = phase_vec[128 + (value_half * 32 + lane_col) * 2 + source_half, word]
            # Lane-private fragment pack/reuse has no cross-lane consumer.
            score_words = frag_words[tid]
            v_words = frag_words[512 + tid]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        token_offset = row_half * 32 + lane_col
        result = inter_acc[r] * al.exp(g[0, chunk_start + token_offset, value_head_idx]) + intra_acc[r]
        out[0, chunk_start + token_offset, value_head_idx, value_base + value_half * 32 + out_col] = al.convert(
            result, al.bf16
        )


def qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact_launch_into(
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
    """Caller-owned diagnostic launch with no fallback."""
    t, num_chunks = _validate_stage6w_chunko_inputs(q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size)
    if (
        output_bf16.dtype != torch.bfloat16
        or tuple(output_bf16.shape) != tuple(v_new_bf16.shape)
        or not output_bf16.is_cuda
        or not output_bf16.is_contiguous()
        or output_bf16.device != q.device
    ):
        raise ValueError("Stage 7A output must be contiguous BF16 [1,T,8,128] on the input device.")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage7a_phase_compact[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    output_bf16 = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact_launch_into(
        q, k, v_new_bf16, h_bf16, g, output_bf16, scale=scale, chunk_size=chunk_size
    )
    return output_bf16


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage7a_phase_compact",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage7a_phase_compact_launch_into",
]

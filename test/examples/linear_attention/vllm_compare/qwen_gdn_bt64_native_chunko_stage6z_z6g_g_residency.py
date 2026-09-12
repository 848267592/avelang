"""Experimental Stage 6Z Z6G g-residency arms.

Both arms are direct forks of the frozen Z5B direct-Q-cache-consumer body.
They keep the Q/K/H/V-new/output graph, the phase-separated accumulators,
the BF16 ABI, and the WG256 launch contract unchanged.

Z6G-S uses a scalar FP32 shared g tile.  Z6G-I uses a legal buffer-load-x4
packet per token, selects the current value-head element, and places that
value in the same CTA-local g tile.  The two modes are separate constexpr
compilations of one source body; they are never enabled together.
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
Z6G_WORKGROUP_CONTRACT = 256
Z6G_STABLE = 0
Z6G_IDEAL = 1


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z6g(
    q_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    vn_ptr: al.Pointer(al.bf16),
    h_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    scale: al.constexpr,
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    g_mode: al.constexpr,
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

    # This is the frozen Z5B 32 KiB allocation.  Q cache rows [0, 256) and
    # the old phase rows [256, 512) remain disjoint.  The g tile is a small
    # additional FP32 shared object and does not alias either BF16 region.
    shared = al.make_shared((2 * 4 * BT, BK), al.bf16)
    q_cache = shared
    q_cache_vec = al.view(q_cache, al.Tensor((2 * 4 * BT, 4, 4), al.i32))
    g_cache = al.make_shared((BT,), al.f32)

    # The Q producer and all Q/K/H/V-new producers are copied from Z5B.
    # No g load is mixed into these loops.
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            q_cache[k_stage * BT + row, col] = al.convert(
                al.convert(q[0, chunk_start + row, key_head_idx, k_stage * BK + col], al.f32) * scale,
                al.bf16,
            )
        al.syncthreads()
    al.syncthreads()

    # Z6G-S: one scalar FP32 global producer per logical token.  Z6G-I:
    # one legal x4 packet per token, selecting the current value-head element
    # before publishing the same logical g tile.  Both arms have one producer
    # region and the same cache-to-consumer lifetime.
    if g_mode == Z6G_STABLE:
        if tid < BT:
            g_cache[tid] = g[0, chunk_start + tid, value_head_idx]
    else:
        g_rsrc = al.amdgpu.make_rsrc(g, num_tokens * H_V * 4)
        zero = al.convert(0, al.i32)
        if tid < BT:
            head_base = (value_head_idx >> 2) << 2
            head_in_packet = value_head_idx - head_base
            g_offset = ((chunk_start + tid) * H_V + head_base) * 4
            g_packet_words = al.amdgpu.raw_buffer_load_x4(
                g_rsrc, zero, al.convert(g_offset, al.i32), 0
            )
            g_packet = al.view(g_packet_words, al.Tensor((4,), al.f32))
            g_cache[tid] = g_packet[head_in_packet]
    al.syncthreads()

    phase = shared
    phase_vec = al.view(phase, al.Tensor((2 * 4 * BT, 4, 4), al.i32))

    # Preserve the Z5B/Z2 phase-separated accumulator order.
    inter_acc = al.full((16,), 0.0, al.f32)

    # Phase A: unchanged H producer and direct persistent Q-cache consumer.
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[4 * BT + 64 + row, col] = h[
                0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col
            ]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
            h_words = phase_vec[4 * BT + 64 + value_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
        al.syncthreads()

    # Keep one target-g scalar live from the cache through both score halves
    # and the final output scale.  It is only one value per lane, not a private
    # 64-element g array.
    token_offset = row_half * 32 + lane_col
    g_target = g_cache[token_offset]

    # Phase B: serialized score halves, unchanged K producer and MFMA order.
    for source_half in al.range(2):
        score_stage_base = source_half * 128
        score_acc = al.full((16,), 0.0, al.f32)
        for k_stage in al.range(4):
            for rep in al.range(4):
                idx = tid + rep * WORKGROUP
                row = idx // BK
                col = idx - row * BK
                phase[4 * BT + score_stage_base + 64 + row, col] = k[
                    0, chunk_start + source_half * 32 + row, key_head_idx, k_stage * BK + col
                ]
            al.syncthreads()

            for kt in al.range(2):
                word = kt * 2 + lane_group
                q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
                k_words = phase_vec[4 * BT + score_stage_base + 64 + lane_col, word]
                if value_half == 0:
                    q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                    k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score_acc)
                    score_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score_acc)
            al.syncthreads()

        if value_half == 0:
            for r in al.range(16):
                out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
                source_offset = source_half * 32 + out_col
                score = al.convert(0.0, al.f32)
                if source_offset <= token_offset:
                    score = score_acc[r] * al.exp(g_target - g_cache[source_offset])
                phase[4 * BT + token_offset * 2 + source_half, out_col] = al.convert(score, al.bf16)
        al.syncthreads()

    # Phase C: unchanged V-new producer, intra MFMA and BF16 output.
    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        local_token = idx - value_offset * BT
        phase[4 * BT + 128 + value_offset * 2 + local_token // 32,
              local_token - (local_token // 32) * 32] = vn[
            0, chunk_start + local_token, value_head_idx, value_base + value_offset
        ]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[4 * BT + (row_half * 32 + lane_col) * 2 + source_half, word]
            v_words = phase_vec[4 * BT + 128 + (value_half * 32 + lane_col) * 2 + source_half, word]
            score_frag = al.view(score_words, al.Tensor((2, 4, 1), al.bf16))
            v_frag = al.view(v_words, al.Tensor((2, 4, 1), al.bf16))
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[0], score_frag[0], intra_acc)
            intra_acc = al.amdgpu.mfma_32x32x8_bf16_f32(v_frag[1], score_frag[1], intra_acc)

    for r in al.range(16):
        out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
        result = inter_acc[r] * al.exp(g_target) + intra_acc[r]
        out[0, chunk_start + token_offset, value_head_idx,
            value_base + value_half * 32 + out_col] = al.convert(result, al.bf16)


def _validate_z6g_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    chunk_size: int,
) -> tuple[int, int]:
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
        raise ValueError("Stage 6Z Z6G output must be contiguous BF16 [1,T,8,128].")
    return t, num_chunks


def _launch_z6g(
    mode: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    output_bf16: torch.Tensor,
    *,
    scale: float | None,
    chunk_size: int,
) -> None:
    if WORKGROUP != Z6G_WORKGROUP_CONTRACT:
        raise RuntimeError("Stage 6Z Z6G shape contract violation: WG must remain 256")
    t, num_chunks = _validate_z6g_inputs(q, k, v_new_bf16, h_bf16, g, output_bf16, chunk_size)
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z6g[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks, mode)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into(
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
    _launch_z6g(
        Z6G_STABLE, q, k, v_new_bf16, h_bf16, g, output_bf16,
        scale=scale, chunk_size=chunk_size,
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into(
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
    _launch_z6g(
        Z6G_IDEAL, q, k, v_new_bf16, h_bf16, g, output_bf16,
        scale=scale, chunk_size=chunk_size,
    )


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    output = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into(
        q, k, v_new_bf16, h_bf16, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new_bf16: torch.Tensor,
    h_bf16: torch.Tensor,
    g: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = BT,
) -> torch.Tensor:
    output = torch.empty_like(v_new_bf16)
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into(
        q, k, v_new_bf16, h_bf16, g, output, scale=scale, chunk_size=chunk_size
    )
    return output


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z6g",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_stable_launch_into",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z6g_ideal_launch_into",
]

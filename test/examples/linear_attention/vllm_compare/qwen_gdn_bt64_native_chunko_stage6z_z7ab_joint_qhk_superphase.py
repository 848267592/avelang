"""Experimental Stage 6Z Z7AB native-shaped Q/H/K A+B joint superphase.

Z7AB is deliberately a single structural arm forked from Z5B.  Q retains the
full persistent LDS cache, while one legal score source-half is paired with
the Q@H phase at each K32 stage.  H and K0 are fetched as BF16x8 packets,
placed into disjoint existing phase slots, and their next packet is issued
before the current stage's phase-buffer release barrier.

The second score source-half remains serialized after the superphase.  This
keeps only ``inter_acc`` and one score accumulator live in the joint loop;
making both score halves live would recreate the rejected Z4C accumulator
overlap experiment.
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
Z7AB_WORKGROUP_CONTRACT = 256
Q_CACHE_ROWS = 4 * BT

# The upper 16 KiB was already phase storage in Z5B.  Z7AB only assigns two
# disjoint, transient slots during A+B0 before score/V-new take ownership.
H_STAGE_BASE = Q_CACHE_ROWS + 64
K0_STAGE_BASE = Q_CACHE_ROWS + 128
K1_STAGE_BASE = Q_CACHE_ROWS + 192


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7ab_joint_qhk_superphase(
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

    # Z5B's 32 KiB allocation remains unchanged.  Q uses rows [0, 256),
    # while the upper half carries transient H/K/score/V-new phase data.
    shared = al.make_shared((2 * Q_CACHE_ROWS, BK), al.bf16)
    q_cache = shared
    q_cache_vec = al.view(q_cache, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))
    phase = shared
    phase_vec = al.view(phase, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))

    # Z5B's sole logical Q global producer is intentionally unchanged.
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

    # Each lane owns one contiguous BF16x8 H packet.  The first half of the
    # CTA owns the matching 32x32 K0 packet.  K0 uses the two otherwise-free
    # 2 KiB phase slots as the minimal ownership-safe ping-pong.  H is issued
    # per current stage: the current AST generator does not preserve a
    # loop-carried raw packet SSA value with the required numerical mapping.
    zero_i32 = al.convert(0, al.i32)
    h_rsrc = al.amdgpu.make_rsrc(h, num_chunks * H_V * V_DIM * K_DIM * 2)
    k_rsrc = al.amdgpu.make_rsrc(k, num_tokens * H_K * K_DIM * 2)
    h_row = tid // 4
    h_col_base = (tid - h_row * 4) * 8
    k_row = tid // 4
    k_col_base = (tid - k_row * 4) * 8

    if tid < 128:
        k0_offset0 = (
            ((chunk_start + k_row) * H_K * K_DIM + key_head_idx * K_DIM + k_col_base) * 2
        )
        k0_initial_words = al.amdgpu.raw_buffer_load_x4(k_rsrc, zero_i32, al.convert(k0_offset0, al.i32), 0)
        k0_initial_packet = al.view(k0_initial_words, al.Tensor((8,), al.bf16))
        for element in al.range(8):
            phase[K0_STAGE_BASE + k_row, k_col_base + element] = k0_initial_packet[element]

    inter_acc = al.full((16,), 0.0, al.f32)
    score0_acc = al.full((16,), 0.0, al.f32)

    # Joint A+B0 superphase.  One q_words/q_frag value is consumed by Q@H
    # and Q@K0 in the same K32 stage; it is not a source-level alias only.
    for k_stage in al.range(4):
        h_current_offset = (
            (((chunk_idx * H_V + value_head_idx) * V_DIM + value_base + h_row) * K_DIM
             + k_stage * BK + h_col_base) * 2
        )
        h_current_words = al.amdgpu.raw_buffer_load_x4(
            h_rsrc, zero_i32, al.convert(h_current_offset, al.i32), 0
        )
        h_packet = al.view(h_current_words, al.Tensor((8,), al.bf16))
        for element in al.range(8):
            phase[H_STAGE_BASE + h_row, h_col_base + element] = h_packet[element]
        al.syncthreads()

        stage_bank = k_stage - (k_stage // 2) * 2
        k0_read_base = K0_STAGE_BASE + stage_bank * BK
        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_words = phase_vec[H_STAGE_BASE + value_half * 32 + lane_col, word]
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
            if value_half == 0:
                k0_words = phase_vec[k0_read_base + lane_col, word]
                k0_frag = al.view(k0_words, al.Tensor((2, 4, 1), al.bf16))
                score0_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k0_frag[0], q_frag[0], score0_acc)
                score0_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k0_frag[1], q_frag[1], score0_acc)

        # K0 next packet is issued before the release barrier.  Current
        # consumers have no dependency on the next K0 values; the next stage
        # commits it only after the barrier.
        if k_stage < 3:
            next_stage = k_stage + 1
            if tid < 128:
                k0_next_offset = (
                    ((chunk_start + k_row) * H_K * K_DIM + key_head_idx * K_DIM
                     + next_stage * BK + k_col_base) * 2
                )
                k0_prefetch_words = al.amdgpu.raw_buffer_load_x4(
                    k_rsrc, zero_i32, al.convert(k0_next_offset, al.i32), 0
                )
                k0_prefetch_packet = al.view(k0_prefetch_words, al.Tensor((8,), al.bf16))
                next_bank = next_stage - (next_stage // 2) * 2
                k0_write_base = K0_STAGE_BASE + next_bank * BK
                for element in al.range(8):
                    phase[k0_write_base + k_row, k_col_base + element] = k0_prefetch_packet[element]
        al.syncthreads()

    # score half 0 completes after the shared A+B superphase.  H slots are
    # dead now, so publishing score to the original rows keeps the LDS size.
    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score0_acc[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[Q_CACHE_ROWS + token_offset * 2, out_col] = al.convert(score, al.bf16)
    al.syncthreads()

    # B1 preserves Z5B's separate accumulator lifetime.  It uses the same
    # BF16x8 packet ownership as K0 but does not keep a second score
    # accumulator live during the A+B0 superphase.
    score1_acc = al.full((16,), 0.0, al.f32)
    for k_stage in al.range(4):
        if tid < 128:
            k1_offset = (
                ((chunk_start + 32 + k_row) * H_K * K_DIM + key_head_idx * K_DIM
                 + k_stage * BK + k_col_base) * 2
            )
            k1_words = al.amdgpu.raw_buffer_load_x4(k_rsrc, zero_i32, al.convert(k1_offset, al.i32), 0)
            k1_packet = al.view(k1_words, al.Tensor((8,), al.bf16))
            for element in al.range(8):
                phase[K1_STAGE_BASE + k_row, k_col_base + element] = k1_packet[element]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            if value_half == 0:
                k1_words = phase_vec[K1_STAGE_BASE + lane_col, word]
                k1_frag = al.view(k1_words, al.Tensor((2, 4, 1), al.bf16))
                score1_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k1_frag[0], q_frag[0], score1_acc)
                score1_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k1_frag[1], q_frag[1], score1_acc)
        al.syncthreads()

    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            source_offset = 32 + out_col
            score = al.convert(0.0, al.f32)
            if source_offset <= token_offset:
                score = score1_acc[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + source_offset, value_head_idx]
                )
            phase[Q_CACHE_ROWS + token_offset * 2 + 1, out_col] = al.convert(score, al.bf16)
    al.syncthreads()

    # C/D/E are byte-for-byte the Z5B source schedule.
    for rep in al.range(16):
        idx = tid + rep * WORKGROUP
        value_offset = idx // BT
        token_offset = idx - value_offset * BT
        phase[Q_CACHE_ROWS + 128 + value_offset * 2 + token_offset // 32,
              token_offset - (token_offset // 32) * 32] = vn[
            0, chunk_start + token_offset, value_head_idx, value_base + value_offset
        ]
    al.syncthreads()

    intra_acc = al.full((16,), 0.0, al.f32)
    for source_half in al.range(2):
        for kt in al.range(2):
            word = kt * 2 + lane_group
            score_words = phase_vec[Q_CACHE_ROWS + (row_half * 32 + lane_col) * 2 + source_half, word]
            v_words = phase_vec[Q_CACHE_ROWS + 128 + (value_half * 32 + lane_col) * 2 + source_half, word]
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


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase_launch_into(
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
    """Caller-owned Z7AB launch with the fixed WG256/no-fallback contract."""
    if WORKGROUP != Z7AB_WORKGROUP_CONTRACT:
        raise RuntimeError("Stage 6Z Z7AB shape contract violation: WG must remain 256")
    t, num_chunks = _validate_stage6w_chunko_inputs(q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size)
    if (
        output_bf16.dtype != torch.bfloat16
        or tuple(output_bf16.shape) != tuple(v_new_bf16.shape)
        or not output_bf16.is_cuda
        or not output_bf16.is_contiguous()
        or output_bf16.device != q.device
    ):
        raise ValueError("Stage 6Z Z7AB output must be contiguous BF16 [1,T,8,128] on the input device.")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7ab_joint_qhk_superphase[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase(
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
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase_launch_into(
        q, k, v_new_bf16, h_bf16, g, output_bf16, scale=scale, chunk_size=chunk_size
    )
    return output_bf16


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7ab_joint_qhk_superphase",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7ab_joint_qhk_superphase_launch_into",
]

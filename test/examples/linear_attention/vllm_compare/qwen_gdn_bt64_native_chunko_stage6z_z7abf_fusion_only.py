"""Stage 6Z Z7AB-F: A+B0 Q-fragment fusion without producer pipelining.

This control starts from the correct Z5B direct-Q-cache kernel.  It keeps one
physical Q fragment live for its adjacent Q@H and Q@K0 consumers, but restores
Z5B's scalar current-stage H/K producers.  In particular it contains no typed
packet producer, no K0 lookahead, no alternate bank, and no pipeline state.
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
Z7ABF_WORKGROUP_CONTRACT = 256
Q_CACHE_ROWS = 4 * BT


@avelang.jit
def _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7abf_fusion_only(
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

    # One 32 KiB allocation: Q cache rows [0, 256), original phase rows
    # [256, 512).  The complete view is required by AveLang memref.cast;
    # indexing keeps the two logical regions disjoint.
    shared = al.make_shared((2 * Q_CACHE_ROWS, BK), al.bf16)
    q_cache = shared
    q_cache_vec = al.view(q_cache, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))

    # The only global Q producer.  Scaling and BF16 rounding match Z5A/Z2.
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

    phase = shared
    phase_vec = al.view(phase, al.Tensor((2 * Q_CACHE_ROWS, 4, 4), al.i32))

    # Phase A+B0 is the only new group.  H and K0 occupy disjoint rows in
    # Z5B's existing upper-half phase storage; no new LDS bank is allocated.
    inter_acc = al.full((16,), 0.0, al.f32)
    score0_acc = al.full((16,), 0.0, al.f32)

    # Restore Z5B's scalar H/K producer ownership and address expressions.
    # The K0 rows [384, 416) are otherwise transient Z5B phase space and are
    # dead before C begins.  They avoid H/K aliasing while preserving 32 KiB.
    for k_stage in al.range(4):
        for rep in al.range(8):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[Q_CACHE_ROWS + 64 + row, col] = h[
                0, chunk_idx, value_head_idx, value_base + row, k_stage * BK + col
            ]
        # Keep the two independent scalar producer regions in distinct shared
        # phases.  Without this boundary the current lowering may merge their
        # stores despite disjoint logical rows, which failed the first F gate.
        al.syncthreads()
        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[Q_CACHE_ROWS + 128 + row, col] = k[
                0, chunk_start + row, key_head_idx, k_stage * BK + col
            ]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
            h_words = phase_vec[Q_CACHE_ROWS + 64 + value_half * 32 + lane_col, word]
            q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
            h_frag = al.view(h_words, al.Tensor((2, 4, 1), al.bf16))
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[0], q_frag[0], inter_acc)
            inter_acc = al.amdgpu.mfma_32x32x8_bf16_f32(h_frag[1], q_frag[1], inter_acc)
            if value_half == 0:
                k0_words = phase_vec[Q_CACHE_ROWS + 128 + lane_col, word]
                k0_frag = al.view(k0_words, al.Tensor((2, 4, 1), al.bf16))
                score0_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k0_frag[0], q_frag[0], score0_acc)
                score0_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k0_frag[1], q_frag[1], score0_acc)
        al.syncthreads()

    # Complete B0 after the fused consumer group, preserving its K32 order.
    if value_half == 0:
        for r in al.range(16):
            out_col = ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            token_offset = row_half * 32 + lane_col
            score = al.convert(0.0, al.f32)
            if out_col <= token_offset:
                score = score0_acc[r] * al.exp(
                    g[0, chunk_start + token_offset, value_head_idx]
                    - g[0, chunk_start + out_col, value_head_idx]
                )
            phase[Q_CACHE_ROWS + token_offset * 2, out_col] = al.convert(score, al.bf16)
        al.syncthreads()

    # B1 is restored byte-for-byte in structure to Z5B.  It creates only the
    # second score accumulator after the fused A+B0 group has ended.
    score1_acc = al.full((16,), 0.0, al.f32)
    for k_stage in al.range(4):
        for rep in al.range(4):
            idx = tid + rep * WORKGROUP
            row = idx // BK
            col = idx - row * BK
            phase[Q_CACHE_ROWS + 192 + row, col] = k[
                0, chunk_start + 32 + row, key_head_idx, k_stage * BK + col
            ]
        al.syncthreads()

        for kt in al.range(2):
            word = kt * 2 + lane_group
            q_words = q_cache_vec[k_stage * BT + row_half * 32 + lane_col, word]
            k_words = phase_vec[Q_CACHE_ROWS + 192 + lane_col, word]
            if value_half == 0:
                q_frag = al.view(q_words, al.Tensor((2, 4, 1), al.bf16))
                k_frag = al.view(k_words, al.Tensor((2, 4, 1), al.bf16))
                score1_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[0], q_frag[0], score1_acc)
                score1_acc = al.amdgpu.mfma_32x32x8_bf16_f32(k_frag[1], q_frag[1], score1_acc)
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

    # Phase C is unchanged from Z5A: V-new, score/V MFMA and BF16 output.
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


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only_launch_into(
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
    """Caller-owned Z7AB-F launch with a hard WG256/no-fallback contract."""
    if WORKGROUP != Z7ABF_WORKGROUP_CONTRACT:
        raise RuntimeError("Stage 6Z Z7AB-F shape contract violation: WG must remain 256")
    t, num_chunks = _validate_stage6w_chunko_inputs(q, k, v_new_bf16, h_bf16, g, chunk_size=chunk_size)
    if (
        output_bf16.dtype != torch.bfloat16
        or tuple(output_bf16.shape) != tuple(v_new_bf16.shape)
        or not output_bf16.is_cuda
        or not output_bf16.is_contiguous()
        or output_bf16.device != q.device
    ):
        raise ValueError("Stage 6Z Z7AB-F output must be contiguous BF16 [1,T,8,128] on the input device.")
    if scale is None:
        scale = K_DIM ** -0.5
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7abf_fusion_only[
        lambda: ((num_chunks * H_V * 2, 1, 1), (WORKGROUP, 1, 1))
    ](q, k, v_new_bf16, h_bf16, g, output_bf16, float(scale), t, num_chunks)


def qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only(
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
    qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only_launch_into(
        q, k, v_new_bf16, h_bf16, g, output_bf16, scale=scale, chunk_size=chunk_size
    )
    return output_bf16


__all__ = [
    "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z7abf_fusion_only",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only",
    "qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z7abf_fusion_only_launch_into",
]

"""Minimal MFMA state-operand backend repro for the v26 performance issue.

This is intentionally not a Qwen GDN implementation.  It isolates one backend
question:

    What is the cost of keeping state in persistent accumulator/register
    fragments, then converting/reusing that state as the B operand of a later
    BF16 MFMA?

Shape:
    BT=64, Kq=32, NT=32, BV in {32,16}

Variants:
    persistent  : state lives in accumulator/register fragments; pred stages
                  those fragments to BF16 LDS before using them as MFMA B.
    lds         : state lives in FP32 LDS; pred stages LDS FP32 to BF16 LDS.
    constant    : state lives in accumulator/register fragments but pred uses a
                  constant BF16 operand, avoiding state->B operand conversion.
    update_only : state lives in accumulator/register fragments; skips pred.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al

BT = 64
KQ = 32
NT = 32
WORKGROUP = 64

VARIANT_PERSISTENT = "persistent"
VARIANT_LDS = "lds"
VARIANT_CONSTANT = "constant"
VARIANT_UPDATE_ONLY = "update_only"
VARIANTS = (VARIANT_PERSISTENT, VARIANT_LDS, VARIANT_CONSTANT, VARIANT_UPDATE_ONLY)


@avelang.jit
def _repro_v26_mfma_state_operand_kernel(
    w_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    fake_v_ptr: al.Pointer(al.bf16),
    out_ptr: al.Pointer(al.f32),
    bv: al.constexpr,
    num_blocks: al.constexpr,
    use_lds_state: al.constexpr,
    constant_state_operand: al.constexpr,
    do_pred: al.constexpr,
):
    w = al.make_tensor(w_ptr, al.bf16, al.make_layout((NT, BT, KQ), (BT * KQ, KQ, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((NT, BT, KQ), (BT * KQ, KQ, 1)))
    fake_v = al.make_tensor(fake_v_ptr, al.bf16, al.make_layout((NT, BT, bv), (BT * bv, bv, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((num_blocks, bv, KQ), (bv * KQ, KQ, 1)))

    lane = al.thread_id(0)
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)

    # Persistent accumulator/register state for BV=32:
    # s00/s01: V 0:16, K 0:16 / 16:32
    # s10/s11: V 16:32, K 0:16 / 16:32
    s00 = al.full((4,), 0.0, al.f32)
    s01 = al.full((4,), 0.0, al.f32)
    s10 = al.full((4,), 0.0, al.f32)
    s11 = al.full((4,), 0.0, al.f32)

    state_f32 = al.make_shared((32, KQ), al.f32)
    state_bf16 = al.make_shared((32, KQ), al.bf16)
    w_tile_bf16 = al.make_shared((16, KQ), al.bf16)
    vnew_t = al.make_shared((32, BT), al.bf16)
    k_all_t = al.make_shared((KQ, BT), al.bf16)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((32, 4, 4), (16, 4, 1)))
    w_vec = al.view(w_tile_bf16, al.i32, al.make_layout((16, 4, 4), (16, 4, 1)))
    vnew_vec = al.view(vnew_t, al.i32, al.make_layout((32, 8, 4), (32, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((KQ, 8, 4), (32, 4, 1)))

    if use_lds_state:
        for rep_init in al.range(16):
            idx_init = lane + rep_init * 64
            if idx_init < bv * KQ:
                row_init = idx_init // KQ
                col_init = idx_init - row_init * KQ
                state_f32[row_init, col_init] = al.convert(0.0, al.f32)

    if constant_state_operand and do_pred:
        for rep_const in al.range(16):
            idx_const = lane + rep_const * 64
            if idx_const < bv * KQ:
                row_const = idx_const // KQ
                col_const = idx_const - row_const * KQ
                state_bf16[row_const, col_const] = al.convert(0.125, al.bf16)

    al.syncthreads()

    for chunk_idx in al.range(NT):
        if do_pred and not constant_state_operand:
            if use_lds_state:
                for rep_stage in al.range(16):
                    idx_stage = lane + rep_stage * 64
                    if idx_stage < bv * KQ:
                        row_stage = idx_stage // KQ
                        col_stage = idx_stage - row_stage * KQ
                        state_bf16[row_stage, col_stage] = al.convert(state_f32[row_stage, col_stage], al.bf16)
            else:
                for r_stage in al.range(4):
                    row0 = lane_group * 4 + r_stage
                    row1 = 16 + row0
                    col0 = lane_col
                    col1 = 16 + lane_col
                    state_bf16[row0, col0] = al.convert(s00[r_stage], al.bf16)
                    state_bf16[row0, col1] = al.convert(s01[r_stage], al.bf16)
                    if bv == 32:
                        state_bf16[row1, col0] = al.convert(s10[r_stage], al.bf16)
                        state_bf16[row1, col1] = al.convert(s11[r_stage], al.bf16)

            al.syncthreads()

        for rep_k in al.range(32):
            idx_k = lane + rep_k * 64
            col_k = idx_k // BT
            tok_k = idx_k - col_k * BT
            k_all_t[col_k, tok_k] = k[chunk_idx, tok_k, col_k]

        al.syncthreads()

        if do_pred:
            for token_tile in al.range(4):
                token_base = token_tile * 16

                for rep_w in al.range(8):
                    idx_w = lane + rep_w * 64
                    row_w = idx_w // KQ
                    col_w = idx_w - row_w * KQ
                    w_tile_bf16[row_w, col_w] = w[chunk_idx, token_base + row_w, col_w]

                al.syncthreads()

                for value_tile in al.range(2):
                    if value_tile * 16 < bv:
                        pred_acc = al.full((4,), 0.0, al.f32)
                        k_vec32 = lane_group
                        a_words = w_vec[lane_col, k_vec32]
                        b_words = state_vec[value_tile * 16 + lane_col, k_vec32]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
                        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

                        for r_pred in al.range(4):
                            token_offset = lane_group * 4 + r_pred
                            value_offset = lane_col
                            local_v = value_tile * 16 + value_offset
                            v_fake = al.convert(fake_v[chunk_idx, token_base + token_offset, local_v], al.f32)
                            vnew_t[local_v, token_base + token_offset] = al.convert(v_fake - pred_acc[r_pred], al.bf16)

                al.syncthreads()
        else:
            for rep_v in al.range(32):
                idx_v = lane + rep_v * 64
                if idx_v < bv * BT:
                    local_v_v = idx_v // BT
                    tok_v = idx_v - local_v_v * BT
                    vnew_t[local_v_v, tok_v] = fake_v[chunk_idx, tok_v, local_v_v]

            al.syncthreads()

        for value_tile_u in al.range(2):
            if value_tile_u * 16 < bv:
                for local_tile in al.range(2):
                    base_k = local_tile * 16
                    acc = al.full((4,), 0.0, al.f32)
                    for token_pack_base in al.range(4):
                        pack0 = token_pack_base * 2
                        pack1 = pack0 + 1
                        if lane_group == 0:
                            a_words_u = vnew_vec[value_tile_u * 16 + lane_col, pack0]
                            b_words_u = kall_vec[base_k + lane_col, pack0]
                            a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                        if lane_group == 1:
                            a_words_u = vnew_vec[value_tile_u * 16 + lane_col, pack0]
                            b_words_u = kall_vec[base_k + lane_col, pack0]
                            a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
                        if lane_group == 2:
                            a_words_u = vnew_vec[value_tile_u * 16 + lane_col, pack1]
                            b_words_u = kall_vec[base_k + lane_col, pack1]
                            a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
                        if lane_group == 3:
                            a_words_u = vnew_vec[value_tile_u * 16 + lane_col, pack1]
                            b_words_u = kall_vec[base_k + lane_col, pack1]
                            a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                            b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                            acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

                    out_col = base_k + lane_col
                    for r_up in al.range(4):
                        vv = value_tile_u * 16 + lane_group * 4 + r_up
                        if use_lds_state:
                            state_f32[vv, out_col] = state_f32[vv, out_col] + acc[r_up]
                        else:
                            if value_tile_u == 0 and local_tile == 0:
                                s00[r_up] = s00[r_up] + acc[r_up]
                            if value_tile_u == 0 and local_tile == 1:
                                s01[r_up] = s01[r_up] + acc[r_up]
                            if value_tile_u == 1 and local_tile == 0:
                                s10[r_up] = s10[r_up] + acc[r_up]
                            if value_tile_u == 1 and local_tile == 1:
                                s11[r_up] = s11[r_up] + acc[r_up]

        al.syncthreads()

    if use_lds_state:
        for rep_out in al.range(16):
            idx_out = lane + rep_out * 64
            if idx_out < bv * KQ:
                row_out = idx_out // KQ
                col_out = idx_out - row_out * KQ
                out[program_id, row_out, col_out] = state_f32[row_out, col_out]
    else:
        for r_out in al.range(4):
            row0_o = lane_group * 4 + r_out
            row1_o = 16 + row0_o
            col0_o = lane_col
            col1_o = 16 + lane_col
            out[program_id, row0_o, col0_o] = s00[r_out]
            out[program_id, row0_o, col1_o] = s01[r_out]
            if bv == 32:
                out[program_id, row1_o, col0_o] = s10[r_out]
                out[program_id, row1_o, col1_o] = s11[r_out]


def _variant_flags(variant: str) -> tuple[bool, bool, bool]:
    if variant == VARIANT_PERSISTENT:
        return False, False, True
    if variant == VARIANT_LDS:
        return True, False, True
    if variant == VARIANT_CONSTANT:
        return False, True, True
    if variant == VARIANT_UPDATE_ONLY:
        return False, False, False
    raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")


def _validate_inputs(w: torch.Tensor, k: torch.Tensor, fake_v: torch.Tensor, *, bv: int, num_blocks: int) -> None:
    if bv not in (16, 32):
        raise ValueError("repro supports bv=16 or bv=32 only.")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if w.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or fake_v.dtype != torch.bfloat16:
        raise ValueError("w/k/fake_v must be torch.bfloat16.")
    for name, tensor in (("w", w), ("k", k), ("fake_v", fake_v)):
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on CUDA/HIP.")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
        if tensor.device != w.device:
            raise ValueError(f"{name} must be on device {w.device}, got {tensor.device}.")
    if tuple(w.shape) != (NT, BT, KQ):
        raise ValueError(f"w must have shape {(NT, BT, KQ)}, got {tuple(w.shape)}.")
    if tuple(k.shape) != (NT, BT, KQ):
        raise ValueError(f"k must have shape {(NT, BT, KQ)}, got {tuple(k.shape)}.")
    if tuple(fake_v.shape) != (NT, BT, bv):
        raise ValueError(f"fake_v must have shape {(NT, BT, bv)}, got {tuple(fake_v.shape)}.")


def run_repro_v26_mfma_state_operand(
    w: torch.Tensor,
    k: torch.Tensor,
    fake_v: torch.Tensor,
    *,
    bv: int,
    variant: str,
    num_blocks: int = 64,
) -> torch.Tensor:
    _validate_inputs(w, k, fake_v, bv=bv, num_blocks=num_blocks)
    use_lds_state, constant_state_operand, do_pred = _variant_flags(variant)
    out = torch.empty((num_blocks, bv, KQ), dtype=torch.float32, device=w.device)
    _repro_v26_mfma_state_operand_kernel[lambda: ((num_blocks, 1, 1), (WORKGROUP, 1, 1))](
        w,
        k,
        fake_v,
        out,
        bv,
        num_blocks,
        use_lds_state,
        constant_state_operand,
        do_pred,
    )
    return out


__all__ = [
    "BT",
    "KQ",
    "NT",
    "VARIANT_PERSISTENT",
    "VARIANT_LDS",
    "VARIANT_CONSTANT",
    "VARIANT_UPDATE_ONLY",
    "VARIANTS",
    "_repro_v26_mfma_state_operand_kernel",
    "run_repro_v26_mfma_state_operand",
]

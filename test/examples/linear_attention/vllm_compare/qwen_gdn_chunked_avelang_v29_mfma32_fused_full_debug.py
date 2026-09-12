"""Debug probes for v29 MFMA32 fused chunk_gdr recurrence.

This file is intentionally narrow: one CTA debugs value_head=0 and V block
0:32 for T=64 or T=128.  It does not modify production v23/v24/v29 files.
"""

from __future__ import annotations

import argparse
from typing import Callable

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (
    BT,
    BV,
    WORKGROUP,
    _make_inputs,
    qwen_gdn_gdr_decay_bt64_reference,
)


MODE_NORMAL = 0
MODE_FEEDBACK_DISABLED = 1
MODE_DECAY_OFF = 2


@avelang.jit
def _qwen_gdn_v29_fused_full_debug_kernel(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    pred_ptr: al.Pointer(al.f32),
    ucorr_ptr: al.Pointer(al.f32),
    vdecay_ptr: al.Pointer(al.f32),
    delta_ptr: al.Pointer(al.f32),
    state_after_ptr: al.Pointer(al.f32),
    state_written_ptr: al.Pointer(al.f32),
    state_readback_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    mode: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))
    u_flat = al.make_tensor(u_ptr, al.f32, al.make_layout((num_tokens * 8 * 128,), (1,)))
    gdr_decay = al.make_tensor(gdr_decay_ptr, al.f32, al.make_layout((1, num_chunks, 8, 64), (num_chunks * 8 * 64, 8 * 64, 64, 1)))
    gdr_g_last_exp = al.make_tensor(gdr_g_last_exp_ptr, al.f32, al.make_layout((1, num_chunks, 8), (num_chunks * 8, 8, 1)))
    initial_state = al.make_tensor(initial_state_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    pred_out = al.make_tensor(pred_ptr, al.f32, al.make_layout((2, 64, 32), (64 * 32, 32, 1)))
    ucorr_out = al.make_tensor(ucorr_ptr, al.f32, al.make_layout((2, 64, 32), (64 * 32, 32, 1)))
    vdecay_out = al.make_tensor(vdecay_ptr, al.f32, al.make_layout((2, 32, 64), (32 * 64, 64, 1)))
    delta_out = al.make_tensor(delta_ptr, al.f32, al.make_layout((2, 32, 128), (32 * 128, 128, 1)))
    state_after_out = al.make_tensor(state_after_ptr, al.f32, al.make_layout((2, 32, 128), (32 * 128, 128, 1)))
    state_written_out = al.make_tensor(state_written_ptr, al.f32, al.make_layout((32, 128), (128, 1)))
    state_readback_out = al.make_tensor(state_readback_ptr, al.f32, al.make_layout((32, 128), (128, 1)))

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_mod32 = lane & 31
    lane_col = lane & 15
    lane_group = lane >> 4

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((2, BV, 64), al.bf16)
    w_bf16 = al.make_shared((2, 32, 64), al.bf16)
    pred_partial = al.make_shared((2, 32, BV), al.f32)
    v_decay_t_bf16 = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((2, BV, 8, 4), (BV * 8 * 4, 8 * 4, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((2, 32, 8, 4), (32 * 8 * 4, 8 * 4, 4, 1)))
    vdecay_vec = al.view(v_decay_t_bf16, al.i32, al.make_layout((BV, 8, 4), (8 * 4, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))

    for rep_init in al.range(32):
        linear_init = tid + rep_init * WORKGROUP
        vv = linear_init // 128
        kk = linear_init - vv * 128
        val = initial_state[0, 0, vv, kk]
        state[vv, kk] = val

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last_exp = gdr_g_last_exp[0, chunk_idx, 0]
        if mode == MODE_DECAY_OFF:
            g_last_exp = al.convert(1.0, al.f32)

        if mode == MODE_FEEDBACK_DISABLED:
            for rep_reset in al.range(32):
                linear_reset = tid + rep_reset * WORKGROUP
                vv_reset = linear_reset // 128
                kk_reset = linear_reset - vv_reset * 128
                state[vv_reset, kk_reset] = initial_state[0, 0, vv_reset, kk_reset]
            al.syncthreads()

        for rep_state in al.range(32):
            linear_s = tid + rep_state * WORKGROUP
            kb_s = linear_s // (BV * 64)
            rem_s = linear_s - kb_s * (BV * 64)
            row_s = rem_s // 64
            col_s = rem_s - row_s * 64
            state_bf16[kb_s, row_s, col_s] = al.convert(state[row_s, kb_s * 64 + col_s], al.bf16)
            if chunk_idx == 1:
                state_written_out[row_s, kb_s * 64 + col_s] = state[row_s, kb_s * 64 + col_s]

        al.syncthreads()

        if chunk_idx == 1:
            for rep_rb in al.range(32):
                linear_rb = tid + rep_rb * WORKGROUP
                kb_rb = linear_rb // (BV * 64)
                rem_rb = linear_rb - kb_rb * (BV * 64)
                row_rb = rem_rb // 64
                col_rb = rem_rb - row_rb * 64
                state_readback_out[row_rb, kb_rb * 64 + col_rb] = al.convert(state_bf16[kb_rb, row_rb, col_rb], al.f32)

        for token_tile in al.range(2):
            token_base = token_tile * 32
            for rep_w in al.range(32):
                linear_w = tid + rep_w * WORKGROUP
                kb_w = linear_w // (32 * 64)
                rem_w = linear_w - kb_w * (32 * 64)
                token_off_w = rem_w // 64
                col_w = rem_w - token_off_w * 64
                token_idx_w = chunk_start + token_base + token_off_w
                w_bf16[kb_w, token_off_w, col_w] = al.convert(w[0, token_idx_w, 0, kb_w * 64 + col_w], al.bf16)

            al.syncthreads()

            pred_acc = al.full((16,), 0.0, al.f32)
            for kpack in al.range(4):
                a_words = w_vec[wave_id, lane_mod32, kpack]
                b_words = state_vec[wave_id, lane_mod32, kpack]
                a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
                pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)

            row_base = lane_col & 7
            col_base = ((lane_col >> 3) * 4) + lane_group
            for acc_i in al.range(16):
                out_row = ((acc_i & 3) * 8) + row_base
                out_col = ((acc_i >> 2) * 8) + col_base
                pred_partial[wave_id, out_row, out_col] = pred_acc[acc_i]

            al.syncthreads()

            for rep_corr in al.range(8):
                linear_corr = tid + rep_corr * WORKGROUP
                token_off = linear_corr // BV
                local_v = linear_corr - token_off * BV
                token_idx = chunk_start + token_base + token_off
                out_offset = token_idx * (8 * 128) + local_v
                pred_value = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
                corrected = u_flat[out_offset] - pred_value
                decay = gdr_decay[0, chunk_idx, 0, token_base + token_off]
                if mode == MODE_DECAY_OFF:
                    decay = al.convert(1.0, al.f32)
                pred_out[chunk_idx, token_base + token_off, local_v] = pred_value
                ucorr_out[chunk_idx, token_base + token_off, local_v] = corrected
                vdecay_out[chunk_idx, local_v, token_base + token_off] = corrected * decay
                v_decay_t_bf16[local_v, token_base + token_off] = al.convert(corrected * decay, al.bf16)

            al.syncthreads()

        for rep_k_all in al.range(64):
            linear_k_all = tid + rep_k_all * WORKGROUP
            k_row_all = linear_k_all // BT
            token_k_all = linear_k_all - k_row_all * BT
            k_all_t[k_row_all, token_k_all] = k[0, chunk_start + token_k_all, 0, k_row_all]

        al.syncthreads()

        for rep_zero in al.range(32):
            linear_zero = tid + rep_zero * WORKGROUP
            vv_zero = linear_zero // 128
            kk_zero = linear_zero - vv_zero * 128
            delta_out[chunk_idx, vv_zero, kk_zero] = al.convert(0.0, al.f32)

        al.syncthreads()

        for v_half in al.range(2):
            for local_tile_u in al.range(4):
                global_tile_u = wave_id * 4 + local_tile_u
                base_k_u = global_tile_u * 16
                update_acc16 = al.full((4,), 0.0, al.f32)

                for token_sub in al.range(4):
                    pack_base = token_sub * 2
                    if lane_group == 0:
                        a_words_u16 = vdecay_vec[v_half * 16 + lane_col, pack_base]
                        b_words_u16 = kall_vec[base_k_u + lane_col, pack_base]
                        a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[0], b_frag_u16[0], update_acc16)
                    if lane_group == 1:
                        a_words_u16 = vdecay_vec[v_half * 16 + lane_col, pack_base]
                        b_words_u16 = kall_vec[base_k_u + lane_col, pack_base]
                        a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[1], b_frag_u16[1], update_acc16)
                    if lane_group == 2:
                        a_words_u16 = vdecay_vec[v_half * 16 + lane_col, pack_base + 1]
                        b_words_u16 = kall_vec[base_k_u + lane_col, pack_base + 1]
                        a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[0], b_frag_u16[0], update_acc16)
                    if lane_group == 3:
                        a_words_u16 = vdecay_vec[v_half * 16 + lane_col, pack_base + 1]
                        b_words_u16 = kall_vec[base_k_u + lane_col, pack_base + 1]
                        a_frag_u16 = al.view(a_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        b_frag_u16 = al.view(b_words_u16, al.Tensor((2, 4, 1), al.bf16))
                        update_acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u16[1], b_frag_u16[1], update_acc16)

                out_col_u = base_k_u + lane_col
                for r_up in al.range(4):
                    out_v_u = v_half * 16 + lane_group * 4 + r_up
                    delta_out[chunk_idx, out_v_u, out_col_u] = update_acc16[r_up]
                    state[out_v_u, out_col_u] = state[out_v_u, out_col_u] * g_last_exp + update_acc16[r_up]

                al.syncthreads()

        for rep_after in al.range(32):
            linear_after = tid + rep_after * WORKGROUP
            vv_after = linear_after // 128
            kk_after = linear_after - vv_after * 128
            state_after_out[chunk_idx, vv_after, kk_after] = state[vv_after, kk_after]

        al.syncthreads()


def run_debug_kernel(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, *, mode: int = MODE_NORMAL):
    num_tokens = k.shape[1]
    if num_tokens not in (64, 128):
        raise ValueError("debug kernel supports only T=64 or T=128")
    num_chunks = num_tokens // BT
    pred = torch.empty((2, 64, 32), device=k.device, dtype=torch.float32)
    ucorr = torch.empty_like(pred)
    vdecay = torch.empty((2, 32, 64), device=k.device, dtype=torch.float32)
    delta = torch.empty((2, 32, 128), device=k.device, dtype=torch.float32)
    state_after = torch.empty_like(delta)
    state_written = torch.empty((32, 128), device=k.device, dtype=torch.float32)
    state_readback = torch.empty_like(state_written)
    _qwen_gdn_v29_fused_full_debug_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        k,
        w,
        u,
        gdr_decay,
        gdr_g_last_exp,
        initial_state,
        pred,
        ucorr,
        vdecay,
        delta,
        state_after,
        state_written,
        state_readback,
        num_tokens,
        num_chunks,
        mode,
        num_warps=2,
    )
    return {
        "pred": pred[:num_chunks],
        "ucorr": ucorr[:num_chunks],
        "vdecay": vdecay[:num_chunks],
        "delta": delta[:num_chunks],
        "state_after": state_after[:num_chunks],
        "state_written": state_written,
        "state_readback": state_readback,
    }


def torch_debug_reference(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, *, mode: int = MODE_NORMAL):
    num_chunks = k.shape[1] // BT
    state = initial_state[0, 0, 0:32].float().clone()
    initial = state.clone()
    pred = torch.empty((num_chunks, 64, 32), device=k.device, dtype=torch.float32)
    ucorr = torch.empty_like(pred)
    vdecay = torch.empty((num_chunks, 32, 64), device=k.device, dtype=torch.float32)
    delta = torch.empty((num_chunks, 32, 128), device=k.device, dtype=torch.float32)
    state_after = torch.empty_like(delta)
    for chunk_idx in range(num_chunks):
        if mode == MODE_FEEDBACK_DISABLED:
            state = initial.clone()
        start = chunk_idx * BT
        state_bf16 = state.to(torch.bfloat16).float()
        w_bf16 = w[0, start : start + BT, 0].to(torch.bfloat16).float()
        pred_chunk = w_bf16 @ state_bf16.t()
        corrected = u[0, start : start + BT, 0, 0:32].float() - pred_chunk
        if mode == MODE_DECAY_OFF:
            decay = torch.ones((BT,), device=k.device, dtype=torch.float32)
            g_last_exp = torch.tensor(1.0, device=k.device)
        else:
            decay = gdr_decay[0, chunk_idx, 0]
            g_last_exp = gdr_g_last_exp[0, chunk_idx, 0]
        vdecay_chunk = (corrected * decay.unsqueeze(-1)).t()
        delta_chunk = vdecay_chunk.to(torch.bfloat16).float() @ k[0, start : start + BT, 0].float()
        state = state * g_last_exp + delta_chunk
        pred[chunk_idx] = pred_chunk
        ucorr[chunk_idx] = corrected
        vdecay[chunk_idx] = vdecay_chunk
        delta[chunk_idx] = delta_chunk
        state_after[chunk_idx] = state
    return {
        "pred": pred,
        "ucorr": ucorr,
        "vdecay": vdecay,
        "delta": delta,
        "state_after": state_after,
    }


def compare_debug(actual: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> dict[str, tuple[float, float]]:
    rows = {}
    for name, ref_tensor in ref.items():
        err = (actual[name] - ref_tensor).abs()
        rows[name] = (err.max().item(), err.mean().item())
    if actual["state_readback"].numel():
        rb_err = (actual["state_readback"] - actual["state_written"].to(torch.bfloat16).float()).abs()
        rows["state_readback_vs_written_bf16"] = (rb_err.max().item(), rb_err.mean().item())
    return rows


def _make_debug_inputs(t: int, *, decay_off: bool = False, seed: int = 0):
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = _make_inputs(t, seed=seed)
    if decay_off:
        g = torch.zeros((1, t, 8), device=k.device, dtype=torch.float32)
        gdr_decay, gdr_g_last_exp = qwen_gdn_gdr_decay_bt64_reference(g)
    return k, w, u, gdr_decay, gdr_g_last_exp, initial_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--mode", choices=["normal", "decay_off", "feedback_disabled"], default="normal")
    args = parser.parse_args()
    mode = {"normal": MODE_NORMAL, "decay_off": MODE_DECAY_OFF, "feedback_disabled": MODE_FEEDBACK_DISABLED}[args.mode]
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = _make_debug_inputs(args.T, decay_off=args.mode == "decay_off", seed=29000 + args.T)
    actual = run_debug_kernel(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, mode=mode)
    torch.cuda.synchronize()
    ref = torch_debug_reference(k, w, u, gdr_decay, gdr_g_last_exp, initial_state, mode=mode)
    torch.cuda.synchronize()
    rows = compare_debug(actual, ref)
    for name, (max_abs, mean_abs) in rows.items():
        print(f"T={args.T} mode={args.mode} tensor={name} max_abs={max_abs:.8e} mean_abs={mean_abs:.8e}")


if __name__ == "__main__":
    main()

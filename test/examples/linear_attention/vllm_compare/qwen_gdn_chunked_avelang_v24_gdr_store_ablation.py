"""v24/v23 chunk_gdr store ablation.

Experimental chunk_gdr-only variants copied from v23.  These are upper-bound
probes; no_h/no_vn variants are not full-pipeline correct.
"""

from __future__ import annotations

import torch

import avelang
import avelang.language as al
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import _num_chunks, _validate_bf16_chunk_gdr_stage
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import _validate_initial_state_v14


BT = 16
BV = 16


@avelang.jit
def _qwen_gdn_chunk_gdr_bf16_kernel_v24_store_ablation(
    k_ptr: al.Pointer(al.bf16),
    w_ptr: al.Pointer(al.f32),
    u_ptr: al.Pointer(al.f32),
    gdr_decay_ptr: al.Pointer(al.f32),
    gdr_g_last_exp_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.f32),
    vn_ptr: al.Pointer(al.f32),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
    do_h_store: al.constexpr,
    do_vn_store: al.constexpr,
):
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((1, num_tokens, 4, 128), (num_tokens * 4 * 128, 4 * 128, 128, 1)))
    w = al.make_tensor(w_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))
    u = al.make_tensor(u_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))
    gdr_decay = al.make_tensor(gdr_decay_ptr, al.f32, al.make_layout((1, num_chunks, 8, 16), (num_chunks * 8 * 16, 8 * 16, 16, 1)))
    gdr_g_last_exp = al.make_tensor(gdr_g_last_exp_ptr, al.f32, al.make_layout((1, num_chunks, 8), (num_chunks * 8, 8, 1)))
    initial_state = al.make_tensor(initial_state_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))
    h = al.make_tensor(h_ptr, al.f32, al.make_layout((1, num_chunks, 8, 128, 128), (num_chunks * 8 * 128 * 128, 8 * 128 * 128, 128 * 128, 128, 1)))
    vn = al.make_tensor(vn_ptr, al.f32, al.make_layout((1, num_tokens, 8, 128), (num_tokens * 8 * 128, 8 * 128, 128, 1)))
    final_state = al.make_tensor(final_state_ptr, al.f32, al.make_layout((1, 8, 128, 128), (8 * 128 * 128, 128 * 128, 128, 1)))

    tid = al.thread_id(0)
    wave_id = tid // 64
    lane = tid - wave_id * 64
    lane_col = lane & 15
    lane_group = lane >> 4
    program_id = al.block_id(0)
    v_block_idx = program_id % 8
    value_head_idx = (program_id // 8) % 8
    value_base = v_block_idx * 16
    key_head_idx = value_head_idx // 2

    state = al.make_shared((BV, 128), al.f32)
    state_bf16 = al.make_shared((BV, 128), al.bf16)
    w_bf16 = al.make_shared((BT, 128), al.bf16)
    v_decay_t = al.make_shared((BV, BT), al.bf16)
    k_all_t = al.make_shared((128, BT), al.bf16)
    pred_partial = al.make_shared((4, BT, BV), al.f32)

    state_vec = al.view(state_bf16, al.i32, al.make_layout((BV, 16, 4), (64, 4, 1)))
    w_vec = al.view(w_bf16, al.i32, al.make_layout((BT, 16, 4), (64, 4, 1)))
    vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 2, 4), (8, 4, 1)))
    kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))

    for rep in al.range(8):
        idx = lane + rep * 64
        vv = idx // 32
        kk_local = idx - vv * 32
        kk = kk_local + wave_id * 32
        global_v = value_base + vv
        if has_initial_state:
            state[vv, kk] = initial_state[0, value_head_idx, global_v, kk]
        else:
            state[vv, kk] = al.convert(0.0, al.f32)

    al.syncthreads()

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * 16
        g_last_exp = gdr_g_last_exp[0, chunk_idx, value_head_idx]

        if do_h_store:
            for rep_h in al.range(8):
                idx_h = lane + rep_h * 64
                vv_h = idx_h // 32
                kk_local_h = idx_h - vv_h * 32
                kk_h = kk_local_h + wave_id * 32
                global_v_h = value_base + vv_h
                h[0, chunk_idx, value_head_idx, global_v_h, kk_h] = state[vv_h, kk_h]

        for rep_pred in al.range(8):
            idx_pred = lane + rep_pred * 64
            row_pred = idx_pred // 32
            kk_local_pred = idx_pred - row_pred * 32
            col_pred = kk_local_pred + wave_id * 32
            token_pred = chunk_start + row_pred
            state_bf16[row_pred, col_pred] = al.convert(state[row_pred, col_pred], al.bf16)
            w_bf16[row_pred, col_pred] = al.convert(w[0, token_pred, value_head_idx, col_pred], al.bf16)

        for rep_k in al.range(8):
            idx_k = lane + rep_k * 64
            row_k = idx_k // 32
            kk_local_k = idx_k - row_k * 32
            col_k = kk_local_k + wave_id * 32
            token_k = chunk_start + row_k
            k_all_t[col_k, row_k] = k[0, token_k, key_head_idx, col_k]

        al.syncthreads()

        pred_acc = al.full((4,), 0.0, al.f32)
        k_vec32 = lane_group + wave_id * 4
        a_words = w_vec[lane_col, k_vec32]
        b_words = state_vec[lane_col, k_vec32]
        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], pred_acc)
        pred_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], pred_acc)

        for r_pred in al.range(4):
            token_offset_pred = lane_group * 4 + r_pred
            value_offset_pred = lane_col
            pred_partial[wave_id, token_offset_pred, value_offset_pred] = pred_acc[r_pred]

        al.syncthreads()

        linear_vn = wave_id * 64 + lane
        token_offset = linear_vn // 16
        value_offset = linear_vn - token_offset * 16
        token_idx = chunk_start + token_offset
        global_v = value_base + value_offset
        pred_value = pred_partial[0, token_offset, value_offset] + pred_partial[1, token_offset, value_offset] + pred_partial[2, token_offset, value_offset] + pred_partial[3, token_offset, value_offset]
        v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
        if do_vn_store:
            vn[0, token_idx, value_head_idx, global_v] = v_new
        decay = gdr_decay[0, chunk_idx, value_head_idx, token_offset]
        v_decay_t[value_offset, token_offset] = al.convert(v_new * decay, al.bf16)

        al.syncthreads()

        for local_tile in al.range(2):
            global_tile = wave_id * 2 + local_tile
            base_k = global_tile * 16
            acc = al.full((4,), 0.0, al.f32)
            if lane_group == 0:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 1:
                a_words_u = vdecay_vec[lane_col, 0]
                b_words_u = kall_vec[base_k + lane_col, 0]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)
            if lane_group == 2:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[0], b_frag_u[0], acc)
            if lane_group == 3:
                a_words_u = vdecay_vec[lane_col, 1]
                b_words_u = kall_vec[base_k + lane_col, 1]
                a_frag_u = al.view(a_words_u, al.Tensor((2, 4, 1), al.bf16))
                b_frag_u = al.view(b_words_u, al.Tensor((2, 4, 1), al.bf16))
                acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag_u[1], b_frag_u[1], acc)

            out_col = base_k + lane_col
            for r_up in al.range(4):
                vv_up = lane_group * 4 + r_up
                state[vv_up, out_col] = state[vv_up, out_col] * g_last_exp + acc[r_up]

        al.syncthreads()

    for rep_final in al.range(8):
        idx_final = lane + rep_final * 64
        vv_final = idx_final // 32
        kk_local_final = idx_final - vv_final * 32
        kk_final = kk_local_final + wave_id * 32
        global_v_final = value_base + vv_final
        final_state[0, value_head_idx, global_v_final, kk_final] = state[vv_final, kk_final]


def qwen_gdn_chunk_gdr_avelang_v24_store_ablation(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gdr_decay: torch.Tensor,
    gdr_g_last_exp: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    variant: str = "baseline_copy",
    chunk_size: int = BT,
):
    num_tokens_for_g = k.shape[1]
    batch_size, num_tokens, num_k_heads, num_v_heads, head_dim_k, head_dim_v = _validate_bf16_chunk_gdr_stage(
        k, w, u, gdr_g_last_exp.new_empty((1, num_tokens_for_g, 8)), chunk_size
    )
    if chunk_size != BT or (batch_size, num_k_heads, num_v_heads, head_dim_k, head_dim_v) != (1, 4, 8, 128, 128):
        raise ValueError("v24 store ablation only supports B=1,Hk=4,Hv=8,K=128,V=128,chunk_size=16.")
    num_chunks = _num_chunks(num_tokens, chunk_size)
    if tuple(gdr_decay.shape) != (1, num_chunks, 8, 16) or tuple(gdr_g_last_exp.shape) != (1, num_chunks, 8):
        raise ValueError("invalid gdr decay shapes")
    initial_state_arg, has_initial_state = _validate_initial_state_v14(initial_state, device=k.device)
    h = torch.empty((1, num_chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    vn = torch.empty_like(u)
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    if initial_state_arg is None:
        initial_state_arg = final_state
    if variant == "baseline_copy":
        do_h_store, do_vn_store = True, True
    elif variant == "no_vn_store":
        do_h_store, do_vn_store = True, False
    elif variant == "no_h_store":
        do_h_store, do_vn_store = False, True
    elif variant == "no_h_no_vn_store":
        do_h_store, do_vn_store = False, False
    else:
        raise ValueError(f"unknown variant {variant}")
    _qwen_gdn_chunk_gdr_bf16_kernel_v24_store_ablation[lambda: ((64, 1, 1), (256, 1, 1))](
        k,
        w,
        u,
        gdr_decay,
        gdr_g_last_exp,
        initial_state_arg,
        h,
        vn,
        final_state,
        num_tokens,
        num_chunks,
        has_initial_state,
        do_h_store,
        do_vn_store,
    )
    return h, vn, final_state


__all__ = [
    "_qwen_gdn_chunk_gdr_bf16_kernel_v24_store_ablation",
    "qwen_gdn_chunk_gdr_avelang_v24_store_ablation",
]


def _smoke_main() -> None:
    import argparse
    import sys
    from pathlib import Path

    AVELANG_DIR = str(Path(__file__).resolve().parent)
    sys.path.insert(0, AVELANG_DIR)
    from bench_qwen_gdn_v24_gdr_store_ablation import prepare_gdr_inputs, time_fn

    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--variant", choices=["baseline_copy", "no_vn_store", "no_h_store", "no_h_no_vn_store"], default="baseline_copy")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    k, w, u, gdr_decay, gdr_g_last_exp, initial_state = prepare_gdr_inputs(args.T)
    latency = time_fn(
        lambda: qwen_gdn_chunk_gdr_avelang_v24_store_ablation(
            k,
            w,
            u,
            gdr_decay,
            gdr_g_last_exp,
            initial_state,
            variant=args.variant,
            chunk_size=16,
        ),
        args.warmup,
        args.repeat,
    )
    print(f"T={args.T} variant={args.variant} chunk_gdr_ms={latency:.6f}")


if __name__ == "__main__":
    _smoke_main()

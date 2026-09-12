"""D0-P: source-expressed 8x8 register-transpose direct-K64 update arm.

This is an experimental direct recurrence-update suffix.  It deliberately
keeps the current BF16 K/V-new, FP32 g/state, BF16 H, FP32 final-state ABI and
the BV32 two-wave ownership used by C0.  It does not touch block-dot lowering.

``c0_reference`` remains the existing persistent typed-block lowering.
``blocked_8x8`` is intentionally unavailable: the source gate showed its
token-major LDS gather cannot become a legal MFMA32 fragment.  The executable
``register_transpose`` arm instead turns every source 8x8 token/feature block
in registers with ``al.shuffle`` before storing a row-contiguous LDS tile.
That permits ordinary packed LDS fragment loads without a generic gather.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al

import repro_qwen_gdn_direct_k64_block_dot_bv32_coop as c0
import repro_qwen_gdn_direct_k64_update_current_abi as base


BT = base.BT
BV = 32
KDIM = base.KDIM
H_K = base.H_K
H_V = base.H_V
WORKGROUP = base.WORKGROUP
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _qwen_gdn_direct_k64_bv32_register_transpose_d0p_kernel(
    k_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    initial_state_ptr: al.Pointer(al.f32),
    h_ptr: al.Pointer(al.bf16),
    final_state_ptr: al.Pointer(al.f32),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
    has_initial_state: al.constexpr,
):
    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_K, KDIM), (num_tokens * H_K * KDIM, H_K * KDIM, KDIM, 1)),
    )
    v_new = al.make_tensor(
        v_new_ptr,
        al.bf16,
        al.make_layout((1, num_tokens, H_V, KDIM), (num_tokens * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, num_tokens, H_V), (num_tokens * H_V, H_V, 1)))
    initial_state = al.make_tensor(
        initial_state_ptr,
        al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )
    h = al.make_tensor(
        h_ptr,
        al.bf16,
        al.make_layout(
            (1, num_chunks, H_V, KDIM, KDIM),
            (num_chunks * H_V * KDIM * KDIM, H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1),
        ),
    )
    final_state = al.make_tensor(
        final_state_ptr,
        al.f32,
        al.make_layout((1, H_V, KDIM, KDIM), (H_V * KDIM * KDIM, KDIM * KDIM, KDIM, 1)),
    )

    tid = al.thread_id(0)
    wave_id = tid >> 6
    lane_in_wave = tid & 63
    lane_col = tid & 31
    lane_group = tid >> 5
    lane8 = tid & 7
    subgroup = tid >> 3
    subgroup_lane_base = lane_in_wave - lane8

    program_id = al.block_id(0)
    v_block_idx = program_id & 3
    value_head_idx = program_id >> 2
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx >> 1

    # Wave 0 owns K[0:64], wave 1 owns K[64:128] for the same V32 rows.
    h_lo = al.full((16,), 0.0, al.f32)
    h_hi = al.full((16,), 0.0, al.f32)
    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_col
        if has_initial_state:
            if wave_id == 0:
                h_lo[acc_i] = initial_state[0, value_head_idx, global_v, local_k]
                h_hi[acc_i] = initial_state[0, value_head_idx, global_v, 32 + local_k]
            else:
                h_lo[acc_i] = initial_state[0, value_head_idx, global_v, 64 + local_k]
                h_hi[acc_i] = initial_state[0, value_head_idx, global_v, 96 + local_k]

    # Both tiles are producer and consumer compact: row is V/K feature and
    # column is token.  A row consumes a contiguous packed BF16 fragment.
    v_stage = al.make_shared((32, 64), al.bf16)
    k_stage = al.make_shared((64, 64), al.bf16)
    v_vec = al.view(v_stage, al.u32, al.make_layout((32, 8, 4), (32, 4, 1)))
    k_vec = al.view(k_stage, al.u32, al.make_layout((64, 8, 4), (32, 4, 1)))
    v_rsrc = al.amdgpu.make_rsrc(v_new, num_tokens * H_V * KDIM * 2)
    k_rsrc = al.amdgpu.make_rsrc(k, num_tokens * H_K * KDIM * 2)
    zero = al.convert(0, al.i32)

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last = g[0, chunk_start + BT - 1, value_head_idx]
        g_last_exp = al.exp(g_last)

        for acc_i in al.range(16):
            local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
            global_v = value_base + lane_col
            if wave_id == 0:
                h[0, chunk_idx, value_head_idx, global_v, local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
            else:
                h[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = al.convert(h_hi[acc_i], al.bf16)

        # Two CTA passes cover 4 V feature blocks x 8 token blocks.  Each
        # lane begins with one token's contiguous BF16x8 then uses an 8-lane
        # DS bpermute transpose so its LDS writes form a contiguous V row.
        for v_pass in al.range(2):
            v_group = subgroup + v_pass * 16
            v_feature_block = v_group & 3
            v_token_block = v_group >> 2
            v_source_token = chunk_start + v_token_block * 8 + lane8
            v_offset = al.convert(
                ((v_source_token * H_V + value_head_idx) * KDIM + value_base + v_feature_block * 8) * 2,
                al.i32,
            )
            v_packed = al.amdgpu.raw_buffer_load_x4(v_rsrc, zero, v_offset, 0)
            for token_in_block in al.range(8):
                source_lane = subgroup_lane_base + token_in_block
                # The destination lane chooses its feature *after* fetching
                # all packed words from the source-token lane. Selecting
                # before shuffle would incorrectly use the source lane's
                # feature index and is not an 8x8 transpose.
                v_word0 = al.shuffle(v_packed[0], source_lane, 64)
                v_word1 = al.shuffle(v_packed[1], source_lane, 64)
                v_word2 = al.shuffle(v_packed[2], source_lane, 64)
                v_word3 = al.shuffle(v_packed[3], source_lane, 64)
                v_selected_word = v_word0
                if (lane8 >> 1) == 1:
                    v_selected_word = v_word1
                if (lane8 >> 1) == 2:
                    v_selected_word = v_word2
                if (lane8 >> 1) == 3:
                    v_selected_word = v_word3
                v_pair = al.view(v_selected_word, al.Tensor((2,), al.bf16))
                v_value_bf16 = v_pair[0]
                if (lane8 & 1) == 1:
                    v_value_bf16 = v_pair[1]
                v_source_g = al.shuffle(g[0, v_source_token, value_head_idx], source_lane, 64)
                v_value = al.convert(
                    al.convert(v_value_bf16, al.f32) * al.exp(g_last - v_source_g), al.f32
                )
                v_stage[v_feature_block * 8 + lane8, v_token_block * 8 + token_in_block] = al.convert(
                    v_value, al.bf16
                )
        al.syncthreads()

        # One K64 half is made persistent while its two 32-column state tiles
        # consume all eight K32 accumulation packs.  Four CTA passes cover
        # 8 K feature blocks x 8 token blocks.
        for k_half in al.range(2):
            for k_pass in al.range(4):
                k_group = subgroup + k_pass * 16
                k_feature_block = k_group & 7
                k_token_block = k_group >> 3
                k_source_token = chunk_start + k_token_block * 8 + lane8
                k_offset = al.convert(
                    ((k_source_token * H_K + key_head_idx) * KDIM + k_half * 64 + k_feature_block * 8) * 2,
                    al.i32,
                )
                k_packed = al.amdgpu.raw_buffer_load_x4(k_rsrc, zero, k_offset, 0)
                for token_in_block in al.range(8):
                    source_lane = subgroup_lane_base + token_in_block
                    k_word0 = al.shuffle(k_packed[0], source_lane, 64)
                    k_word1 = al.shuffle(k_packed[1], source_lane, 64)
                    k_word2 = al.shuffle(k_packed[2], source_lane, 64)
                    k_word3 = al.shuffle(k_packed[3], source_lane, 64)
                    k_selected_word = k_word0
                    if (lane8 >> 1) == 1:
                        k_selected_word = k_word1
                    if (lane8 >> 1) == 2:
                        k_selected_word = k_word2
                    if (lane8 >> 1) == 3:
                        k_selected_word = k_word3
                    k_pair = al.view(k_selected_word, al.Tensor((2,), al.bf16))
                    k_value = k_pair[0]
                    if (lane8 & 1) == 1:
                        k_value = k_pair[1]
                    k_stage[k_feature_block * 8 + lane8, k_token_block * 8 + token_in_block] = k_value
            al.syncthreads()

            for col_half in al.range(2):
                update_acc = al.full((16,), 0.0, al.f32)
                if wave_id == k_half:
                    for token_half in al.range(2):
                        for k_pack in al.range(2):
                            word = token_half * 4 + k_pack * 2 + lane_group
                            a_words = v_vec[lane_col, word]
                            b_words = k_vec[col_half * 32 + lane_col, word]
                            a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                            b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                            update_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], update_acc)
                            update_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], update_acc)

                    for acc_i in al.range(16):
                        if k_half == 0:
                            if col_half == 0:
                                h_lo[acc_i] = h_lo[acc_i] * g_last_exp + update_acc[acc_i]
                            else:
                                h_hi[acc_i] = h_hi[acc_i] * g_last_exp + update_acc[acc_i]
                        else:
                            if col_half == 0:
                                h_lo[acc_i] = h_lo[acc_i] * g_last_exp + update_acc[acc_i]
                            else:
                                h_hi[acc_i] = h_hi[acc_i] * g_last_exp + update_acc[acc_i]
            al.syncthreads()

    for acc_i in al.range(16):
        local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
        global_v = value_base + lane_col
        if wave_id == 0:
            final_state[0, value_head_idx, global_v, local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 32 + local_k] = h_hi[acc_i]
        else:
            final_state[0, value_head_idx, global_v, 64 + local_k] = h_lo[acc_i]
            final_state[0, value_head_idx, global_v, 96 + local_k] = h_hi[acc_i]


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    target = _HSACO_DUMP_DIR / "_qwen_gdn_direct_k64_bv32_register_transpose_d0p_kernel.hsaco"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and "register_transpose_d0p" in src.fn.fn.__name__:
            target.write_bytes(binary)
            dumped = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError("no D0-P register-transpose kernel matched HSACO dump")


def qwen_gdn_direct_k64_bv32_layout_feasibility_d0p(
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    arm: str = "register_transpose",
) -> tuple[torch.Tensor, torch.Tensor]:
    if arm == "c0_reference":
        os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"
        os.environ["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = "persistent_typed_block"
        return c0.qwen_gdn_direct_k64_block_dot_bv32_coop(k, v_new, g, initial_state)
    if arm == "blocked_8x8":
        raise RuntimeError(
            "blocked_8x8 is unavailable: token-major packed LDS requires a non-contiguous "
            "gather-to-MFMA fragment that fails the D0-P source expressibility gate."
        )
    if arm != "register_transpose":
        raise ValueError(f"unknown D0-P arm {arm!r}")

    tokens, num_chunks = base._validate_inputs(k, v_new, g, initial_state)
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_direct_k64_bv32_register_transpose_d0p_kernel[lambda: ((H_V * 4, 1, 1), (WORKGROUP, 1, 1))](
            k,
            v_new,
            g,
            initial_state,
            h,
            final_state,
            tokens,
            num_chunks,
            has_initial_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch)
    launch()
    return h, final_state


def run_case(tokens: int, *, arm: str, seed: int, warmup: int, repeat: int, check: bool) -> dict[str, object]:
    k, v_new, g, initial = base._make_inputs(tokens, seed)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return qwen_gdn_direct_k64_bv32_layout_feasibility_d0p(k, v_new, g, initial, arm=arm)

    h, final_state = launch()
    torch.cuda.synchronize()
    row: dict[str, object] = {"T": tokens, "arm": arm}
    if check:
        ref_h, ref_final = base.qwen_gdn_direct_k64_update_reference(k, v_new, g, initial)
        row["h_max_abs"] = float((h.float() - ref_h.float()).abs().max().item())
        row["h_mean_abs"] = float((h.float() - ref_h.float()).abs().mean().item())
        row["final_max_abs"] = float((final_state - ref_final).abs().max().item())
        row["final_mean_abs"] = float((final_state - ref_final).abs().mean().item())
        row["finite"] = bool(torch.isfinite(h.float()).all() and torch.isfinite(final_state).all())
    row["median_ms"] = base._benchmark_ms(launch, warmup=warmup, repeat=repeat)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["c0_reference", "register_transpose", "blocked_8x8"], default="register_transpose")
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [
        run_case(tokens, arm=args.arm, seed=args.seed, warmup=args.warmup, repeat=args.repeat, check=not args.no_check)
        for tokens in args.T
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print("d0p," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()

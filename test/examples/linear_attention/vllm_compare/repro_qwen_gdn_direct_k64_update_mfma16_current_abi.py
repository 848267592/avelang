"""Direct-K64 MFMA16 recurrence-update control repro for gfx942.

This is deliberately an isolated recurrence *suffix* experiment, not a new
full Qwen GDN implementation.  It keeps the current-vLLM storage boundary:

* K and corrected V-new are BF16.
* g and persistent state are FP32.
* H snapshots are BF16 and the final state is FP32.

This is the MFMA16 control for ``repro_qwen_gdn_direct_k64_update_current_abi``.
It uses the identical external ABI, CTA ownership, persistent 32x32 FP32 state
fragments, state/output layouts, and direct K64 source blocks.  Only the
update dot decomposition changes: each 32x32 BF16 MFMA is decomposed into
16x16x16 BF16 MFMAs.

Each CTA owns one [V=64, K=128] state block for one value head. Two waves
keep the state in four 32x32 FP32 accumulator fragments per wave:

    wave 0: V 0:32, K 0:64 and K 64:128
    wave 1: V 32:64, K 0:64 and K 64:128

For each chunk the update consumes direct K blocks:

    K[0:64, token:token+64]
    K[64:128, token:token+64]

There is intentionally no ``k_all_t[128, 64]`` and no generic K shared view.
The source stages only immediate 16x16 direct-K operands.  A temporary
``delta_stage[V64, K128]`` converts the native MFMA16 output mapping back into
the same persistent 32x32 H1/H2 fragment layout used by the MFMA32 control.

The pred/W-to-V-new phase is intentionally outside this repro.  That avoids
conflating direct-K/update lowering with the unresolved v29 pred semantics.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


BT = 64
BV = 64
KDIM = 128
H_K = 4
H_V = 8
WORKGROUP = 128
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _qwen_gdn_direct_k64_update_mfma16_current_abi_kernel(
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
    lane = tid & 63
    lane_col = lane & 31
    lane_group = lane >> 5
    lane_col16 = lane & 15
    lane_group16 = lane >> 4

    program_id = al.block_id(0)
    v_block_idx = program_id & 1
    value_head_idx = program_id >> 1
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx >> 1

    # This is the Stage-6Z 32x32 BF16 MFMA output layout for the operand
    # arrangement below: rows come from lane_col and the two lane groups own
    # interleaved four-column output groups.  It covers every 32x32 element.
    # Each wave holds V32 x K128 as four 32x32 FP32 accumulator fragments.
    h1_lo = al.full((16,), 0.0, al.f32)
    h1_hi = al.full((16,), 0.0, al.f32)
    h2_lo = al.full((16,), 0.0, al.f32)
    h2_hi = al.full((16,), 0.0, al.f32)

    for acc_i in al.range(16):
        local_v = lane_col
        local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
        global_v = value_base + wave_id * 32 + local_v
        if has_initial_state:
            h1_lo[acc_i] = initial_state[0, value_head_idx, global_v, local_k]
            h1_hi[acc_i] = initial_state[0, value_head_idx, global_v, 32 + local_k]
            h2_lo[acc_i] = initial_state[0, value_head_idx, global_v, 64 + local_k]
            h2_hi[acc_i] = initial_state[0, value_head_idx, global_v, 96 + local_k]

    # MFMA16 has a different output ownership than the persistent MFMA32
    # H1/H2 fragments above.  Keep its immediate [V16,T16] / [K16,T16]
    # operands compact, then use this temporary delta tile only to transpose
    # back into the unchanged H1/H2 register layout after all update tiles.
    a_stage = al.make_shared((2, 16, 16), al.bf16)
    b_stage = al.make_shared((16, 16), al.bf16)
    delta_stage = al.make_shared((64, 128), al.f32)
    a_vec = al.view(a_stage, al.i32, al.make_layout((2, 16, 2, 4), (16 * 2 * 4, 2 * 4, 4, 1)))
    b_vec = al.view(b_stage, al.i32, al.make_layout((16, 2, 4), (2 * 4, 4, 1)))

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last_exp = al.exp(g[0, chunk_start + BT - 1, value_head_idx])

        # Snapshot the chunk-entry state in the native BF16 H contract.
        for acc_i in al.range(16):
            local_v = lane_col
            local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
            global_v = value_base + wave_id * 32 + local_v
            h[0, chunk_idx, value_head_idx, global_v, local_k] = al.convert(h1_lo[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = al.convert(h1_hi[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = al.convert(h2_lo[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = al.convert(h2_hi[acc_i], al.bf16)

        for v_half in al.range(2):
            for k_tile in al.range(8):
                update_acc = al.full((4,), 0.0, al.f32)
                for token_sub in al.range(4):
                    # Exactly one [V16,T16] tile for each wave and one direct
                    # [K16,T16] tile. No all-K transpose or generic view exists.
                    for rep_a in al.range(4):
                        linear_a = tid + rep_a * WORKGROUP
                        a_wave = linear_a // (16 * 16)
                        a_rem = linear_a - a_wave * (16 * 16)
                        a_row = a_rem // 16
                        a_tok = a_rem - a_row * 16
                        token_idx = chunk_start + token_sub * 16 + a_tok
                        global_v = value_base + a_wave * 32 + v_half * 16 + a_row
                        decay = al.exp(g[0, chunk_start + BT - 1, value_head_idx] - g[0, token_idx, value_head_idx])
                        a_stage[a_wave, a_row, a_tok] = al.convert(
                            al.convert(v_new[0, token_idx, value_head_idx, global_v], al.f32) * decay,
                            al.bf16,
                        )

                    for rep_b in al.range(2):
                        linear_b = tid + rep_b * WORKGROUP
                        b_row = linear_b // 16
                        b_tok = linear_b - b_row * 16
                        b_stage[b_row, b_tok] = k[
                            0,
                            chunk_start + token_sub * 16 + b_tok,
                            key_head_idx,
                            k_tile * 16 + b_row,
                        ]

                    al.syncthreads()

                    # The four 16-lane subgroups supply the four K=4 operand
                    # slices required by one MFMA16. This is the proven v23
                    # operand mapping, now fed only by the direct K16 stage.
                    if lane_group16 == 0:
                        a_words = a_vec[wave_id, lane_col16, 0]
                        b_words = b_vec[lane_col16, 0]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], update_acc)
                    if lane_group16 == 1:
                        a_words = a_vec[wave_id, lane_col16, 0]
                        b_words = b_vec[lane_col16, 0]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], update_acc)
                    if lane_group16 == 2:
                        a_words = a_vec[wave_id, lane_col16, 1]
                        b_words = b_vec[lane_col16, 1]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[0], b_frag[0], update_acc)
                    if lane_group16 == 3:
                        a_words = a_vec[wave_id, lane_col16, 1]
                        b_words = b_vec[lane_col16, 1]
                        a_frag = al.view(a_words, al.Tensor((2, 4, 1), al.bf16))
                        b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
                        update_acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag[1], b_frag[1], update_acc)

                    al.syncthreads()

                for r_up in al.range(4):
                    local_v_delta = wave_id * 32 + v_half * 16 + lane_group16 * 4 + r_up
                    delta_stage[local_v_delta, k_tile * 16 + lane_col16] = update_acc[r_up]

                al.syncthreads()

        # Return from the MFMA16 [V16,K16] ownership to the unchanged 32x32
        # H1/H2 register fragments. The load has no alias with future staging.
        for acc_i in al.range(16):
            local_v = wave_id * 32 + lane_col
            local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
            h1_lo[acc_i] = h1_lo[acc_i] * g_last_exp + delta_stage[local_v, local_k]
            h1_hi[acc_i] = h1_hi[acc_i] * g_last_exp + delta_stage[local_v, 32 + local_k]
            h2_lo[acc_i] = h2_lo[acc_i] * g_last_exp + delta_stage[local_v, 64 + local_k]
            h2_hi[acc_i] = h2_hi[acc_i] * g_last_exp + delta_stage[local_v, 96 + local_k]

        al.syncthreads()

    for acc_i in al.range(16):
        local_v = lane_col
        local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
        global_v = value_base + wave_id * 32 + local_v
        final_state[0, value_head_idx, global_v, local_k] = h1_lo[acc_i]
        final_state[0, value_head_idx, global_v, 32 + local_k] = h1_hi[acc_i]
        final_state[0, value_head_idx, global_v, 64 + local_k] = h2_lo[acc_i]
        final_state[0, value_head_idx, global_v, 96 + local_k] = h2_hi[acc_i]


def _require_cuda_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/HIP tensor.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_inputs(
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int]:
    for name, tensor in (("k", k), ("v_new", v_new), ("g", g)):
        _require_cuda_contiguous(name, tensor)
    if k.dtype != torch.bfloat16 or v_new.dtype != torch.bfloat16:
        raise ValueError("k and v_new must be BF16.")
    if g.dtype != torch.float32:
        raise ValueError("g must be FP32.")
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[2:] != (H_K, KDIM):
        raise ValueError(f"k must be [1,T,{H_K},{KDIM}], got {tuple(k.shape)}.")
    t = int(k.shape[1])
    if t == 0 or t % BT:
        raise ValueError("T must be a positive multiple of 64.")
    if tuple(v_new.shape) != (1, t, H_V, KDIM):
        raise ValueError(f"v_new must be [1,{t},{H_V},{KDIM}], got {tuple(v_new.shape)}.")
    if tuple(g.shape) != (1, t, H_V):
        raise ValueError(f"g must be [1,{t},{H_V}], got {tuple(g.shape)}.")
    if v_new.device != k.device or g.device != k.device:
        raise ValueError("k, v_new, and g must share a device.")
    if initial_state is not None:
        _require_cuda_contiguous("initial_state", initial_state)
        if initial_state.dtype != torch.float32 or tuple(initial_state.shape) != (1, H_V, KDIM, KDIM):
            raise ValueError("initial_state must be contiguous FP32 [1,8,128,128].")
        if initial_state.device != k.device:
            raise ValueError("initial_state must share k.device.")
    return t, t // BT


def qwen_gdn_direct_k64_update_mfma16_current_abi(
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the direct-K64 update suffix without fallback."""
    t, num_chunks = _validate_inputs(k, v_new, g, initial_state)
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_direct_k64_update_mfma16_current_abi_kernel[lambda: ((H_V * 2, 1, 1), (WORKGROUP, 1, 1))](
            k,
            v_new,
            g,
            initial_state,
            h,
            final_state,
            t,
            num_chunks,
            has_initial_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch, "_qwen_gdn_direct_k64_update_mfma16_current_abi_kernel")
    launch()
    return h, final_state


def qwen_gdn_direct_k64_update_reference(
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference matching the BF16 V-decay boundary of the repro."""
    t, num_chunks = _validate_inputs(k, v_new, g, initial_state)
    state = torch.zeros((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    if initial_state is not None:
        state.copy_(initial_state)
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * BT
        for head in range(H_V):
            h[0, chunk_idx, head] = state[0, head].to(torch.bfloat16)
            decay = torch.exp(g[0, chunk_start + BT - 1, head] - g[0, chunk_start : chunk_start + BT, head])
            v_decay = (v_new[0, chunk_start : chunk_start + BT, head].float() * decay[:, None]).to(torch.bfloat16)
            key_head = head // 2
            delta = v_decay.float().transpose(0, 1) @ k[0, chunk_start : chunk_start + BT, key_head].float()
            state[0, head] = state[0, head] * torch.exp(g[0, chunk_start + BT - 1, head]) + delta
    return h, state


def _benchmark_ms(launch, *, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_substr: str) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    if list(dump_dir.glob(f"{kernel_substr}*.hsaco")):
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if not dumped and kernel_substr in kernel_name:
            path = dump_dir / f"{kernel_name}.hsaco"
            path.write_bytes(binary)
            dumped = True
            print(f"dumped_hsaco: {path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no compiled kernel matched {kernel_substr!r} for hsaco dump")


def _make_inputs(t: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn((1, t, H_K, KDIM), device="cuda", dtype=torch.bfloat16, generator=generator)
    v_new = torch.randn((1, t, H_V, KDIM), device="cuda", dtype=torch.bfloat16, generator=generator)
    g = torch.randn((1, t, H_V), device="cuda", dtype=torch.float32, generator=generator) * 0.02
    initial = torch.randn((1, H_V, KDIM, KDIM), device="cuda", dtype=torch.float32, generator=generator) * 0.1
    return k, v_new, g.contiguous(), initial


def run_case(t: int, *, seed: int, warmup: int, repeat: int, check: bool) -> dict[str, float | int | bool]:
    k, v_new, g, initial = _make_inputs(t, seed)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return qwen_gdn_direct_k64_update_mfma16_current_abi(k, v_new, g, initial)

    h, final_state = launch()
    torch.cuda.synchronize()
    result: dict[str, float | int | bool] = {"T": t}
    if check:
        ref_h, ref_final = qwen_gdn_direct_k64_update_reference(k, v_new, g, initial)
        torch.cuda.synchronize()
        result["h_max_abs"] = float((h.float() - ref_h.float()).abs().max().item())
        result["h_mean_abs"] = float((h.float() - ref_h.float()).abs().mean().item())
        result["final_max_abs"] = float((final_state - ref_final).abs().max().item())
        result["final_mean_abs"] = float((final_state - ref_final).abs().mean().item())
        result["finite"] = bool(torch.isfinite(h).all() and torch.isfinite(final_state).all())
    result["median_ms"] = _benchmark_ms(launch, warmup=warmup, repeat=repeat)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [run_case(t, seed=args.seed, warmup=args.warmup, repeat=args.repeat, check=not args.no_check) for t in args.T]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print(
            "direct_k64_update_mfma16,"
            + ",".join(f"{key}={value}" for key, value in row.items())
        )


if __name__ == "__main__":
    main()

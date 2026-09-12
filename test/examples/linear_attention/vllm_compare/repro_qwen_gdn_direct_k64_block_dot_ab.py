"""Same-source generic/specialized ``block_dot_bf16_f32`` direct-K64 repro.

This is an isolated current-vLLM-ABI recurrence-update suffix.  Its high-level
kernel has exactly one block-dot source expression per K64 half.  The compiler
chooses the late expansion using ``AVELANG_BLOCK_DOT_LOWERING``:

* ``generic`` reproduces the original output-tile-first staging order;
* ``specialized`` stages one typed V-decay tile before its two K32 consumers.

Both routes retain BF16 K/V-new, FP32 gate/state, BT64, BV64, WG128, the
two-wave persistent H1/H2 layout, and the direct-K64 state-update math.
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

import repro_qwen_gdn_direct_k64_update_current_abi as base


BT = base.BT
BV = base.BV
KDIM = base.KDIM
H_K = base.H_K
H_V = base.H_V
WORKGROUP = base.WORKGROUP
_HSACO_DUMP_DIR: Path | None = None


def set_block_dot_lowering(lowering: str) -> None:
    if lowering not in {"generic", "specialized"}:
        raise ValueError("lowering must be generic or specialized")
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = lowering


@avelang.jit
def _qwen_gdn_direct_k64_block_dot_ab_kernel(
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

    program_id = al.block_id(0)
    v_block_idx = program_id & 1
    value_head_idx = program_id >> 1
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx >> 1

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

    # The source has no scalar fill, i32 view, packed fragment extraction, or
    # explicit MFMA loop. The experimental compiler op owns exactly that
    # direct-K64 block-dot expansion after AveLang-to-memref conversion.
    a_stage = al.make_shared((2, 32, 32), al.bf16)
    b_stage = al.make_shared((32, 32), al.bf16)

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last = g[0, chunk_start + BT - 1, value_head_idx]

        for acc_i in al.range(16):
            local_v = lane_col
            local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
            global_v = value_base + wave_id * 32 + local_v
            h[0, chunk_idx, value_head_idx, global_v, local_k] = al.convert(h1_lo[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = al.convert(h1_hi[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = al.convert(h2_lo[acc_i], al.bf16)
            h[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = al.convert(h2_hi[acc_i], al.bf16)

        # One op returns the two K32 output tiles for this K64 half.  The
        # generic/specialized selection is exclusively a compiler late pass.
        for k_half in al.range(2):
            if k_half == 0:
                update_pair = al.amdgpu.block_dot_bf16_f32(
                    a_stage, b_stage, k, v_new, g, tid, chunk_start,
                    value_head_idx, key_head_idx, value_base, k_half, g_last,
                    h1_lo, h1_hi,
                )
                for acc_i in al.range(16):
                    h1_lo[acc_i] = update_pair[acc_i]
                    h1_hi[acc_i] = update_pair[16 + acc_i]
            else:
                update_pair = al.amdgpu.block_dot_bf16_f32(
                    a_stage, b_stage, k, v_new, g, tid, chunk_start,
                    value_head_idx, key_head_idx, value_base, k_half, g_last,
                    h2_lo, h2_hi,
                )
                for acc_i in al.range(16):
                    h2_lo[acc_i] = update_pair[acc_i]
                    h2_hi[acc_i] = update_pair[16 + acc_i]
        al.syncthreads()

    for acc_i in al.range(16):
        local_v = lane_col
        local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
        global_v = value_base + wave_id * 32 + local_v
        final_state[0, value_head_idx, global_v, local_k] = h1_lo[acc_i]
        final_state[0, value_head_idx, global_v, 32 + local_k] = h1_hi[acc_i]
        final_state[0, value_head_idx, global_v, 64 + local_k] = h2_lo[acc_i]
        final_state[0, value_head_idx, global_v, 96 + local_k] = h2_hi[acc_i]


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    if list(dump_dir.glob("*block_dot_ab*.hsaco")):
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if not dumped and "block_dot_ab" in kernel_name:
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
        raise RuntimeError("no block-dot kernel matched HSACO dump")


def qwen_gdn_direct_k64_block_dot_ab(
    k: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    lowering: str = "generic",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the same high-level direct-K64 source under one late lowering."""
    set_block_dot_lowering(lowering)
    t, num_chunks = base._validate_inputs(k, v_new, g, initial_state)
    h = torch.empty((1, num_chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_direct_k64_block_dot_ab_kernel[lambda: ((H_V * 2, 1, 1), (WORKGROUP, 1, 1))](
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

    _maybe_dump_hsaco(launch)
    launch()
    return h, final_state


def run_case(
    t: int,
    *,
    lowering: str,
    seed: int,
    warmup: int,
    repeat: int,
    check: bool,
) -> dict[str, float | int | bool | str]:
    set_block_dot_lowering(lowering)
    k, v_new, g, initial = base._make_inputs(t, seed)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return qwen_gdn_direct_k64_block_dot_ab(k, v_new, g, initial, lowering=lowering)

    h, final_state = launch()
    torch.cuda.synchronize()
    row: dict[str, float | int | bool | str] = {"T": t, "lowering": lowering}
    if check:
        ref_h, ref_final = base.qwen_gdn_direct_k64_update_reference(k, v_new, g, initial)
        torch.cuda.synchronize()
        row["h_max_abs"] = float((h.float() - ref_h.float()).abs().max().item())
        row["h_mean_abs"] = float((h.float() - ref_h.float()).abs().mean().item())
        row["final_max_abs"] = float((final_state - ref_final).abs().max().item())
        row["final_mean_abs"] = float((final_state - ref_final).abs().mean().item())
        row["finite"] = bool(torch.isfinite(h).all() and torch.isfinite(final_state).all())
    row["median_ms"] = base._benchmark_ms(launch, warmup=warmup, repeat=repeat)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowering", choices=["generic", "specialized"], required=True)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [
        run_case(
            t,
            lowering=args.lowering,
            seed=args.seed + t,
            warmup=args.warmup,
            repeat=args.repeat,
            check=not args.no_check,
        )
        for t in args.T
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print("block_dot_ab," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()

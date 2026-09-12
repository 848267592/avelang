"""BV32 direct-K64 control with BF16 V-decay prepared outside body timing.

This experimental control keeps the specialized block-dot K staging, MFMA32,
CTA ownership, persistent H layout, and state recurrence fixed.  Its only
semantic boundary change is that ``v_decay = bf16(v_new * exp(g_last - g))``
is created before timing and supplied directly to the block-dot lowering.
The kernel still applies the FP32 ``exp(g_last)`` persistent-state scale.
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
BV = 32
KDIM = base.KDIM
H_K = base.H_K
H_V = base.H_V
WORKGROUP = base.WORKGROUP
_HSACO_DUMP_DIR: Path | None = None


def set_block_dot_lowering() -> None:
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"


def precompute_v_decay(v_new: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Materialize the existing BF16 V-decay contract outside body timing."""
    tokens = int(v_new.shape[1])
    if tokens % BT:
        raise ValueError("V-decay control requires T divisible by 64")
    chunk_last = g[:, BT - 1 :: BT, :]
    g_last = torch.repeat_interleave(chunk_last, BT, dim=1)
    return (v_new.float() * torch.exp(g_last.unsqueeze(-1) - g.unsqueeze(-1))).to(torch.bfloat16)


def qwen_gdn_direct_k64_precomputed_vdecay_reference(
    k: torch.Tensor,
    v_decay: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 recurrence reference for an already-rounded BF16 V-decay input."""
    tokens, chunks = base._validate_inputs(k, v_decay, g, initial_state)
    state = torch.zeros((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    if initial_state is not None:
        state.copy_(initial_state)
    h = torch.empty((1, chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    for chunk_idx in range(chunks):
        chunk_start = chunk_idx * BT
        for head in range(H_V):
            h[0, chunk_idx, head] = state[0, head].to(torch.bfloat16)
            key_head = head // 2
            delta = (
                v_decay[0, chunk_start : chunk_start + BT, head].float().transpose(0, 1)
                @ k[0, chunk_start : chunk_start + BT, key_head].float()
            )
            state[0, head] = state[0, head] * torch.exp(g[0, chunk_start + BT - 1, head]) + delta
    return h, state


@avelang.jit
def _qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay_kernel(
    k_ptr: al.Pointer(al.bf16),
    v_decay_ptr: al.Pointer(al.bf16),
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
    v_decay = al.make_tensor(
        v_decay_ptr,
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
    v_block_idx = program_id & 3
    value_head_idx = program_id >> 2
    value_base = v_block_idx * BV
    key_head_idx = value_head_idx >> 1

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

    a_stage = al.make_shared((1, 32, 32), al.bf16)
    b_stage = al.make_shared((32, 32), al.bf16)

    for chunk_idx in al.range(num_chunks):
        chunk_start = chunk_idx * BT
        g_last = g[0, chunk_start + BT - 1, value_head_idx]

        for acc_i in al.range(16):
            local_k = ((acc_i >> 2) * 8) + lane_group * 4 + (acc_i & 3)
            global_v = value_base + lane_col
            if wave_id == 0:
                h[0, chunk_idx, value_head_idx, global_v, local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 32 + local_k] = al.convert(h_hi[acc_i], al.bf16)
            else:
                h[0, chunk_idx, value_head_idx, global_v, 64 + local_k] = al.convert(h_lo[acc_i], al.bf16)
                h[0, chunk_idx, value_head_idx, global_v, 96 + local_k] = al.convert(h_hi[acc_i], al.bf16)

        for k_half in al.range(2):
            update_pair = al.amdgpu.block_dot_bf16_f32_precomputed_vdecay(
                a_stage,
                b_stage,
                k,
                v_decay,
                g,
                tid,
                chunk_start,
                value_head_idx,
                key_head_idx,
                value_base,
                k_half,
                g_last,
                h_lo,
                h_hi,
            )
            for acc_i in al.range(16):
                if k_half == 0:
                    if wave_id == 0:
                        h_lo[acc_i] = update_pair[acc_i]
                        h_hi[acc_i] = update_pair[16 + acc_i]
                else:
                    if wave_id == 1:
                        h_lo[acc_i] = update_pair[acc_i]
                        h_hi[acc_i] = update_pair[16 + acc_i]
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

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    if list(dump_dir.glob("*precomputed_vdecay*.hsaco")):
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if not dumped and "precomputed_vdecay" in kernel_name:
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
        raise RuntimeError("no precomputed-V-decay kernel matched HSACO dump")


def qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay(
    k: torch.Tensor,
    v_decay: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the isolated control; ``v_decay`` must be precomputed BF16."""
    set_block_dot_lowering()
    tokens, chunks = base._validate_inputs(k, v_decay, g, initial_state)
    h = torch.empty((1, chunks, H_V, KDIM, KDIM), device=k.device, dtype=torch.bfloat16)
    final_state = torch.empty((1, H_V, KDIM, KDIM), device=k.device, dtype=torch.float32)
    has_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = final_state

    def launch() -> None:
        _qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay_kernel[
            lambda: ((H_V * 4, 1, 1), (WORKGROUP, 1, 1))
        ](
            k,
            v_decay,
            g,
            initial_state,
            h,
            final_state,
            tokens,
            chunks,
            has_initial_state,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch)
    launch()
    return h, final_state


def run_case(t: int, *, seed: int, warmup: int, repeat: int, check: bool) -> dict[str, float | int | bool | str]:
    set_block_dot_lowering()
    k, v_new, g, initial = base._make_inputs(t, seed)
    v_decay = precompute_v_decay(v_new, g)

    def launch() -> tuple[torch.Tensor, torch.Tensor]:
        return qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay(k, v_decay, g, initial)

    h, final_state = launch()
    torch.cuda.synchronize()
    row: dict[str, float | int | bool | str] = {
        "T": t,
        "implementation": "bv32_precomputed_vdecay_specialized",
    }
    if check:
        ref_h, ref_final = qwen_gdn_direct_k64_precomputed_vdecay_reference(k, v_decay, g, initial)
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
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [
        run_case(t, seed=args.seed + t, warmup=args.warmup, repeat=args.repeat, check=not args.no_check)
        for t in args.T
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        print("precomputed_vdecay," + ",".join(f"{key}={value}" for key, value in row.items()))


if __name__ == "__main__":
    main()

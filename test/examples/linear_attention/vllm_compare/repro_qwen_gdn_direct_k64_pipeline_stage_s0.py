#!/usr/bin/env python3
"""S0: opaque gfx942 K64 distributed-stage lookahead, isolated from recurrence.

The AveLang source is intentionally identical for the two arms.  It creates
two opaque K64 stage tokens around the existing C0 ``preloaded_k`` consumer:

  current stage -> commit -> current direct-K64 MFMA32 consumer
  next stage (issued) -> current consumer -> commit -> next consumer

``AVELANG_QWEN_K64_PIPELINE_LOWERING=immediate`` materializes each load only
at commit.  ``distributed`` materializes four BF16x8 packets/lane at stage
load and commits the same packets after the current consumer.  No source-side
``k_next`` local tensor exists in either arm.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 128
H_K = 4
H_V = 8
WORKGROUP = 128
TOKENS = 128

_HSACO_DUMP_DIR: Path | None = None


def _set_lowering(mode: str) -> None:
    if mode not in {"immediate", "distributed"}:
        raise ValueError(f"unsupported S0 lowering mode: {mode}")
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"
    os.environ["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = "persistent_typed_block"
    os.environ["AVELANG_QWEN_K64_PIPELINE_LOWERING"] = mode


@avelang.jit
def _qwen_gdn_direct_k64_pipeline_stage_s0_kernel(
    k_ptr: al.Pointer(al.bf16),
    v_new_ptr: al.Pointer(al.bf16),
    g_ptr: al.Pointer(al.f32),
    current_out_ptr: al.Pointer(al.f32),
    next_out_ptr: al.Pointer(al.f32),
    bank_snapshot_ptr: al.Pointer(al.bf16),
):
    """One K64 half, two waves, one K LDS bank and two C0 consumers.

    This is intentionally not a recurrence and not a standalone alternative
    algorithm.  Its sole purpose is to make the load/consumer/commit ordering
    observable while preserving the existing preloaded-K C0 MFMA32 consumer.
    """

    k = al.make_tensor(
        k_ptr,
        al.bf16,
        al.make_layout((1, TOKENS, H_K, KDIM), (TOKENS * H_K * KDIM, H_K * KDIM, KDIM, 1)),
    )
    v_new = al.make_tensor(
        v_new_ptr,
        al.bf16,
        al.make_layout((1, TOKENS, H_V, KDIM), (TOKENS * H_V * KDIM, H_V * KDIM, KDIM, 1)),
    )
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((1, TOKENS, H_V), (TOKENS * H_V, H_V, 1)))
    current_out = al.make_tensor(
        current_out_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    next_out = al.make_tensor(next_out_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1)))
    bank_snapshot = al.make_tensor(
        bank_snapshot_ptr, al.bf16, al.make_layout((64, 64), (64, 1))
    )

    tid = al.thread_id(0)

    # The C0 staged/preloaded contract is frozen: [V32,T64] plus a two-half
    # [K64,T64] bank.  S0 uses only K half 0 but leaves the ABI untouched.
    vdecay_stage = al.make_shared((1, BV, BT), al.bf16)
    k_bank = al.make_shared((2, BT, 64), al.bf16)

    # This is deliberately ordinary common work, not part of the A/B.  It
    # builds a fixed BF16 V32xT64 operand once; the C0 consumer will reuse it
    # across both current and next K64 blocks.
    for rep in al.range(16):
        linear = tid + rep * WORKGROUP
        local_v = linear // BT
        token_off = linear - local_v * BT
        vdecay_stage[0, local_v, token_off] = v_new[0, token_off, 0, local_v]
    al.syncthreads()

    # Prologue: no frontend-visible K tensor is created.  The late pass either
    # emits packets here (distributed) or delays them to the commit below.
    current_stage = al.amdgpu.qwen_k64_pipeline_stage_load(k, tid, 0, 0, 0)
    al.amdgpu.qwen_k64_pipeline_stage_commit(current_stage, k_bank)
    al.syncthreads()

    # The sole experimental ordering point: issue the next K64 packets before
    # current K is consumed.  The token cannot be indexed or unpacked here.
    next_stage = al.amdgpu.qwen_k64_pipeline_stage_load(k, tid, BT, 0, 0)
    # Keep the block-dot ABI explicitly F32. Accumulators begin at zero, so
    # this value only types the unchanged state-scale term in this isolated
    # consumer check.
    g_last = g[0, BT - 1, 0]

    current_low = al.full((16,), 0.0, al.f32)
    current_high = al.full((16,), 0.0, al.f32)
    current = al.amdgpu.block_dot_bf16_f32_staged_vdecay_preloaded_k(
        vdecay_stage,
        k_bank,
        k,
        v_new,
        g,
        tid,
        0,
        0,
        0,
        0,
        0,
        g_last,
        current_low,
        current_high,
    )

    # The current preloaded-K consumer has no remaining reads after this
    # barrier.  Only now may the late pass commit the next tile to the same
    # workgroup K bank.
    al.syncthreads()
    al.amdgpu.qwen_k64_pipeline_stage_commit(next_stage, k_bank)
    al.syncthreads()

    next_low = al.full((16,), 0.0, al.f32)
    next_high = al.full((16,), 0.0, al.f32)
    next_value = al.amdgpu.block_dot_bf16_f32_staged_vdecay_preloaded_k(
        vdecay_stage,
        k_bank,
        k,
        v_new,
        g,
        tid,
        BT,
        0,
        0,
        0,
        0,
        g_last,
        next_low,
        next_high,
    )

    # Keep both consumers and the tail commit observable without introducing
    # a recurrence result or a second K bank.
    for item in al.range(32):
        current_out[tid, item] = current[item]
        next_out[tid, item] = next_value[item]
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear // 64
        token_off = linear - row * 64
        bank_snapshot[row, token_off] = k_bank[0, row, token_off]


def _make_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    k = torch.randn((1, TOKENS, H_K, KDIM), generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    v_new = torch.randn((1, TOKENS, H_V, KDIM), generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn((1, TOKENS, H_V), generator=generator, device="cuda", dtype=torch.float32) * 0.05
    return k.contiguous(), v_new.contiguous(), g.contiguous()


def _maybe_dump_hsaco(launch: Callable[[], None]) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    _HSACO_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    if list(_HSACO_DUMP_DIR.glob("*pipeline_stage_s0*.hsaco")):
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target, options)
        name = src.fn.fn.__name__
        if not dumped and "pipeline_stage_s0" in name:
            path = _HSACO_DUMP_DIR / f"{name}.hsaco"
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
        raise RuntimeError("no S0 kernel matched HSACO dump")


def run_s0(
    mode: str,
    *,
    seed: int,
    warmup: int = 0,
    repeat: int = 0,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    _set_lowering(mode)
    k, v_new, g = _make_inputs(seed)
    current = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    next_value = torch.empty_like(current)
    snapshot = torch.empty((64, 64), device="cuda", dtype=torch.bfloat16)

    def launch() -> None:
        _qwen_gdn_direct_k64_pipeline_stage_s0_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
            k, v_new, g, current, next_value, snapshot, num_warps=2
        )

    _maybe_dump_hsaco(launch)
    launch()
    torch.cuda.synchronize()
    expected_snapshot = k[0, BT : 2 * BT, 0, :BT].transpose(0, 1).contiguous()
    samples: list[float] = []
    if repeat:
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            start.record()
            launch()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    row: dict[str, object] = {
        "mode": mode,
        "seed": seed,
        "snapshot_byte_equal": bool(torch.equal(snapshot, expected_snapshot)),
        "snapshot_max_abs": float((snapshot.float() - expected_snapshot.float()).abs().max().item()),
        "current_finite": bool(torch.isfinite(current).all().item()),
        "next_finite": bool(torch.isfinite(next_value).all().item()),
        "median_ms": statistics.median(samples) if samples else None,
        "samples_ms": samples,
    }
    outputs = {
        "current": current.detach().cpu(),
        "next": next_value.detach().cpu(),
        "snapshot": snapshot.detach().cpu(),
    }
    return row, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["immediate", "distributed"], required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--dump-hsaco-dir", type=Path)
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    row, outputs = run_s0(args.mode, seed=args.seed, warmup=args.warmup, repeat=args.repeat)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, args.save)
        row["save"] = str(args.save)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()

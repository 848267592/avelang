"""C15 real gfx942 physical-tile numerical closure, V-first.

This is an experimental correctness repro, not a chunk-o benchmark.  The
source uses the existing generic ``block_dot_bf16_f32_operand`` operation.  A
compiler environment gate makes its third operand an ordinary global BF16
[64, 64] V tile and routes it through the C13/C14 physical recipe:

    global V -> distributed ownership -> packed rotating LDS -> barrier
        -> encoded LDS read -> MFMA32

The kernel also writes a diagnostic LDS readback to ``debug``.  The readback
is useful for separating a physical-tile transport error from an MFMA fragment
mapping error.  No performance measurement is performed here.
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


TILE = 64
WORKGROUP = 128


@avelang.jit
def _c15_real_v_tile_kernel(
    v_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((TILE, TILE), (TILE, 1)))
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((TILE, TILE), (TILE, 1))
    )
    raw = al.make_tensor(
        raw_ptr, al.f32, al.make_layout((WORKGROUP, 16), (16, 1))
    )
    # Keep the ordinary block-dot ABI valid even though C15's V-only branch
    # does not consume G.  The compiler still verifies the fifth operand as
    # the regular logical [1, 64, 8] FP32 tensor.
    g = al.make_tensor(
        g_ptr,
        al.f32,
        al.make_layout((1, TILE, 8), (TILE * 8, 8, 1)),
    )

    a_stage = al.make_shared((TILE, TILE), al.bf16)
    b_stage = al.make_shared((TILE, TILE), al.bf16)

    tid = al.thread_id(0)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)

    # Static identity-like A: with the MFMA output writeback oracle, every
    # visible V element affects one raw accumulator slot.  The whole A tile
    # is produced by the same CTA before the real V producer runs.
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        value = al.convert(0, al.bf16)
        if row == col:
            value = al.convert(1, al.bf16)
        a_stage[row, col] = value
    al.syncthreads()

    acc = al.full((16,), 0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage,
        b_stage,
        v,
        debug,
        g,
        tid,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_f32,
        acc,
        acc,
    )

    # Preserve the raw MFMA fragment.  The host oracle uses the documented
    # row/column accumulator writeback mapping from the AveLang MFMA tutorial.
    for r in al.range(16):
        raw[tid, r] = result[r]


def _patterns() -> dict[str, torch.Tensor]:
    base = torch.zeros((TILE, TILE), device="cuda", dtype=torch.bfloat16)
    out: dict[str, torch.Tensor] = {"all_zero": base.clone()}

    one = base.clone()
    one[17, 29] = 1
    out["one_hot_element"] = one

    token = base.clone()
    token[17, :] = 1
    out["one_hot_token_row"] = token

    column = base.clone()
    column[:, 29] = 1
    out["one_hot_value_column"] = column

    rows = torch.arange(TILE, device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(TILE, device="cuda", dtype=torch.float32)[None, :]
    out["token_distinct"] = (rows.remainder(8)).to(torch.bfloat16).expand(TILE, TILE).clone()
    out["value_distinct"] = (cols.remainder(8)).to(torch.bfloat16).expand(TILE, TILE).clone()
    out["packet_distinct"] = ((rows.remainder(16) + cols.remainder(4))).to(torch.bfloat16)
    out["wave_distinct"] = ((rows >= 32).to(torch.float32)).to(torch.bfloat16).expand(TILE, TILE).clone()
    out["lane_distinct"] = ((rows.remainder(32) + cols.remainder(8))).to(torch.bfloat16)
    out["small_integer"] = ((rows.remainder(4) * 4 + cols.remainder(4))).to(torch.bfloat16)
    return out


def _raw_reference(v: torch.Tensor) -> torch.Tensor:
    """Reference for the tutorial's raw 32x32 MFMA accumulator mapping.

    C15 uses two waves.  Wave ``w`` owns the 32-row output block and the
    corresponding 32-row B block.  The identity A makes C[row, col] equal to
    V[row, col] in that block.  The formula below is deliberately kept in
    Python rather than hidden in the compiler lowering.
    """

    ref = torch.zeros((WORKGROUP, 16), device=v.device, dtype=torch.float32)
    vf = v.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        lane_col = lane & 31
        lane_group = lane >> 5
        row = wave * 32 + lane_col
        for r in range(16):
            col = wave * 32 + ((r >> 2) << 3) + lane_group * 4 + (r & 3)
            ref[tid, r] = vf[row, col]
    return ref


def _launch(v: torch.Tensor, debug: torch.Tensor, raw: torch.Tensor) -> None:
    g = torch.zeros((1, TILE, 8), device=v.device, dtype=torch.float32)
    _c15_real_v_tile_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        v, debug, raw, g, num_warps=2
    )


def run_correctness() -> dict[str, object]:
    results: dict[str, object] = {}
    for name, v in _patterns().items():
        debug = torch.full_like(v, torch.tensor(float("nan"), device=v.device))
        raw = torch.empty((WORKGROUP, 16), device=v.device, dtype=torch.float32)
        _launch(v, debug, raw)
        torch.cuda.synchronize()
        debug_ok = torch.equal(debug, v)
        ref = _raw_reference(v)
        raw_max = float((raw - ref).abs().max().item())
        results[name] = {
            "debug_byte_exact": bool(debug_ok),
            "raw_max_abs": raw_max,
            "raw_exact": bool(raw_max == 0.0),
            "raw_finite": bool(torch.isfinite(raw).all().item()),
        }
        print(
            f"{name}: debug_byte_exact={debug_ok} raw_max_abs={raw_max:.8g} "
            f"raw_exact={raw_max == 0.0} "
            f"raw_finite={bool(torch.isfinite(raw).all().item())}"
        )
    return results


def _dump_hsaco(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    # The backend's ordinary HSACO dump is enabled by the caller through the
    # existing repro helper environment.  Keep a small identity manifest here
    # so the report can bind the numerical run to this source.
    (directory / "source_identity.json").write_text(
        json.dumps(
            {
                "source": str(Path(__file__).resolve()),
                "tile": [64, 64],
                "workgroup": 128,
                "mfma": "v_mfma_f32_32x32x8_bf16",
                "mode": "c15_real_v",
                "env_gate": "AVELANG_C15_REAL_TILE=1",
                "lowering": "specialized",
            },
            indent=2,
        )
        + "\n"
    )


_HSACO_DUMP_DIR: Path | None = None


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_substr: str) -> None:
    """Capture the actual runtime code object for the numerical closure.

    This is deliberately a one-shot compile hook.  It does not benchmark the
    kernel and is removed immediately after the first matching compilation.
    """
    if _HSACO_DUMP_DIR is None:
        return

    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"no compiled kernel matched {kernel_substr!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()

    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_dir

    os.environ.setdefault("AVELANG_C15_REAL_TILE", "1")
    os.environ.setdefault("AVELANG_BLOCK_DOT_LOWERING", "specialized")
    os.environ.setdefault("AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT", "1")

    if args.dump_dir is not None:
        _dump_hsaco(args.dump_dir)

        # Compile once before the pattern matrix so the captured HSACO is
        # bound to the exact source/flags used by the correctness run.
        first_v = next(iter(_patterns().values()))
        first_debug = torch.full_like(first_v, torch.tensor(float("nan"), device=first_v.device))
        first_raw = torch.empty((WORKGROUP, 16), device=first_v.device, dtype=torch.float32)
        _maybe_dump_hsaco(
            lambda: _launch(first_v, first_debug, first_raw),
            "_c15_real_v_tile_kernel",
        )

    results = run_correctness()
    if not all(
        item["debug_byte_exact"] and item["raw_exact"] and item["raw_finite"]
        for item in results.values()
    ):
        raise SystemExit(2)
    if args.dump_dir is not None:
        (args.dump_dir / "correctness.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

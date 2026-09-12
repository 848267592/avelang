"""Experimental WG128 score@V physical-consumer contract oracle.

This is a correctness/identity probe only.  It reuses the existing generic
``block_dot_bf16_f32_operand`` ABI and the C15 V producer, while the
environment-gated late lowering keeps two independent 32-column output halves
instead of the historical one-chain-and-duplicate boundary.

The score operand is an identity matrix in CTA shared memory.  Consequently
the raw accumulator fragment exposes the exact V row/column selected by the
consumer.  Distinct V patterns and both output halves make orientation,
source-row, and accumulator ownership mistakes observable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

import avelang
import avelang.language as al


TILE = 64
WORKGROUP = 128


@avelang.jit
def scorev_fragment_oracle_wg128_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
):
    source = al.make_tensor(
        source_ptr, al.bf16, al.make_layout((TILE, TILE), (TILE, 1))
    )
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((TILE, TILE), (TILE, 1))
    )
    raw = al.make_tensor(
        raw_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    # The generic ABI requires a valid G tensor although this isolated V
    # producer/consumer contract does not use it.
    g = al.make_tensor(
        g_ptr, al.f32, al.make_layout((1, TILE, 8), (TILE * 8, 8, 1))
    )

    a_stage = al.make_shared((TILE, TILE), al.bf16)
    b_stage = al.make_shared((TILE, TILE), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)

    # Identity score tile: score[row, k] = 1 only when row == k.  The C15
    # producer fills b_stage from the real global V source.
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if row == col:
            a_stage[row, col] = one

    al.syncthreads()
    acc_low = al.full((16,), 0.0, al.f32)
    acc_high = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage,
        b_stage,
        source,
        debug,
        g,
        tid,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_f32,
        acc_low,
        acc_high,
    )
    for r in al.range(32):
        raw[tid, r] = result[r]


def _raw_reference(source: torch.Tensor) -> torch.Tensor:
    ref = torch.zeros((WORKGROUP, 32), device=source.device, dtype=torch.float32)
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        row = wave * 32 + (lane & 31)
        lane_group = lane >> 5
        for output_half in range(2):
            base = output_half * 32
            for r in range(16):
                col = base + ((r >> 2) << 3) + lane_group * 4 + (r & 3)
                ref[tid, output_half * 16 + r] = src[row, col]
    return ref


def _patterns() -> dict[str, torch.Tensor]:
    rows = torch.arange(TILE, device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(TILE, device="cuda", dtype=torch.float32)[None, :]
    return {
        "zero": torch.zeros((TILE, TILE), device="cuda", dtype=torch.bfloat16),
        "one_hot_low": ((rows == 17) & (cols == 9)).to(torch.bfloat16),
        "one_hot_high": ((rows == 17) & (cols == 41)).to(torch.bfloat16),
        "row_col_code": (
            rows.remainder(8) * 16
            + cols.remainder(16)
            + (cols >= 32).to(torch.float32) * 64
        ).to(torch.bfloat16),
        "checker": ((rows + cols).remainder(2)).to(torch.bfloat16),
    }


def _launch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    debug = torch.full_like(source, torch.tensor(float("nan"), device="cuda"))
    raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    g = torch.zeros((1, TILE, 8), device="cuda", dtype=torch.float32)
    scorev_fragment_oracle_wg128_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        source, debug, raw, g, num_warps=2
    )
    torch.cuda.synchronize()
    return debug, raw


def _objdump() -> str:
    for candidate in (
        "/opt/rocm/llvm/bin/llvm-objdump",
        "/opt/rocm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("llvm-objdump was not found")


def _capture(directory: Path) -> None:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    directory.mkdir(parents=True, exist_ok=True)
    kfrag_dir = directory / "ir" / "vfrag"
    recurrence_dir = directory / "ir" / "recurrence"
    kfrag_dir.mkdir(parents=True, exist_ok=True)
    recurrence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT"] = "1"
    os.environ["AVELANG_QWEN_KFRAG_AB_DUMP_DIR"] = str(kfrag_dir)
    os.environ["AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR"] = str(recurrence_dir)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    hsaco = directory / "scorev_wg128_full64.hsaco"

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if not hsaco.exists():
            hsaco.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        _launch(torch.ones((TILE, TILE), device="cuda", dtype=torch.bfloat16))
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
        for name in (
            "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR",
            "AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
        ):
            os.environ.pop(name, None)

    isa = directory / "scorev_wg128_full64.isa.s"
    isa.write_text(
        subprocess.run(
            [_objdump(), "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (hsaco, isa)
    }
    (directory / "artifact_hashes.sha256").write_text(
        "".join(f"{name}  {digest}\n" for name, digest in hashes.items())
    )
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "contract": "WG128 score@V full64 experimental consumer",
                "source_shape": [64, 64],
                "logical_score_shape": [64, 64],
                "logical_v_shape": [64, 64],
                "mfma": "v_mfma_f32_32x32x8_bf16",
                "waves": 2,
                "independent_accumulator_halves": 2,
                "hsaco_sha256": hashes[hsaco.name],
                "isa_sha256": hashes[isa.name],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()
    os.environ["AVELANG_C15_REAL_TILE"] = "1"
    os.environ["AVELANG_C15_SCOREV_FULL64"] = "1"
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"

    if args.dump_dir is not None:
        _capture(args.dump_dir)

    observations = []
    for name, source in _patterns().items():
        debug, raw = _launch(source)
        expected = _raw_reference(source)
        debug_mismatch = torch.nonzero(debug != source)
        mismatch = torch.nonzero((raw - expected).abs() != 0)
        max_abs = float((raw - expected).abs().max().item())
        first = None
        if mismatch.numel():
            tid, slot = [int(value) for value in mismatch[0].tolist()]
            first = {
                "tid": tid,
                "slot": slot,
                "actual": float(raw[tid, slot].item()),
                "expected": float(expected[tid, slot].item()),
            }
        item = {
            "pattern": name,
            "debug_byte_exact": bool(torch.equal(debug, source)),
            "debug_mismatch_count": int(debug_mismatch.shape[0]),
            "raw_exact": bool(max_abs == 0.0),
            "max_abs": max_abs,
            "finite": bool(torch.isfinite(raw).all().item()),
            "wave0_low": float(raw[0, 0].item()),
            "wave0_high": float(raw[0, 16].item()),
            "wave1_low": float(raw[64, 0].item()),
            "wave1_high": float(raw[64, 16].item()),
            "first_mismatch": first,
        }
        observations.append(item)
        print(
            f"{name}: debug={item['debug_byte_exact']} "
            f"exact={item['raw_exact']} max_abs={max_abs:.8g} "
            f"finite={item['finite']} first={first}"
        )

    result = {
        "contract": {
            "target": "gfx942",
            "workgroup": WORKGROUP,
            "waves": 2,
            "operation": "score@V fragment ownership",
            "score_shape": [64, 64],
            "v_shape": [64, 64],
            "output": [64, 64],
            "mfma": "v_mfma_f32_32x32x8_bf16",
        },
        "observations": observations,
        "all_exact": all(
            item["debug_byte_exact"] and item["raw_exact"] and item["finite"]
            for item in observations
        ),
    }
    output = Path(
        "test/examples/linear_attention/compile_bug/"
        "qwen_t8192_qh_wg128_closure/scorev_wg128_fragment_observation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    if not result["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

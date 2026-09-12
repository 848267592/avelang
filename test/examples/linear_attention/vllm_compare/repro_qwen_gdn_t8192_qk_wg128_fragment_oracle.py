"""Experimental two-wave WG128 Q@K consumer contract oracle.

This probe is deliberately smaller than chunk-o.  It uses the existing
``block_dot_bf16_f32_operand`` ABI, a [32, 64] BF16 K source, and an identity
Q-side stage with different row-half weights.  The source pattern separates K
rows, low/high output columns, and the two physical output waves, so a wrong
K orientation, output-half ownership, or accumulator chain cannot pass by
accident.  It is a correctness/identity probe only; it is not a benchmark.
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


WORKGROUP = 128
TILE = 64


@avelang.jit
def qk_fragment_oracle_wg128_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
):
    source = al.make_tensor(
        source_ptr, al.bf16, al.make_layout((32, 64), (64, 1))
    )
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((32, 64), (64, 1))
    )
    raw = al.make_tensor(
        raw_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    g = al.make_tensor(
        g_ptr, al.f32, al.make_layout((1, TILE, 8), (TILE * 8, 8, 1))
    )
    a_stage = al.make_shared((TILE, TILE), al.bf16)
    b_stage = al.make_shared((TILE, TILE), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)
    three = al.convert(3, al.bf16)

    # A[row, row % 32] is 1 for the first physical output row half and 3 for
    # the second.  The K source is the B operand; the compiler gate must
    # transport all 32x64 source elements into the B stage.
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if col < 32 and row == col:
            a_stage[row, col] = one
        if col < 32 and row == col + 32:
            a_stage[row, col] = three

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


def expected_qk(source: torch.Tensor) -> torch.Tensor:
    expected = torch.empty((WORKGROUP, 32), device=source.device, dtype=torch.float32)
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        output_row = wave * 32 + (lane & 31)
        row_weight = 1.0 if output_row < 32 else 3.0
        k_row = output_row & 31
        for r in range(16):
            low_col = ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
            expected[tid, r] = src[k_row, low_col] * row_weight
            expected[tid, 16 + r] = src[k_row, low_col + 32] * row_weight
    return expected


def _patterns() -> dict[str, torch.Tensor]:
    rows = torch.arange(32, device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(64, device="cuda", dtype=torch.float32)[None, :]
    return {
        "zero": torch.zeros((32, 64), device="cuda", dtype=torch.bfloat16),
        "one_hot_low": ((rows == 17) & (cols == 9)).to(torch.bfloat16),
        "one_hot_high": ((rows == 17) & (cols == 41)).to(torch.bfloat16),
        "row_col_code": (
            rows.remainder(8) * 16
            + cols.remainder(16)
            + (cols >= 32).to(torch.float32) * 64
        ).to(torch.bfloat16),
        "checker": ((rows + cols).remainder(2)).to(torch.bfloat16),
    }


def launch_case(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    debug = torch.full_like(source, torch.tensor(float("nan"), device="cuda"))
    raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    g = torch.zeros((1, TILE, 8), device="cuda", dtype=torch.float32)
    qk_fragment_oracle_wg128_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
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
    kfrag_dir = directory / "ir" / "kfrag"
    recurrence_dir = directory / "ir" / "recurrence"
    kfrag_dir.mkdir(parents=True, exist_ok=True)
    recurrence_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT"] = "1"
    os.environ["AVELANG_QWEN_KFRAG_AB_DUMP_DIR"] = str(kfrag_dir)
    os.environ["AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR"] = str(recurrence_dir)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    hsaco = directory / "qk_wg128_full64.hsaco"

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if not hsaco.exists():
            hsaco.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch_case(torch.ones((32, 64), device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
        for name in (
            "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR",
            "AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
        ):
            os.environ.pop(name, None)

    isa = directory / "qk_wg128_full64.isa.s"
    isa.write_text(
        subprocess.run(
            [_objdump(), "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (hsaco, isa)}
    (directory / "artifact_hashes.sha256").write_text(
        "".join(f"{name}  {digest}\n" for name, digest in hashes.items())
    )
    (directory / "identity.json").write_text(
        json.dumps(
            {
                "contract": "WG128 Q@K full64 experimental consumer",
                "source_shape": [32, 64],
                "logical_q_shape": [64, 32],
                "logical_k_shape": [32, 64],
                "mfma": "v_mfma_f32_32x32x8_bf16",
                "waves": 2,
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
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = "K"
    os.environ["AVELANG_C16_WG128_QH"] = "1"
    os.environ["AVELANG_C16_WG128_QK_FULL64"] = "1"
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"

    if args.dump_dir is not None:
        _capture(args.dump_dir)

    observations = []
    for name, source in _patterns().items():
        debug, raw = launch_case(source)
        expected = expected_qk(source)
        max_abs = float((raw - expected).abs().max().item())
        mismatch = torch.nonzero((raw - expected).abs() > 0)
        debug_mismatch = torch.nonzero(debug != source)
        first_debug = None
        if debug_mismatch.numel():
            debug_row, debug_col = [int(value) for value in debug_mismatch[0].tolist()]
            first_debug = {
                "row": debug_row,
                "col": debug_col,
                "actual_bf16": float(debug[debug_row, debug_col].float().item()),
                "expected_bf16": float(source[debug_row, debug_col].float().item()),
            }
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
            "first_debug_mismatch": first_debug,
            "raw_exact": bool(max_abs == 0.0),
            "max_abs": max_abs,
            "finite": bool(torch.isfinite(raw).all().item()),
            "first_mismatch": first,
            "wave0_low": float(raw[0, 0].item()),
            "wave1_low": float(raw[64, 0].item()),
            "wave0_high": float(raw[0, 16].item()),
            "wave1_high": float(raw[64, 16].item()),
        }
        observations.append(item)
        print(
            f"{name}: debug={item['debug_byte_exact']} "
            f"first_debug={first_debug} exact={item['raw_exact']} "
            f"max_abs={max_abs:.8g} finite={item['finite']} first={first}"
        )

    result = {
        "contract": {
            "target": "gfx942",
            "workgroup": WORKGROUP,
            "waves": 2,
            "operation": "Q@K fragment ownership",
            "q_shape": [64, 32],
            "k_shape": [32, 64],
            "mfma": "v_mfma_f32_32x32x8_bf16",
            "output": [64, 64],
        },
        "observations": observations,
        "all_exact": all(
            item["debug_byte_exact"] and item["raw_exact"] and item["finite"]
            for item in observations
        ),
    }
    output = Path(
        "test/examples/linear_attention/compile_bug/"
        "qwen_t8192_qh_wg128_closure/qk_wg128_fragment_observation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output}")
    if not result["all_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

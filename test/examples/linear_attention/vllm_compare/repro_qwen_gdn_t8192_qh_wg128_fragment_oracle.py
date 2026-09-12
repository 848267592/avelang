"""Two-wave WG128 Q@H fragment oracle for the T=8192 parity audit.

This is an experimental compile/correctness probe only.  It keeps the current
WG128 Q/H producer mapping and uses a logical identity B tile so the raw
32-element result of the block-dot operation can be checked without a full
chunk-o schedule.  The two accumulator inputs are deliberately distinct;
passing the same accumulator twice would only test a single-accumulator
closure and could hide output-tile ownership mistakes.
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
RESULT_PATH = Path(
    "test/examples/linear_attention/compile_bug/"
    "qwen_t8192_qh_wg128_closure/qh_wg128_full_tile_observation.json"
)


@avelang.jit
def qh_fragment_oracle_wg128_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    b_high: al.constexpr,
):
    source = al.make_tensor(
        source_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
    )
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
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
    two = al.convert(b_high, al.bf16)

    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if row < 32 and col == row:
            b_stage[row, col] = one
        if row < 32 and col == row + 32:
            b_stage[row, col] = two
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


def expected_q_identity(source: torch.Tensor, *, wave_rows: bool) -> torch.Tensor:
    """Expected legacy one-accumulator writeback used by the old closure."""

    expected = torch.empty((WORKGROUP, 32), device=source.device, dtype=torch.float32)
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        row = (wave * 32 if wave_rows else 0) + (lane & 31)
        for r in range(16):
            col = ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
            expected[tid, r] = src[row, col]
            expected[tid, 16 + r] = src[row, col]
    return expected


def expected_q_full64(source: torch.Tensor, b_high: float) -> torch.Tensor:
    """Expected two-accumulator writeback for a 64x64 Q@H tile."""
    expected = torch.empty((WORKGROUP, 32), device=source.device, dtype=torch.float32)
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        row = wave * 32 + (lane & 31)
        for r in range(16):
            source_col = ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
            expected[tid, r] = src[row, source_col] * 1.0
            expected[tid, 16 + r] = src[row, source_col] * b_high
    return expected


def launch_case(source: torch.Tensor, b_high: float) -> torch.Tensor:
    """Run one row-half/column-half disambiguation case."""
    debug = torch.full_like(source, torch.tensor(float("nan"), device="cuda"))
    raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    g = torch.zeros((1, TILE, 8), device="cuda", dtype=torch.float32)

    # The kernel's B stage is initialized from the same source code for every
    # case.  A low output half has weight 1 and a high output half has the
    # caller-selected weight, so row and column ownership cannot be confused.
    qh_fragment_oracle_wg128_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        source, debug, raw, g, b_high=b_high, num_warps=2
    )
    torch.cuda.synchronize()
    if not torch.equal(debug, source):
        raise AssertionError("WG128 full-tile oracle debug readback is not exact")
    if not torch.isfinite(raw).all().item():
        raise AssertionError("WG128 full-tile oracle produced non-finite output")
    return raw


def _objdump() -> str:
    for candidate in (
        "/opt/rocm/llvm/bin/llvm-objdump",
        "/opt/rocm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("llvm-objdump was not found")


def _capture_full64(directory: Path) -> None:
    """Capture one opt-in full64 compile and its audit snapshots."""
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
    hsaco = directory / "qh_wg128_full64.hsaco"

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if not hsaco.exists():
            hsaco.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        source = torch.ones((64, 32), device="cuda", dtype=torch.bfloat16)
        launch_case(source, 2.0)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
        for name in (
            "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR",
            "AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
        ):
            os.environ.pop(name, None)

    isa = directory / "qh_wg128_full64.isa.s"
    isa.write_text(
        subprocess.run(
            [_objdump(), "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )
    (directory / "artifact_hashes.sha256").write_text(
        "".join(
            f"{path.name}  {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
            for path in (hsaco, isa)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = "Q"
    os.environ["AVELANG_C16_WG128_QH"] = "1"
    os.environ["AVELANG_C16_WG128_QH_FULL64"] = "1"
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"

    if args.dump_dir is not None:
        _capture_full64(args.dump_dir)

    cols = torch.zeros((1, 32), device="cuda", dtype=torch.float32)
    low_source = torch.ones((64, 32), device="cuda", dtype=torch.float32)
    low_source[32:, :] = 7.0
    row_code = (torch.arange(64, device="cuda", dtype=torch.float32)[:, None] * 32 + cols)
    row_code = row_code.to(torch.bfloat16)
    half_code = low_source.to(torch.bfloat16)

    observations = []
    for name, source in (("row_code", row_code), ("row_half", half_code)):
        for b_high in (1.0, 2.0):
            raw = launch_case(source, b_high)
            expected = expected_q_full64(source, b_high)
            exact = bool(torch.equal(raw, expected))
            max_abs = float((raw - expected).abs().max().item())
            values = {
                "wave0_low": float(raw[0, 0].item()),
                "wave1_low": float(raw[64, 0].item()),
                "wave0_high": float(raw[0, 16].item()),
                "wave1_high": float(raw[64, 16].item()),
            }
            print(
                f"{name} B_high={b_high:g}: "
                f"wave0_low={values['wave0_low']:.8g} "
                f"wave1_low={values['wave1_low']:.8g} "
                f"wave0_high={values['wave0_high']:.8g} "
                f"wave1_high={values['wave1_high']:.8g}"
            )
            observations.append({
                "source_pattern": name,
                "b_high_weight": b_high,
                "observed": values,
                "full64_expected_exact": exact,
                "full64_expected_max_abs": max_abs,
            })
            print(
                f"{name} B_high={b_high:g}: full64_expected_exact={exact} "
                f"max_abs={max_abs:.8g}"
            )

    result = {
        "contract": {
            "target": "gfx942",
            "workgroup": 128,
            "waves": 2,
            "operation": "Q@H fragment ownership",
            "source_shape": [64, 32],
            "b_logical_shape": [64, 64],
            "low_output_column_weight": 1.0,
        },
        "producer_gate": {
            "debug_readback": "checked exact for every case",
            "finite": "checked for every case",
        },
        "observations": observations,
        "interpretation": {
            "wave0_source_row_half": "low rows in this closure",
            "wave1_source_row_half": "not proven by a full chunk contract; minimal closure observes low-row values",
            "wave1_output_column_half": "high column half is selected when B_high_weight=2",
            "status": "ownership_mapping_not_closed",
            "full64_contract_gate": "pending until both source-row and output-column cases are exact",
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()

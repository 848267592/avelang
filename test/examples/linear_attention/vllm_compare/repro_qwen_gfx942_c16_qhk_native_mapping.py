"""C16 selected-T2048 Q/H/K physical mapping closure.

This is an experimental numerical repro only.  It deliberately uses the
existing ``block_dot_bf16_f32_operand`` source ABI and lets the compiler's
``c16.real_tile_role`` gate select Q, H, or K.  The source contains no
Qwen-specific runtime address table: the role only selects the generic C13
physical plan for the selected T2048/WG256 Triton encoding.

The first observable is a BF16 LDS readback.  The second is the raw MFMA
fragment.  The readback makes a producer/shared mapping failure unambiguous;
the raw result is compared with the ordinary 32x32 MFMA accumulator writeback
oracle for the corresponding Q@I, I@H, or I@K closure.
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


WORKGROUP = 256
TILE = 64


@avelang.jit
def _c16_q_real_tile_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
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
    # A 64x64 shared identity tile has 4096 elements.  With 256 threads,
    # sixteen rounds cover rows 0..63 exactly; more rounds would write past
    # the shared allocation and contaminate the other operand stage.
    for rep in al.range(16):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if row < 32 and (col == row or col == row + 32):
            b_stage[row, col] = one
    al.syncthreads()
    acc = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage, b_stage, source, debug, g, tid, zero_i32, zero_i32,
        zero_i32, zero_i32, zero_i32, zero_f32, acc, acc
    )
    for r in al.range(32):
        raw[tid, r] = result[r]


@avelang.jit
def _c16_q_dual_consumer_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_a_ptr: al.Pointer(al.f32),
    raw_b_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
):
    source = al.make_tensor(
        source_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
    )
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
    )
    raw_a = al.make_tensor(
        raw_a_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    raw_b = al.make_tensor(
        raw_b_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    g = al.make_tensor(
        g_ptr, al.f32, al.make_layout((1, TILE, 8), (TILE * 8, 8, 1))
    )
    a_stage = al.make_shared((TILE, TILE), al.bf16)
    b_stage = al.make_shared((TILE, TILE), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)
    for rep in al.range(16):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if row < 32 and (col == row or col == row + 32):
            b_stage[row, col] = one
    al.syncthreads()
    acc_a = al.full((16,), 0.0, al.f32)
    acc_b = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    # Both consumers use the same source/stage SSA values.  The C16 dual
    # compiler gate emits the Q producer only for the first op; the second op
    # is a consumer-only MFMA closure over that same physical LDS tile.
    result_a = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage, b_stage, source, debug, g, tid, zero_i32, zero_i32,
        zero_i32, zero_i32, zero_i32, zero_f32, acc_a, acc_a
    )
    # Use the existing generic transposed spelling only to give the second
    # consumer a distinct source identity.  Under AVELANG_C16_Q_DUAL the
    # compiler still selects the Q physical role, so no second Q producer or
    # H/K mapping is introduced.
    result_b = al.amdgpu.block_dot_bf16_f32_operand_transposed(
        a_stage, b_stage, source, debug, g, tid, zero_i32, zero_i32,
        zero_i32, zero_i32, zero_i32, zero_f32, acc_b, acc_b
    )
    for r in al.range(32):
        raw_a[tid, r] = result_a[r]
        raw_b[tid, r] = result_b[r]


@avelang.jit
def _c16_h_real_tile_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
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
    # Keep the identity-stage initialization in bounds: 16 rounds cover the
    # complete 64x64 tile for a 256-thread CTA.
    for rep in al.range(16):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if col < 32 and (row == col or row == col + 32):
            a_stage[row, col] = one
    al.syncthreads()
    acc = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage, b_stage, source, debug, g, tid, zero_i32, zero_i32,
        zero_i32, zero_i32, zero_i32, zero_f32, acc, acc
    )
    for r in al.range(32):
        raw[tid, r] = result[r]


@avelang.jit
def _c16_k_real_tile_kernel(
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
    # Keep the identity-stage initialization in bounds: 16 rounds cover the
    # complete 64x64 tile for a 256-thread CTA.
    for rep in al.range(16):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if col < 32 and (row == col or row == col + 32):
            a_stage[row, col] = one
    al.syncthreads()
    acc = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage, b_stage, source, debug, g, tid, zero_i32, zero_i32,
        zero_i32, zero_i32, zero_i32, zero_f32, acc, acc
    )
    for r in al.range(32):
        raw[tid, r] = result[r]


_KERNELS = {"Q": _c16_q_real_tile_kernel, "H": _c16_h_real_tile_kernel,
            "K": _c16_k_real_tile_kernel}


def _patterns(role: str) -> dict[str, torch.Tensor]:
    shape = (32, 64) if role == "K" else (64, 32)
    rows = torch.arange(shape[0], device="cuda", dtype=torch.float32)[:, None]
    cols = torch.arange(shape[1], device="cuda", dtype=torch.float32)[None, :]
    return {
        "zero": torch.zeros(shape, device="cuda", dtype=torch.bfloat16),
        "one_hot": ((rows == 17) & (cols == 9)).to(torch.bfloat16),
        "row_code": (rows.remainder(16) + cols.remainder(4)).to(torch.bfloat16),
        "col_code": (rows.remainder(4) * 4 + cols.remainder(16)).to(torch.bfloat16),
        "checker": ((rows + cols).remainder(2)).to(torch.bfloat16),
    }


def _raw_reference(role: str, source: torch.Tensor) -> torch.Tensor:
    """Reference for the 2x2 MFMA32 accumulator writeback layout."""

    ref = torch.zeros((WORKGROUP, 16), device=source.device, dtype=torch.float32)
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        row = (wave // 2) * 32 + (lane & 31)
        for r in range(16):
            col = (wave & 1) * 32 + ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
            if role == "Q":
                # The probe's B tile has B[k,k] = B[k,k+32] = 1.
                # Therefore Q @ I selects Q[row, col % 32] for both output
                # column halves.
                ref[tid, r] = src[row, col % 32]
            elif role == "H":
                # A[row,k] is one at k=row % 32, and H is stored as
                # [output, K], so I @ H^T selects H[col, row % 32].
                ref[tid, r] = src[col, row % 32]
            else:
                # A[row,k] is one at k=row % 32, and K is stored as
                # [K, output], so I @ K selects K[row % 32, col].
                ref[tid, r] = src[row % 32, col]
    return ref


def _launch(role: str, source: torch.Tensor, debug: torch.Tensor, raw: torch.Tensor) -> None:
    # The role is a compiler representation gate, so every compile and every
    # correctness launch must set it explicitly.  This avoids accidentally
    # running all three kernels under the role left by the last HSACO capture.
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = role
    g = torch.zeros((1, TILE, 8), device=source.device, dtype=torch.float32)
    _KERNELS[role][lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        source, debug, raw, g, num_warps=4
    )


def _launch_q_dual(
    source: torch.Tensor,
    debug: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
) -> None:
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = "Q"
    os.environ["AVELANG_C16_Q_DUAL"] = "1"
    g = torch.zeros((1, TILE, 8), device=source.device, dtype=torch.float32)
    _c16_q_dual_consumer_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        source, debug, raw_a, raw_b, g, num_warps=4
    )


def run_correctness() -> dict[str, object]:
    results: dict[str, object] = {}
    os.environ.pop("AVELANG_C16_Q_DUAL", None)
    for role in ("Q", "H", "K"):
        for name, source in _patterns(role).items():
            debug = torch.full_like(source, torch.tensor(float("nan"), device="cuda"))
            raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
            _launch(role, source, debug, raw)
            torch.cuda.synchronize()
            ref = _raw_reference(role, source)
            raw_low = raw[:, :16]
            raw_high = raw[:, 16:]
            raw_max = float((raw_low - ref).abs().max().item())
            high_max = float((raw_high - ref).abs().max().item())
            mismatch = torch.nonzero(
                (raw_low - ref).abs() > 0
            )
            first_mismatch = None
            if mismatch.numel():
                mtid, mslot = [int(x) for x in mismatch[0].tolist()]
                first_mismatch = {
                    "tid": mtid,
                    "slot": mslot,
                    "actual": float(raw_low[mtid, mslot].item()),
                    "expected": float(ref[mtid, mslot].item()),
                }
            key = f"{role}_{name}"
            results[key] = {
                "debug_byte_exact": bool(torch.equal(debug, source)),
                "raw_low_max_abs": raw_max,
                "raw_high_max_abs": high_max,
                "raw_low_exact": bool(raw_max == 0.0),
                "raw_high_exact": bool(high_max == 0.0),
                "raw_exact": bool(raw_max == 0.0 and high_max == 0.0),
                "raw_finite": bool(torch.isfinite(raw).all().item()),
                "first_mismatch": first_mismatch,
            }
            print(
                f"{key}: debug_byte_exact={results[key]['debug_byte_exact']} "
                f"raw_low={raw_max:.8g} raw_high={high_max:.8g} "
                f"finite={results[key]['raw_finite']} "
                f"first_mismatch={first_mismatch}"
            )

    # One physical Q producer must feed two independent MFMA consumers.  The
    # compiler-side C16_Q_DUAL gate suppresses only the second producer; the
    # two returned raw fragments remain separately checked against the same
    # independent logical Q reference.
    for name, source in _patterns("Q").items():
        debug = torch.empty_like(source)
        raw_a = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
        raw_b = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
        _launch_q_dual(source, debug, raw_a, raw_b)
        torch.cuda.synchronize()
        ref = _raw_reference("Q", source)
        raw_a_low = raw_a[:, :16]
        raw_b_low = raw_b[:, :16]
        raw_a_high = raw_a[:, 16:]
        raw_b_high = raw_b[:, 16:]
        key = f"Q_dual_{name}"
        results[key] = {
            "debug_byte_exact": bool(torch.equal(debug, source)),
            "consumer_a_raw_exact": bool(torch.equal(raw_a_low, ref)),
            "consumer_b_raw_exact": bool(torch.equal(raw_b_low, ref)),
            "consumer_a_high_exact": bool(torch.equal(raw_a_high, ref)),
            "consumer_b_high_exact": bool(torch.equal(raw_b_high, ref)),
            "consumer_a_finite": bool(torch.isfinite(raw_a).all().item()),
            "consumer_b_finite": bool(torch.isfinite(raw_b).all().item()),
            "single_physical_producer": True,
        }
        print(
            f"{key}: debug_byte_exact={results[key]['debug_byte_exact']} "
            f"consumer_a={results[key]['consumer_a_raw_exact']} "
            f"consumer_b={results[key]['consumer_b_raw_exact']} "
            f"finite={results[key]['consumer_a_finite'] and results[key]['consumer_b_finite']}"
        )
    os.environ.pop("AVELANG_C16_Q_DUAL", None)
    return results


def _dump_identity(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source_identity.json").write_text(
        json.dumps(
            {
                "source": str(Path(__file__).resolve()),
                "roles": ["Q", "H", "K"],
                "selected_target": "T2048_WG256_gfx942",
                "encoding": {
                    "Q_H": {
                        "shape": [64, 32],
                        "sizePerThread": [1, 8],
                        "threadsPerWarp": [16, 4],
                        "warpsPerCTA": [4, 1],
                        "order": [1, 0],
                    },
                    "K": {
                        "shape": [32, 64],
                        "sizePerThread": [8, 1],
                        "threadsPerWarp": [4, 16],
                        "warpsPerCTA": [1, 4],
                        "order": [0, 1],
                    },
                },
                "mfma": "v_mfma_f32_32x32x8_bf16",
                "workgroup": WORKGROUP,
                "env_gate": "AVELANG_C16_REAL_TILE_ROLE=Q|H|K",
            },
            indent=2,
        )
        + "\n"
    )


def _capture_hsaco(directory: Path) -> None:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    seen: set[str] = set()

    def wrapped_compile(self, src, target, options=None):
        # `_make_kernel` returns a JIT wrapper whose `.fn` is already the
        # Python function, whereas directly decorated functions in older
        # repros expose one more wrapper level.  The launch loop sets the
        # role explicitly, so use that compiler gate as the capture identity
        # instead of relying on wrapper internals.
        role = os.environ.get("AVELANG_C16_REAL_TILE_ROLE", "unknown")
        binary = original_compile(self, src, target, options)
        if role not in seen:
            (directory / f"c16_{role.lower()}.hsaco").write_bytes(binary)
            seen.add(role)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        for role in ("Q", "H", "K"):
            os.environ["AVELANG_C16_REAL_TILE_ROLE"] = role
            source = next(iter(_patterns(role).values()))
            debug = torch.empty_like(source)
            raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
            _launch(role, source, debug, raw)
            torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()
    os.environ.setdefault("AVELANG_BLOCK_DOT_LOWERING", "specialized")
    os.environ.setdefault("AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT", "1")
    if args.dump_dir is not None:
        _dump_identity(args.dump_dir)
        _capture_hsaco(args.dump_dir)
    results = run_correctness()
    if args.dump_dir is not None:
        (args.dump_dir / "correctness.json").write_text(json.dumps(results, indent=2) + "\n")

    def passed(item: dict[str, object]) -> bool:
        if "raw_exact" in item:
            return bool(
                item["debug_byte_exact"]
                and item["raw_exact"]
                and item["raw_finite"]
            )
        return bool(
            item["debug_byte_exact"]
            and item["consumer_a_raw_exact"]
            and item["consumer_b_raw_exact"]
            and item["consumer_a_high_exact"]
            and item["consumer_b_high_exact"]
            and item["consumer_a_finite"]
            and item["consumer_b_finite"]
            and item["single_physical_producer"]
        )

    if not all(
        passed(item)
        for item in results.values()
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

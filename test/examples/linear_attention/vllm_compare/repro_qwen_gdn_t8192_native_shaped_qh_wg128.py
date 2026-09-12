"""Experimental T=8192 native-shaped AveLang Q@H path.

This is an isolated Q@H machine-contract probe, not a chunk-o implementation.
It deliberately uses the already closed C16 WG128/full64 lowering:

    gfx942 / wave64 / WG128 / two waves
    Q[64, 32] @ H[32, 64] -> FP32[64, 64]

The source keeps the producer, shared stage and two accumulator halves explicit.
The opt-in compiler gates select the existing native-shaped physical plan.  Z5B,
Q@K, score@V, scheduling and benchmark paths are not imported or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

import avelang
import avelang.language as al


WORKGROUP = 128
TILE = 64
DEFAULT_OUT = Path(
    "test/examples/linear_attention/compile_bug/"
    "qwen_t8192_native_shaped_qh_wg128"
)


def _packed_h_enabled() -> bool:
    return os.environ.get("AVELANG_C16_WG128_QH_PACKED_H") == "1"


@avelang.jit
def native_shaped_qh_t8192_wg128_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    b_high: al.constexpr,
):
    """Q@H-only source with the selected native two-wave contract."""

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
    q_stage = al.make_shared((TILE, TILE), al.bf16)
    h_stage = al.make_shared((TILE, TILE), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)
    high = al.convert(b_high, al.bf16)

    # The two waves jointly initialize Q.  The baseline H arm keeps the
    # ordinary row-major identity tile.  The packed-H arm instead writes the
    # inverse of the selected native #shared1 packet map: one vector<2xi32>
    # is four contiguous BF16 values at h_base xor {0,16,32,48}.
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        q_stage[row, col] = zero
        h_stage[row, col] = zero
        if row < 32 and col == row:
            h_stage[row, col] = one
        if row < 32 and col == row + 32:
            h_stage[row, col] = high
    al.syncthreads()

    acc_low = al.full((16,), 0.0, al.f32)
    acc_high = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        q_stage,
        h_stage,
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


@avelang.jit
def native_shaped_qh_t8192_wg128_packed_h_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
    b_high: al.constexpr,
):
    """Same Q@H probe with a fixed native-shaped packed H operand."""

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
    q_stage = al.make_shared((TILE, TILE), al.bf16)
    # Native #shared1 stores the logical H[32,64] operand in a compact
    # [64,32] BF16 physical region.  The packet addresses below are the
    # selected Triton producer contract, not the older self-consistent
    # compact-H recipe.  The consumer reads the same address set with the
    # four MFMA packet offsets {0,16,32,48}.
    h_stage = al.make_shared((TILE, 32), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)
    high = al.convert(b_high, al.bf16)

    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        q_stage[row, col] = zero

    h_words = al.view(
        h_stage, al.u32, al.make_layout((TILE, 8, 2), (16, 2, 1))
    )
    # Keep the selected Triton packet map explicit in the source IR.  The
    # old packet loop constructed one packet and immediately stored it, which
    # left four independent store regions after lowering.  This form computes
    # one pbase, constructs all four 64-bit packets first, and emits the two
    # sibling store pairs only after every packet is ready.
    producer_tid = tid & (WORKGROUP - 1)
    pbase = ((producer_tid << 4) & 2032) ^ (producer_tid & 56)

    # fragment 0: addr0 = pbase
    h_byte0 = pbase
    consumer_tid0 = producer_tid >> 2
    consumer_t1130 = (consumer_tid0 << 6) & 1984
    consumer_t1270 = (consumer_tid0 >> 2) & 8
    consumer_t1250 = (consumer_tid0 << 5) & 2048
    logical_h_base0 = consumer_t1130 | consumer_t1270 | consumer_t1250
    logical_byte0 = logical_h_base0 + ((producer_tid & 3) << 4)
    physical_element0 = h_byte0 >> 1
    physical_row0 = physical_element0 >> 5
    physical_group0 = (physical_element0 & 31) >> 2
    logical_row0 = (logical_byte0 >> 1) & 31
    logical_col0 = logical_byte0 >> 6
    packet0 = al.full((2,), 0, al.u32)
    for pair0 in al.range(2):
        row00 = logical_row0 + (pair0 << 1)
        row01 = row00 + 1
        value00 = zero
        value01 = zero
        if row00 < 32 and logical_col0 == row00:
            value00 = one
        if row00 < 32 and logical_col0 == row00 + 32:
            value00 = high
        if row01 < 32 and logical_col0 == row01:
            value01 = one
        if row01 < 32 and logical_col0 == row01 + 32:
            value01 = high
        bits00 = al.convert(al.bitcast(value00, al.u16), al.u32)
        bits01 = al.convert(al.bitcast(value01, al.u16), al.u32)
        packet0[pair0] = bits00 | (bits01 << 16)

    # fragment 1: addr1 = pbase | 2048
    h_byte1 = pbase | 2048
    consumer_tid1 = (producer_tid >> 2) | 64
    consumer_t1131 = (consumer_tid1 << 6) & 1984
    consumer_t1271 = (consumer_tid1 >> 2) & 8
    consumer_t1251 = (consumer_tid1 << 5) & 2048
    logical_h_base1 = consumer_t1131 | consumer_t1271 | consumer_t1251
    logical_byte1 = logical_h_base1 + ((producer_tid & 3) << 4)
    physical_element1 = h_byte1 >> 1
    physical_row1 = physical_element1 >> 5
    physical_group1 = (physical_element1 & 31) >> 2
    logical_row1 = (logical_byte1 >> 1) & 31
    logical_col1 = logical_byte1 >> 6
    packet1 = al.full((2,), 0, al.u32)
    for pair1 in al.range(2):
        row10 = logical_row1 + (pair1 << 1)
        row11 = row10 + 1
        value10 = zero
        value11 = zero
        if row10 < 32 and logical_col1 == row10:
            value10 = one
        if row10 < 32 and logical_col1 == row10 + 32:
            value10 = high
        if row11 < 32 and logical_col1 == row11:
            value11 = one
        if row11 < 32 and logical_col1 == row11 + 32:
            value11 = high
        bits10 = al.convert(al.bitcast(value10, al.u16), al.u32)
        bits11 = al.convert(al.bitcast(value11, al.u16), al.u32)
        packet1[pair1] = bits10 | (bits11 << 16)

    # fragment 2: addr2 = pbase ^ 8
    h_byte2 = pbase ^ 8
    consumer_tid2 = (producer_tid >> 2) | 32
    consumer_t1132 = (consumer_tid2 << 6) & 1984
    consumer_t1272 = (consumer_tid2 >> 2) & 8
    consumer_t1252 = (consumer_tid2 << 5) & 2048
    logical_h_base2 = consumer_t1132 | consumer_t1272 | consumer_t1252
    logical_byte2 = logical_h_base2 + ((producer_tid & 3) << 4)
    physical_element2 = h_byte2 >> 1
    physical_row2 = physical_element2 >> 5
    physical_group2 = (physical_element2 & 31) >> 2
    logical_row2 = (logical_byte2 >> 1) & 31
    logical_col2 = logical_byte2 >> 6
    packet2 = al.full((2,), 0, al.u32)
    for pair2 in al.range(2):
        row20 = logical_row2 + (pair2 << 1)
        row21 = row20 + 1
        value20 = zero
        value21 = zero
        if row20 < 32 and logical_col2 == row20:
            value20 = one
        if row20 < 32 and logical_col2 == row20 + 32:
            value20 = high
        if row21 < 32 and logical_col2 == row21:
            value21 = one
        if row21 < 32 and logical_col2 == row21 + 32:
            value21 = high
        bits20 = al.convert(al.bitcast(value20, al.u16), al.u32)
        bits21 = al.convert(al.bitcast(value21, al.u16), al.u32)
        packet2[pair2] = bits20 | (bits21 << 16)

    # fragment 3: addr3 = (pbase ^ 8) | 2048
    h_byte3 = (pbase ^ 8) | 2048
    consumer_tid3 = (producer_tid >> 2) | 32 | 64
    consumer_t1133 = (consumer_tid3 << 6) & 1984
    consumer_t1273 = (consumer_tid3 >> 2) & 8
    consumer_t1253 = (consumer_tid3 << 5) & 2048
    logical_h_base3 = consumer_t1133 | consumer_t1273 | consumer_t1253
    logical_byte3 = logical_h_base3 + ((producer_tid & 3) << 4)
    # h_byte3 is exactly h_byte2 + 2048 for this fixed packet map.  Express
    # that relation through packet2's physical view coordinates so the LLVM
    # producer can retain one shared base and a fixed +2048 sibling offset.
    # The effective byte address is unchanged.
    physical_row3 = physical_row2 + 32
    physical_group3 = physical_group2
    logical_row3 = (logical_byte3 >> 1) & 31
    logical_col3 = logical_byte3 >> 6
    packet3 = al.full((2,), 0, al.u32)
    for pair3 in al.range(2):
        row30 = logical_row3 + (pair3 << 1)
        row31 = row30 + 1
        value30 = zero
        value31 = zero
        if row30 < 32 and logical_col3 == row30:
            value30 = one
        if row30 < 32 and logical_col3 == row30 + 32:
            value30 = high
        if row31 < 32 and logical_col3 == row31:
            value31 = one
        if row31 < 32 and logical_col3 == row31 + 32:
            value31 = high
        bits30 = al.convert(al.bitcast(value30, al.u16), al.u32)
        bits31 = al.convert(al.bitcast(value31, al.u16), al.u32)
        packet3[pair3] = bits30 | (bits31 << 16)

    # Keep the two sibling pairs adjacent in the source IR.  The effective
    # byte addresses are exactly the frozen pbase map above.
    h_words[physical_row0, physical_group0] = packet0
    h_words[physical_row1, physical_group1] = packet1
    h_words[physical_row2, physical_group2] = packet2
    h_words[physical_row3, physical_group3] = packet3

    al.syncthreads()
    acc_low = al.full((16,), 0.0, al.f32)
    acc_high = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        q_stage,
        h_stage,
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


def expected_qh(source: torch.Tensor, b_high: float) -> torch.Tensor:
    expected = torch.empty(
        (WORKGROUP, 32), device=source.device, dtype=torch.float32
    )
    src = source.float()
    for tid in range(WORKGROUP):
        wave = tid // 64
        lane = tid & 63
        row = wave * 32 + (lane & 31)
        for r in range(16):
            col = ((r >> 2) << 3) + ((lane >> 5) * 4) + (r & 3)
            expected[tid, r] = src[row, col]
            expected[tid, 16 + r] = src[row, col] * b_high
    return expected


def launch_case(source: torch.Tensor, b_high: float) -> tuple[torch.Tensor, float]:
    debug = torch.full_like(source, torch.tensor(float("nan"), device="cuda"))
    raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    g = torch.zeros((1, TILE, 8), device="cuda", dtype=torch.float32)
    kernel = (
        native_shaped_qh_t8192_wg128_packed_h_kernel
        if _packed_h_enabled()
        else native_shaped_qh_t8192_wg128_kernel
    )
    kernel[
        lambda: ((1, 1, 1), (WORKGROUP, 1, 1))
    ](source, debug, raw, g, b_high=b_high, num_warps=2)
    torch.cuda.synchronize()
    if not torch.equal(debug, source):
        raise AssertionError("native-shaped Q@H debug readback is not BF16-exact")
    if not bool(torch.isfinite(raw).all().item()):
        raise AssertionError("native-shaped Q@H output is not finite")
    expected = expected_qh(source, b_high)
    max_abs = float((raw - expected).abs().max().item())
    if not torch.equal(raw, expected):
        raise AssertionError(f"native-shaped Q@H max_abs={max_abs}")
    return raw, max_abs


def _tool(name: str) -> str:
    for candidate in (
        f"/opt/rocm/llvm/bin/{name}",
        f"/opt/rocm/bin/{name}",
        shutil.which(name),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(f"{name} was not found")


def _capture_compile(out: Path, source: torch.Tensor) -> None:
    """Capture compiler-owned IR plus the exact binary used by the oracle."""
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ir_dir = out / "compiler_ir"
    ir_dir.mkdir(exist_ok=True)
    link_debug = out / "link_debug"
    link_debug.mkdir(exist_ok=True)
    os.environ["AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT"] = "1"
    os.environ["AVELANG_QWEN_KFRAG_AB_DUMP_DIR"] = str(ir_dir / "kfrag")
    os.environ["AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR"] = str(
        ir_dir / "recurrence"
    )
    os.environ["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(link_debug)
    packed_h = _packed_h_enabled()
    kernel_name = (
        "native_shaped_qh_t8192_wg128_packed_h_kernel"
        if packed_h
        else "native_shaped_qh_t8192_wg128_kernel"
    )
    hsaco = out / (
        "native_shaped_qh_wg128_packed_h.hsaco"
        if packed_h
        else "native_shaped_qh_wg128.hsaco"
    )
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if not hsaco.exists():
            hsaco.write_bytes(binary)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch_case(source, 2.0)
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
        for name in (
            "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR",
            "AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR",
            "AVELANG_AMDGPU_LINK_DEBUG_DIR",
        ):
            os.environ.pop(name, None)

    objdump = _tool("llvm-objdump")
    readelf = _tool("llvm-readelf")
    isa = out / "final_isa.s"
    notes = out / "code_object_notes.txt"
    isa.write_text(
        subprocess.run(
            [objdump, "-d", "--no-show-raw-insn", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )
    notes.write_text(
        subprocess.run(
            [readelf, "--notes", str(hsaco)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    )

    isa_text = isa.read_text()
    counts = {
        "mfma32": len(re.findall(r"\bv_mfma_f32_32x32x8_bf16\b", isa_text)),
        "global_or_buffer_load": len(
            re.findall(r"\b(?:global_load|buffer_load)", isa_text)
        ),
        "ds_read": len(re.findall(r"\bds_read", isa_text)),
        "ds_write": len(re.findall(r"\bds_write", isa_text)),
        "v_perm_b32": len(re.findall(r"\bv_perm_b32\b", isa_text)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", isa_text)),
        "s_waitcnt": len(re.findall(r"\bs_waitcnt\b", isa_text)),
    }
    (out / "static_isa_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    (out / "artifact_hashes.sha256").write_text(
        "".join(
            f"{path.name}  {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
            for path in (hsaco, isa, notes)
        )
    )

    # The compiler hook normally emits exact linker argv.  Replay it when it
    # is available; otherwise the report records MIR as unavailable instead of
    # pretending that an llc stop point is linker-LTO MIR.
    argv = sorted(link_debug.glob("*.argv.txt"))
    if argv:
        replay = Path(__file__).resolve().parents[1] / "compile_bug" / "qwen_mfma32_lowering_ladder" / "replay_qwen_v29_lto_mir.py"
        exact = out / "exact_lto"
        exact.mkdir(exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(replay), "--argv-file", str(argv[0]), "--out-dir", str(exact), "--kernel", kernel_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out / "exact_lto_replay.stdout.txt").write_text(result.stdout)
        (out / "exact_lto_replay.returncode").write_text(str(result.returncode) + "\n")
    else:
        (out / "mir.status.txt").write_text(
            "exact linker argv was not emitted by the active runtime\n"
        )

    manifest = sorted(
        str(path.relative_to(out))
        for path in out.rglob("*")
        if path.is_file()
    )
    (out / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "source": str(Path(__file__).resolve()),
                "pipeline": ["source", "compiler_ir", "llvm", "mir_if_available", "isa", "hsaco"],
                "files": manifest,
                "no_benchmark": True,
                "z5b_modified": False,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = "Q"
    os.environ["AVELANG_C16_WG128_QH"] = "1"
    os.environ["AVELANG_C16_WG128_QH_FULL64"] = "1"
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"

    row_code = (
        torch.arange(64, device="cuda", dtype=torch.float32)[:, None]
        * 32
        + torch.zeros((1, 32), device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    row_half = torch.ones((64, 32), device="cuda", dtype=torch.bfloat16)
    row_half[32:, :] = 7

    args.dump_dir = args.dump_dir.resolve()
    # Capture before any correctness case can populate the JIT cache.  The
    # capture wrapper must observe the first compilation.
    _capture_compile(args.dump_dir, row_code)

    observations = []
    for name, source in (("row_code", row_code), ("row_half", row_half)):
        for b_high in (1.0, 2.0):
            _, max_abs = launch_case(source, b_high)
            observations.append(
                {"source_pattern": name, "b_high": b_high, "max_abs": max_abs}
            )
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    (args.dump_dir / "correctness.json").write_text(
        json.dumps(
            {
                "contract": {
                    "target": "gfx942",
                    "T": 8192,
                    "logical_operation": "Q[64,32] @ H[32,64] -> f32[64,64]",
                    "workgroup": 128,
                    "waves": 2,
                    "num_warps": 2,
                    "mfma": "v_mfma_f32_32x32x8_bf16",
                    "z5b_modified": False,
                    "qk_scorev_touched": False,
                    "packed_h_enabled": _packed_h_enabled(),
                },
                "observations": observations,
                "all_max_abs_zero": all(x["max_abs"] == 0.0 for x in observations),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"out_dir": str(args.dump_dir), "observations": observations}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the final A70 object against the existing C67 BW16 golden ledger."""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = ROOT / "test/examples/kda/reports/A_20260927_70_BW16_TRUE_W4_REBUILD"
HSACO = REPORT_DIR / "A_20260927_70_T4096_final.hsaco"
ISA = REPORT_DIR / "A_20260927_70_T4096_final.isa"
READOBJ = REPORT_DIR / "A_20260927_70_T4096_final.readobj.txt"
SOURCE = ROOT / "test/examples/kda/avelang_impl/prefill/A_20260927_70_BW16_TRUE_W4_REBUILD.py"
OUT = REPORT_DIR / "A_20260927_70_RESOURCE_AUDIT.json"
SYMBOL = "_a70_bw16_k2_persistent"

C67_GOLDEN = {
    "ordinary_vgpr": 126,
    "agpr": 0,
    "unified_span": 126,
    "accum_offset": 128,
    "numbered_sgpr": 84,
    "total_sgpr": 90,
    "max_physical_vgpr": "v125",
    "lds_bytes": 30720,
    "occupancy_blocks_per_cu": 2,
    "steady_state": {
        "mfma": 24,
        "mfma_bf16": 20,
        "mfma_f16": 4,
        "vmem_load": 12,
        "vmem_store": 1,
        "lds_read": 44,
        "lds_write": 30,
        "barrier": 12,
        "waitcnt": 36,
        "v_perm": 12,
    },
}


def _kernel_block(text: str) -> str:
    marker = "    .group_segment_fixed_size:"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("missing kernel metadata block")
    end = text.find("\namdhsa.target:", start)
    if end < 0:
        raise RuntimeError("missing end of kernel metadata")
    return text[start:end]


def _metadata(block: str, name: str) -> int:
    match = re.search(rf"^\s+(?:-\s+)?\.{re.escape(name)}:\s+(\d+)\s*$", block, re.M)
    if not match:
        raise RuntimeError(f"missing .{name} metadata")
    return int(match.group(1))


def _symbol_hex(text: str, name: str) -> int:
    match = re.search(
        rf"Name: {re.escape(SYMBOL)}\.{re.escape(name)} .*?\n\s+Value: 0x([0-9A-Fa-f]+)",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError(f"missing {name} symbol metadata")
    return int(match.group(1), 16)


def _function_span(text: str) -> tuple[int, int]:
    match = re.search(
        rf"Name: {re.escape(SYMBOL)} \([^\n]+\)\n\s+Value: 0x([0-9A-Fa-f]+)\n\s+Size: (\d+)",
        text,
    )
    if not match:
        raise RuntimeError(f"missing {SYMBOL} function symbol")
    return int(match.group(1), 16), int(match.group(2))


def _instruction_lines(text: str) -> list[tuple[int, str]]:
    result = []
    for line in text.splitlines():
        match = re.search(r"//\s*00000000([0-9A-Fa-f]+):", line)
        if match:
            result.append((int(match.group(1), 16), line))
    return result


def _physical_max(lines: list[tuple[int, str]], prefix: str) -> int:
    maximum = -1
    pattern = re.compile(rf"\b{prefix}\[(\d+):(\d+)\]|\b{prefix}(\d+)\b")
    for _, line in lines:
        for match in pattern.finditer(line):
            if match.group(3) is not None:
                maximum = max(maximum, int(match.group(3)))
            else:
                maximum = max(maximum, int(match.group(1)), int(match.group(2)))
    if maximum < 0:
        raise RuntimeError(f"no physical {prefix} register found")
    return maximum


def _count(lines: list[tuple[int, str]], pattern: str) -> int:
    rx = re.compile(pattern)
    return sum(1 for _, line in lines if rx.search(line))


def _loop(lines: list[tuple[int, str]], function_start: int) -> tuple[list[tuple[int, str]], int, int]:
    backward = []
    for address, line in lines:
        match = re.search(rf"<{re.escape(SYMBOL)}\+0x([0-9a-f]+)>", line)
        if match:
            target = function_start + int(match.group(1), 16)
            if target < address:
                backward.append((address, target))
    if not backward:
        raise RuntimeError("missing dynamic loop backedge")
    loop_end, loop_start = backward[-1]
    return [(addr, line) for addr, line in lines if loop_start <= addr <= loop_end], loop_start, loop_end


def _occupancy() -> dict[str, int]:
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleLoad.restype = ctypes.c_int
    hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    hip.hipModuleGetFunction.restype = ctypes.c_int
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    ]
    hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.restype = ctypes.c_int
    hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
    hip.hipModuleUnload.restype = ctypes.c_int
    module = ctypes.c_void_p()
    rc = hip.hipModuleLoad(ctypes.byref(module), str(HSACO).encode())
    if rc:
        raise RuntimeError(f"hipModuleLoad rc={rc}")
    try:
        function = ctypes.c_void_p()
        rc = hip.hipModuleGetFunction(ctypes.byref(function), module, SYMBOL.encode())
        if rc:
            raise RuntimeError(f"hipModuleGetFunction rc={rc}")
        blocks = ctypes.c_int()
        rc = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks), function, 256, 0,
        )
        if rc:
            raise RuntimeError(f"hipModuleOccupancyMaxActiveBlocksPerMultiprocessor rc={rc}")
        return {"blocks_per_cu": blocks.value, "block_size": 256, "dynamic_lds_bytes": 0}
    finally:
        hip.hipModuleUnload(module)


def main() -> None:
    readobj = READOBJ.read_text()
    isa = ISA.read_text()
    block = _kernel_block(readobj)
    function_start, function_size = _function_span(readobj)
    all_instructions = _instruction_lines(isa)
    kernel = [(addr, line) for addr, line in all_instructions if function_start <= addr < function_start + function_size]
    loop, loop_start, loop_end = _loop(kernel, function_start)
    vgpr = _metadata(block, "vgpr_count")
    agpr = _metadata(readobj, "agpr_count")
    numbered_sgpr = _symbol_hex(readobj, "numbered_sgpr")
    total_sgpr = _metadata(block, "sgpr_count")
    lds = _metadata(block, "group_segment_fixed_size")
    private = _metadata(block, "private_segment_fixed_size")
    vgpr_spill = _metadata(block, "vgpr_spill_count")
    sgpr_spill = _metadata(block, "sgpr_spill_count")
    steady = {
        "mfma": _count(loop, r"v_mfma_f32_16x16x16_(?:bf16|f16)\b"),
        "mfma_bf16": _count(loop, r"v_mfma_f32_16x16x16_bf16\b"),
        "mfma_f16": _count(loop, r"v_mfma_f32_16x16x16_f16\b"),
        "vmem_load": _count(loop, r"(?:global|flat|buffer)_load(?:_|\b)"),
        "vmem_store": _count(loop, r"(?:global|flat|buffer)_store(?:_|\b)"),
        "lds_read": _count(loop, r"ds_read(?:_|\d|\b)"),
        "lds_write": _count(loop, r"ds_write(?:_|\b)"),
        "barrier": _count(loop, r"s_barrier\b"),
        "waitcnt": _count(loop, r"s_waitcnt\b"),
        "v_perm": _count(loop, r"v_perm(?:_|\b)"),
        "dpp": _count(loop, r"(?:\bdpp\b|_dpp\b)"),
        "loop_instruction_lines": len(loop),
        "loop_start": hex(loop_start),
        "loop_end": hex(loop_end),
    }
    occupancy = _occupancy()
    resources = {
        "ordinary_vgpr": vgpr,
        "agpr": agpr,
        "unified_span": vgpr + agpr,
        "accum_offset": "not emitted; AGPR=0",
        "numbered_sgpr": numbered_sgpr,
        "total_sgpr": total_sgpr,
        "max_physical_vgpr": f"v{_physical_max(kernel, 'v')}",
        "max_physical_sgpr": f"s{_physical_max(kernel, 's')}",
        "vgpr_spill": vgpr_spill,
        "sgpr_spill": sgpr_spill,
        "private_scratch_bytes": private,
        "lds_bytes": lds,
        "occupancy": occupancy,
        "steady_state_isa": steady,
    }
    hard_gates = {
        "agpr_zero": agpr == 0,
        "no_vgpr_spill": vgpr_spill == 0,
        "no_sgpr_spill": sgpr_spill == 0,
        "private_scratch_zero": private == 0,
        "occupancy_at_least_2": occupancy["blocks_per_cu"] >= 2,
        "mfma_20_bf16_plus_4_fp16": steady["mfma_bf16"] == 20 and steady["mfma_f16"] == 4,
    }
    report = {
        "artifact": "A_20260927_70_BW16_TRUE_W4_REBUILD",
        "object": {
            "hsaco": str(HSACO.relative_to(ROOT)),
            "hsaco_sha256": hashlib.sha256(HSACO.read_bytes()).hexdigest(),
            "isa": str(ISA.relative_to(ROOT)),
            "readobj": str(READOBJ.relative_to(ROOT)),
            "source": str(SOURCE.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "symbol": SYMBOL,
        },
        "resources": resources,
        "hard_gates": hard_gates,
        "hard_gates_pass": all(hard_gates.values()),
        "C67_existing_golden": C67_GOLDEN,
        "golden_provenance": "Existing C_20260925_67_BW16_W4_MACHINE_GOLDEN_AUDIT ledger; not regenerated here.",
        "four_wave_machine_equivalence": "verified by separate A70 dynamic ownership gate",
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only HIP occupancy queries for the three C69 machine families.

This loads modules and asks the occupancy API for metadata only. It never
launches a kernel and never performs timing or correctness work.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "occupancy_queries.json"
OBJ = ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit"

HIP = ctypes.CDLL("libamdhip64.so")
HIP.hipInit.argtypes = [ctypes.c_uint]
HIP.hipSetDevice.argtypes = [ctypes.c_int]
HIP.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
HIP.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
]
HIP.hipModuleUnload.argtypes = [ctypes.c_void_p]


FAMILIES = {
    "A_T32_W4_S2": {
        "hsaco": OBJ / "k1_winner_w4_s1_5c0a24b34425219f.hsaco",
        "block": 256,
        "grid": [1, 12, 1],
        "triton_hash": "42d944a8b207fea563aa684bf867d4ebd2f1e97a65c0d3c1e735a68f77a778fa",
        "config": "w4_s2",
    },
    "B_GENERIC_W4": {
        "hsaco": OBJ / "k1_winner_w4_s1_34419742ff3240a0.hsaco",
        "block": 256,
        "grid": [2, 12, 1],
        "triton_hash": "8dd7fa31ef878420a1f7bf60982721aa2221671908291fc761b6597731d1ac13",
        "config": "w4_s1 representative",
    },
    "C_LONG_W2_S2": {
        "hsaco": OBJ / "k1_winner_w2_s1_68844813aee59496.hsaco",
        "block": 128,
        "grid": [64, 12, 1],
        "triton_hash": "cef0a88f4d9ce72fcbe18b85287dfdb6a441f91b36d5dc9a611ad91e3a5fe477",
        "config": "w2_s2 representative",
    },
}


def query(data: bytes, symbol: bytes, block: int, dynamic_lds: int) -> dict:
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    storage = (ctypes.c_char * len(data)).from_buffer_copy(data)
    rc_load = HIP.hipModuleLoadData(ctypes.byref(module), ctypes.cast(storage, ctypes.c_void_p))
    if rc_load != 0:
        return {"rc_load": rc_load, "rc_function": None, "rc_occupancy": None}
    rc_function = HIP.hipModuleGetFunction(ctypes.byref(function), module, symbol)
    if rc_function != 0:
        HIP.hipModuleUnload(module)
        return {"rc_load": rc_load, "rc_function": rc_function, "rc_occupancy": None}
    blocks = ctypes.c_int()
    rc_occ = HIP.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks), function, block, dynamic_lds
    )
    result = {
        "rc_load": rc_load,
        "rc_function": rc_function,
        "rc_occupancy": rc_occ,
        "active_blocks_per_cu": blocks.value if rc_occ == 0 else None,
    }
    HIP.hipModuleUnload(module)
    return result


def main() -> None:
    HIP.hipInit(0)
    HIP.hipSetDevice(0)
    output = {
        "device": "AMD Instinct MI300X gfx942",
        "symbol": "_flash_kda_prepare_kernel",
        "dynamic_lds_bytes": 8192,
        "queries_are_metadata_only": True,
        "families": {},
    }
    for name, spec in FAMILIES.items():
        data = spec["hsaco"].read_bytes()
        rows = {}
        for block in (64, 128, 192, 256):
            for lds in (0, 8192, 16384, 32768, 65536):
                rows[f"block{block}_lds{lds}"] = query(
                    data, b"_flash_kda_prepare_kernel", block, lds
                )
        actual = rows[f"block{spec['block']}_lds8192"]
        output["families"][name] = {
            "hsaco": str(spec["hsaco"]),
            "hsaco_sha256": hashlib.sha256(data).hexdigest(),
            "config": spec["config"],
            "triton_hash": spec["triton_hash"],
            "grid": spec["grid"],
            "block": spec["block"],
            "actual_query": actual,
            "sensitivity_queries": rows,
            "resident_waves_per_cu_from_actual_api": (
                actual["active_blocks_per_cu"] * spec["block"] // 64
                if actual.get("active_blocks_per_cu") is not None else None
            ),
        }
    OUT.write_text(json.dumps(output, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()

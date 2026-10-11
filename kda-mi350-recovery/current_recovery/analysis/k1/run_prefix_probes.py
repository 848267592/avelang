#!/usr/bin/env python3
"""Direct timing of C71 diagnostic prefix copies only.

The copies terminate at explicit machine anchors and are not correctness or
production kernels. This runner never loads the original C69/C26 object for a
timed row; it loads only the C71 prefix HSACOs.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import struct
import statistics
import sys

import torch


ROOT = Path("/workspace/project/avelang")
HERE = ROOT / "test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT"
PREFIX = HERE / "prefix_probes_preserved_metadata"
OUT = HERE / "prefix_probe_timing.json"
T_LIST = [int(x) for x in os.environ.get("C71_T_LIST", "32,128,512,1024").split(",") if x]
WARMUP = int(os.environ.get("C71_WARMUP", "10"))
REPEATS = int(os.environ.get("C71_REPEATS", "30"))
ROUNDS = int(os.environ.get("C71_ROUNDS", "3"))
H = 12
K = 128
C = 32
BC = 16
SCALE = 1.0 / (K ** 0.5)
LOWER_BOUND = -5.0

sys.path.insert(0, str(ROOT / "test/examples/kda"))
sys.path.insert(0, str(ROOT / "test/examples/kda/tests"))
from bench_common import make_kda_prefill_inputs  # noqa: E402
from avelang_impl.prefill.kda_prefill_v0 import _allocate_workspace  # noqa: E402

HIP = ctypes.CDLL("libamdhip64.so")
HIP.hipInit(0)
HIP.hipSetDevice(0)
HIP.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
HIP.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
HIP.hipModuleLaunchKernel.argtypes = [
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
]
HIP.hipModuleUnload.argtypes = [ctypes.c_void_p]


FAMILIES = {
    "C26": {"block": 128, "lds": 0},
    "W4_A": {"block": 256, "lds": 8192},
    "W4_B": {"block": 256, "lds": 8192},
}
CUTS = ["frontend_to_first_bf16", "bf16_to_first_fp32", "fp32_to_first_store"]


def load(path: Path):
    module = ctypes.c_void_p()
    rc = HIP.hipModuleLoad(ctypes.byref(module), str(path).encode())
    if rc:
        raise RuntimeError(f"hipModuleLoad {path} rc={rc}")
    fn = ctypes.c_void_p()
    symbol = b"_kda_native_k1_kernel_v2" if path.name.startswith("C26_") else b"_flash_kda_prepare_kernel"
    rc = HIP.hipModuleGetFunction(ctypes.byref(fn), module, symbol)
    if rc:
        HIP.hipModuleUnload(module)
        raise RuntimeError(f"hipModuleGetFunction {path} rc={rc}")
    return module, fn


def c26_config(inp, ws):
    vals = [
        inp["q"].data_ptr(), inp["k"].data_ptr(), inp["raw_g"].data_ptr(),
        inp["raw_beta"].float().contiguous().data_ptr(), inp["A_log"].data_ptr(), inp["dt_bias"].view(-1).contiguous().data_ptr(),
        ws.kd.data_ptr(), ws.qd.data_ptr(), ws.kr.data_ptr(), ws.gt.data_ptr(), ws.inv_mqk.data_ptr(),
    ]
    raw = struct.pack("=QQQQQQQQQQQ", *vals)
    arr = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
    size = ctypes.c_size_t(len(raw))
    cfg = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(1), ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p),
        ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), ctypes.c_void_p(3),
    )
    return cfg, arr, size


def w4_config(inp, ws, T, nt):
    beta = inp["raw_beta"].float().contiguous()
    dt_bias = inp["dt_bias"].view(-1).contiguous()
    vals = [
        inp["q"].data_ptr(), inp["k"].data_ptr(), inp["raw_g"].data_ptr(),
        beta.data_ptr(), inp["A_log"].data_ptr(), dt_bias.data_ptr(),
        ws.kd.data_ptr(), ws.qd.data_ptr(), ws.kr.data_ptr(), ws.gt.data_ptr(), ws.inv_mqk.data_ptr(),
    ]
    raw = struct.pack(
        "=QQQQQQQQQQQffIII4xQQ", *vals, SCALE, LOWER_BOUND, T, nt, nt, 0, 0
    )
    arr = (ctypes.c_char * len(raw)).from_buffer_copy(raw)
    size = ctypes.c_size_t(len(raw))
    cfg = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(1), ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p),
        ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), ctypes.c_void_p(3),
    )
    return cfg, arr, size


def main():
    result = {
        "artifact": "C_20260927_71_DIAGNOSTIC_PREFIX_PHASE_PROBES",
        "protocol": {"warmup": WARMUP, "repeats": REPEATS, "rounds": ROUNDS, "default_stream": True, "interleaved": True},
        "identity_scope": "C71 diagnostic prefix copies; not exact production HSACO identities after truncation/reassembly",
        "rows": [],
    }
    for T in T_LIST:
        nt = (T + C - 1) // C
        inp = make_kda_prefill_inputs(T=T, H=H, seed=71000 + T)
        for cut in CUTS:
            compiled = {}
            for family in FAMILIES:
                module, fn = load(PREFIX / f"{family}_{cut}.hsaco")
                ws = _allocate_workspace(T=T, device="cuda:0")
                cfg = c26_config(inp, ws) if family == "C26" else w4_config(inp, ws, T, nt)
                compiled[family] = {"module": module, "fn": fn, "ws": ws, "cfg": cfg}

            def launch(family):
                item = compiled[family]
                p = FAMILIES[family]
                rc = HIP.hipModuleLaunchKernel(
                    item["fn"], nt, H, 1, p["block"], 1, 1, p["lds"],
                    ctypes.c_void_p(0), ctypes.c_void_p(0), item["cfg"][0],
                )
                if rc:
                    raise RuntimeError(f"hipModuleLaunchKernel {family} T={T} cut={cut} rc={rc}")

            for family in FAMILIES:
                for _ in range(WARMUP):
                    launch(family)
            torch.cuda.synchronize()

            samples = {family: [] for family in FAMILIES}
            round_medians = {family: [] for family in FAMILIES}
            for r in range(ROUNDS):
                order = ["C26", "W4_A", "W4_B"] if r % 2 == 0 else ["W4_B", "C26", "W4_A"]
                for family in order:
                    local = []
                    for _ in range(REPEATS):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        launch(family)
                        end.record()
                        end.synchronize()
                        local.append(float(start.elapsed_time(end) * 1000.0))
                    samples[family].extend(local)
                    round_medians[family].append(float(statistics.median(local)))
            for family in FAMILIES:
                item = compiled[family]
                HIP.hipModuleUnload(item["module"])
            result["rows"].append({
                "T": T, "NT": nt, "cut": cut,
                "round_medians_us": round_medians,
                "samples_us": samples,
                "median_us": {k: float(statistics.median(v)) for k, v in samples.items()},
                "semantic_anchor": "prefix ends at the exact opcode recorded in prefix_probe_manifest.json; not claimed as full C69 identity",
            })
            OUT.write_text(json.dumps(result, indent=2))
            print("checkpoint", T, cut, result["rows"][-1]["median_us"], flush=True)
    result["status"] = "COMPLETE"
    OUT.write_text(json.dumps(result, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()

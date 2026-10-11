#!/usr/bin/env python3
"""Correctness gate for full metadata-preserved timestamp diagnostics."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import struct
import sys

import torch


ROOT = Path("/workspace/project/avelang")
HERE = ROOT / "test/examples/kda/reports/C_20260927_71_K1_TRUE_BEST_MACHINE_REALIGNMENT_AUDIT"
DIAG = HERE / "timestamp_probes_preserved_metadata"
H = 12
K = 128
C = 32

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

T = int(os.environ.get("C71_CORRECTNESS_T", "32"))
FAMILY = "W4_A" if T == 32 else "W4_B"
PHASES = ["frontend_to_first_bf16", "bf16_to_first_fp32", "first_fp32_to_first_store"]

ORIGINAL = {
    "C26": (ROOT / "test/examples/kda/artifacts/kda_native_k1_c26.hsaco", b"_kda_native_k1_kernel_v2", 128, 0),
    "W4_A": (ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_5c0a24b34425219f.hsaco", b"_flash_kda_prepare_kernel", 256, 8192),
    "W4_B": (ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_34419742ff3240a0.hsaco", b"_flash_kda_prepare_kernel", 256, 8192),
}


def load(path: Path, symbol: bytes):
    mod = ctypes.c_void_p()
    rc = HIP.hipModuleLoad(ctypes.byref(mod), str(path).encode())
    if rc:
        raise RuntimeError(f"hipModuleLoad {path} rc={rc}")
    fn = ctypes.c_void_p()
    rc = HIP.hipModuleGetFunction(ctypes.byref(fn), mod, symbol)
    if rc:
        HIP.hipModuleUnload(mod)
        raise RuntimeError(f"hipModuleGetFunction {path} rc={rc}")
    return mod, fn


def c26_config(inp, ws, inv_ptr=None, extra_ptr=None):
    vals = [inp["q"].data_ptr(), inp["k"].data_ptr(), inp["raw_g"].data_ptr(), inp["raw_beta"].float().contiguous().data_ptr(), inp["A_log"].data_ptr(), inp["dt_bias"].view(-1).contiguous().data_ptr(), ws.kd.data_ptr(), ws.qd.data_ptr(), ws.kr.data_ptr(), ws.gt.data_ptr(), ws.inv_mqk.data_ptr() if inv_ptr is None else inv_ptr]
    if extra_ptr is None:
        raw = struct.pack("=QQQQQQQQQQQ", *vals)
    else:
        raw = struct.pack("=QQQQQQQQQQQQ", *vals, extra_ptr)
    arr = (ctypes.c_char * len(raw)).from_buffer_copy(raw); size = ctypes.c_size_t(len(raw))
    cfg = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p), ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), ctypes.c_void_p(3))
    return cfg, arr, size


def w4_config(inp, ws, T, nt, timer_ptr=None, family=None):
    family = FAMILY if family is None else family
    beta = inp["raw_beta"].float().contiguous(); dt_bias = inp["dt_bias"].view(-1).contiguous()
    ptrs = [inp["q"].data_ptr(), inp["k"].data_ptr(), inp["raw_g"].data_ptr(), beta.data_ptr(), inp["A_log"].data_ptr(), dt_bias.data_ptr(), ws.kd.data_ptr(), ws.qd.data_ptr(), ws.kr.data_ptr(), ws.gt.data_ptr(), ws.inv_mqk.data_ptr()]
    if family == "W4_A" and T == 32:
        base_vals = ptrs + [1.0/(K**0.5), -5.0, T, 0, 0 if timer_ptr is None else timer_ptr]
        fmt = "=QQQQQQQQQQQffI4xQQ"
    else:
        base_vals = ptrs + [1.0/(K**0.5), -5.0, T, nt, nt, 0 if timer_ptr is None else timer_ptr, 0 if timer_ptr is None else timer_ptr]
        fmt = "=QQQQQQQQQQQffIII4xQQ"
    if timer_ptr is None:
        raw = struct.pack(fmt, *base_vals)
    else:
        raw = struct.pack(fmt + "Q", *base_vals, timer_ptr)
    arr = (ctypes.c_char * len(raw)).from_buffer_copy(raw); size = ctypes.c_size_t(len(raw))
    cfg = (ctypes.c_void_p * 5)(ctypes.c_void_p(1), ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p), ctypes.c_void_p(2), ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), ctypes.c_void_p(3))
    return cfg, arr, size


def launch(fn, grid, block, lds, cfg):
    rc = HIP.hipModuleLaunchKernel(fn, grid[0], grid[1], grid[2], block, 1, 1, lds, ctypes.c_void_p(0), ctypes.c_void_p(0), cfg)
    if rc: raise RuntimeError(f"hipModuleLaunchKernel rc={rc}")
    torch.cuda.synchronize()


def diff(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max().item()), "different": int((d != 0).sum().item()), "finite": bool(torch.isfinite(a.float()).all().item() and torch.isfinite(b.float()).all().item())}


def main():
    nt = (T + C - 1) // C
    inp = make_kda_prefill_inputs(T=T, H=H, seed=71000 + T)
    ws_base = _allocate_workspace(T=T, device="cuda:0")
    orig_path, orig_symbol, block, lds = ORIGINAL[FAMILY]
    print("C71 original launch", orig_path, "family", FAMILY, "T", T, flush=True)
    orig_mod, orig_fn = load(orig_path, orig_symbol)
    base_cfg = c26_config(inp, ws_base)[0] if FAMILY == "C26" else w4_config(inp, ws_base, T, nt, family=FAMILY)[0]
    launch(orig_fn, [nt, H, 1], block, lds, base_cfg)
    print("C71 original launch complete", flush=True)
    result = {"T": T, "family": FAMILY, "status": "PASS", "phases": {}}
    for phase in PHASES:
        diag_path = DIAG / f"{FAMILY}_{phase}.hsaco"
        print("C71 diagnostic load", phase, diag_path, flush=True)
        diag_mod, diag_fn = load(diag_path, orig_symbol)
        ws = _allocate_workspace(T=T, device="cuda:0")
        timer = torch.empty((4,), device="cuda", dtype=torch.float32)
        inv_storage = None
        cfg = c26_config(inp, ws, extra_ptr=timer.data_ptr())[0] if FAMILY == "C26" else w4_config(inp, ws, T, nt, timer.data_ptr(), FAMILY)[0]
        timer_base = timer.data_ptr()
        timer_size = timer.numel() * timer.element_size()
        print(
            "C71 diagnostic pointers",
            {"base": hex(timer_base),
             "allocation_end": hex(timer_base + timer_size),
             "store_start": hex(timer_base),
             "store_end": hex(timer_base + 8)},
            flush=True,
        )
        print("C71 diagnostic launch", phase, flush=True)
        launch(diag_fn, [nt, H, 1], block, lds, cfg)
        print("C71 diagnostic launch complete", phase, flush=True)
        row = {name: diff(getattr(ws, name), getattr(ws_base, name)) for name in ["kd", "qd", "kr", "gt"]}
        row["inv_mqk"] = diff(ws.inv_mqk, ws_base.inv_mqk)
        row["pass"] = all(x["different"] == 0 for x in row.values() if isinstance(x, dict))
        result["phases"][phase] = row
        if not row["pass"]: result["status"] = "FAIL"
        HIP.hipModuleUnload(diag_mod)
    HIP.hipModuleUnload(orig_mod)
    out = HERE / f"timestamp_correctness_T{T}.json"; out.write_text(json.dumps(result, indent=2)); print(out); print(json.dumps(result, indent=2))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()

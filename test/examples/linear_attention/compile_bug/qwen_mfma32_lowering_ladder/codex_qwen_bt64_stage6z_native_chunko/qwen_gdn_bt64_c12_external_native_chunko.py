"""Experimental external selected-native chunk-o control for C12.

This module is an upper-bound control only.  It loads the exact captured
T2048 WG256 or T8192 WG128 Triton code object and never presents it as an
AveLang compiler result.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch


BT = 64
HK = 4
HV = 8
K = 128
V = 128
HERE = Path(__file__).resolve().parent

CONFIGS = {
    2048: {
        "hsaco": HERE / "native/T2048/trace_capture/selected/chunk_fwd_kernel_o.hsaco",
        "sha256": "9cc107ecf4f9b8529694fd07f4532174ba98593aab206fbddf628dbf53bfb95c",
        "grid": (2, 32, 8),
        "block": 256,
        "shared_bytes": 24576,
        "num_warps": 4,
        "num_stages": 3,
    },
    8192: {
        "hsaco": HERE / "native/T8192/trace_capture/selected/chunk_fwd_kernel_o.hsaco",
        "sha256": "e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5",
        "grid": (2, 128, 8),
        "block": 128,
        "shared_bytes": 12288,
        "num_warps": 2,
        "num_stages": 2,
    },
}


@lru_cache(maxsize=1)
def _bridge() -> ctypes.CDLL:
    path = Path(os.environ.get("C12_NATIVE_CHUNK_O_BRIDGE", HERE / "libc12_native_chunk_o_bridge.so"))
    library = ctypes.CDLL(str(path))
    library.c12_native_chunk_o_launch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        *([ctypes.c_void_p] * 6),
        ctypes.c_float,
        ctypes.c_int32,
        *([ctypes.c_int32] * 4),
        ctypes.c_uint32,
    ]
    library.c12_native_chunk_o_launch.restype = ctypes.c_int
    library.c12_native_chunk_o_last_error.restype = ctypes.c_char_p
    library.c12_native_chunk_o_module_load_count.restype = ctypes.c_uint
    return library


def selected_config(t: int) -> dict[str, object]:
    if t not in CONFIGS:
        raise ValueError("C12 external native control has exact captures only for T=2048 and T=8192")
    return CONFIGS[t]


@lru_cache(maxsize=2)
def _checked_hsaco(t: int) -> str:
    config = selected_config(t)
    path = Path(os.environ.get(f"C12_NATIVE_CHUNK_O_HSACO_T{t}", str(config["hsaco"])))
    if not path.is_file():
        raise ValueError(f"missing captured native chunk-o HSACO: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = os.environ.get(f"C12_NATIVE_CHUNK_O_SHA256_T{t}", str(config["sha256"]))
    if actual != expected:
        raise ValueError(f"native chunk-o HSACO SHA256 mismatch for T={t}: {actual} != {expected}")
    return str(path)


def _validate(q, k, v_new, h, g, out):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(q.device).gcnArchName.startswith("gfx942"):
        raise ValueError("C12 native chunk-o control requires gfx942")
    t = int(q.shape[1]) if q.ndim == 4 else -1
    if t not in CONFIGS:
        raise ValueError("C12 native chunk-o control requires T=2048 or T=8192")
    expected = {
        "q": ((1, t, HK, K), torch.bfloat16, q),
        "k": ((1, t, HK, K), torch.bfloat16, k),
        "v_new": ((1, t, HV, V), torch.bfloat16, v_new),
        "h": ((1, t // BT, HV, V, K), torch.bfloat16, h),
        "g": ((1, t, HV), torch.float32, g),
        "out": ((1, t, HV, V), torch.bfloat16, out),
    }
    for name, (shape, dtype, value) in expected.items():
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_cuda or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous CUDA {dtype} with shape {shape}")
        if value.device != q.device:
            raise ValueError(f"{name} must be on {q.device}")
    return t


def launch_into(q, k, v_new, h, g, output_bf16, *, scale: float | None = None) -> dict[str, object]:
    t = _validate(q, k, v_new, h, g, output_bf16)
    path = _checked_hsaco(t)
    config = selected_config(t)
    if scale is None:
        scale = K ** -0.5
    stream = torch.cuda.current_stream(q.device).cuda_stream
    library = _bridge()
    result = library.c12_native_chunk_o_launch(
        os.fsencode(path),
        int(stream),
        *[ctypes.c_void_p(value.data_ptr()) for value in (q, k, v_new, h, g, output_bf16)],
        float(scale),
        t,
        *config["grid"],
        config["block"],
        config["shared_bytes"],
    )
    if result:
        message = library.c12_native_chunk_o_last_error().decode()
        raise RuntimeError(f"external selected native chunk-o launch failed: {message}")
    return {
        "T": t,
        "hsaco": path,
        "sha256": config["sha256"],
        "grid": list(config["grid"]),
        "block": config["block"],
        "shared_bytes": config["shared_bytes"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        "module_load_count": int(library.c12_native_chunk_o_module_load_count()),
    }


__all__ = ["CONFIGS", "launch_into", "selected_config"]

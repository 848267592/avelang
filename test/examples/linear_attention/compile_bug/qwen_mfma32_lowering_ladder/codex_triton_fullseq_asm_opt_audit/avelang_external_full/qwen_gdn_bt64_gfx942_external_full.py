"""Audit-only full-sequence dispatch for the frozen vLLM BT64/gfx942 HSACO."""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


BT, HK, HV, KDIM, VDIM = 64, 4, 8, 128, 128
EXPECTED_SHA256 = "cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9"
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _bridge() -> ctypes.CDLL:
    path = Path(os.environ.get("QWEN_TRITON_FULLSEQ_EXTERNAL_BRIDGE", HERE / "libqwen_triton_external_full_bridge.so"))
    library = ctypes.CDLL(str(path))
    library.qwen_triton_external_full_launch.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int32] + [ctypes.c_void_p] * 8
    library.qwen_triton_external_full_launch.restype = ctypes.c_int
    library.qwen_triton_external_full_last_error.restype = ctypes.c_char_p
    library.qwen_triton_external_full_module_load_count.restype = ctypes.c_uint
    return library


def module_load_count() -> int:
    return int(_bridge().qwen_triton_external_full_module_load_count())


@lru_cache(maxsize=4)
def _verify_hsaco(value: str, expected: str) -> str:
    path = Path(value)
    if not path.is_file(): raise ValueError(f"full-sequence external HSACO does not exist: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected: raise ValueError(f"full-sequence external HSACO SHA256 mismatch: expected {expected}, got {actual}")
    return str(path)


def _validate(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None) -> int:
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(k.device).gcnArchName.startswith("gfx942"):
        raise ValueError("external full-sequence HSACO requires gfx942 HIP device")
    t = k.shape[1]
    if t < BT or t % BT: raise ValueError("external full-sequence HSACO requires T divisible by 64")
    expected = (("k", k, (1, t, HK, KDIM), torch.bfloat16), ("w", w, (1, t, HV, KDIM), torch.float32), ("u", u, (1, t, HV, VDIM), torch.float32), ("g", g, (1, t, HV), torch.float32))
    for name, value, shape, dtype in expected:
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_cuda or not value.is_contiguous() or value.device != k.device:
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape} on {k.device}")
    if initial_state is None or tuple(initial_state.shape) != (1, HV, VDIM, KDIM) or initial_state.dtype != torch.float32 or not initial_state.is_cuda or not initial_state.is_contiguous() or initial_state.device != k.device:
        raise ValueError("external full-sequence HSACO requires contiguous FP32 initial_state [1,8,128,128]")
    return t


def qwen_gdn_bt64_gfx942_external_full(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None, *, fallback: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        t = _validate(k, w, u, g, initial_state)
        hsaco = _verify_hsaco(os.environ["QWEN_TRITON_FULLSEQ_HSACO_PATH"], os.environ.get("QWEN_TRITON_FULLSEQ_HSACO_SHA256", EXPECTED_SHA256))
    except (KeyError, ValueError):
        if fallback is not None: return fallback()
        raise
    h = torch.empty((1, t // BT, HV, VDIM, KDIM), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, HV, VDIM, KDIM), dtype=torch.float32, device=k.device)
    _launch_preallocated(hsaco, t, k, w, u, g, initial_state, h, v_new, final_state)
    return h, v_new, final_state


def _launch_preallocated(hsaco: str, t: int, k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor, h: torch.Tensor, v_new: torch.Tensor, final_state: torch.Tensor) -> None:
    if tuple(h.shape) != (1, t // BT, HV, VDIM, KDIM) or h.dtype != torch.bfloat16 or not h.is_contiguous():
        raise ValueError("h must be contiguous BF16 [1,T/64,8,128,128]")
    if tuple(v_new.shape) != tuple(u.shape) or v_new.dtype != torch.float32 or not v_new.is_contiguous():
        raise ValueError("v_new must be contiguous FP32 [1,T,8,128]")
    if tuple(final_state.shape) != (1, HV, VDIM, KDIM) or final_state.dtype != torch.float32 or not final_state.is_contiguous():
        raise ValueError("final_state must be contiguous FP32 [1,8,128,128]")
    result = _bridge().qwen_triton_external_full_launch(os.fsencode(hsaco), int(torch.cuda.current_stream(k.device).cuda_stream), t, *[ctypes.c_void_p(value.data_ptr()) for value in (k, u, w, v_new, g, h, initial_state, final_state)])
    if result: raise RuntimeError(_bridge().qwen_triton_external_full_last_error().decode())


def qwen_gdn_bt64_gfx942_external_full_preallocated(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, initial_state: torch.Tensor | None, h: torch.Tensor, v_new: torch.Tensor, final_state: torch.Tensor) -> None:
    """Audit benchmark helper: launch the frozen artifact without allocation or module load."""
    t = _validate(k, w, u, g, initial_state)
    hsaco = _verify_hsaco(os.environ["QWEN_TRITON_FULLSEQ_HSACO_PATH"], os.environ.get("QWEN_TRITON_FULLSEQ_HSACO_SHA256", EXPECTED_SHA256))
    _launch_preallocated(hsaco, t, k, w, u, g, initial_state, h, v_new, final_state)

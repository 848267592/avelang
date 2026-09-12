"""Experimental external-HSACO dispatch for one fixed vLLM Qwen BT64 step.

This is not an AveLang lowering and does not vendor a Triton binary. Callers
must opt in by setting ``QWEN_TRITON_HSACO_PATH`` to a locally audited artifact.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


BT, HK, HV, KDIM, VDIM = 64, 4, 8, 128, 128
HERE = Path(__file__).resolve().parent
EXTRACTED_HSACO_SHA256 = "cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9"


@lru_cache(maxsize=1)
def _bridge() -> ctypes.CDLL:
    path = Path(os.environ.get("QWEN_TRITON_EXTERNAL_BRIDGE", HERE / "libqwen_triton_external_bridge.so"))
    library = ctypes.CDLL(str(path))
    library.qwen_triton_external_launch.argtypes = [ctypes.c_char_p, ctypes.c_uint64] + [ctypes.c_void_p] * 8
    library.qwen_triton_external_launch.restype = ctypes.c_int
    library.qwen_triton_external_last_error.restype = ctypes.c_char_p
    library.qwen_triton_external_module_load_count.restype = ctypes.c_uint
    return library


def module_load_count() -> int:
    """Return the bridge's process-local HIP module-load count for audit tests."""
    return int(_bridge().qwen_triton_external_module_load_count())


@lru_cache(maxsize=4)
def _verify_hsaco(path_value: str, expected: str) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"external Triton HSACO does not exist: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"external Triton HSACO SHA256 mismatch: expected {expected}, got {actual}")
    return str(path)


def _checked_hsaco_path() -> str:
    try:
        path_value = os.environ["QWEN_TRITON_HSACO_PATH"]
    except KeyError as exc:
        raise ValueError("QWEN_TRITON_HSACO_PATH must name the audited extracted HSACO") from exc
    expected = os.environ.get("QWEN_TRITON_HSACO_SHA256", EXTRACTED_HSACO_SHA256)
    return _verify_hsaco(path_value, expected)


def _validate(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
              initial_state: torch.Tensor | None) -> None:
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(k.device).gcnArchName.startswith("gfx942"):
        raise ValueError("external Triton HSACO requires a gfx942 HIP device")
    expected = {
        "k": ((1, BT, HK, KDIM), torch.bfloat16, k),
        "w": ((1, BT, HV, KDIM), torch.float32, w),
        "u": ((1, BT, HV, VDIM), torch.float32, u),
        "g": ((1, BT, HV), torch.float32, g),
    }
    for name, (shape, dtype, value) in expected.items():
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_cuda or not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} with shape {shape}")
        if value.device != k.device:
            raise ValueError(f"{name} must be on {k.device}")
    if initial_state is None or tuple(initial_state.shape) != (1, HV, VDIM, KDIM) or initial_state.dtype != torch.float32 or not initial_state.is_cuda or not initial_state.is_contiguous() or initial_state.device != k.device:
        raise ValueError("external HSACO specialization requires contiguous FP32 initial_state [1,8,128,128]")


def qwen_gdn_bt64_gfx942_external(
    k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
    initial_state: torch.Tensor | None, *, fallback: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the extracted fixed specialization; optionally call a supplied fallback on guard failure."""
    try:
        _validate(k, w, u, g, initial_state)
        hsaco = _checked_hsaco_path()
    except ValueError:
        if fallback is not None:
            return fallback()
        raise
    h = torch.empty((1, 1, HV, VDIM, KDIM), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, HV, VDIM, KDIM), dtype=torch.float32, device=k.device)
    stream = torch.cuda.current_stream(k.device).cuda_stream
    library = _bridge()
    result = library.qwen_triton_external_launch(
        os.fsencode(hsaco), int(stream), *[ctypes.c_void_p(value.data_ptr()) for value in (k, u, w, v_new, g, h, initial_state, final_state)]
    )
    if result:
        raise RuntimeError(library.qwen_triton_external_last_error().decode())
    return h, v_new, final_state

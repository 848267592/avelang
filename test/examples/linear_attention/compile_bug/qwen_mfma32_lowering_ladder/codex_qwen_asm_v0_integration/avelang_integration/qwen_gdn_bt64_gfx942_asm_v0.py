"""Opt-in Avelang runtime dispatch for the fixed Qwen BT64 gfx942 asm v0.

This is deliberately an opaque external-HSACO route.  It does not construct
an AveLang memref/vector/MFMA graph, so the Triton-derived core kernel reaches
HIP as a fixed code object rather than being re-lowered by LLVM.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable

import torch


BT = 64
H_K = 4
H_V = 8
K_DIM = 128
V_DIM = 128
WORKGROUP = 256
GRID = (4, 8, 1)
DYNAMIC_LDS_BYTES = 57_344
KERNEL_SYMBOL = "qwen_gdn_bt64_gfx942_asm_v0"

HERE = Path(__file__).resolve().parent
AUDIT_ROOT = HERE.parent
ASSEMBLY_DIR = AUDIT_ROOT / "assembly"
DEFAULT_HSACO = ASSEMBLY_DIR / "qwen_gdn_bt64_gfx942_asm_v0.hsaco"
DEFAULT_SHA256 = ASSEMBLY_DIR / "qwen_gdn_bt64_gfx942_asm_v0.hsaco.sha256"
DEFAULT_BRIDGE = HERE / "libqwen_gdn_bt64_gfx942_asm_v0_bridge.so"


@lru_cache(maxsize=1)
def _bridge() -> ctypes.CDLL:
    path = Path(os.environ.get("QWEN_GDN_BT64_GFX942_ASM_V0_BRIDGE", DEFAULT_BRIDGE))
    library = ctypes.CDLL(str(path))
    library.qwen_gdn_bt64_gfx942_asm_v0_launch.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.c_int32,
        *([ctypes.c_void_p] * 8),
    ]
    library.qwen_gdn_bt64_gfx942_asm_v0_launch.restype = ctypes.c_int
    library.qwen_gdn_bt64_gfx942_asm_v0_last_error.restype = ctypes.c_char_p
    library.qwen_gdn_bt64_gfx942_asm_v0_module_load_count.restype = ctypes.c_uint
    return library


def module_load_count() -> int:
    """Return the process-local HIP module-cache load count for test coverage."""
    return int(_bridge().qwen_gdn_bt64_gfx942_asm_v0_module_load_count())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sha256() -> str:
    configured = os.environ.get("QWEN_GDN_BT64_GFX942_ASM_V0_SHA256")
    if configured is not None:
        return configured
    try:
        match = re.search(r"[0-9a-fA-F]{64}", DEFAULT_SHA256.read_text())
    except OSError as error:
        raise ValueError(f"asm v0 SHA256 manifest is unavailable: {DEFAULT_SHA256}") from error
    if match is None:
        raise ValueError(f"asm v0 SHA256 manifest is invalid: {DEFAULT_SHA256}")
    return match.group(0).lower()


@lru_cache(maxsize=4)
def _verified_hsaco(path_value: str, expected: str) -> str:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"asm v0 HSACO does not exist: {path}")
    actual = _hash_file(path)
    if actual != expected:
        raise ValueError(f"asm v0 HSACO SHA256 mismatch: expected {expected}, got {actual}")
    return str(path)


def _validate_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be on a CUDA/HIP device")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")


def _validate_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> int:
    if k.ndim != 4:
        raise ValueError("k must have rank 4 with layout [1,T,4,128]")
    t = int(k.shape[1])
    if t < BT or t % BT != 0:
        raise ValueError("asm v0 requires T divisible by 64")
    if not torch.cuda.is_available():
        raise ValueError("asm v0 requires a gfx942 HIP device")
    properties = torch.cuda.get_device_properties(k.device)
    if not str(properties.gcnArchName).startswith("gfx942"):
        raise ValueError(f"asm v0 requires gfx942, got {properties.gcnArchName}")
    _validate_tensor("k", k, (1, t, H_K, K_DIM), torch.bfloat16, k.device)
    _validate_tensor("w", w, (1, t, H_V, K_DIM), torch.float32, k.device)
    _validate_tensor("u", u, (1, t, H_V, V_DIM), torch.float32, k.device)
    _validate_tensor("g", g, (1, t, H_V), torch.float32, k.device)
    if initial_state is None:
        raise ValueError("asm v0 requires a non-null initial_state")
    _validate_tensor("initial_state", initial_state, (1, H_V, V_DIM, K_DIM), torch.float32, k.device)
    return t


def _resolve_hsaco() -> str:
    path = Path(os.environ.get("QWEN_GDN_BT64_GFX942_ASM_V0_HSACO", DEFAULT_HSACO))
    return _verified_hsaco(str(path), _expected_sha256())


def _launch_preallocated(
    hsaco: str,
    t: int,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    v_new: torch.Tensor,
    final_state: torch.Tensor,
) -> None:
    _validate_tensor("h", h, (1, t // BT, H_V, V_DIM, K_DIM), torch.bfloat16, k.device)
    _validate_tensor("v_new", v_new, tuple(u.shape), torch.float32, k.device)
    _validate_tensor("final_state", final_state, (1, H_V, V_DIM, K_DIM), torch.float32, k.device)
    result = _bridge().qwen_gdn_bt64_gfx942_asm_v0_launch(
        os.fsencode(hsaco),
        int(torch.cuda.current_stream(k.device).cuda_stream),
        t,
        *[
            ctypes.c_void_p(value.data_ptr())
            for value in (k, u, w, v_new, g, h, initial_state, final_state)
        ],
    )
    if result:
        message = _bridge().qwen_gdn_bt64_gfx942_asm_v0_last_error()
        raise RuntimeError(message.decode() if message else "asm v0 HIP launch failed")


def qwen_gdn_bt64_gfx942_asm_v0(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    fallback: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the opaque BT64 recurrence with the native Triton-compatible ABI.

    The raw result contract is `(h_bf16, v_new_fp32, final_state_fp32)`, which
    matches the frozen vLLM `chunk_delta_h` operator.  Guard failures take the
    explicit supplied fallback; a launch failure is never silently redirected.
    """
    try:
        t = _validate_inputs(k, w, u, g, initial_state)
        hsaco = _resolve_hsaco()
    except (KeyError, ValueError):
        if fallback is not None:
            return fallback()
        raise
    assert initial_state is not None
    h = torch.empty((1, t // BT, H_V, V_DIM, K_DIM), dtype=torch.bfloat16, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty((1, H_V, V_DIM, K_DIM), dtype=torch.float32, device=k.device)
    _launch_preallocated(hsaco, t, k, w, u, g, initial_state, h, v_new, final_state)
    return h, v_new, final_state


def qwen_gdn_bt64_gfx942_asm_v0_preallocated(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    h: torch.Tensor,
    v_new: torch.Tensor,
    final_state: torch.Tensor,
) -> None:
    """Benchmark helper that excludes module load and output allocation."""
    t = _validate_inputs(k, w, u, g, initial_state)
    assert initial_state is not None
    _launch_preallocated(_resolve_hsaco(), t, k, w, u, g, initial_state, h, v_new, final_state)


def qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    fallback: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expose the raw BT64 result with Avelang-style FP32 `h` container.

    `h` is intentionally widened after the external call; its values are the
    BF16 materialization required by the vLLM/Triton contract.  This adapter
    is opt-in and is not wired into v24 because v24 remains a BT16 pipeline.
    """
    h_bf16, v_new, final_state = qwen_gdn_bt64_gfx942_asm_v0(
        k, w, u, g, initial_state, fallback=fallback
    )
    return h_bf16.to(torch.float32), v_new, final_state


def contract() -> dict[str, object]:
    """Return the fixed opaque-dispatch contract for reports and callers."""
    return {
        "symbol": KERNEL_SYMBOL,
        "shape": "B=1,T%64=0,Hk=4,Hv=8,K=V=128",
        "input_dtypes": {"k": "bf16", "w": "fp32", "u": "fp32", "g": "fp32", "initial_state": "fp32"},
        "output_dtypes": {"h": "bf16", "v_new": "fp32", "final_state": "fp32"},
        "grid": GRID,
        "workgroup": WORKGROUP,
        "dynamic_lds_bytes": DYNAMIC_LDS_BYTES,
    }


__all__ = [
    "BT",
    "DYNAMIC_LDS_BYTES",
    "GRID",
    "KERNEL_SYMBOL",
    "WORKGROUP",
    "contract",
    "module_load_count",
    "qwen_gdn_bt64_gfx942_asm_v0",
    "qwen_gdn_bt64_gfx942_asm_v0_preallocated",
    "qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0",
]

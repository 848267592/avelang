"""Experimental-only BT64 full graph using the Stage 6R BF16 recurrence bridge.

This module intentionally keeps every Stage 4 math kernel unchanged.  It
only materializes the three explicit dtype boundaries required by the frozen
Stage 6S contract: W/U FP32 -> BF16 before the external recurrence, then
V-new BF16 -> FP32 before the unchanged Stage 4 chunk-o kernel.
"""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import torch

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (
    H_K,
    H_V,
    K_DIM,
    V_DIM,
    BT,
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import qwen_gdn_chunk_cumsum_avelang_v6_standalone


_ROOT = Path(__file__).resolve().parents[1] / "compile_bug/qwen_mfma32_lowering_ladder"
_STAGE6R = _ROOT / "codex_qwen_bt64_recurrence_reconciliation_stage6r"
_BRIDGE_LIBRARY = _STAGE6R / "libstage6r_external_bridge.so"
_HSACO = _STAGE6R / "current_kernels/vllm/kernel.hsaco"
_SYMBOL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_EXPECTED_HSACO_SHA256 = "632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e"
_GRID_X, _GRID_Y, _WORKGROUP, _DYNAMIC_LDS = 4, 8, 128, 40960
_STAGE6S = _ROOT / "codex_qwen_bt64_bf16_recurrence_full_contract_stage6s"
_SOLVE_BRIDGE_LIBRARY = _STAGE6S / "libstage6s_external_solve_bridge.so"
_SOLVE_HSACO = _ROOT / "codex_qwen_bt64_hierarchical_solve_stage5b/rocprof_s0_v1/isa/qwen_bt64_hierarchical_fp32_s0.hsaco"
_SOLVE_SYMBOL = "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1"
_EXPECTED_SOLVE_HSACO_SHA256 = "b8c2dee0b1b94269170ff333da8fa912a38ae6198abd71de80d0898645091536"
_SOLVE_WORKGROUP = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage6s_contract() -> dict[str, object]:
    """Return the frozen opt-in bridge ABI without importing vLLM."""
    return {
        "experimental_only": True,
        "target": "gfx942",
        "shape": "B=1,Hk=4,Hv=8,K=V=128,BT=64,T%64=0",
        "hsaco": str(_HSACO),
        "hsaco_sha256": _EXPECTED_HSACO_SHA256,
        "symbol": _SYMBOL,
        "grid": (_GRID_X, _GRID_Y, 1),
        "workgroup": _WORKGROUP,
        "dynamic_lds": _DYNAMIC_LDS,
        "solve_hsaco": str(_SOLVE_HSACO),
        "solve_hsaco_sha256": _EXPECTED_SOLVE_HSACO_SHA256,
        "solve_symbol": _SOLVE_SYMBOL,
        "solve_workgroup": _SOLVE_WORKGROUP,
        "bridge_inputs": {"k": "bf16", "w": "bf16", "u": "bf16", "g": "fp32", "initial_state": "fp32"},
        "bridge_outputs": {"h": "bf16", "v_new": "bf16", "final_state": "fp32"},
        "boundaries": ("w:fp32->bf16", "u:fp32->bf16", "v_new:bf16->fp32"),
    }


def _require_target(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    if not torch.cuda.is_available() or q.device.type != "cuda":
        raise ValueError("Stage 6S requires a gfx942 HIP/CUDA device.")
    t = int(q.shape[1]) if q.ndim == 4 else -1
    if t < BT or t % BT:
        raise ValueError("Stage 6S requires T >= 64 and T divisible by BT=64.")
    required = (
        ("q", q, torch.bfloat16, (1, t, H_K, K_DIM)),
        ("k", k, torch.bfloat16, (1, t, H_K, K_DIM)),
        ("v", v, torch.bfloat16, (1, t, H_V, V_DIM)),
        ("g", g, torch.float32, (1, t, H_V)),
        ("beta", beta, torch.float32, (1, t, H_V)),
    )
    for name, value, dtype, shape in required:
        if value.dtype != dtype or tuple(value.shape) != shape or not value.is_contiguous() or value.device != q.device:
            raise ValueError(f"Stage 6S requires contiguous {name}={dtype} with shape={shape} on one device.")
    if initial_state is None:
        return torch.zeros((1, H_V, V_DIM, K_DIM), dtype=torch.float32, device=q.device)
    if (initial_state.dtype != torch.float32 or tuple(initial_state.shape) != (1, H_V, V_DIM, K_DIM) or
            not initial_state.is_contiguous() or initial_state.device != q.device):
        raise ValueError("Stage 6S initial_state must be contiguous FP32 [1,8,128,128].")
    return initial_state


class _Stage6RBridge:
    """Thin ABI wrapper for the unchanged Stage 6R external launcher."""

    def __init__(self) -> None:
        if not _BRIDGE_LIBRARY.is_file():
            raise RuntimeError(f"Stage 6S requires the prebuilt audit bridge: {_BRIDGE_LIBRARY}")
        if not _HSACO.is_file():
            raise RuntimeError(f"Stage 6S requires the captured current-vLLM HSACO: {_HSACO}")
        actual_hash = _sha256(_HSACO)
        if actual_hash != _EXPECTED_HSACO_SHA256:
            raise RuntimeError(f"Stage 6S HSACO hash guard failed: {actual_hash} != {_EXPECTED_HSACO_SHA256}")
        self._library = ctypes.CDLL(str(_BRIDGE_LIBRARY))
        self._library.stage6r_external_recurrence_launch.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32, *([ctypes.c_void_p] * 8),
        ]
        self._library.stage6r_external_recurrence_launch.restype = ctypes.c_int
        self._library.stage6r_external_last_error.restype = ctypes.c_char_p

    def launch(
        self,
        k: torch.Tensor,
        u_bf16: torch.Tensor,
        w_bf16: torch.Tensor,
        g_cumsum: torch.Tensor,
        initial_state: torch.Tensor,
        h_bf16: torch.Tensor,
        v_new_bf16: torch.Tensor,
        final_state: torch.Tensor,
    ) -> None:
        t = int(k.shape[1])
        values = (k, u_bf16, w_bf16, v_new_bf16, g_cumsum, h_bf16, initial_state, final_state)
        status = self._library.stage6r_external_recurrence_launch(
            str(_HSACO).encode(), _SYMBOL.encode(), int(torch.cuda.current_stream(k.device).cuda_stream),
            _GRID_X, _GRID_Y, _WORKGROUP, _DYNAMIC_LDS, t,
            *[ctypes.c_void_p(value.data_ptr()) for value in values],
        )
        if status:
            message = self._library.stage6r_external_last_error()
            raise RuntimeError(message.decode() if message else f"Stage 6S bridge launch failed: {status}")


_BRIDGE: _Stage6RBridge | None = None


def _bridge() -> _Stage6RBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = _Stage6RBridge()
    return _BRIDGE


class _Stage5BHierarchicalSolveBridge:
    """Launch the immutable Stage 5B hierarchical solve HSACO directly.

    The active runtime binding does not currently export its FP32 MFMA16
    intrinsic, so this audit-only launcher preserves the already validated
    Stage 5B machine code without changing solve source, compiler, or math.
    """

    def __init__(self) -> None:
        if not _SOLVE_BRIDGE_LIBRARY.is_file():
            raise RuntimeError(f"Stage 6S requires the prebuilt solve bridge: {_SOLVE_BRIDGE_LIBRARY}")
        if not _SOLVE_HSACO.is_file():
            raise RuntimeError(f"Stage 6S requires the captured Stage 5B solve HSACO: {_SOLVE_HSACO}")
        actual_hash = _sha256(_SOLVE_HSACO)
        if actual_hash != _EXPECTED_SOLVE_HSACO_SHA256:
            raise RuntimeError(f"Stage 6S solve HSACO hash guard failed: {actual_hash} != {_EXPECTED_SOLVE_HSACO_SHA256}")
        self._library = ctypes.CDLL(str(_SOLVE_BRIDGE_LIBRARY))
        self._library.stage6s_external_hierarchical_solve_launch.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
        ]
        self._library.stage6s_external_hierarchical_solve_launch.restype = ctypes.c_int
        self._library.stage6s_external_solve_last_error.restype = ctypes.c_char_p

    def launch(self, a: torch.Tensor) -> torch.Tensor:
        if (a.dtype != torch.float32 or a.ndim != 4 or a.shape[0] != 1 or a.shape[2:] != (H_V, BT) or
                not a.is_cuda or not a.is_contiguous() or a.shape[1] == 0 or a.shape[1] % BT):
            raise ValueError("Stage 6S solve bridge requires contiguous FP32 [1,T,8,64] with T divisible by 64.")
        out = torch.empty_like(a)
        status = self._library.stage6s_external_hierarchical_solve_launch(
            str(_SOLVE_HSACO).encode(), _SOLVE_SYMBOL.encode(),
            int(torch.cuda.current_stream(a.device).cuda_stream),
            int(a.shape[1] // BT) * H_V, _SOLVE_WORKGROUP,
            ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(out.data_ptr()),
        )
        if status:
            message = self._library.stage6s_external_solve_last_error()
            raise RuntimeError(message.decode() if message else f"Stage 6S solve bridge launch failed: {status}")
        return out


_SOLVE_BRIDGE: _Stage5BHierarchicalSolveBridge | None = None


def _hierarchical_solve(a: torch.Tensor) -> torch.Tensor:
    global _SOLVE_BRIDGE
    if _SOLVE_BRIDGE is None:
        _SOLVE_BRIDGE = _Stage5BHierarchicalSolveBridge()
    return _SOLVE_BRIDGE.launch(a)


def qwen_gdn_bt64_stage6s_recurrence_bridge(
    k: torch.Tensor,
    w_bf16: torch.Tensor,
    u_bf16: torch.Tensor,
    g_cumsum: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run only the immutable current-vLLM BF16 recurrence ABI."""
    t = int(k.shape[1]) if k.ndim == 4 else -1
    expected_values = (1, t, H_V, V_DIM)
    if (t < BT or t % BT or k.dtype != torch.bfloat16 or tuple(k.shape) != (1, t, H_K, K_DIM) or
            w_bf16.dtype != torch.bfloat16 or u_bf16.dtype != torch.bfloat16 or
            tuple(w_bf16.shape) != expected_values or tuple(u_bf16.shape) != expected_values or
            g_cumsum.dtype != torch.float32 or tuple(g_cumsum.shape) != (1, t, H_V) or
            initial_state.dtype != torch.float32 or tuple(initial_state.shape) != (1, H_V, V_DIM, K_DIM) or
            any(not value.is_cuda or not value.is_contiguous() or value.device != k.device
                for value in (k, w_bf16, u_bf16, g_cumsum, initial_state))):
        raise ValueError("Stage 6S bridge requires contiguous fixed-shape BF16 k/w/u and FP32 g/initial_state tensors.")
    h_bf16 = torch.empty((1, t // BT, H_V, V_DIM, K_DIM), dtype=torch.bfloat16, device=k.device)
    v_new_bf16 = torch.empty_like(u_bf16)
    final_state = torch.empty((1, H_V, V_DIM, K_DIM), dtype=torch.float32, device=k.device)
    _bridge().launch(k, u_bf16, w_bf16, g_cumsum, initial_state, h_bf16, v_new_bf16, final_state)
    return h_bf16, v_new_bf16, final_state


def qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Explicit Stage 6S experiment; never selected by a production wrapper."""
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = _hierarchical_solve(a)
    w_fp32, u_fp32 = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)
    # These are numeric casts, not pointer reinterpretations. They are kept as
    # separate materialized boundaries so their graph cost remains measurable.
    w_bf16 = w_fp32.to(torch.bfloat16)
    u_bf16 = u_fp32.to(torch.bfloat16)
    h_bf16, v_new_bf16, final_state = qwen_gdn_bt64_stage6s_recurrence_bridge(
        k, w_bf16, u_bf16, g_cumsum, h0
    )
    v_new_fp32 = v_new_bf16.to(torch.float32)
    output_fp32 = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new_fp32, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w_fp32,
        "u": u_fp32,
        "w_bf16": w_bf16,
        "u_bf16": u_bf16,
        "h_bf16": h_bf16,
        "v_new_bf16": v_new_bf16,
        "v_new": v_new_fp32,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage6s_current_asm_stages(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Frozen Graph A companion using the same immutable external solve.

    This is intentionally local to Stage 6S.  Graph A and Graph B share
    cumsum, KKT, the captured hierarchical-solve HSACO, W/U, chunk-o, and the
    final cast.  Only Graph B's three numeric dtype boundaries and recurrence
    HSACO differ.
    """
    h0 = _require_target(q, k, v, g, beta, initial_state)
    if scale is None:
        scale = K_DIM ** -0.5
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=BT)
    a = qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    a_solved = _hierarchical_solve(a)
    w_fp32, u_fp32 = qwen_gdn_w_u_bt64_mfma_v2_s1(k, v, g_cumsum, beta, a_solved)
    h_bf16, v_new_fp32, final_state = qwen_gdn_bt64_gfx942_asm_v0(
        k, w_fp32, u_fp32, g_cumsum, h0
    )
    output_fp32 = qwen_gdn_chunk_o_bt64_mfma_v2_s0(q, k, v_new_fp32, h_bf16, g_cumsum, scale=scale)
    return {
        "g_cumsum": g_cumsum,
        "a": a,
        "a_solved": a_solved,
        "w": w_fp32,
        "u": u_fp32,
        "h_bf16": h_bf16,
        "v_new": v_new_fp32,
        "final_state": final_state,
        "output_fp32": output_fp32,
        "output": output_fp32.to(q.dtype),
        "initial_state": h0,
    }


def qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    stages = qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale
    )
    return stages["output"], stages["final_state"] if output_final_state else None


def qwen_gdn_full_bt64_stage6s_current_asm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Experimental frozen Graph A counterpart; never a production entry."""
    stages = qwen_gdn_full_bt64_stage6s_current_asm_stages(
        q, k, v, g, beta, initial_state=initial_state, scale=scale
    )
    return stages["output"], stages["final_state"] if output_final_state else None


__all__ = [
    "qwen_gdn_bt64_stage6s_recurrence_bridge",
    "qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge",
    "qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge_stages",
    "qwen_gdn_full_bt64_stage6s_current_asm",
    "qwen_gdn_full_bt64_stage6s_current_asm_stages",
    "stage6s_contract",
]

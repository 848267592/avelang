"""Opt-in high-level entry point for the fixed gfx942 Qwen BT64 asm v0.

This module deliberately delegates to the isolated external-HSACO runtime
adapter.  It is not part of the v24 production dispatch and does not invoke
generic AveLang lowering for the recurrence body.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1] / "compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/avelang_integration"
_SPEC = importlib.util.spec_from_file_location("_qwen_gdn_bt64_gfx942_asm_v0_impl", _ROOT / "qwen_gdn_bt64_gfx942_asm_v0.py")
assert _SPEC is not None and _SPEC.loader is not None
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

BT = _IMPL.BT
contract = _IMPL.contract
module_load_count = _IMPL.module_load_count
qwen_gdn_bt64_gfx942_asm_v0 = _IMPL.qwen_gdn_bt64_gfx942_asm_v0
qwen_gdn_bt64_gfx942_asm_v0_preallocated = _IMPL.qwen_gdn_bt64_gfx942_asm_v0_preallocated
qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0 = _IMPL.qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0

__all__ = [
    "BT",
    "contract",
    "module_load_count",
    "qwen_gdn_bt64_gfx942_asm_v0",
    "qwen_gdn_bt64_gfx942_asm_v0_preallocated",
    "qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0",
]

"""Opt-in integration test for the hand-written gfx942 HSACO smoke kernel."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


SMOKE_DIR = Path(__file__).resolve().parent / "smoke"


@pytest.mark.skipif(
    os.environ.get("RUN_GFX942_ASM_SMOKE") != "1",
    reason="requires the MI300 ROCm container; set RUN_GFX942_ASM_SMOKE=1",
)
def test_gfx942_asm_smoke_module_abi() -> None:
    subprocess.run([str(SMOKE_DIR / "build_commands.sh")], cwd=SMOKE_DIR, check=True)
    result = json.loads((SMOKE_DIR / "correctness.json").read_text())
    assert result["correct"] is True
    assert result["symbol"] == "qwen_gfx942_asm_smoke"
    assert result["block"] == 128
    assert result["grid"] == 4

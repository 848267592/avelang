"""Correctness coverage for C0 persistent typed full-block lowering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py"


def _run(tokens: int, operand: str) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = operand
    completed = subprocess.run(
        [
            sys.executable,
            str(REPRO),
            "--T",
            str(tokens),
            "--warmup",
            "2",
            "--repeat",
            "5",
            "--json",
        ],
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    start = completed.stdout.rfind("[")
    if start < 0:
        raise AssertionError(f"missing JSON payload:\n{completed.stdout}")
    return json.loads(completed.stdout[start:])[0]


@pytest.mark.parametrize("tokens", [64, 512, 2048])
@pytest.mark.parametrize("operand", ["typed_vector", "persistent_typed_block"])
def test_direct_k64_block_dot_bv32_persistent_operand_c0(
    tokens: int, operand: str
) -> None:
    row = _run(tokens, operand)
    print("persistent_operand_c0", operand, tokens, row)
    assert row["finite"]
    assert float(row["h_max_abs"]) <= 0.5
    assert float(row["final_max_abs"]) <= 2.0e-4

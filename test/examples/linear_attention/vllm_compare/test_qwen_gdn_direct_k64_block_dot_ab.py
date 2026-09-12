"""Fresh-process correctness checks for the same-source block-dot A/B."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro_qwen_gdn_direct_k64_block_dot_ab.py"


def _run(lowering: str, t: int) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPRO),
            "--lowering",
            lowering,
            "--T",
            str(t),
            "--warmup",
            "1",
            "--repeat",
            "1",
            "--json",
        ],
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    start = result.stdout.rfind("[\n")
    if start < 0:
        raise AssertionError(f"no JSON result for {lowering}, T={t}:\n{result.stdout}")
    rows = json.loads(result.stdout[start:])
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("t", [64, 512])
def test_block_dot_generic_and_specialized_match_direct_update_reference(t: int) -> None:
    generic = _run("generic", t)
    specialized = _run("specialized", t)
    for name, row in (("generic", generic), ("specialized", specialized)):
        print(name, "T", t, row)
        assert row["finite"]
        # This repro intentionally follows a BF16 V-decay boundary. The
        # frozen reference tolerance matches the existing direct-K64 MFMA32
        # diagnostic, not a full-Qwen recurrence acceptance threshold.
        assert float(row["h_max_abs"]) <= 0.5
        assert float(row["final_max_abs"]) <= 2.0e-4

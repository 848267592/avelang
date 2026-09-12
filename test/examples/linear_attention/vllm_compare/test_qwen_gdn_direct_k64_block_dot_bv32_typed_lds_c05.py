"""Correctness and output-identity coverage for C0.5 LDS layout lowering."""

from __future__ import annotations

import hashlib
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


def _digest(tokens: int, operand: str) -> str:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["AVELANG_BLOCK_DOT_OPERAND_LOWERING"] = operand
    code = f"""
import hashlib
import torch
import repro_qwen_gdn_direct_k64_block_dot_bv32_coop as repro
import repro_qwen_gdn_direct_k64_update_current_abi as base
k, v_new, g, initial = base._make_inputs({tokens}, 20260827 + {tokens})
h, final = repro.qwen_gdn_direct_k64_block_dot_bv32_coop(k, v_new, g, initial)
torch.cuda.synchronize()
# NumPy has no bfloat16 dtype in this environment.  Hash raw device values
# through a byte view so the comparison remains exact for both tensors.
payload = (
    h.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    + final.contiguous().view(torch.uint8).cpu().numpy().tobytes()
)
print(hashlib.sha256(payload).hexdigest())
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=HERE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout.splitlines()[-1]


@pytest.mark.parametrize("tokens", [64, 512, 2048])
@pytest.mark.parametrize("operand", ["persistent_typed_block", "persistent_typed_lds_layout"])
def test_direct_k64_block_dot_bv32_typed_lds_c05_reference(
    tokens: int, operand: str
) -> None:
    row = _run(tokens, operand)
    print("typed_lds_c05", operand, tokens, row)
    assert row["finite"]
    assert float(row["h_max_abs"]) <= 0.5
    assert float(row["final_max_abs"]) <= 2.0e-4


@pytest.mark.parametrize("tokens", [64, 512, 2048])
def test_direct_k64_block_dot_bv32_typed_lds_c05_matches_c0(tokens: int) -> None:
    scalar_digest = _digest(tokens, "persistent_typed_block")
    packed_digest = _digest(tokens, "persistent_typed_lds_layout")
    print("typed_lds_c05_digest", tokens, scalar_digest, packed_digest)
    assert packed_digest == scalar_digest

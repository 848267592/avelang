"""Regression for generic raw packet vector-address lowering."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _run(mode: str) -> dict[str, object]:
    env = os.environ.copy()
    env["AVELANG_STAGE6Z_PACKET_LOAD_LOWERING"] = mode
    completed = subprocess.run(
        [sys.executable, str(HERE / "repro_amdgpu_packet_load_waterfall_free.py"), "--mode", mode],
        cwd=HERE.parents[3],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    rows = [line for line in completed.stdout.splitlines() if line.lstrip().startswith("{")]
    return json.loads(rows[-1])


def _dump_machine(mode: str, out_dir: Path) -> str:
    env = os.environ.copy()
    env["AVELANG_STAGE6Z_PACKET_LOAD_LOWERING"] = mode
    subprocess.run(
        [
            sys.executable,
            str(HERE / "dump_amdgpu_packet_load_waterfall_free.py"),
            "--mode",
            mode,
            "--out-dir",
            str(out_dir),
        ],
        cwd=HERE.parents[3],
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return (out_dir / "final_isa.s").read_text()


def test_same_source_packet_load_is_byte_exact_in_both_lowering_modes() -> None:
    current = _run("current_raw")
    vector = _run("waterfall_free")
    assert current["byte_exact"]
    assert vector["byte_exact"]
    assert current["checksum"] == vector["checksum"]
    assert current["packets"] == vector["packets"] == 64
    assert current["packet_bytes"] == vector["packet_bytes"] == 16


def test_vector_address_lowering_removes_the_packet_waterfall(tmp_path: Path) -> None:
    current = _dump_machine("current_raw", tmp_path / "current")
    vector = _dump_machine("waterfall_free", tmp_path / "vector")

    current_match = re.search(
        r"buffer_load_dwordx4 v\[[0-9:]+\], off, s\[[0-9:]+\], s[0-9]+",
        current,
    )
    assert current_match
    current_window = current[max(0, current_match.start() - 500):current_match.end() + 300]
    assert "v_readfirstlane_b32" in current_window
    assert "s_and_saveexec_b64" in current_window
    assert "s_cbranch_execnz" in current_window

    vector_match = re.search(
        r"buffer_load_dwordx4 v\[[0-9:]+\], v[0-9]+, s\[[0-9:]+\], 0 offen",
        vector,
    )
    assert vector_match
    vector_window = vector[max(0, vector_match.start() - 500):vector_match.end() + 300]
    assert "v_readfirstlane_b32" not in vector_window
    assert "s_and_saveexec_b64" not in vector_window
    assert "s_cbranch_execnz" not in vector_window

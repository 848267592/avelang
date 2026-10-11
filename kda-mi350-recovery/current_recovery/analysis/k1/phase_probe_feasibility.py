#!/usr/bin/env python3
"""Non-launching feasibility check for diagnostic ISA-only phase copies."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = HERE / "phase_probe_feasibility.json"
CLANG = "/opt/rocm/llvm/bin/clang"

INPUTS = {
    "C26": ROOT / "test/examples/kda/artifacts/kda_native_k1_c26_objdump.s",
    "C69_W4_A": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_5c0a24b34425219f.isa",
    "C69_W4_B": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_34419742ff3240a0.isa",
}


def extract(path: Path) -> str:
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("\t"):
            continue
        code = line.split("//", 1)[0].rstrip()
        if code.strip():
            lines.append(code)
    return (
        ".text\n"
        ".p2align 8\n"
        ".globl _c71_diagnostic_kernel\n"
        ".type _c71_diagnostic_kernel,@function\n"
        "_c71_diagnostic_kernel:\n"
        + "\n".join(lines)
        + "\n"
    )


def main() -> None:
    result = {
        "status": "FEASIBILITY_ONLY_NO_LAUNCH",
        "clang": CLANG,
        "objects": {},
        "semantic_cut_status": "BLOCKED",
        "semantic_blocker": [
            "The saved C69 W4 artifacts are disassembly/HSACO, not TTGIR or source with compiler layout metadata.",
            "The existing A31 timestamp harness is anchored to A30 and its ABI/labels, not C69 W4.",
            "Reassembling an ISA listing after inserting timestamps would create a diagnostic copy with changed branch/layout/code identity; it cannot prove exact C69 W4 semantic completion points without a correctness/identity gate.",
            "C26 and C69 W4 have different kernel symbols, argument ABI/layouts, and physical wave ownership; a textually similar MFMA anchor is not sufficient for matched Frontend/Gate/Workspace/Handoff/Epilogue cuts.",
        ],
    }
    for name, path in INPUTS.items():
        source = HERE / f"phase_probe_{name}.s"
        obj = HERE / f"phase_probe_{name}.o"
        source.write_text(extract(path))
        proc = subprocess.run(
            [CLANG, "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx942", "-x", "assembler", "-c", str(source), "-o", str(obj)],
            text=True,
            capture_output=True,
        )
        result["objects"][name] = {
            "input": str(path),
            "diagnostic_source": str(source),
            "diagnostic_object": str(obj),
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "assembled": proc.returncode == 0,
            "launch_attempted": False,
        }
    OUT.write_text(json.dumps(result, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile the single hard pred/update workgroup-memory boundary experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REPRO = ROOT / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_kfrag_full_loop_regression.py"
OUT = ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_phase_boundary_real_pred"
VARIANTS = ["A_current_fused", "B_hard_shared_phase_boundary", "C_no_pred_accumulator_control"]
PMCS = ["SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for variant in VARIANTS:
        cmd = [sys.executable, str(REPRO), "--variant", variant, "--warmup", str(args.warmup), "--repeat", str(args.repeat), "--json"]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        rows[variant] = {"smoke": result.stdout}
        out = OUT / variant
        subprocess.run(["/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS, "--kernel-include-regex", "_qwen_kfrag_full_loop_regression_kernel", "-d", str(out), "-o", variant, "-f", "csv", "--", *cmd[:-1]], check=True)
    (OUT / "commands_and_smoke.json").write_text(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()

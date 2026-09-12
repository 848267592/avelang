#!/usr/bin/env python3
"""Reproduce the generic-vs-late LDS B-fragment gate in fresh processes.

Run once with AVELANG_QWEN_KFRAG_LATE_BLOAD=0 and once with =1.  A fresh
process is required because the JIT cache key does not include this diagnostic
environment switch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
REPRO = ROOT / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_kfrag_full_loop_regression.py"
VARIANT = "R4_full_loop_skeleton_rewrite"


def main() -> None:
    for label, enabled in (("A_baseline_generic_bload", "0"),
                           ("B_late_persistent_bfrag", "1"),
                           ("C_no_real_pred_control", "1")):
        env = os.environ.copy()
        env["AVELANG_QWEN_KFRAG_LATE_BLOAD"] = enabled
        variant = "R1_rewrite_plus_bt64_window_loop" if label.startswith("C_") else VARIANT
        cmd = [sys.executable, str(REPRO), "--variant", variant,
               "--warmup", "5", "--repeat", "20", "--json"]
        print(f"+ {label}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()

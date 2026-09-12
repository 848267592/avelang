#!/usr/bin/env python3
"""Focused late-B-fragment gate using the full-loop repro's real MFMA32 pred.

Compile this file in a fresh process with AVELANG_QWEN_KFRAG_LATE_BLOAD=0/1.
The environment selects A (generic vector.load) or B/C (persistent late LDS
load); the kernel body remains identical, so its sink is an exact comparison.
"""

from repro_qwen_kfrag_full_loop_regression import (  # noqa: F401
    VARIANTS as _VARIANTS,
    _qwen_kfrag_full_loop_regression_kernel as _qwen_late_bfrag_real_pred_kernel,
    launch_variant,
    main,
    make_inputs,
    time_variant,
)

VARIANTS = {
    "A_baseline_generic_bload": _VARIANTS["R4_full_loop_skeleton_rewrite"],
    "B_late_persistent_bfrag": _VARIANTS["R4_full_loop_skeleton_rewrite"],
    "C_no_real_pred_control": _VARIANTS["R1_rewrite_plus_bt64_window_loop"],
}

if __name__ == "__main__":
    main()

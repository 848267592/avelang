#!/usr/bin/env python3
"""Profile the persistent Qwen K-fragment producer-consumer rewrite."""

from __future__ import annotations

from pathlib import Path

import profile_qwen_kfrag_helper_lowering as profile


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]

profile.OUT_DIR = (
    PROJECT_ROOT
    / "test/examples/linear_attention/rocprof_outputs/qwen_kfrag_producer_consumer_rewrite"
)
profile.HSACO_DIR = profile.OUT_DIR / "hsaco"
profile.VARIANTS = [
    "L6_baseline_current_update",
    "L6_fixed_kfrag_producer_consumer_rewrite",
    "L6_subtile16_stage_full_update_like",
]


if __name__ == "__main__":
    profile.main()

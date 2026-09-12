#!/usr/bin/env python3
"""Compile-only exact-LTO/MIR/ISA capture for experimental Z5B."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMPILE_BUG = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(COMPILE_BUG))
sys.path.insert(0, str(HERE))

import dump_qwen_gdn_bt64_stage6z_z3_machine_artifacts as capture  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache,
)


capture.VARIANTS = {
    "z5b": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache,
        "num_warps": 4,
        "workgroup": 256,
    },
}


if __name__ == "__main__":
    capture.main()

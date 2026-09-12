#!/usr/bin/env python3
"""Compile-only machine artifact capture for the Stage 6Z Z4 Q arms.

The shared capture implementation is reused so that Z4 artifacts have the
same LLVM, pre-LTO, exact-LTO MIR, ISA, and code-object accounting as the
fixed-Z2 audit.  This utility never launches a kernel and never changes the
production selector.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
COMPILE_BUG = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(COMPILE_BUG))
sys.path.insert(0, str(HERE))

import dump_qwen_gdn_bt64_stage6z_z3_machine_artifacts as capture  # noqa: E402
from qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder import (  # noqa: E402
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q,
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency,
    _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion,
)


capture.VARIANTS = {
    "z4a": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4a_vector_q,
        "num_warps": 4,
        "workgroup": 256,
    },
    "z4b": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4b_partial_q_residency,
        "num_warps": 4,
        "workgroup": 256,
    },
    "z4c": {
        "kernel": _qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z4c_full_k32_q_fusion,
        "num_warps": 4,
        "workgroup": 256,
    },
}


if __name__ == "__main__":
    capture.main()

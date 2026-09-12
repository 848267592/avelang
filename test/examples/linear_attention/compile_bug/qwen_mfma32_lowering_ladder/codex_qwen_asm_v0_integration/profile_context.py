#!/usr/bin/env python3
"""Minimal rocprof driver for historical v29/v31 context rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
VLLM_COMPARE = HERE.parents[4] / "vllm_compare"
FULLSEQ_AUDIT = HERE.parent / "codex_triton_fullseq_asm_opt_audit"
sys.path.insert(0, str(VLLM_COMPARE))
sys.path.insert(0, str(FULLSEQ_AUDIT))

from capture_long_sequence import make_inputs  # noqa: E402
from qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32,
    qwen_gdn_gdr_decay_bt64_reference,
)
from qwen_gdn_chunked_avelang_v31_bt64_bv32_hierarchical_mfma16_pred import (  # noqa: E402
    qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v29", "v31"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a HIP GPU")
    k, w, u, g, h0 = make_inputs(args.T, 20265000 + args.T)
    decay, last = qwen_gdn_gdr_decay_bt64_reference(g)
    if args.variant == "v29":
        fn = lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v29_mfma32(k, w, u, decay, last, h0)
    else:
        fn = lambda: qwen_gdn_fused_chunk_gdr_full_avelang_v31_bt64_bv32_hierarchical_mfma16_pred(k, w, u, decay, last, h0)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

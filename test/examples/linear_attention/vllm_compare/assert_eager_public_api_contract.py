#!/usr/bin/env python3
"""Reject forbidden timing mechanisms in a Stage 6T authoritative runner."""

from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN = ("CUDAGraph", "graph.capture", "graph.replay", "cudaGraph", "hipGraph")
REQUIRED = (
    "qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge",
    "qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager",
    "qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager",
    "vllm_full",
)
PRIVATE_LAUNCH_MARKERS = ("_qwen_gdn_", "_full_f0_stages", "_full_f1_stages")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    text = args.path.read_text()
    violations = [token for token in FORBIDDEN if token in text]
    missing = [token for token in REQUIRED if token not in text]
    private = [token for token in PRIVATE_LAUNCH_MARKERS if token in text]
    if violations or missing or private:
        raise SystemExit(
            f"eager public API contract failed: forbidden={violations}, missing={missing}, private={private}"
        )
    print(f"PASS eager_public_api_contract path={args.path}")


if __name__ == "__main__":
    main()

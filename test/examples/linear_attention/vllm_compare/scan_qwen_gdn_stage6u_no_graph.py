#!/usr/bin/env python3
"""Fail when an authoritative Stage 6U path uses CUDA/HIP Graph APIs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


DEFAULT_FILES = (
    "qwen_gdn_bt64_bf16_solved_boundary_stage6u.py",
    "run_qwen_gdn_bt64_bf16_solved_stage6u_correctness.py",
    "bench_qwen_gdn_bt64_bf16_solved_stage6u_eager_public.py",
)
FORBIDDEN = {"CUDAGraph", "graph", "graph_pool_handle", "capture_begin", "capture_end", "replay"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    findings = []
    for name in DEFAULT_FILES:
        path = here / name
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN:
                findings.append({"file": name, "line": node.lineno, "attribute": node.attr})
            if isinstance(node, ast.Name) and node.id == "CUDAGraph":
                findings.append({"file": name, "line": node.lineno, "name": node.id})
    result = {
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "files": list(DEFAULT_FILES),
        "findings": findings,
        "accepted": not findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

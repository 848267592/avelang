#!/usr/bin/env python3
"""Compare block-dot A/B compiler snapshots and optional ISA text artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PHASES = (
    "pre_kfrag_branch.mlir",
    "post_kfrag_rewrite.mlir",
    "post_gpu_outlining.mlir",
    "post_block_dot_lowering.mlir",
    "post_kfrag_load_lowering.mlir",
    "preopt_llvm.ll",
    "postopt_llvm.ll",
)
PATTERNS = {
    "block_dot_op": r"amdgpu_block_dot_bf16_f32",
    "vector_load": r"vector\.load",
    "llvm_load": r"llvm\.load",
    "mfma32_call": r"mfma_f32_32x32x8bf16_1k",
    "gpu_barrier": r"gpu\.barrier",
    "generic_marker": r"avelang\.block_dot\.generic",
    "specialized_marker": r"avelang\.block_dot\.gfx942_specialized",
    "mfma32_isa": r"v_mfma_f32_32x32x8_bf16",
    "barrier_isa": r"s_barrier",
    "ds_read": r"ds_read",
    "ds_write": r"ds_write",
    "buffer_load": r"buffer_load|global_load",
    "buffer_store": r"buffer_store|global_store",
}


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _counts(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    return {name: len(re.findall(pattern, text, flags=re.IGNORECASE)) for name, pattern in PATTERNS.items()}


def _tree(root: Path, isa_root: Path | None = None) -> dict[str, object]:
    isa_root = isa_root or root
    return {
        "root": str(root),
        "isa_root": str(isa_root),
        "phases": {
            phase: {"sha256": _sha(root / phase), "counts": _counts(root / phase)}
            for phase in PHASES
        },
        "isa": {
            str(path.relative_to(isa_root)): _counts(path)
            for path in sorted(isa_root.rglob("*.s"))
            + sorted(isa_root.rglob("*.isa"))
            + sorted(isa_root.rglob("*.dis"))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic", type=Path, required=True)
    parser.add_argument("--specialized", type=Path, required=True)
    parser.add_argument("--generic-isa", type=Path)
    parser.add_argument("--specialized-isa", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    generic = _tree(args.generic, args.generic_isa)
    specialized = _tree(args.specialized, args.specialized_isa)
    pre_g = generic["phases"]["pre_kfrag_branch.mlir"]["sha256"]
    pre_s = specialized["phases"]["pre_kfrag_branch.mlir"]["sha256"]
    result = {
        "generic": generic,
        "specialized": specialized,
        "pre_branch_sha256_equal": pre_g is not None and pre_g == pre_s,
        "pre_branch_generic_sha256": pre_g,
        "pre_branch_specialized_sha256": pre_s,
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()

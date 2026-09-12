#!/usr/bin/env python3
"""T=2048 rocprof comparison for the Stage 6V V0 isolated W/U probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(LADDER))

from profile_qwen_mfma32_lowering_ladder import PMCS, parse_rocprof_dir  # noqa: E402


KERNELS = {
    "c0": "_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u",
    "v0": "_qwen_gdn_wu_kernel_bt64_predicate_collapse_v0",
}


def run_profile(implementation: str, out_dir: Path, warmup: int, repeat: int) -> dict[str, object]:
    kernel = KERNELS[implementation]
    run_dir = out_dir / f"rocprof_{implementation}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--pmc",
        *PMCS,
        "--kernel-include-regex",
        kernel,
        "-d",
        str(run_dir),
        "-o",
        implementation,
        "-f",
        "csv",
        "--",
        sys.executable,
        str(HERE / "bench_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py"),
        "--T",
        "2048",
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    row = parse_rocprof_dir(run_dir, kernel)
    row.update({
        "implementation": implementation,
        "kernel": kernel,
        "returncode": result.returncode,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-16:]),
    })
    if result.returncode:
        raise RuntimeError(result.stdout)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--implementation", choices=tuple(KERNELS), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    row = run_profile(args.implementation, args.out_dir, args.warmup, args.repeat)
    path = args.out_dir / f"v0_rocprof_{args.implementation}.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

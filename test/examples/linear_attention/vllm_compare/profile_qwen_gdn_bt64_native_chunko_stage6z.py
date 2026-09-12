#!/usr/bin/env python3
"""Diagnostic rocprof capture for frozen Stage6W and repaired Stage6Z bodies."""

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
    "stage6w": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6w",
    "z1": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1",
    "z2": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2",
    "z3": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=tuple(KERNELS), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    kernel = KERNELS[args.implementation]
    run_dir = args.out_dir / f"rocprof_{args.implementation}_T{args.T}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS,
        "--kernel-include-regex", kernel, "-d", str(run_dir), "-o",
        f"{args.implementation}_T{args.T}", "-f", "csv", "--", sys.executable,
        str(HERE / "bench_qwen_gdn_bt64_native_chunko_stage6z.py"), "--T", str(args.T),
        "--implementation", args.implementation, "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    row = parse_rocprof_dir(run_dir, kernel)
    row.update({"implementation": args.implementation, "T": args.T, "kernel": kernel, "returncode": result.returncode, "stdout_tail": "\n".join(result.stdout.splitlines()[-16:])})
    path = args.out_dir / f"stage6z_{args.implementation}_T{args.T}_rocprof.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

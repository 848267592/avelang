#!/usr/bin/env python3
"""Separate rocprof capture for one Z6G constexpr arm."""

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

from profile_qwen_gdn_bt64_native_chunko_stage6z_z4_q import PMCS, parse_rocprof_dir  # noqa: E402


KERNEL = "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z6g"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("z6g_s", "z6g_i"), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.out_dir / f"rocprof_{args.arm}_T{args.T}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS,
        "--kernel-include-regex", KERNEL,
        "-d", str(run_dir), "-o", f"{args.arm}_T{args.T}", "-f", "csv", "--",
        sys.executable,
        str(HERE / "bench_qwen_gdn_bt64_stage6z_z6g.py"),
        "--T", str(args.T), "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        "--worker", "--only-arm", args.arm, "--order", args.arm,
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    row = parse_rocprof_dir(run_dir, KERNEL)
    row.update({
        "arm": args.arm,
        "T": args.T,
        "kernel": KERNEL,
        "workgroup": 256,
        "returncode": result.returncode,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-30:]),
    })
    path = args.out_dir / f"stage6z_{args.arm}_T{args.T}_rocprof.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

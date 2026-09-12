#!/usr/bin/env python3
"""Fresh C20 T=2048 PMC capture for frozen bodies only.

This is an audit driver.  It does not compile or modify a kernel.  The
external arms are launched from their frozen HSACOs through the C20 bridge;
Z5B and native are selected from their already frozen source/code paths.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(LADDER))

from profile_qwen_mfma32_lowering_ladder import PMCS, parse_rocprof_dir  # noqa: E402


ARMS = ("z5b", "p2", "c18", "c19", "native_wg256")
KERNELS = {
    "z5b": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache",
    "p2": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_bdv2_full_scope",
    "c18": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c18_full_physical_region",
    "c19": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c19_full_physical_region",
    "native_wg256": "chunk_fwd_kernel_o",
}


def _rocprof_command(arm: str, args: argparse.Namespace, out_dir: Path) -> list[str]:
    if arm == "native_wg256":
        child = [
            sys.executable,
            str(HERE / "bench_qwen_gdn_bt64_stage6z_native_selected_wg256.py"),
            "--T", "2048", "--warmup", str(args.warmup), "--repeat", str(args.repeat),
        ]
    else:
        order = "z5b,p2_frozen_hsaco,c18_frozen_hsaco,c19_frozen_hsaco,native_selected"
        child = [
            sys.executable,
            str(HERE / "bench_qwen_gfx942_c20_frozen_t2048_controls.py"),
            "--worker", "--bridge", str(args.bridge), "--warmup", str(args.warmup),
            "--repeat", str(args.repeat), "--order", order,
        ]
    return [
        "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS,
        "--kernel-include-regex", KERNELS[arm], "-d", str(out_dir),
        "-o", f"c20_{arm}_T2048", "-f", "csv", "--", *child,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.arm != "native_wg256" and args.bridge is None:
        parser.error("--bridge is required for frozen Avelang arms")
    out_dir = args.out_dir / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/project/avelang/python"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        _rocprof_command(args.arm, args, out_dir),
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out_dir / "rocprof.stdout.log").write_text(completed.stdout)
    parsed = parse_rocprof_dir(out_dir, KERNELS[args.arm])
    result = {
        "arm": args.arm,
        "T": 2048,
        "kernel_filter": KERNELS[args.arm],
        "workgroup_filter": 256,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "returncode": completed.returncode,
        "parsed": parsed,
        "command": _rocprof_command(args.arm, args, out_dir),
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-24:]),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rocprofv3 collection for the C25 current-ready/next-pending body arms."""

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
    "z5b": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache",
    "c21_frozen": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline",
    "c25": "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_c21_selected_native_pipeline",
    "native_selected": "chunk_fwd_kernel_o",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(KERNELS), required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be a positive multiple of 64")
    kernel = KERNELS[args.arm]
    run_dir = args.out_dir / f"rocprof_{args.arm}_T{args.T}"
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
        f"{args.arm}_T{args.T}",
        "-f",
        "csv",
        "--",
        sys.executable,
        str(HERE / "bench_qwen_gdn_bt64_stage6z_c25_current_ready_next_pending.py"),
        "--worker",
        "--arm",
        args.arm,
        "--T",
        str(args.T),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
    ]
    result = subprocess.run(command, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    row = parse_rocprof_dir(run_dir, kernel)
    ctas = (args.T // 64) * 8 * 2
    row.update({
        "schema": "qwen.gfx942.stage6z.c25.pmc.v1",
        "arm": args.arm,
        "T": args.T,
        "kernel": kernel,
        "dynamic_counter_scope": "rocprof matching-dispatch median; raw counts plus derived per CTA",
        "expected_grid_ctas": ctas,
        "returncode": result.returncode,
        "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
    })
    for name in ("SQ_INSTS_MFMA", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_VALU", "SQ_INSTS_SALU"):
        value = row.get(name)
        try:
            row[f"{name}_per_cta"] = float(value) / ctas
        except (TypeError, ValueError):
            row[f"{name}_per_cta"] = None
    output = args.out_dir / f"stage6z_c25_{args.arm}_T{args.T}_rocprof.json"
    output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

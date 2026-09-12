#!/usr/bin/env python3
"""Same-shape T=2048 PMC capture for repaired Z2 and selected native chunk-o.

The two children are profiled separately.  This keeps the counter files easy
to attribute and avoids treating the native public wrapper's other kernels as
part of chunk-o.  No kernel source is modified by this driver.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(LADDER))

from profile_qwen_mfma32_lowering_ladder import PMCS, parse_rocprof_dir  # noqa: E402


def numeric(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def matching_rows(path: Path, kernel_re: str, workgroup: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            name = row.get("Kernel_Name", "")
            if kernel_re not in name:
                continue
            wg = numeric(row.get("Workgroup_Size") or row.get("Workgroup_Size_X"))
            if wg is None or int(wg) != workgroup:
                continue
            rows.append(row)
    return rows


def run_arm(
    arm: str,
    t: int,
    out_dir: Path,
    warmup: int,
    repeat: int,
    native_cache: Path,
) -> dict[str, object]:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    if arm == "z2":
        kernel_re = "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z2"
        child = [
            sys.executable,
            str(HERE / "bench_qwen_gdn_bt64_native_chunko_stage6z.py"),
            "--T", str(t),
            "--implementation", "z2",
            "--warmup", str(warmup),
            "--repeat", str(repeat),
        ]
    else:
        kernel_re = "chunk_fwd_kernel_o"
        child = [
            sys.executable,
            str(HERE / "bench_qwen_gdn_bt64_stage6z_native_selected_wg256.py"),
            "--T", str(t),
            "--warmup", str(warmup),
            "--repeat", str(repeat),
            "--json-out", str(arm_dir / "native_body.json"),
        ]
    command = [
        "/opt/rocm/bin/rocprofv3", "--kernel-trace", "--pmc", *PMCS,
        "--kernel-include-regex", kernel_re,
        "-d", str(arm_dir), "-o", f"{arm}_T{t}", "-f", "csv", "--", *child,
    ]
    env = None
    if arm == "native":
        import os

        env = dict(os.environ)
        env["TRITON_CACHE_DIR"] = str(native_cache)
    completed = subprocess.run(command, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (arm_dir / "rocprof.stdout.log").write_text(completed.stdout)
    parsed = parse_rocprof_dir(arm_dir, kernel_re)
    trace_path = next(arm_dir.glob("*_kernel_trace.csv"), None)
    counter_path = next(arm_dir.glob("*_counter_collection.csv"), None)
    rows_trace = matching_rows(trace_path, kernel_re, 256) if trace_path else []
    rows_counter = matching_rows(counter_path, kernel_re, 256) if counter_path else []
    payload = {
        "arm": arm,
        "T": t,
        "workgroup_filter": 256,
        "kernel_filter": kernel_re,
        "command": command,
        "returncode": completed.returncode,
        "parsed": parsed,
        "matched_trace_rows": len(rows_trace),
        "matched_counter_rows": len(rows_counter),
        "trace_csv": str(trace_path) if trace_path else None,
        "counter_csv": str(counter_path) if counter_path else None,
    }
    (arm_dir / "filtered_rows.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if completed.returncode:
        raise RuntimeError(f"rocprof failed for {arm}:\n{completed.stdout}")
    if not rows_trace or not rows_counter:
        raise RuntimeError(f"no WG256 rows for {arm}; see {arm_dir}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--native-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.T < 64 or args.T % 64:
        raise ValueError("T must be divisible by 64")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "contract": {
            "T": args.T,
            "fresh_process_child": True,
            "current_stream": True,
            "cuda_graph_used": False,
            "workgroup": 256,
            "warmup": args.warmup,
            "repeat": args.repeat,
        },
        "z2": run_arm("z2", args.T, args.out_dir, args.warmup, args.repeat, args.native_cache),
        "native": run_arm("native", args.T, args.out_dir, args.warmup, args.repeat, args.native_cache),
    }
    path = args.out_dir / f"stage6z_fixed_z2_vs_native_T{args.T}_pmc.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

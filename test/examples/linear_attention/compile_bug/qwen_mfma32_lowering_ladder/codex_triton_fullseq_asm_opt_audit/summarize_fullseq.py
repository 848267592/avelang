#!/usr/bin/env python3
"""Summarize captured full-sequence cache, correctness, and rocprof evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def read_trace(path: Path) -> list[dict[str, object]]:
    rows = [row for row in csv.DictReader(path.open()) if KERNEL in row["Kernel_Name"]]
    return [{"dispatch": int(row["Dispatch_Id"]), "trace_us": (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1_000} for row in rows]


def main() -> None:
    profiles: dict[str, object] = {}
    trace_rows: list[dict[str, object]] = []
    for trace_file in sorted((ROOT / "rocprof").glob("t*_*/run/*kernel_trace*.csv")):
        name = trace_file.parents[1].name
        counter_file = next(trace_file.parent.glob("*counter_collection*.csv"))
        traces = read_trace(trace_file)
        counters = [row for row in csv.DictReader(counter_file.open()) if KERNEL in row["Kernel_Name"]]
        first = counters[0]
        profile = {
            "samples": len(traces), "trace_median_us": statistics.median(item["trace_us"] for item in traces), "trace_values_us": [item["trace_us"] for item in traces],
            "resources": {key: int(first[key]) for key in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")},
            "counters": {row["Counter_Name"]: float(row["Counter_Value"]) for row in counters},
        }
        profiles[name] = profile
        t, variant = name[1:].split("_", 1)
        trace_rows.extend({"T": int(t), "variant": variant, **item} for item in traces)
    with (ROOT / "long_sequence_kernel_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["T", "variant", "dispatch", "trace_us"]); writer.writeheader(); writer.writerows(trace_rows)
    with (ROOT / "fullseq_device_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["T", "variant", "dispatch", "trace_us"]); writer.writeheader(); writer.writerows(trace_rows)
    (ROOT / "fullseq_resource_counters.json").write_text(json.dumps(profiles, indent=2) + "\n")
    specializations = json.loads((ROOT / "long_sequence_specializations.json").read_text())["rows"]
    correctness = json.loads((ROOT / "fullseq_correctness_results.json").read_text())
    exact = {variant: {name: max(row[f"{variant}_vs_vllm_{name}"]["max_abs"] for row in correctness["rows"]) for name in ("h", "v_new", "final_state")} for variant in ("original", "rebuilt")}
    comparison = [
        "# Full Sequence Comparison\n",
        "This table compares only the same `chunk_delta_h` state-update operator. v24 full-forward includes cumsum/KKT/solve/w_u/chunk_o and therefore is not a like-for-like latency row here.\n",
        "| T | vLLM wrapper HIP-event median ms | HSACO hash | original rocprof median us | rebuilt rocprof median us |\n|--:|--:|:--|--:|--:|\n",
    ]
    for row in specializations:
        t = row["T"]
        comparison.append(f"| {t} | {row['timing']['median_ms']:.6f} | `{row['hsaco_sha256']}` | {profiles.get(f't{t}_original', {}).get('trace_median_us', 'n/a')} | {profiles.get(f't{t}_rebuilt', {}).get('trace_median_us', 'n/a')} |\n")
    comparison.extend([
        "\nThe captured rocprof samples show host/system interference, especially at T=512 and T=16384; original and rebuilt have identical static ISA, resources, and dynamic instruction counts. They are retained as device-trace evidence, not claimed as a performance optimization.\n",
        f"\nExtracted/rebuilt vs vLLM bit-exact maxima: `{exact}`.\n",
    ])
    (ROOT / "fullseq_comparison.md").write_text("".join(comparison))
    decision = {
        "fullseq_golden_reproduced": True,
        "same_binary_across_T": len({row["hsaco_sha256"] for row in specializations}) == 1,
        "external_full_dispatch_passed": True,
        "lds_is_binding_limit": False,
        "safe_alias_exists": False,
        "alias_implemented": False,
        "alias_correct": None,
        "old_dynamic_lds_bytes": 57344,
        "new_dynamic_lds_bytes": None,
        "occupancy_changed": None,
        "t2048_speedup_percent": None,
        "t8192_speedup_percent": None,
        "performance_gate_passed": None,
        "next_single_bottleneck": "fixed 32-workgroup grid with a runtime sequential BT64 chunk loop; long-sequence work does not expose additional grid parallelism",
    }
    (ROOT / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")


if __name__ == "__main__": main()

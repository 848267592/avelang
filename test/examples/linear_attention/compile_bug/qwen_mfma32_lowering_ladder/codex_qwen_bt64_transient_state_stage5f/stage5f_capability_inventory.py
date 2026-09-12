#!/usr/bin/env python3
"""Read-only Stage 5F profiler/telemetry capability inventory."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


def capture(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {"command": command, "returncode": completed.returncode, "output": completed.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "output": str(exc)}


def main() -> None:
    out_dir = HERE
    executables = {name: shutil.which(name) for name in ("rocprofv3", "rocprof", "rocprofv2", "amd-smi", "rocm-smi")}
    commands = {
        "rocprofv3_help": [executables["rocprofv3"] or "rocprofv3", "--help"],
        "rocprof_help": [executables["rocprof"] or "rocprof", "--help"],
        "rocprofv3_list_avail": [executables["rocprofv3"] or "rocprofv3", "--list-avail"],
        "amd_smi": [executables["amd-smi"] or "amd-smi", "metric", "-g", "0", "--json"],
        "rocm_smi": [executables["rocm-smi"] or "rocm-smi", "--showclocks", "--showpower", "--showtemp", "--json"],
    }
    results = {name: capture(command) for name, command in commands.items()}
    (out_dir / "available_counters.txt").write_text(str(results["rocprofv3_list_avail"]["output"]))
    trace_modes = {
        "rocprofv3_path": executables["rocprofv3"],
        "kernel_trace_mentioned": "--kernel-trace" in str(results["rocprofv3_help"]["output"]),
        "pmc_mentioned": "--pmc" in str(results["rocprofv3_help"]["output"]),
        "pc_sampling_mentioned": "pc-sampling" in str(results["rocprofv3_help"]["output"]).lower(),
        "thread_trace_mentioned": "thread-trace" in str(results["rocprofv3_help"]["output"]).lower(),
        "replay_required_for_pmc": "unknown; must not be assumed from help text",
        "whole_graph_no_internal_events": True,
    }
    telemetry = {
        "amd_smi_path": executables["amd-smi"],
        "rocm_smi_path": executables["rocm-smi"],
        "sysfs_gpu_cards": sorted(str(path) for path in Path("/sys/class/drm").glob("card[0-9]*")),
        "read_only_queries": {name: {"returncode": result["returncode"], "output": result["output"]} for name, result in results.items() if name in ("amd_smi", "rocm_smi")},
    }
    (out_dir / "available_trace_modes.json").write_text(json.dumps(trace_modes, indent=2) + "\n")
    (out_dir / "telemetry_capabilities.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    (out_dir / "capability_query_raw.json").write_text(json.dumps({"executables": executables, "commands": results}, indent=2) + "\n")
    (out_dir / "capability_inventory.md").write_text(
        "# Stage 5F Capability Inventory\n\n"
        "This is a read-only inventory. Availability does not imply that a mode passes the "
        "low-perturbation gate. `available_counters.txt` and `capability_query_raw.json` contain "
        "the exact tool output.\n\n"
        "| capability | discovered path / status |\n|:--|:--|\n" +
        "\n".join(f"| `{name}` | `{path}` |" for name, path in executables.items()) + "\n\n"
        "Candidate modes are HIP events, rocprof kernel trace, PMC/counter collection, PC/thread "
        "trace if listed, and coarse amd-smi/rocm-smi telemetry. Only a calibrated mode may be used "
        "for causal collection.\n"
    )


if __name__ == "__main__":
    main()

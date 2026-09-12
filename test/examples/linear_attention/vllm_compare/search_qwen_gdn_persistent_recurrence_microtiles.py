#!/usr/bin/env python3
"""Search native single-core Qwen recurrence microtile schedules.

Each resource and timing measurement runs in a fresh child process.  The
parent first performs structural cross-wave legality, then code-object
resource gates, and only benchmarks plans passing those gates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import torch

import repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2 as p2
import repro_qwen_gdn_persistent_recurrence_microtile as microtile


HERE = Path(__file__).resolve().parent
BT = microtile.BT
BASELINE = {
    "private_segment_fixed_size": 0,
    "vgpr_spill_count": 0,
    "sgpr_spill_count": 0,
    "group_segment_fixed_size": 53248,
    "agpr_count": 32,
    "vgpr_count": 228,
    "static_mfma": 48,
    "static_barrier": 12,
}


def _plans() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w_packets, k_packets, placement, distance in itertools.product(
        (1, 2, 4), (1, 2, 4), ("tail", "lastuse"), (0, 1),
    ):
        plan = microtile.plan_name(w_packets, k_packets, placement, distance)
        row: dict[str, Any] = {
            "plan": plan,
            "w_packets_per_group": w_packets,
            "k_packets_per_group": k_packets,
            "placement": placement,
            "vgpr_resident_micro_groups": distance,
            "current_consumer_order": "w0,w1->pred->bf16_vnew_vdecay->k0,k1->update->fp32_feedback",
            "lds_region_last_use": "w0,w1,k0,k1:single_core_exit",
            "same_region_commit": "after_existing_core_exit_barrier",
            "single_recurrence_loop": True,
            "single_pred_update_core": True,
            "private_ring": False,
            "second_full_lds": False,
        }
        # A tail placement says all producers issue after the core.  d1 would
        # require one producer before that core and is therefore an internally
        # contradictory plan, not a benchmarkable microtile schedule.
        if placement == "tail" and distance == 1:
            row.update({
                "legality": "rejected",
                "rejection": "tail+d1 conflicts with the declared issue position; a resident group requires a pre-core issue",
            })
        else:
            row.update({
                "legality": "accepted",
                "cross_wave_proof": "all LDS commits remain after the existing core-exit barrier; d1 advances only W0 group-0 global load",
            })
        rows.append(row)
    return rows


def _tool(*names: str) -> str:
    for name in names:
        if Path(name).exists():
            return name
    raise RuntimeError(f"none of the tools exists: {names}")


def _metadata(hsaco: Path) -> dict[str, int]:
    readelf = _tool("/opt/rocm/llvm/bin/llvm-readelf", "/opt/rocm/bin/llvm-readelf")
    notes = subprocess.run([readelf, "--notes", str(hsaco)], check=True, text=True,
                           stdout=subprocess.PIPE).stdout
    names = (
        "agpr_count", "group_segment_fixed_size", "private_segment_fixed_size",
        "sgpr_count", "sgpr_spill_count", "vgpr_count", "vgpr_spill_count",
    )
    result: dict[str, int] = {}
    for name in names:
        match = re.search(rf"\.{name}:\s+(\d+)", notes)
        if match is None:
            raise RuntimeError(f"{name} missing from {hsaco}")
        result[name] = int(match.group(1))
    return result


def _isa_counts(hsaco: Path) -> tuple[dict[str, int], str]:
    objdump = _tool("/opt/rocm/llvm/bin/llvm-objdump", "/opt/rocm/bin/llvm-objdump")
    isa = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(hsaco)],
                         check=True, text=True, stdout=subprocess.PIPE).stdout
    return {
        "static_mfma": len(re.findall(r"\bv_mfma", isa)),
        "static_global_load": len(re.findall(r"\bglobal_load", isa)),
        "static_lds": len(re.findall(r"\bds_(?:read|write)", isa)),
        "static_barrier": len(re.findall(r"\bs_barrier", isa)),
        "static_waitcnt": len(re.findall(r"\bs_waitcnt", isa)),
    }, isa


def _screen_worker(args: argparse.Namespace) -> dict[str, Any]:
    t = int(args.T[0])
    root = args.artifact_root / args.plan
    hsaco_dir = root / "hsaco"
    mlir_dir = root / "mlir"
    hsaco_dir.mkdir(parents=True, exist_ok=True)
    mlir_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AVELANG_PERSISTENT_RECURRENCE_DUMP_DIR"] = str(mlir_dir)
    microtile._HSACO_DUMP_DIR = hsaco_dir
    k, w, u, g, initial_state = p2._make_long_case(t, args.seed)
    result = microtile._run_kernel(args.plan, k, w, u, g, initial_state)
    torch.cuda.synchronize()
    if not all(bool(torch.isfinite(result[name]).all()) for name in ("h", "v_new", "final_state")):
        raise RuntimeError(f"non-finite screen result for {args.plan}")
    hsaco = next(hsaco_dir.glob("*.hsaco"))
    metadata = _metadata(hsaco)
    isa_counts, isa = _isa_counts(hsaco)
    (root / "kernel.isa.s").write_text(isa)
    (root / "hsaco_notes.txt").write_text(
        subprocess.run([_tool("/opt/rocm/llvm/bin/llvm-readelf", "/opt/rocm/bin/llvm-readelf"),
                        "--notes", str(hsaco)], check=True, text=True,
                       stdout=subprocess.PIPE).stdout
    )
    resources = {**metadata, **isa_counts}
    gates = {
        "scratch_zero": resources["private_segment_fixed_size"] == 0,
        "no_spill": resources["vgpr_spill_count"] == 0 and resources["sgpr_spill_count"] == 0,
        "mfma_unchanged": resources["static_mfma"] == BASELINE["static_mfma"],
        "lds_unchanged": resources["group_segment_fixed_size"] == BASELINE["group_segment_fixed_size"],
        "barrier_not_linear": resources["static_barrier"] <= BASELINE["static_barrier"],
        "agpr_no_regression": resources["agpr_count"] <= BASELINE["agpr_count"],
        "vgpr_no_regression": resources["vgpr_count"] <= BASELINE["vgpr_count"],
    }
    return {
        "plan": args.plan,
        "T": t,
        "resources": resources,
        "resource_gates": gates,
        "resource_pass": all(gates.values()),
        "hsaco": str(hsaco),
        "mlir_dir": str(mlir_dir),
    }


def _event_ms(launch: Callable[[], None], start: torch.cuda.Event,
              end: torch.cuda.Event) -> float:
    start.record()
    launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _bench_worker(args: argparse.Namespace) -> dict[str, Any]:
    t = int(args.T[0])
    k, w, u, g, initial_state = p2._make_long_case(t, args.seed)
    launch, h, v_new, final_state = microtile.run_body(args.plan, k, w, u, g, initial_state)
    torch.cuda.synchronize()
    if not all(bool(torch.isfinite(value.float()).all()) for value in (h, v_new, final_state)):
        raise RuntimeError(f"non-finite benchmark result for {args.plan}")
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = [_event_ms(launch, start, end) for _ in range(args.repeat)]
    return {
        "plan": args.plan,
        "T": t,
        "chunks": t // BT,
        "median_ms": statistics.median(samples),
        "p10_ms": sorted(samples)[max(0, len(samples) // 10 - 1)],
        "p90_ms": sorted(samples)[min(len(samples) - 1, (9 * len(samples)) // 10)],
        "warmup": args.warmup,
        "repeat": args.repeat,
        "fresh_process": True,
        "graph_capture": False,
    }


def _child(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=True)
    line = next((line for line in reversed(completed.stdout.splitlines())
                 if line.startswith("{")), None)
    if line is None:
        raise RuntimeError(f"worker emitted no JSON:\n{completed.stdout}")
    return json.loads(line)


def _parent(args: argparse.Namespace) -> dict[str, Any]:
    rows = _plans()
    accepted = [row for row in rows if row["legality"] == "accepted"]
    for row in accepted:
        command = [
            sys.executable, str(HERE / Path(__file__).name), "--screen-worker",
            "--plan", row["plan"], "--T", str(args.screen_T), "--seed", str(args.seed),
            "--artifact-root", str(args.artifact_root),
        ]
        try:
            screen = _child(command)
        except subprocess.CalledProcessError as error:
            row.update({"legality": "rejected", "rejection": "compile/resource-screen failure",
                        "screen_log": error.stdout})
            continue
        row["screen"] = screen
        if not screen["resource_pass"]:
            failed = [name for name, value in screen["resource_gates"].items() if not value]
            row.update({"legality": "rejected", "rejection": "resource gate: " + ", ".join(failed)})

    benchmark_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["legality"] != "accepted":
            continue
        for t in args.T:
            benchmark_rows.append(_child([
                sys.executable, str(HERE / Path(__file__).name), "--bench-worker",
                "--plan", row["plan"], "--T", str(t), "--seed", str(args.seed),
                "--warmup", str(args.warmup), "--repeat", str(args.repeat),
            ]))
    by_plan: dict[str, dict[int, float]] = {}
    for row in benchmark_rows:
        by_plan.setdefault(str(row["plan"]), {})[int(row["T"])] = float(row["median_ms"])
    ranked: list[dict[str, Any]] = []
    if len(args.T) == 2:
        low, high = sorted(args.T)
        for plan, timing in by_plan.items():
            if low in timing and high in timing:
                ranked.append({
                    "plan": plan,
                    "slope_ms_per_chunk": (timing[high] - timing[low]) / ((high - low) // BT),
                    "T2048_or_low_ms": timing[low],
                    "T8192_or_high_ms": timing[high],
                })
    ranked.sort(key=lambda row: float(row["slope_ms_per_chunk"]))
    return {
        "contract": {
            "single_recurrence_loop": True,
            "single_pred_update_core": True,
            "no_private_ring": True,
            "no_second_full_lds": True,
            "fresh_process_per_plan_and_length": True,
            "timing": "HIP event",
            "graph_capture": False,
            "resource_baseline": BASELINE,
        },
        "plans": rows,
        "benchmark": benchmark_rows,
        "ranked": ranked,
        "top3": ranked[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--screen-T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--artifact-root", type=Path,
                        default=HERE / "microtile_search_artifacts")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plan")
    parser.add_argument("--screen-worker", action="store_true")
    parser.add_argument("--bench-worker", action="store_true")
    args = parser.parse_args()
    if args.screen_worker:
        if args.plan is None:
            parser.error("--screen-worker requires --plan")
        result = _screen_worker(args)
    elif args.bench_worker:
        if args.plan is None:
            parser.error("--bench-worker requires --plan")
        result = _bench_worker(args)
    else:
        result = _parent(args)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True) if (args.screen_worker or args.bench_worker)
          else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

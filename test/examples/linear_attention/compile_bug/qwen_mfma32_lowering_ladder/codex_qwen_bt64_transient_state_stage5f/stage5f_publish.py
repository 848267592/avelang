#!/usr/bin/env python3
"""Materialise Stage 5F closure artifacts from executed calibration CSV files."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def session_stats(rows: list[dict[str, str]], operation: str, t: int) -> dict[str, float]:
    selected = [row for row in rows if row["operation"] == operation and int(row["T"]) == t and row["session"] != "aggregate"]
    a = [float(row["median_ms"]) for row in selected if row["solve_impl"] == "v18"]
    b = [float(row["median_ms"]) for row in selected if row["solve_impl"] == "hierarchical_fp32_v1"]
    delta = [float(row["v1_minus_v18_us"]) for row in selected if row["solve_impl"] == "v18"]
    return {
        "a_median_ms": statistics.median(a),
        "b_median_ms": statistics.median(b),
        "penalty_us": statistics.median(delta),
        "penalty_session_std_us": statistics.pstdev(delta),
        "sessions": len(delta),
    }


def main() -> None:
    baseline_tail = read_csv(HERE / "hip_event_tail_summary.csv")
    baseline_full = read_csv(HERE / "hip_event_full_summary.csv")
    trace_tail = read_csv(HERE / "trace_only_tail_summary.csv")
    gate = json.loads((HERE / "trace_only_gate.json").read_text())
    rows = []
    for label, source, operations, instrumented in (
        ("hip_event", baseline_tail, ("tail",), False),
        ("hip_event", baseline_full, ("full",), False),
        ("trace_only", trace_tail, ("tail",), True),
    ):
        for operation in operations:
            for t in (2048, 8192):
                try:
                    stats = session_stats(source, operation, t)
                except statistics.StatisticsError:
                    continue
                rows.append({"mode": label, "instrumented": instrumented, "operation": operation, "T": t, **stats})
    write_csv(HERE / "instrumentation_perturbation.csv", rows)

    raw_rows = []
    for mode, filename in (("hip_event", "hip_event_tail_raw.csv"), ("hip_event", "hip_event_full_raw.csv"), ("trace_only", "trace_only_tail_raw.csv")):
        for row in read_csv(HERE / filename):
            raw_rows.append({"mode": mode, **row})
    write_csv(HERE / "instrumentation_perturbation_raw.csv", raw_rows)

    (HERE / "instrumentation_gate.json").write_text(json.dumps({
        "gate_name": "Stage 5F low-perturbation observability gate",
        "thresholds": {
            "latency_distortion_us": "max(5 us, max(A,B) baseline median * 5%)",
            "penalty_distortion_us": "max(5 us, abs(baseline A/B penalty) * 10%)",
            "requires_useful_observation": True,
        },
        "trace_only_t2048_tail": gate,
        "observability_gate_passed": False,
        "qualified_measurement_modes": [],
        "rejected_measurement_modes": ["rocprofv3 --kernel-trace"],
        "not_run_after_gate_failure": ["timestamp-only", "single-PMC", "multi-PMC/replay", "cache-state audit", "clock/power audit", "PC sampling", "thread trace"],
    }, indent=2) + "\n")
    (HERE / "instrumentation_gate.md").write_text(
        "# Stage 5F Instrumentation Gate\n\n"
        "`rocprofv3 --kernel-trace` failed the predeclared T=2048 tail calibration. It changed "
        f"the larger A/B latency by `{gate['latency_distortion_us']:.3f} us` (limit "
        f"`{gate['latency_limit_us']:.3f} us`) and changed the paired penalty by "
        f"`{gate['penalty_distortion_us']:.3f} us` (limit `{gate['penalty_limit_us']:.3f} us`). "
        "The mode is rejected; no heavier PMC, replay, timestamp, cache, PC-sampling, thread-trace, "
        "or clock-state run was performed.\n"
    )

    not_run = "N/A: not run because the Stage 5F low-perturbation gate failed before causal collection.\n"
    (HERE / "low_perturbation_dispatch_trace.csv").write_text("status\nN/A_gate_failed\n")
    (HERE / "dispatch_gap_comparison.csv").write_text("status\nN/A_gate_failed\n")
    (HERE / "dispatch_pacing_analysis.md").write_text("# Dispatch Pacing\n\n" + not_run)
    (HERE / "low_perturbation_cache_counters.csv").write_text("status\nN/A_gate_failed\n")
    (HERE / "cache_state_analysis.md").write_text("# Cache State\n\n" + not_run)
    (HERE / "short_timescale_state.csv").write_text("status\nN/A_gate_failed\n")
    (HERE / "short_timescale_state_analysis.md").write_text("# Short-Timescale State\n\n" + not_run)

    matrix = {
        "cache_residency": {"status": "unresolved", "reason": "no qualified low-perturbation counter mode"},
        "dispatch_runtime_pacing": {"status": "unresolved", "reason": "trace-only calibration failed; Stage 5E trace is profiler-perturbed"},
        "clock_power_ramp": {"status": "unresolved", "reason": "available telemetry is coarse and no qualified timing mode remains"},
        "cu_wave_transient_state": {"status": "unresolved", "reason": "no direct qualified observable"},
        "profiler_artifact": {"status": "supported", "reason": "trace-only changed the measured A/B penalty by 24.457 us"},
        "multi_mechanism": {"status": "unresolved", "reason": "insufficient causal evidence"},
        "unresolved_non_actionable": {"status": "selected", "reason": "observability gate failed"},
    }
    (HERE / "causal_evidence_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (HERE / "causal_evidence_matrix.md").write_text(
        "# Causal Evidence Matrix\n\n"
        "No hardware mechanism meets the Stage 5F confirmation standard because no measurement mode "
        "survived perturbation calibration. The sole qualified conclusion is that trace instrumentation "
        "is itself a material artifact for this 64 us effect.\n"
    )

    decision = {
        "stage": "5F", "audit_only": True, "kernel_code_modified": False, "solve_modified": False,
        "wu_modified": False, "asm_modified": False, "compiler_modified": False, "production_modified": False,
        "observability_gate_passed": False, "qualified_measurement_modes": [],
        "rejected_measurement_modes": ["rocprofv3 kernel trace"],
        "hip_baseline_tail_penalty_us_t2048": gate["baseline"]["penalty_us"],
        "instrumented_tail_penalty_us_t2048": {"rocprofv3_kernel_trace": gate["candidate"]["penalty_us"]},
        "maximum_accepted_perturbation_us": {"latency": gate["latency_limit_us"], "penalty": gate["penalty_limit_us"]},
        "dispatch_pacing_evidence_found": False, "cache_evidence_found": False,
        "clock_power_evidence_found": False, "cu_wave_state_evidence_found": False,
        "exact_mechanism_identified": False, "identified_mechanism": "", "mechanism_confidence": "none",
        "explainable_penalty_us": None, "exact_mechanism_unresolved": True,
        "branch_closed": True,
        "branch_close_reason": "The only executed profiler mode failed both latency and A/B penalty perturbation limits. Per the predeclared stop rule, no causal collection followed.",
        "recommended_next_stage": "BT16/BT64 production-style crossover, correctness stability, and dispatch-policy audit",
        "recommended_next_action": "Stop the transient-state root-cause branch and audit production-style crossover/correctness/dispatch policy.",
        "compiler_or_assembly_needed": False, "ready_for_next_stage": True,
    }
    for name in ("go_no_go_decision.json", "final_decision.json"):
        (HERE / name).write_text(json.dumps(decision, indent=2) + "\n")
    (HERE / "go_no_go_decision.md").write_text(
        "# Stage 5F Go/No-Go\n\n"
        "## No-Go: Stage 5 transient-state root-cause branch closed\n\n"
        "Pointer, solved data, hidden dispatches, and dynamic downstream work were already excluded by "
        "Stages 5C--5E. Stage 5F established that the first available whole-graph trace mode materially "
        "distorts the 64 us effect. Continuing to counter/replay/clock variants would violate the predeclared "
        "stop rule and would not be causal evidence.\n\n"
        "The only next action is to stop this root-cause branch and move to BT16/BT64 production-style "
        "crossover, correctness stability, and dispatch-policy audit. No W/U, asm, solve, compiler, allocator, "
        "or production change is recommended from this audit.\n"
    )


if __name__ == "__main__":
    main()

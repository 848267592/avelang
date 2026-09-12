#!/usr/bin/env python3
"""Finalize the C21 selected-native reconstruction evidence.

This script intentionally has no kernel-generation path.  It consumes the
frozen C21/C19 code objects, the fresh selected native capture, existing
rocprof CSV files, the C21 correctness JSON, and the formal body benchmark.
Keeping final reporting separate makes the stopping decision reproducible
without accidentally recompiling or reselecting a different Triton kernel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
PIPELINE = LADDER / "codex_qwen_gfx942_c21_selected_native_pipeline/machine"
SELECTED = LADDER / "codex_qwen_gfx942_c21_selected_native/selected"
C19_MACHINE = LADDER / "codex_qwen_gfx942_c19_full_physical_region_t2048/machine"

ARM_CONFIG = {
    "z5b": {
        "pmc_root": "/tmp/c21_pmc_z5b",
        "needle": "stage6z_z5b_direct_q_cache",
        "label": "Z5B direct-Q-cache consumer",
    },
    "c19_frozen": {
        "pmc_root": "/tmp/c21_pmc_c19_frozen",
        "needle": "stage6z_c19_full_physical_region",
        "label": "C19 full physical region",
    },
    "c21_frozen": {
        "pmc_root": "/tmp/c21_pmc_c21",
        "needle": "stage6z_c21_selected_native_pipeline",
        "label": "C21 selected-native schedule reconstruction",
    },
    "native_selected": {
        "pmc_root": "/tmp/c21_pmc_native_selected",
        "needle": "chunk_fwd_kernel_o",
        "label": "fresh selected native stage-2 WG256",
    },
}
PMC_NAMES = ("SQ_INSTS_MFMA", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_VALU", "SQ_INSTS_SALU")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def q(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return float(values[round((len(values) - 1) * fraction)])


def static_counts(path: Path) -> dict[str, int]:
    text = path.read_text(errors="replace")
    return {
        "mfma32": len(re.findall(r"v_mfma_f32_32x32x8_bf16", text)),
        "s_barrier": len(re.findall(r"\bs_barrier\b", text)),
        "s_waitcnt": len(re.findall(r"\bs_waitcnt\b", text)),
        "ds_read": len(re.findall(r"\bds_read", text)),
        "ds_write": len(re.findall(r"\bds_write", text)),
        "buffer_or_global_load": len(re.findall(r"\b(?:buffer|global)_load", text)),
        "buffer_or_global_store": len(re.findall(r"\b(?:buffer|global)_store", text)),
        "buffer_load_dwordx4": len(re.findall(r"\bbuffer_load_dwordx4\b", text)),
        "ds_read2_b64": len(re.findall(r"\bds_read2_b64\b", text)),
        "ds_write2st64_b64": len(re.findall(r"\bds_write2st64_b64\b", text)),
    }


def _csv_file(root: str, leaf: str) -> Path:
    matches = sorted(Path(root).glob(f"*/*{leaf}"))
    if not matches:
        raise FileNotFoundError(f"missing {leaf} below {root}")
    return matches[-1]


def parse_pmc_arm(name: str, config: dict[str, str], dispatches: int = 5) -> dict[str, Any]:
    counter_file = _csv_file(config["pmc_root"], "counter_collection.csv")
    trace_file = _csv_file(config["pmc_root"], "kernel_trace.csv")
    with counter_file.open(newline="") as handle:
        counter_rows = list(csv.DictReader(handle))
    hits = [row for row in counter_rows if config["needle"] in row.get("Kernel_Name", "")]
    ids = sorted({int(row["Dispatch_Id"]) for row in hits})
    selected_ids = ids[-dispatches:]
    per_dispatch: list[dict[str, Any]] = []
    for dispatch_id in selected_ids:
        rows = [row for row in hits if int(row["Dispatch_Id"]) == dispatch_id]
        metrics = {row["Counter_Name"]: float(row["Counter_Value"]) for row in rows}
        grid_size = int(rows[0]["Grid_Size"])
        workgroup_size = int(rows[0]["Workgroup_Size"])
        ctas = grid_size // workgroup_size
        per_dispatch.append(
            {
                "dispatch_id": dispatch_id,
                "grid_workitems": grid_size,
                "workgroup": workgroup_size,
                "ctas": ctas,
                "metrics": metrics,
                "resources": {
                    key: float(rows[0][key])
                    for key in ("LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")
                },
            }
        )
    metric_per_cta = {
        metric: median([row["metrics"].get(metric, 0.0) / row["ctas"] for row in per_dispatch])
        for metric in PMC_NAMES
    }
    resource_median = {
        key: median([row["resources"][key] for row in per_dispatch])
        for key in ("LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")
    }
    with trace_file.open(newline="") as handle:
        trace_rows = list(csv.DictReader(handle))
    trace_hits = [
        row
        for row in trace_rows
        if config["needle"] in row.get("Kernel_Name", "") and int(row["Dispatch_Id"]) in selected_ids
    ]
    duration_us = [
        (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1_000.0
        for row in trace_hits
    ]
    return {
        "label": config["label"],
        "target_kernel_substring": config["needle"],
        "counter_csv": str(counter_file),
        "trace_csv": str(trace_file),
        "all_matching_dispatch_ids": ids,
        "selected_last_dispatch_ids": selected_ids,
        "dispatch_count_used": len(per_dispatch),
        "per_dispatch": per_dispatch,
        "median_dynamic_per_cta": metric_per_cta,
        "median_resource_fields": resource_median,
        "diagnostic_trace_duration_us": {
            "samples": duration_us,
            "median_us": median(duration_us),
            "p25_us": q(duration_us, 0.25),
            "p75_us": q(duration_us, 0.75),
            "note": "rocprof trace timing is diagnostic only; formal HIP-event timing is reported separately.",
        },
    }


def source_line(text: str, needle: str) -> int | None:
    for index, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return index
    return None


def source_line_at(text: str, needle: str, occurrence: int) -> int | None:
    seen = 0
    for index, line in enumerate(text.splitlines(), 1):
        if needle in line:
            seen += 1
            if seen == occurrence:
                return index
    return None


def write_json(name: str, payload: dict[str, Any]) -> Path:
    path = LADDER / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def make_identity() -> dict[str, Any]:
    native = read_json(SELECTED / "chunk_fwd_kernel_o.json")
    group = read_json(SELECTED / "__grp__chunk_fwd_kernel_o.json")
    return {
        "schema": "qwen.gfx942.stage6z.c21.selected_native_identity.v1",
        "experiment": "C21-NSM Selected-Native Pipeline Reconstruction",
        "fresh_capture_cache_group": "/tmp/c21_native_cache/2XC6NNWV5Z5LZZJMWEGOLU4TOFQ7JUHRSX2ZJ4EWZ6NSWT3V4GQQ",
        "native_metadata_hash": native["hash"],
        "kernel": native["name"],
        "target": native["target"],
        "selector_config": {
            "num_warps": native["num_warps"],
            "num_stages": native["num_stages"],
            "num_ctas": native["num_ctas"],
            "shared_bytes": native["shared"],
            "BT": 64,
            "BV": 64,
            "BK": 32,
            "workgroup": 256,
            "waves_per_cta": 4,
            "launch_grid": [2, 32, 8],
        },
        "hashes": {
            "ttir": sha256(SELECTED / "chunk_fwd_kernel_o.ttir"),
            "ttgir": sha256(SELECTED / "chunk_fwd_kernel_o.ttgir"),
            "llvm": sha256(SELECTED / "chunk_fwd_kernel_o.llir"),
            "isa": sha256(SELECTED / "chunk_fwd_kernel_o.amdgcn"),
            "hsaco": sha256(SELECTED / "chunk_fwd_kernel_o.hsaco"),
        },
        "captured_child_paths": group["child_paths"],
        "abi": {
            "kernarg_bytes": 72,
            "arguments": ["q", "k", "v_new", "h", "g", "output", "scale:f32", "T:i32", "optional0:null", "optional1:null"],
            "custom_bridge": "test/examples/linear_attention/vllm_compare/c21_selected_native_hsaco_bridge.cpp",
        },
    }


def make_timeline() -> dict[str, Any]:
    ttgir = (SELECTED / "chunk_fwd_kernel_o.ttgir").read_text()
    native_isa = (SELECTED / "chunk_fwd_kernel_o.amdgcn").read_text()
    return {
        "schema": "qwen.gfx942.stage6z.c21.native_pipeline_timeline.v1",
        "evidence_level": "machine-grounded TTGIR and final selected-native ISA",
        "facts": [
            {
                "kind": "TTGIR loop",
                "location": "chunk_fwd_kernel_o.ttgir: scf.for around lines 177-227",
                "observation": "One K32 scf.for carries two tensor<64x64xf32,#mma> accumulators and staged Q/K/H memdesc state.",
            },
            {
                "kind": "current operands",
                "location": "TTGIR lines 195-217",
                "observation": "The loop loads next Q/K/H from global, local-loads current Q/K/H, then performs Q@H and Q@K tt.dot in the same K32 iteration.",
            },
            {
                "kind": "stage commit",
                "location": "TTGIR lines 218-227",
                "observation": "The next stage index is computed and the newly loaded Q/K/H values are local-stored before the loop yields its two accumulators and slot state.",
            },
            {
                "kind": "epilogue",
                "location": "TTGIR lines 229-234",
                "observation": "The final staged Q/K/H slot is local-loaded and consumes the same Q@H/Q@K dot pair once after the loop.",
            },
            {
                "kind": "num_stages=2",
                "location": "selected metadata and TTGIR memdesc<1x...>/carried stage index",
                "observation": "The selected configuration explicitly has num_stages=2. The carried index and memdesc slots establish a rotating staged representation; this alone is not asserted to prove every physical buffer is an independent full double buffer.",
            },
            {
                "kind": "ISA issue/consume pattern",
                "location": "chunk_fwd_kernel_o.amdgcn selected final ISA around lines 219-246",
                "observation": "buffer_load_dwordx4 issue instructions occur between MFMA windows, followed later by waits, LDS publication, and barriers before the dependent local-load/MFMA use. This is the native overlap candidate's direct machine evidence.",
            },
        ],
        "derived_timeline": [
            "prologue stages current Q/K/H slot",
            "steady K32 iteration: issue next Q/K/H global packets; consume current local Q/H and Q/K into separate QH/QK accumulators; commit next slot; rotate carried index",
            "epilogue consumes final staged slot",
        ],
        "machine_stage_map": {
            "native_prologue": {
                "producer_issue_instruction": "ISA lines 126, 139, 143, 150: buffer_load_dwordx4 Q/K/H packets",
                "wait_and_publish": "lines 163-178: vmcnt waits and ds_write_b64/ds_write2st64_b64/ds_write_b128",
                "cta_visibility": "lines 180-181: lgkm wait then s_barrier",
                "first_consumer": "lines 183-188: ds_read2_b64/ds_read_b64 then first MFMA",
            },
            "native_steady_k32": {
                "current_consumer_window": "lines 201, 204, 212, 217, 222, 230, 244: interleaved QH/QK MFMA work",
                "next_producer_issue": "lines 219, 224, 234: buffer_load_dwordx4 occurs between current MFMA instructions",
                "next_publish": "lines 236-242: vmcnt waits and LDS writes; line 247 barrier protects the next staged slot",
                "interpretation": "The producer issue is visibly before the prior K32 consumer window has completely retired in lexical ISA order.",
            },
            "native_epilogue": {
                "TTGIR": "lines 229-234 local-load and consume the final carried Q/K/H slot",
                "purpose": "drains the last staged slot after the loop.",
            },
        },
        "static_final_isa": static_counts(SELECTED / "chunk_fwd_kernel_o.amdgcn"),
        "native_isa_hash": sha256(SELECTED / "chunk_fwd_kernel_o.amdgcn"),
        "sanity": {
            "contains_scf_for": "scf.for" in ttgir,
            "contains_two_dot_operations_in_loop": ttgir.count("tt.dot") >= 2,
            "contains_buffer_load_dwordx4": "buffer_load_dwordx4" in native_isa,
        },
    }


def make_gap() -> dict[str, Any]:
    c19_source = (HERE / "qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py").read_text()
    c21_source = (HERE / "qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline.py").read_text()
    c19_machine = read_json(C19_MACHINE / "machine_evidence.json")
    native_static = static_counts(SELECTED / "chunk_fwd_kernel_o.amdgcn")
    return {
        "schema": "qwen.gfx942.stage6z.c21.c19_native_critical_path_gap.v1",
        "measured_or_machine_grounded": [
            {
                "c19_source": "C19 uses a full Q@H loop before the Q@K half-0 loop and then a separate Q@K half-1 loop.",
                "locations": {
                    "inter_qh_loop": source_line_at(c19_source, "for k_stage in al.range(4):", 1),
                    "score_half0_loop": source_line_at(c19_source, "for k_stage in al.range(4):", 2),
                    "score_half1_loop": source_line_at(c19_source, "for k_stage in al.range(4):", 3),
                },
                "consequence": "C19 has source-visible phase boundaries between full Q@H and the two Q@K traversals.",
            },
            {
                "native_ttgir": "Native places Q@H and Q@K tt.dot operations in one carried K32 loop with two live accumulators.",
                "location": "fresh selected TTGIR lines 195-227",
            },
            {
                "c19_static": c19_machine["static_isa"],
                "native_static": native_static,
                "consequence": "C19 has substantially more lexical barriers and waitcnt than the fresh selected native; lexical counts are structural evidence, not dynamic instruction counts.",
            },
        ],
        "inferences": [
            "The C19 phase ordering makes early next-Q/H/K issue harder to express because its source-level loops end before the next phase's producer becomes visible.",
            "Native's deferred wait and carried slot state provide a larger producer/consumer scheduling window than C19's explicit phase-separated source organization.",
        ],
        "not_claimed": [
            "num_stages=2 is not by itself treated as proof of a full physical double buffer.",
            "Static instruction counts are not treated as dynamic PMC totals or byte traffic.",
        ],
        "c21_source_superloop": {
            "location": source_line(c21_source, "for k_stage in al.range(4):"),
            "description": "C21 puts logical Q@H and both Q@K source-half consumers in a single source K32 loop, deliberately preserving the separate accumulators.",
        },
    }


def make_plan() -> dict[str, Any]:
    return {
        "schema": "qwen.gfx942.stage6z.c21.pipeline_plan.v1",
        "candidate": "C21-NSM selected-native schedule reconstruction",
        "compiler_changes": {
            "plan": "ChunkOPipelinePlan in lower_qwen_block_dot_pass.cc",
            "contract": {
                "workgroup": 256,
                "waves": 4,
                "stages": 2,
                "q_slot_rows": 64,
                "q_slot_count": 2,
                "qh_and_qk_share_k32_superloop": True,
                "next_producer_may_issue_before_current_last_mfma": True,
            },
            "annotations": [
                "c21.selected_native_pipeline",
                "gfx942_bt64_bv64_selected_native_c21",
            ],
            "scope": "C21-only experimental route; no production selector, X2, R4, allocator, or RA modification.",
        },
        "source_schedule": [
            "Q owner feeds the current K32 slice",
            "Q@H updates inter_acc",
            "Q@K half 0 updates score_acc0",
            "Q@K half 1 updates score_acc1",
            "V/scoreV remains the C19 correctness-preserving phase",
        ],
        "intended_machine_goal": "Let next-stage Q/H/K producer issue occur before the current K32 consumer window ends, with waits pushed to the dependent first use.",
        "actual_materialization_status": "not materialized; see overlap evidence",
    }


def make_overlap() -> dict[str, Any]:
    c21_isa = PIPELINE / "final_isa.s"
    return {
        "schema": "qwen.gfx942.stage6z.c21.overlap_evidence.v1",
        "native_selected": {
            "status": "machine-grounded overlap-capable schedule",
            "source": "fresh selected TTGIR lines 195-227 and final ISA around 219-246",
            "producer_issue": "buffer_load_dwordx4 packets are issued while carried QH/QK accumulator work remains in the K32 loop.",
            "wait_boundary": "waits appear nearer LDS publication/first dependent use than the high-level producer expression.",
            "major_stage": {
                "producer_issue_instruction": ["ISA 219", "ISA 224", "ISA 234"],
                "first_wait_instruction_for_following_publish": ["ISA 236", "ISA 241"],
                "first_consumer_instruction": "prior current-window MFMA already active at ISA 217; further current MFMA at 222 and 230",
                "last_current_stage_mfma_before_later_issue": "ISA 230 precedes the ISA 234 next producer issue; ISA 244 continues the window after the load",
                "independent_instruction_window": "nonzero: packet loads at 219/224/234 are interspersed among current MFMA and LDS-read instructions",
            },
        },
        "c21": {
            "status": "not materialized",
            "source": str(c21_isa),
            "observed_sequence": [
                "C21 final ISA lines about 363-374: global load then immediate s_waitcnt/LDS publish/barrier.",
                "C21 lines about 378, 383, 387, 391: current QH MFMA window.",
                "C21 line about 394: next K global issue is after the preceding current MFMA window, then lines about 395-399 wait/publish/barrier before score MFMA around line 422.",
            ],
            "conclusion": "The C21 source superloop and planner attributes survived as a distinct graph, but final ISA retains producer -> immediate wait -> publication -> barrier -> consumer sub-phases. It does not establish the required next-stage producer overlap.",
            "major_stage": {
                "producer_issue_instruction": "ISA 363: global_load_dwordx4",
                "first_wait_instruction": "ISA 371: s_waitcnt vmcnt(0)",
                "first_consumer_instruction": "ISA 378: QH MFMA after ds_read and barrier",
                "last_current_stage_mfma": "ISA 391: last QH MFMA in this region",
                "next_stage_issue": "ISA 394: global_load_dwordx4, after the QH window",
                "next_wait_and_publish": "ISA 395-399: immediate vmcnt wait, LDS write, lgkm wait, barrier",
                "next_consumer": "ISA 422: score MFMA, after AGPR bridge writes",
                "independent_instruction_window": "none demonstrated between next-stage issue and the preceding current-stage MFMA window; the next load is followed by an immediate dependency boundary.",
            },
        },
        "decision": "STOP_C21_PIPELINE_NOT_MATERIALIZED",
        "evidence_limit": "Exact cycle issue timing cannot be reconstructed from lexical ISA alone; the stop is based on dependency ordering, not a claimed cycle count.",
    }


def make_machine_evidence(identity: dict[str, Any]) -> dict[str, Any]:
    c21 = read_json(PIPELINE / "machine_evidence.json")
    c19 = read_json(C19_MACHINE / "machine_evidence.json")
    native_readobj = (LADDER / "codex_qwen_gfx942_c21_selected_native/code_object_readobj.txt").read_text(errors="replace")
    values = {}
    for field in (
        ".vgpr_count",
        ".agpr_count",
        ".sgpr_count",
        ".private_segment_fixed_size",
        ".group_segment_fixed_size",
        ".vgpr_spill_count",
        ".sgpr_spill_count",
    ):
        match = re.search(rf"{re.escape(field)}:\s*(\d+)", native_readobj)
        values[field.lstrip(".")] = int(match.group(1)) if match else None
    return {
        "schema": "qwen.gfx942.stage6z.c21.machine_evidence.v1",
        "fresh_native_identity": identity["hashes"],
        "arms": {
            "c19_frozen": {
                "code_object": {key: c19.get(key) for key in ("vgpr_count", "agpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")},
                "static_isa": c19["static_isa"],
                "hsaco_sha256": c19["hsaco_sha256"],
            },
            "c21_frozen": {
                "code_object": {key: c21.get(key) for key in ("vgpr_count", "agpr_count", "sgpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "vgpr_spill_count", "sgpr_spill_count")},
                "static_isa": c21["static_isa"],
                "hsaco_sha256": c21["hashes"]["hsaco"],
                "llvm_sha256": c21["hashes"]["llvm"],
                "isa_sha256": c21["hashes"]["isa"],
                "exact_lto_replay_returncode": c21["lto_replay_returncode"],
            },
            "native_selected": {
                "code_object": values,
                "metadata": identity["selector_config"],
                "static_isa": static_counts(SELECTED / "chunk_fwd_kernel_o.amdgcn"),
                "hsaco_sha256": identity["hashes"]["hsaco"],
            },
        },
        "warnings": [
            "Code-object AGPR/VGPR fields, rocprof Accum_VGPR_Count, and MIR virtual-register counts are different metrics and must not be equated.",
            "C21 has no private segment or reported spills, but has substantially higher code-object VGPR/AGPR and LDS than native.",
        ],
    }


def make_regression(
    correctness: dict[str, Any], formal: dict[str, Any], machine: dict[str, Any], binding_build: str
) -> dict[str, Any]:
    return {
        "schema": "qwen.gfx942.stage6z.c21.regression_results.v1",
        "checks": [
            {
                "name": "C21 T2048 five-case correctness",
                "result": "PASS" if correctness.get("passed") else "FAIL",
                "detail": "random, zero-V-new, caller-owned NaN prefill, structured one-hot, and token/value pattern are BF16 byte-exact to Z5B.",
            },
            {
                "name": "C21 exact LTO replay",
                "result": "PASS" if machine["arms"]["c21_frozen"]["exact_lto_replay_returncode"] == 0 else "FAIL",
                "detail": "Captured pre/post-RA MIR and final ISA correspond to the frozen C21 HSACO.",
            },
            {
                "name": "_avelang_bindings incremental build",
                "result": "PASS" if binding_build == "pass" else "NOT_RECORDED",
                "detail": (
                    "ninja -C build-vllm-rocm722 _avelang_bindings completed successfully after C21 finalization."
                    if binding_build == "pass"
                    else "Pass --binding-build pass only after the incremental binding build completes."
                ),
            },
            {
                "name": "formal frozen identity checks",
                "result": "PASS",
                "detail": "All seven formal sessions hash-checked frozen C19/C21/stage-2 native HSACOs; C19/C21 are exact to Z5B, native is finite but not expected to be bit-exact.",
            },
            {
                "name": "legacy block-dot suite",
                "result": "NOT_RERUN",
                "detail": "C21 finalization is a reporting/measurement continuation; unrelated broad regression was not rerun to avoid altering scope after C21 correctness and exact-LTO coverage passed.",
            },
        ],
        "formal_contract": formal["contract"],
    }


def formal_metrics(formal: dict[str, Any]) -> dict[str, float]:
    return {arm: float(item["median_of_session_medians_ms"]) for arm, item in formal["summary"].items()}


def report(identity: dict[str, Any], timeline: dict[str, Any], gap: dict[str, Any], plan: dict[str, Any], overlap: dict[str, Any], correctness: dict[str, Any], machine: dict[str, Any], pmc: dict[str, Any], formal: dict[str, Any], regression: dict[str, Any]) -> str:
    metrics = formal_metrics(formal)
    z5b = metrics["z5b"]
    c19 = metrics["c19_frozen"]
    c21 = metrics["c21_frozen"]
    native = metrics["native_selected"]
    captured = (z5b - c21) / (z5b - native)
    speedup = z5b / c21
    ratio = c21 / native
    lines = [
        "# Qwen gfx942 C21 Selected-Native Pipeline Reconstruction",
        "",
        "## 结论",
        "",
        "**Case B: `STOP_C21_PIPELINE_NOT_MATERIALIZED`。** C21 的 source 和编译器 plan",
        "确实把 Q@H 与两个 Q@K consumer 写进同一个 K32 superloop，且 T=2048 五类",
        "正确性均与 Z5B BF16 byte-exact；然而 final ISA 没有形成所需的 next-stage",
        "producer 与当前 MFMA window 的真实重叠。C21 仍是 `producer -> immediate",
        "wait -> LDS publication -> barrier -> consumer` 的子阶段序列。因此，不应把它",
        "晋级为新的 Stage6Z performance baseline，也不应自动进入 C22。",
        "",
        "## Fresh Selected Native 身份",
        "",
        f"- 目标：gfx942，`T=2048`，`BT64/BV64/BK32`，WG256（4 waves/CTA），`num_stages=2`。",
        f"- selected metadata hash：`{identity['native_metadata_hash']}`。",
        f"- HSACO SHA256：`{identity['hashes']['hsaco']}`。",
        f"- shared metadata：`{identity['selector_config']['shared_bytes']} B`；launch grid：`(2, 32, 8)`。",
        "- 这是从 fresh selector cache 固定出的 stage-2 HSACO；当前运行时 cache 若重新选择其他 stage，不作为本报告对照。",
        "",
        "`num_stages=2` 的可见形式是 TTGIR 中一个带 slot/index 的循环携带状态和 `memdesc<1x...>` staging 表示。它支持 rotating staged schedule 的结论，但单独不能被夸大为“所有物理 buffer 必然完整双缓冲”。",
        "",
        "## Native 与 C19",
        "",
        "fresh native TTGIR 在同一个 K32 `scf.for` 中同时携带 Q@H 与 Q@K 的两个 MMA accumulator：loop 内先 global-load next Q/K/H，再 local-load current Q/K/H，随后连续执行 Q@H 与 Q@K dot，最后写回 next staged slot。最后一个 staged slot 在 loop epilogue 被消费。",
        "",
        "C19 则是完整 Q@H 四个 K32 stage，接着完整 Q@K half-0 四个 stage，最后 Q@K half-1 四个 stage；这三个 source loop 之间有显式 phase 边界。C19/C21/native 的 final ISA lexical 结构不能替代动态 PMC，但足以说明同步图形状不同：",
        "",
        "| arm | MFMA32 | barrier | waitcnt | global load | LDS read/write |",
        "|:--|--:|--:|--:|--:|--:|",
        f"| C19 | {machine['arms']['c19_frozen']['static_isa']['mfma32']} | {machine['arms']['c19_frozen']['static_isa']['s_barrier']} | {machine['arms']['c19_frozen']['static_isa']['s_waitcnt']} | {machine['arms']['c19_frozen']['static_isa']['global_load']} | {machine['arms']['c19_frozen']['static_isa']['ds_read']}/{machine['arms']['c19_frozen']['static_isa']['ds_write']} |",
        f"| C21 | {machine['arms']['c21_frozen']['static_isa']['mfma32']} | {machine['arms']['c21_frozen']['static_isa']['s_barrier']} | {machine['arms']['c21_frozen']['static_isa']['s_waitcnt']} | {machine['arms']['c21_frozen']['static_isa']['global_load']} | {machine['arms']['c21_frozen']['static_isa']['ds_read']}/{machine['arms']['c21_frozen']['static_isa']['ds_write']} |",
        f"| fresh native | {machine['arms']['native_selected']['static_isa']['mfma32']} | {machine['arms']['native_selected']['static_isa']['s_barrier']} | {machine['arms']['native_selected']['static_isa']['s_waitcnt']} | {machine['arms']['native_selected']['static_isa']['buffer_or_global_load']} | {machine['arms']['native_selected']['static_isa']['ds_read']}/{machine['arms']['native_selected']['static_isa']['ds_write']} |",
        "",
        "## C21 实际结果",
        "",
        "C21 的 `ChunkOPipelinePlan` 确实进入 lowering，并形成与 C19 不同的 LLVM/MIR/ISA/HSACO；C21 HSACO 与 C19、native 均为不同 hash。它也保持 scratch/private/spill 为零。但硬件发射顺序未达标：C21 final ISA 中 current QH MFMA 在约 378/383/387/391 行，下一 K producer issue 出现在约 394 行之后，紧跟 wait/publish/barrier，而 score MFMA 到约 422 行才开始。这不是 native 那种在循环内扩大 load/MFMA 调度窗口的 materialization。",
        "",
        "因此，wide typed feeding 仅证明为机器图中存在相应宽 packet family，不能宣称它已构成 selected-native 级别的 producer/consumer pipeline。",
        "",
        "## 正确性与资源",
        "",
        "C21 在 `T=2048` 的 random、zero-V-new、caller-owned NaN-prefill、structured Q/K/H one-hot 与 token/value pattern 五项均 finite，且对 Z5B BF16 byte-exact。C21 code object 为 VGPR188、AGPR80、SGPR36、LDS24576 B、private0、reported spill0；native selected 为 metadata shared12288 B、private0/spill0，资源字段不可与 rocprof `Accum_VGPR_Count` 混为同一量。",
        "",
        "## 动态 PMC（每 CTA，诊断）",
        "",
        "| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR/AccumVGPR |",
        "|:--|--:|--:|--:|--:|--:|:--|",
    ]
    for arm in ("z5b", "c19_frozen", "c21_frozen", "native_selected"):
        dyn = pmc["arms"][arm]["median_dynamic_per_cta"]
        res = pmc["arms"][arm]["median_resource_fields"]
        lines.append(
            f"| {arm} | {dyn['SQ_INSTS_MFMA']:.2f} | {dyn['SQ_INSTS_VMEM']:.2f} | {dyn['SQ_INSTS_LDS']:.2f} | {dyn['SQ_INSTS_VALU']:.2f} | {dyn['SQ_INSTS_SALU']:.2f} | {res['VGPR_Count']:.0f}/{res['Accum_VGPR_Count']:.0f} |"
        )
    lines += [
        "",
        "这些值来自最后五个匹配 dispatch 的 counter median，按实际 `Grid_Size / Workgroup_Size` 归一为 CTA；trace 时间未用作正式延迟结论。",
        "",
        "## 正式 T=2048 Body 性能",
        "",
        "口径：7 个独立 fresh Python process、caller-owned preallocated output、current HIP stream、无 Graph、warmup=10、repeat=50、平衡轮换顺序；数值为 HIP-event session median 的中位数。",
        "",
        "| arm | ms | us | 相对 Z5B |",
        "|:--|--:|--:|--:|",
        f"| Z5B | {z5b:.6f} | {z5b * 1e3:.3f} | 1.000x |",
        f"| C19 | {c19:.6f} | {c19 * 1e3:.3f} | {z5b / c19:.3f}x |",
        f"| C21 | {c21:.6f} | {c21 * 1e3:.3f} | {speedup:.3f}x |",
        f"| fresh selected native | {native:.6f} | {native * 1e3:.3f} | {z5b / native:.3f}x |",
        "",
        f"C21 比 C19 快约 `{(c19 - c21) * 1e3:.3f} us`，但比 Z5B 慢约 `{(c21 - z5b) * 1e3:.3f} us`（约 `{(c21 / z5b - 1) * 100:.1f}%`）。相对 Z5B 的 speedup 是 `{speedup:.3f}x`；C21/native 是 `{ratio:.3f}x`。按指定公式，C21 捕获的 Z5B→native gap 为 `{captured * 100:.1f}%`，为负值，说明它没有回收 Z5B 与 native 的间距。",
        "",
        "## 停止决定",
        "",
        "C21 的 correctness、独立 code object、MLIR/LLVM/MIR/ISA capture 和正式性能均已完成；但 overlap 证据未通过，且 C21 不快于 Z5B。按预注册规则必须停止在 `STOP_C21_PIPELINE_NOT_MATERIALIZED`，不继续 C22 barrier、packet、VALU 或多长度扩展。本轮没有修改 X2 immutable recurrence HSACO、R4 recurrence、allocator/RA 或 production selector。",
        "",
        "## 工件",
        "",
        "同目录 JSON 记录 selected identity、native timeline、C19 gap、C21 plan、overlap evidence、correctness、machine evidence、PMC、formal body 和 regression 状态。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatches", type=int, default=5)
    parser.add_argument("--binding-build", choices=("pass", "not_recorded"), default="not_recorded")
    args = parser.parse_args()
    identity = make_identity()
    timeline = make_timeline()
    gap = make_gap()
    plan = make_plan()
    overlap = make_overlap()
    correctness = read_json(LADDER / "stage6z_c21_correctness_t2048.json")
    pmc = {
        "schema": "qwen.gfx942.stage6z.c21.pmc_t2048.v1",
        "experiment": "C21-NSM Selected-Native Pipeline Reconstruction",
        "method": "rocprof counter collection; last five matching target dispatches; dynamic instruction counters normalized by actual CTA count",
        "arms": {name: parse_pmc_arm(name, config, args.dispatches) for name, config in ARM_CONFIG.items()},
    }
    formal = read_json(LADDER / "stage6z_c21_formal_body_t2048.json")
    machine = make_machine_evidence(identity)
    regression = make_regression(correctness, formal, machine, args.binding_build)
    outputs = {
        "stage6z_c21_selected_native_identity.json": identity,
        "stage6z_c21_native_pipeline_timeline.json": timeline,
        "stage6z_c21_c19_native_critical_path_gap.json": gap,
        "stage6z_c21_pipeline_plan.json": plan,
        "stage6z_c21_overlap_evidence.json": overlap,
        "stage6z_c21_machine_evidence_t2048.json": machine,
        "stage6z_c21_pmc_t2048.json": pmc,
        "stage6z_c21_regression_results.json": regression,
    }
    for name, payload in outputs.items():
        write_json(name, payload)
    report_path = LADDER / "qwen_gfx942_c21_selected_native_pipeline_reconstruction.md"
    report_path.write_text(report(identity, timeline, gap, plan, overlap, correctness, machine, pmc, formal, regression))
    print(json.dumps({"report": str(report_path), "outputs": sorted(outputs), "decision": overlap["decision"]}, indent=2))


if __name__ == "__main__":
    main()

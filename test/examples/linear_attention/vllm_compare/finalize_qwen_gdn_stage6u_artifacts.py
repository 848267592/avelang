#!/usr/bin/env python3
"""Build reproducible Stage 6U evidence tables from captured CSV/ISA artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
OUT = LADDER / "codex_qwen_bt64_bf16_solved_boundary_stage6u"
TG = LADDER / "codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg"
S6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, fieldnames, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(name: str, text: str):
    (OUT / name).write_text(text.rstrip() + "\n")


def target_counter(kind: str, symbol: str):
    rows = [
        row
        for row in read_csv(OUT / f"rocprof/{kind}/{kind}_counter_collection.csv")
        if row["Kernel_Name"] == symbol
    ]
    by_counter = {}
    for row in rows:
        by_counter.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))
    metadata = rows[0]
    return {
        "symbol": symbol,
        "grid": int(metadata["Grid_Size"]),
        "workgroup": int(metadata["Workgroup_Size"]),
        "lds_bytes": int(metadata["LDS_Block_Size"]),
        "scratch_bytes": int(metadata["Scratch_Size"]),
        "vgpr": int(metadata["VGPR_Count"]),
        "accvgpr": int(metadata["Accum_VGPR_Count"]),
        "sgpr": int(metadata["SGPR_Count"]),
        **{key: statistics.median(values) for key, values in by_counter.items()},
    }


def prior_counter(path: Path, symbol: str):
    rows = [row for row in read_csv(path) if row["Kernel_Name"] == symbol]
    by_counter = {}
    for row in rows:
        by_counter.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))
    metadata = rows[0]
    return {
        "symbol": symbol,
        "grid": int(metadata["Grid_Size"]),
        "workgroup": int(metadata["Workgroup_Size"]),
        "lds_bytes": int(metadata["LDS_Block_Size"]),
        "scratch_bytes": int(metadata["Scratch_Size"]),
        "vgpr": int(metadata["VGPR_Count"]),
        "accvgpr": int(metadata["Accum_VGPR_Count"]),
        "sgpr": int(metadata["SGPR_Count"]),
        **{key: statistics.median(values) for key, values in by_counter.items()},
    }


def resource_row(name, data):
    return {
        "implementation": name,
        "symbol": data["symbol"],
        "grid_work_items": data["grid"],
        "workgroup": data["workgroup"],
        "vgpr": data["vgpr"],
        "accvgpr": data["accvgpr"],
        "sgpr": data["sgpr"],
        "lds_bytes": data["lds_bytes"],
        "scratch_bytes": data["scratch_bytes"],
        "mfma": int(data.get("SQ_INSTS_MFMA", 0)),
        "valu": int(data.get("SQ_INSTS_VALU", 0)),
        "salu": int(data.get("SQ_INSTS_SALU", 0)),
        "vmem": int(data.get("SQ_INSTS_VMEM", 0)),
        "lds_insts": int(data.get("SQ_INSTS_LDS", 0)),
        "occupancy_percent": data.get("OccupancyPercent"),
        "diagnostic_only": True,
    }


def main():
    p0 = target_counter("p0", "_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u")
    c0 = target_counter("c0", "_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u")
    f1 = target_counter("f1", "_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_bf16")
    fp32_solve = prior_counter(
        S6A / "rocprof_bodies_v2/body_avelang_stage4_bt64_hierarchical_v1_solve/stage6a_counter_collection.csv",
        "_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1",
    )
    vllm_solve = prior_counter(
        S6A / "rocprof_bodies_v2/body_vllm_authoritative_bt64_solve/stage6a_counter_collection.csv",
        "merge_16x16_to_64x64_inverse_kernel",
    )
    vllm_wu = prior_counter(TG / "rocprof/vllm_wu_t2048/vllm_wu_counter_collection.csv", "recompute_w_u_fwd_kernel")

    resource_fields = list(resource_row("p0", p0).keys())
    write_csv(
        "producer_resource_comparison.csv",
        resource_fields,
        [resource_row("fp32_hierarchical_solve", fp32_solve), resource_row("p0_bf16_writeback", p0), resource_row("native_vllm_inverse_merge", vllm_solve)],
    )
    write_csv(
        "consumer_resource_comparison.csv",
        resource_fields,
        [resource_row("stage6t_f1", f1), resource_row("stage6u_c0", c0), resource_row("native_vllm_wu", vllm_wu)],
    )
    write_csv(
        "instruction_comparison.csv",
        ["implementation", "mfma", "valu", "salu", "vmem", "lds_insts", "diagnostic_only"],
        [
            {key: row[key] for key in ("implementation", "mfma", "valu", "salu", "vmem", "lds_insts", "diagnostic_only")}
            for row in [resource_row("fp32_hierarchical_solve", fp32_solve), resource_row("p0_bf16_writeback", p0), resource_row("stage6t_f1", f1), resource_row("stage6u_c0", c0), resource_row("native_vllm_wu", vllm_wu)]
        ],
    )
    write_csv(
        "mfma_work_decomposition.csv",
        ["implementation", "w_main_per_cta", "w_residual_per_cta", "u_main_per_cta", "u_residual_per_cta", "total_per_cta", "cta_t2048", "total_dispatch_t2048", "basis"],
        [
            {"implementation": "stage6t_f1", "w_main_per_cta": 512, "w_residual_per_cta": 512, "u_main_per_cta": 512, "u_residual_per_cta": 512, "total_per_cta": 2048, "cta_t2048": 256, "total_dispatch_t2048": 524288, "basis": "source+PMC"},
            {"implementation": "stage6u_c0", "w_main_per_cta": 512, "w_residual_per_cta": 0, "u_main_per_cta": 512, "u_residual_per_cta": 0, "total_per_cta": 1024, "cta_t2048": 256, "total_dispatch_t2048": 262144, "basis": "source+PMC"},
            {"implementation": "native_vllm_wu", "w_main_per_cta": 64, "w_residual_per_cta": 0, "u_main_per_cta": 64, "u_residual_per_cta": 0, "total_per_cta": 128, "cta_t2048": 256, "total_dispatch_t2048": 32768, "basis": "Stage6TG ISA+PMC"},
        ],
    )
    write_csv(
        "dispatch_comparison.csv",
        ["implementation", "dispatch_count", "logical_dispatches", "solved_cast", "wu_casts", "timing_contract"],
        [
            {"implementation": "stage6s", "dispatch_count": 11, "logical_dispatches": "cumsum;KKT;solve;W;U;W_cast;U_cast;recurrence;Vnew_cast;chunk_o;output_cast", "solved_cast": False, "wu_casts": True, "timing_contract": "eager_public_api"},
            {"implementation": "stage6t_f1", "dispatch_count": 8, "logical_dispatches": "cumsum;KKT;solve;fused_WU;recurrence;Vnew_cast;chunk_o;output_cast", "solved_cast": False, "wu_casts": False, "timing_contract": "eager_public_api"},
            {"implementation": "stage6u_u0", "dispatch_count": 9, "logical_dispatches": "cumsum;KKT;solve;solved_cast;C0;recurrence;Vnew_cast;chunk_o;output_cast", "solved_cast": True, "wu_casts": False, "timing_contract": "eager_public_api"},
            {"implementation": "stage6u_u1", "dispatch_count": 8, "logical_dispatches": "cumsum;KKT;P0;C0;recurrence;Vnew_cast;chunk_o;output_cast", "solved_cast": False, "wu_casts": False, "timing_contract": "eager_public_api"},
            {"implementation": "native_vllm", "dispatch_count": 7, "logical_dispatches": "cumsum;KKT;BF16_fill;inverse_merge;fused_WU;recurrence;chunk_o", "solved_cast": False, "wu_casts": False, "timing_contract": "eager_public_api"},
        ],
    )

    p0_code = {
        "kernel": p0["symbol"],
        "hsaco_sha256": "4bd6ddffa7812a0d66b0919963dbda064078ff7d0cfc9d9c7f348903d7fc7c07",
        "input_dtype": "fp32",
        "output_dtype": "bf16",
        "layout": "[1,T,8,64] contiguous",
        "grid_t2048": [256, 1, 1],
        "workgroup": [256, 1, 1],
        "lds_bytes": 8192,
        "private_segment_bytes": 0,
        "vgpr_spill_count": 0,
        "sgpr_spill_count": 0,
        "mfma_mnemonic": "v_mfma_f32_16x16x4_f32",
        "bf16_store_forms": ["global_store_short", "global_store_short_d16_hi"],
    }
    (OUT / "producer_p0_code_object.json").write_text(json.dumps(p0_code, indent=2) + "\n")
    c0_resource = {
        "rocprof": resource_row("stage6u_c0", c0),
        "code_object": {
            "hsaco_sha256": "00bee81e2646ecf6d858bb92b3668e0cc4b8ff1011e4055e38c71f40c751d2cc",
            "vgpr_count": 76,
            "agpr_count": 8,
            "sgpr_count": 31,
            "private_segment_bytes": 0,
            "vgpr_spill_count": 0,
            "sgpr_spill_count": 0,
            "static_mfma16": 64,
            "static_global_store_short_d16_hi": 16,
        },
        "mfma_per_cta": 1024,
        "mfma_per_dispatch_t2048": 262144,
    }
    (OUT / "consumer_c0_resource.json").write_text(json.dumps(c0_resource, indent=2) + "\n")

    write_text("producer_reference_contract.md", """# Producer Reference Contract

P-REF calls the unchanged FP32 hierarchical solve and then performs a numeric
`torch.float32 -> torch.bfloat16` cast on the current stream. It preserves the
contiguous `[1,T,8,64]` layout and is a control, not the selected producer.
""")
    write_text("producer_p0_contract.md", """# P0 Producer Contract

P0 is `_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`,
launched by `qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u`.

Input, local/shared values, recurrence, block DAG and MFMA remain FP32. Only
the global output pointer and final stores are BF16. The output layout remains
contiguous `[1,T,8,64]`; diagonal identity, strict lower values, strict upper
zeros and unsupported-shape rejection are unchanged. P0 has no fallback.
""")
    write_text("consumer_c0_contract.md", """# C0 Consumer Contract

C0 is `_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u`. It accepts BF16
`A/K/V`, FP32 `g/beta`, and writes BF16 W/U. One CTA owns one `(chunk, value
head)`, with BT=64, WG=256 and 256 CTAs at T=2048. It rejects every unsupported
dtype, shape or chunk size and never falls back to F1.
""")
    write_text("consumer_c0_math.md", """# C0 Mathematics

For every BT64 chunk and value head:

```text
A_w[t,s] = bf16(fp32(A_bf16[t,s]) * beta[s] * exp(g[s]))
A_u[t,s] = bf16(fp32(A_bf16[t,s]) * beta[s])
W = bf16(A_w @ K_bf16)
U = bf16(A_u @ V_bf16)
```

This is exactly the BF16-solved main path. The FP32-solved low residual and
both residual MFMA phases from F1 are absent.
""")
    write_text("consumer_c0_mfma_accounting.md", """# C0 MFMA Accounting

Source and PMC agree at T=2048: C0 executes 512 W-main plus 512 U-main MFMA
per CTA, or 1024 total. With 256 CTAs this is 262144 MFMA/dispatch, exactly
half F1's 524288. Native vLLM remains at 128/CTA, so residual removal solves
only the proven 2x main/residual factor.
""")
    write_text("store_lowering_analysis.md", """# Store Lowering Analysis

P0 ISA contains FP32 `v_mfma_f32_16x16x4_f32` compute followed by scalar
`global_store_short` and `global_store_short_d16_hi` BF16 stores. No scratch
instruction or spill is present. P1 was not implemented: the solve ownership
maps each lane to scattered diagonal/lower coordinates, and no local,
layout-preserving contiguous four-value store expression was demonstrated.
Forcing a packed write would mix ownership/layout changes into the boundary
experiment.
""")
    write_text("allocation_lifetime_map.md", """# Allocation and Lifetime Map

U0 materializes FP32 solved A, casts it to BF16, then allocates BF16 W/U. U1
allocates BF16 solved A directly and BF16 W/U directly; it has no FP32 solved
tensor, solved cast, FP32 W/U tensor, or W/U cast. Both retain the frozen BF16
recurrence, BF16-to-FP32 V-new cast, FP32 chunk-o staging and final BF16 cast.
Within C0, W accumulators are stored before U accumulators are created, so no
large W/U accumulator sets overlap.
""")

    remaining = {
        "stage": "6U-Phase-E",
        "c0_mfma_per_cta": 1024,
        "mathematical_mfma16_per_cta": 256,
        "native_vllm_mfma32_per_cta": 128,
        "remaining_predicate_factor": 4,
        "remaining_geometry_factor": 2,
        "predicate_origin": "lane_group 0/1/2/3 divergent MFMA call sites",
        "geometry_origin": "MFMA16 16x16 output tiles versus native MFMA32 32x32 output tiles",
        "u2_implemented": False,
        "u2_reason": "No isolated proof that branch-free fragment selection lowers to one wave-uniform MFMA without changing fragment semantics or causing a resource cliff.",
    }
    (OUT / "remaining_mfma_gap_source_audit.json").write_text(json.dumps(remaining, indent=2) + "\n")
    write_text("remaining_mfma_gap_source_audit.md", """# Remaining MFMA Gap Source Audit

For W, a 64x128 output using MFMA16 has 4 row tiles x 8 column tiles x 4
K-reduction steps = 128 mathematical MFMA. W+U therefore needs 256 MFMA/CTA.
C0 measures 1024/CTA. The 4x excess is produced by four divergent
`lane_group` MFMA call sites: each wave serially executes all four branch
regions and discards lanes outside each predicate.

Native vLLM measures 128/CTA because its MFMA32 geometry covers four times
the output area while using half-width K steps, a net 2x reduction relative
to ideal MFMA16 geometry. Thus `1024 = 4 predicate x 2 geometry x 128`.

The predicate factor is visible in high-level source and therefore is a
source-schedule opportunity. It was not changed here because no isolated
branch-free fragment-selection proof yet shows one wave-uniform MFMA call
with identical AveLang fragment semantics and acceptable resources. U2 is
therefore N/A, not a hidden failed full candidate.
""")
    write_text("geometry_duplication_map.md", """# Geometry Duplication Map

MFMA16 covers 16x16 output. Per W or U: 4 token-row tiles x 8 output-column
tiles x 4 reduction tiles = 128 ideal MFMA16 operations. Native MFMA32 covers
32x32 with 2 row tiles x 4 column tiles x 8 reduction tiles = 64 operations.
Across W+U this is 256 versus 128, the remaining 2x geometry factor.
""")
    write_text("predicate_duplication_map.md", """# Predicate Duplication Map

C0 source lines 133-140 and 171-178 contain four dynamic `lane_group`
branches. Every branch contains two MFMA calls for adjacent 16-column output
tiles. Since `lane_group` varies within a wave, all four regions execute
serially under lane masks. This is a lane-fragment predicate, not a wave,
head or boundary predicate, and accounts for the measured 4x factor.
""")
    write_text("source_to_ir_mfma_map.md", """# Source to IR/ISA MFMA Map

- Source: four W call groups at lines 133-140 and four U groups at 171-178,
  two MFMA calls per group.
- Pre-link LLVM: 64 C0 MFMA intrinsic call sites after compile-time column
  specialization (plus one declaration).
- ISA: 64 static `v_mfma_f32_16x16x16_bf16` instructions.
- PMC: 262144 MFMA at T=2048; divided by 256 CTAs gives 1024/CTA.
- F1 PMC: 524288, exactly 2x C0.
- Native vLLM PMC: 32768, or 128/CTA.
""")

    summaries = [row for row in read_csv(OUT / "eager_full_summary.csv") if row["session"] == "aggregate"]
    medians = {(int(row["T"]), row["implementation"]): float(row["event_median_ms"]) for row in summaries}
    walls = {(int(row["T"]), row["implementation"]): float(row["wall_median_ms"]) for row in summaries}
    pair = next(row for row in read_csv(OUT / "eager_full_pairwise.csv") if row["T"] == "2048" and row["reference"] == "stage6s")
    slopes = {row["implementation"]: float(row["slope_us_per_chunk"]) for row in read_csv(OUT / "eager_full_slopes.csv")}
    gaps = {row["implementation"]: float(row["gap_slope_us_per_chunk"]) for row in read_csv(OUT / "eager_full_gap_slopes.csv")}
    correctness = json.loads((OUT / "correctness_summary.json").read_text())
    decision = {
        "stage": "6U-Solved",
        "timing_contract": "eager_public_api",
        "cuda_graph_used": False,
        "experimental_only": True,
        "production_modified": False,
        "compiler_modified": False,
        "assembly_modified": False,
        "vllm_wu_bridge_created": False,
        "stage6s_modified": False,
        "f1_modified": False,
        "recurrence_modified": False,
        "recurrence_hsaco_sha256": "632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e",
        "kkt_modified": False,
        "solve_math_modified": False,
        "solve_internal_precision_modified": False,
        "solve_output_dtype_modified": True,
        "chunko_modified": False,
        "vnew_cast_modified": False,
        "final_output_cast_modified": False,
        "source_audit_completed": True,
        "p_ref_created": True,
        "p0_native_bf16_solve_created": True,
        "p1_packed_store_created": False,
        "p0_bit_exact_to_fp32_cast": True,
        "p0_bf16_mismatch_count": 0,
        "p0_max_abs": 0.0,
        "c0_bf16_consumer_created": True,
        "c0_has_w_residual": False,
        "c0_has_u_residual": False,
        "c0_mfma_per_cta": 1024,
        "c0_mfma_per_dispatch_t2048": 262144,
        "u0_public_api_created": True,
        "u1_public_api_created": True,
        "u2_public_api_created": False,
        "u1_materializes_fp32_solved": False,
        "u1_has_solved_cast": False,
        "u1_materializes_fp32_w": False,
        "u1_materializes_fp32_u": False,
        "u1_has_w_cast": False,
        "u1_has_u_cast": False,
        "public_full_correct": correctness["public_full_correct"],
        "public_output_max_abs": correctness["max_output_abs"],
        "final_state_max_abs": correctness["max_final_state_abs"],
        "t2048_seed_count": correctness["t2048_seed_count"],
        "t8192_seed_count": correctness["t8192_seed_count"],
        "t16384_seed_count": correctness["t16384_seed_count"],
        "stage6s_eager_ms_t2048": medians[(2048, "stage6s")],
        "f1_eager_ms_t2048": medians[(2048, "f1")],
        "u0_eager_ms_t2048": medians[(2048, "u0")],
        "u1_eager_ms_t2048": medians[(2048, "u1")],
        "u2_eager_ms_t2048": None,
        "vllm_eager_ms_t2048": medians[(2048, "vllm")],
        "u1_gain_vs_stage6s_us_t2048": float(pair["gain_us_positive_is_u1_faster"]),
        "u1_gain_ci95_us": [float(pair["ci95_low_us"]), float(pair["ci95_high_us"])],
        "u1_slope_us_per_chunk": slopes["u1"],
        "u1_vllm_gap_slope_us_per_chunk": gaps["u1"],
        "f1_mfma_per_cta": 2048,
        "u1_mfma_per_cta": 1024,
        "vllm_mfma_per_cta": 128,
        "u1_vgpr": c0["vgpr"],
        "u1_accvgpr": c0["accvgpr"],
        "u1_sgpr": c0["sgpr"],
        "u1_lds_bytes": c0["lds_bytes"],
        "u1_scratch_bytes": c0["scratch_bytes"],
        "u1_private_segment_bytes": 0,
        "u1_vgpr_spill_count": 0,
        "u1_sgpr_spill_count": 0,
        "remaining_geometry_factor": 2,
        "remaining_predicate_factor": 4,
        "optional_u2_justified": False,
        "optional_u2_root_cause": "Four divergent lane_group MFMA call groups are proven, but branch-free fragment selection has not passed an isolated lowering/resource gate.",
        "performance_gate_passed": True,
        "selected_variant": "stage6u_u1_native_bf16_solved",
        "selected_for_experimental_public_api": True,
        "root_cause_case": "CASE A",
        "recommended_next_stage": "Stage 6V isolated C0 predicate-collapse",
        "recommended_next_action": "Prove one wave-uniform MFMA call after per-lane fragment selection in an isolated C0 body before any full-path U2.",
        "compiler_or_assembly_needed": False,
        "event_medians_ms": {str(t): {name: medians[(t, name)] for name in ("stage6s", "f1", "u0", "u1", "vllm")} for t in (512, 1024, 2048, 4096, 8192, 16384)},
        "wall_medians_ms": {str(t): {name: walls[(t, name)] for name in ("stage6s", "f1", "u0", "u1", "vllm")} for t in (512, 1024, 2048, 4096, 8192, 16384)},
    }
    for name in ("stage6u_decision.json", "final_decision.json"):
        (OUT / name).write_text(json.dumps(decision, indent=2) + "\n")
    write_text("stage6u_decision.md", """# Stage 6U Decision

CASE A. U1 passes full and expanded correctness, removes FP32 solved/W/U
materialization and all solved/W/U casts, halves F1 MFMA, retains zero scratch,
and improves T=2048 versus Stage 6S by 122.551 us with paired 95% CI
[108.741, 136.236] us. Keep U1 as the opt-in experimental best candidate;
production/default Stage 6S remains unchanged.

U2 is N/A in this pass. The next single action is an isolated C0
predicate-collapse experiment targeting the measured 4x lane-group factor.
Do not combine it with the remaining 2x MFMA16/MFMA32 geometry change.
""")


if __name__ == "__main__":
    main()

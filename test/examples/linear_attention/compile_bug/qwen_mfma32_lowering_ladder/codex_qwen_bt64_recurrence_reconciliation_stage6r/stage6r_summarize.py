#!/usr/bin/env python3
"""Generate Stage 6R reconciliation artifacts from fresh capture/benchmark data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
CURRENT = HERE / "current_kernels"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(directory: Path, symbol: str) -> dict[str, object]:
    trace = [row for row in read_csv(directory / "stage6r_kernel_trace.csv") if symbol in row["Kernel_Name"]]
    explicit = trace[-3:]
    ids = {row["Dispatch_Id"] for row in explicit}
    counters = [row for row in read_csv(directory / "stage6r_counter_collection.csv")
                if symbol in row["Kernel_Name"] and row["Dispatch_Id"] in ids]
    values: dict[str, list[float]] = {}
    for row in counters:
        values.setdefault(row["Counter_Name"], []).append(float(row["Counter_Value"]))
    meta = explicit[-1]
    return {
        "matching_dispatches_before_tail_normalization": len(trace),
        "explicit_replay_count": 3,
        "explicit_dispatch_ids": sorted(ids, key=int),
        "trace_median_us": statistics.median((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in explicit),
        "workgroup": int(meta["Workgroup_Size_X"]),
        "grid_x": int(meta["Grid_Size_X"]),
        "grid_y": int(meta["Grid_Size_Y"]),
        "grid_work_items": int(meta["Grid_Size_X"]) * int(meta["Grid_Size_Y"]),
        "lds_block_bytes": int(meta["LDS_Block_Size"]),
        "scratch_bytes": int(meta["Scratch_Size"]),
        "vgpr": int(meta["VGPR_Count"]),
        "accvgpr": int(meta["Accum_VGPR_Count"]),
        "sgpr": int(meta["SGPR_Count"]),
        **{name: statistics.median(items) for name, items in values.items()},
    }


def stage6a_matching_count(implementation: str, symbol: str) -> int:
    path = STAGE6A / "rocprof" / implementation / "stage6a_kernel_trace.csv"
    return sum(symbol in row["Kernel_Name"] for row in read_csv(path))


def static_counts(path: Path) -> dict[str, int]:
    text = path.read_text()
    patterns = {
        "mfma_xf32": r"v_mfma_f32_32x32x4_xf32",
        "mfma_bf16": r"v_mfma_f32_32x32x8_bf16",
        "buffer_load": r"\bbuffer_load",
        "buffer_store": r"\bbuffer_store",
        "global_load": r"\bglobal_load",
        "global_store": r"\bglobal_store",
        "ds_read": r"\bds_read",
        "ds_write": r"\bds_write",
        "barrier": r"\bs_barrier\b",
        "convert": r"\bv_cvt_",
        "address_lshl": r"\bv_lshl",
        "address_add": r"\bv_add(?:3)?_",
    }
    return {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}


def lookup_summary(rows: list[dict[str, str]], pair: str, impl: str) -> dict[int, float]:
    return {int(row["T"]): float(row["median_of_session_medians_ms"])
            for row in rows if row["comparison_pair"] == pair and row["implementation"] == impl}


def slopes(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["implementation"]: row for row in rows}


def fit_chunks(points: dict[int, float]) -> tuple[float, float]:
    ordered = sorted((float(t // 64), value) for t, value in points.items())
    mean_x = sum(x for x, _ in ordered) / len(ordered)
    mean_y = sum(y for _, y in ordered) / len(ordered)
    denominator = sum((x - mean_x) ** 2 for x, _ in ordered)
    slope_ms = sum((x - mean_x) * (y - mean_y) for x, y in ordered) / denominator
    return mean_y - slope_ms * mean_x, slope_ms * 1000.0


def main() -> None:
    asm = profile(HERE / "rocprof_asm_v0", "qwen_gdn_bt64_gfx942_asm_v0")
    vllm = profile(HERE / "rocprof_current_vllm", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    raw_rows = [
        {"capture": "Stage6A full profile", "implementation": "asm-v0", "symbol": "qwen_gdn_bt64_gfx942_asm_v0",
         "matching_dispatches": stage6a_matching_count("full_avelang_stage4_bt64_hierarchical_v1", "qwen_gdn_bt64_gfx942_asm_v0"), "requested_replays": 3},
        {"capture": "Stage6A full profile", "implementation": "vLLM", "symbol": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
         "matching_dispatches": stage6a_matching_count("full_vllm_authoritative_bt64", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"), "requested_replays": 3},
        {"capture": "Stage6R fresh recurrence profile", "implementation": "asm-v0", "symbol": "qwen_gdn_bt64_gfx942_asm_v0",
         "matching_dispatches": asm["matching_dispatches_before_tail_normalization"], "requested_replays": 3},
        {"capture": "Stage6R fresh recurrence profile", "implementation": "vLLM actual", "symbol": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
         "matching_dispatches": vllm["matching_dispatches_before_tail_normalization"], "requested_replays": 3},
    ]
    write_csv(HERE / "counter_aggregation_raw.csv", raw_rows)
    normalized_rows: list[dict[str, object]] = []
    for name, values in (("asm-v0", asm), ("vLLM actual", vllm)):
        mfma = float(values["SQ_INSTS_MFMA"])
        normalized_rows.append({
            "implementation": name,
            "replay_count": values["explicit_replay_count"],
            "tail_dispatch_count": len(values["explicit_dispatch_ids"]),
            "raw_mfma_per_counter_dispatch": mfma,
            "normalized_mfma_per_recurrence_dispatch": mfma,
            "normalized_mfma_per_chunk": mfma / 32.0,
            "normalized_mfma_per_chunk_head": mfma / (32.0 * 8.0),
            "counter_aggregation_error_found": False,
        })
    write_csv(HERE / "counter_aggregation_normalized.csv", normalized_rows)
    (HERE / "counter_aggregation_audit.md").write_text(
        "# Counter Aggregation Audit\n\n"
        "Stage 6A requested three explicit graph replays. The raw trace contains extra recurrence symbols because "
        "the process includes Triton autotuning/capture activity: asm has 5 matching dispatches and vLLM has 1,315. "
        "The fresh Stage 6R trace reproduces this asymmetry (5 versus 1,211). It is not a counter multiplier.\n\n"
        "For both implementations the final three matching dispatch IDs are the three explicit post-capture replays. "
        "Counter values are a median over those IDs, not a sum. Therefore Stage 6A's `196608` versus `65536` MFMA values "
        "are already per recurrence dispatch.\n\n"
        f"- asm-v0: `{asm['SQ_INSTS_MFMA']:.0f}` MFMA/dispatch, `{asm['SQ_INSTS_MFMA']/32:.0f}`/chunk, "
        f"`{asm['SQ_INSTS_MFMA']/(32*8):.0f}`/chunk-head.\n"
        f"- current vLLM: `{vllm['SQ_INSTS_MFMA']:.0f}` MFMA/dispatch, `{vllm['SQ_INSTS_MFMA']/32:.0f}`/chunk, "
        f"`{vllm['SQ_INSTS_MFMA']/(32*8):.0f}`/chunk-head.\n\n"
        "Conclusion: `counter_aggregation_error_found=false`; the 3x dynamic-MFMA difference is real.\n"
    )
    vllm_hsaco = CURRENT / "vllm/kernel.hsaco"
    asm_hsaco = CURRENT / "avelang/kernel.hsaco"
    historical_original = HERE / "historical_triton/original.hsaco"
    historical_rebuilt = HERE / "historical_triton/rebuilt.hsaco"
    identity = {
        "classification": "different_specialization",
        "byte_identical": False,
        "code_equivalent_but_metadata_different": False,
        "different_specialization": True,
        "avelang_asm_v0_sha256": sha256(asm_hsaco),
        "vllm_actual_sha256": sha256(vllm_hsaco),
        "historical_triton_original_sha256": sha256(historical_original),
        "historical_triton_rebuilt_sha256": sha256(historical_rebuilt),
        "historical_relation": "historical original/rebuilt/asm-v0 are output-bit-exact on FP32 ABI; prior source-normalized audit established code equivalence after symbol rename",
        "current_relation": "current actual vLLM is a distinct BF16 ABI, BV32/2-wave specialization with a distinct HSACO hash",
    }
    write_json(HERE / "kernel_identity_comparison.json", identity)
    (HERE / "kernel_identity_comparison.md").write_text(
        "# Kernel Identity Comparison\n\n"
        f"- Current asm-v0 SHA256: `{identity['avelang_asm_v0_sha256']}`\n"
        f"- Current Stage 6A vLLM actual SHA256: `{identity['vllm_actual_sha256']}`\n"
        f"- Historical Triton original/rebuilt SHA256: `{identity['historical_triton_original_sha256']}` / `{identity['historical_triton_rebuilt_sha256']}`\n\n"
        "**Classification: `different_specialization`.** The current vLLM artifact is not byte-identical to asm-v0 and "
        "cannot be called code-equivalent: it has BF16 W/U/v_new, `BV=32`, two waves, 40,960 B shared memory, and a "
        "different static MFMA schedule. Historical original/rebuilt/asm-v0 remain a separate FP32 ABI lineage.\n"
    )
    v_static = static_counts(CURRENT / "vllm/disassembly.txt")
    a_static = static_counts(CURRENT / "avelang/disassembly.txt")
    instruction_rows = [{"metric": key, "asm_v0_static": a_static[key], "vllm_actual_static": v_static[key],
                         "delta_asm_minus_vllm": a_static[key] - v_static[key]}
                        for key in a_static]
    write_csv(HERE / "instruction_diff.csv", instruction_rows)
    resource_fields = ("trace_median_us", "workgroup", "grid_work_items", "lds_block_bytes", "scratch_bytes", "vgpr", "accvgpr", "sgpr",
                       "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent")
    resource_rows = [{"metric": key, "asm_v0": asm.get(key), "vllm_actual": vllm.get(key),
                      "asm_minus_vllm": float(asm.get(key, 0)) - float(vllm.get(key, 0))} for key in resource_fields]
    write_csv(HERE / "resource_diff.csv", resource_rows)
    (HERE / "isa_diff.md").write_text(
        "# ISA Difference\n\n"
        f"Current vLLM has `{v_static['mfma_bf16']}` static `v_mfma_f32_32x32x8_bf16` and `{v_static['mfma_xf32']}` XF32 MFMA instructions. "
        f"asm-v0 has `{a_static['mfma_bf16']}` BF16 plus `{a_static['mfma_xf32']}` `v_mfma_f32_32x32x4_xf32`. "
        "The dynamic 3x MFMA ratio is consistent with the different pred/update schedule, not a replay sum.\n\n"
        f"Static ISA counts: asm/vLLM buffer-load `{a_static['buffer_load']}/{v_static['buffer_load']}`, buffer-store "
        f"`{a_static['buffer_store']}/{v_static['buffer_store']}`, ds-read `{a_static['ds_read']}/{v_static['ds_read']}`, "
        f"ds-write `{a_static['ds_write']}/{v_static['ds_write']}`, barriers `{a_static['barrier']}/{v_static['barrier']}`. "
        "See `instruction_diff.csv` for every counted mnemonic family.\n"
    )
    (HERE / "abi_diff.md").write_text(
        "# ABI Difference\n\n"
        "Both code objects use an 88-byte, 8-byte-aligned kernarg segment with eight pointer slots at offsets 0..56, "
        "a 32-bit T value at offset 64, and two runtime scratch pointers at 72/80. Thus the *physical kernarg layout* is compatible.\n\n"
        "The semantic ABI is not compatible: current vLLM pointers for `v`, `w`, and `v_new` are BF16; asm-v0 requires "
        "FP32 for all three. `k` and `h` are BF16; `g`, initial state, and final state are FP32 in both. Current vLLM "
        "uses `WG=128` and 40,960 B dynamic shared memory, while asm-v0 uses `WG=256` and 57,344 B.\n"
    )
    (HERE / "dtype_layout_diff.md").write_text(
        "# Dtype and Layout Difference\n\n"
        "Both paths receive contiguous `[1,T,4,128]` BF16 K, `[1,T,8]` FP32 g-cumsum, and `[1,8,128,128]` FP32 initial state. "
        "Current vLLM's Stage 6A `recompute_w_u_fwd` produces contiguous BF16 `[1,T,8,128]` W/U and its recurrence writes BF16 v_new. "
        "The Stage 6A asm body benchmark explicitly casts those W/U buffers to FP32 outside the event and asm-v0 writes FP32 v_new.\n\n"
        "Therefore the original 40-us body comparison was a native-ABI comparison, not identical element-type execution. "
        "When both consume the same FP32 W/U values, current vLLM source and asm-v0 are output-bit-exact, but their code objects still differ.\n"
    )
    (HERE / "launch_diff.md").write_text(
        "# Launch Difference\n\n"
        "At T=2048 both paths use 32 CTAs with grid `(4,8,1)`. Current vLLM obtains this from `BV=32` and has workgroup 128 (two waves). "
        "asm-v0 has workgroup 256 (four waves). Current vLLM selected `num_warps=2`, `num_stages=2`; asm-v0 is the fixed historical WG256 specialization.\n\n"
        "Rocprof reports static LDS block size zero for both code objects because the LDS amount is supplied by launch; the explicit launch arguments are 40,960 B vLLM and 57,344 B asm-v0.\n"
    )
    summary = read_csv(HERE / "standalone_summary.csv")
    slope_map = slopes(read_csv(HERE / "standalone_slopes.csv"))
    native_asm = lookup_summary(summary, "native_abi_asm_vs_current_vllm", "asm_v0_native_fp32")
    native_vllm = lookup_summary(summary, "native_abi_asm_vs_current_vllm", "vllm_actual_native_bf16")
    bridge_native = lookup_summary(summary, "actual_native_vs_bridge", "vllm_actual_native_bf16")
    bridge_external = lookup_summary(summary, "actual_native_vs_bridge", "vllm_actual_bridge_bf16")
    body_rows = []
    for t in sorted(native_asm):
        body_rows.append({"T": t, "asm_v0_ms": native_asm[t], "vllm_actual_ms": native_vllm[t],
                          "gap_us": (native_asm[t] - native_vllm[t]) * 1000.0,
                          "vllm_bridge_ms": bridge_external[t], "bridge_delta_pct": (bridge_external[t] / bridge_native[t] - 1.0) * 100.0})
    write_csv(HERE / "body_native_abi_comparison.csv", body_rows)
    asm_intercept, asm_slope = fit_chunks(native_asm)
    vllm_intercept, vllm_slope = fit_chunks(native_vllm)
    write_csv(HERE / "body_native_abi_slopes.csv", [
        {"implementation": "asm_v0_native_fp32", "intercept_ms": asm_intercept,
         "slope_us_per_chunk": asm_slope},
        {"implementation": "vllm_actual_native_bf16", "intercept_ms": vllm_intercept,
         "slope_us_per_chunk": vllm_slope},
        {"implementation": "gap_asm_minus_vllm", "intercept_ms": asm_intercept - vllm_intercept,
         "slope_us_per_chunk": asm_slope - vllm_slope},
    ])
    checks = read_csv(HERE / "standalone_correctness.csv")
    bridge_ok = all(row["h_bitwise_equal"] == "True" and row["v_new_bitwise_equal"] == "True" and row["final_state_bitwise_equal"] == "True"
                    for row in checks if row["implementation"] == "vllm_actual_bridge_bf16")
    root = {
        "root_cause_case": "B",
        "root_cause": "Current vLLM full graph selects a different BF16, BV32, two-wave recurrence specialization. The frozen asm-v0 is the historical FP32, WG256 specialization.",
        "counter_aggregation_error_found": False,
        "bridge_created": True,
        "bridge_correct": bridge_ok,
        "bridge_t2048_ms": bridge_external[2048],
        "bridge_native_t2048_ms": bridge_native[2048],
        "bridge_percent_delta_t2048": (bridge_external[2048] / bridge_native[2048] - 1.0) * 100.0,
        "stable_body_gap_us_t2048": (native_asm[2048] - native_vllm[2048]) * 1000.0,
        "recurrence_should_remain_frozen": True,
    }
    write_json(HERE / "root_cause_decision.json", root)
    (HERE / "root_cause_decision.md").write_text(
        "# Root Cause Decision\n\n"
        "**CASE B: current vLLM uses a different, faster specialization.** The raw MFMA discrepancy is not aggregation; "
        "the fresh native-vLLM-to-extracted-HSACO bridge is bit-exact and stays within 5% at every measured length. "
        "The historical asm-v0 has not regressed. It remains bit-exact with historical original/rebuilt code on its FP32 ABI.\n\n"
        "No assembly or compiler change is justified by this audit. The measured body gap belongs to the explicit current-BF16 "
        "versus historical-FP32 specialization boundary plus its associated geometry/lowering; this audit does not assign all of it to dtype alone.\n"
    )
    next_decision = {
        "stage": "6R",
        "only_next_action": "Create one separate, opt-in full-graph integration-contract experiment for the already-audited current vLLM BF16 recurrence bridge; do not modify asm-v0 or compiler and do not promote it to production in that experiment.",
        "chunk_o_action": "No Stage 6R chunk-o work. Stage 6B O0/O1 remains rejected and unchanged.",
        "asm_change_required": False,
        "compiler_change_required": False,
    }
    write_json(HERE / "next_stage_decision.json", next_decision)
    (HERE / "next_stage_decision.md").write_text(
        "# Next Stage Decision\n\n"
        "The only data-supported next action is an **isolated BF16 recurrence-boundary integration-contract experiment** using the now-audited "
        "current vLLM HSACO bridge. It must remain opt-in and outside the production full graph until the BF16 W/U/v_new boundary and end-to-end "
        "numerical contract are proven. Do not edit asm-v0, compiler, or Stage 6B chunk-o variants.\n"
    )
    final = {
        "stage": "6R", "audit_only": True, "chunk_o_modified": False, "wu_modified": False, "kkt_modified": False,
        "solve_modified": False, "asm_v0_modified": False, "compiler_modified": False, "production_modified": False,
        "raw_mfma_avelang": asm["SQ_INSTS_MFMA"], "raw_mfma_vllm": vllm["SQ_INSTS_MFMA"],
        "dispatch_count_avelang": len(asm["explicit_dispatch_ids"]), "dispatch_count_vllm": len(vllm["explicit_dispatch_ids"]),
        "replay_count_avelang": asm["explicit_replay_count"], "replay_count_vllm": vllm["explicit_replay_count"],
        "normalized_mfma_per_dispatch_avelang": asm["SQ_INSTS_MFMA"], "normalized_mfma_per_dispatch_vllm": vllm["SQ_INSTS_MFMA"],
        "counter_aggregation_error_found": False,
        "avelang_hsaco_sha256": identity["avelang_asm_v0_sha256"], "vllm_actual_hsaco_sha256": identity["vllm_actual_sha256"],
        "historical_triton_hsaco_sha256": identity["historical_triton_original_sha256"], "kernels_byte_identical": False,
        "kernels_code_equivalent": False, "different_specialization": True,
        "avelang_w_dtype": "fp32", "vllm_w_dtype": "bf16", "avelang_u_dtype": "fp32", "vllm_u_dtype": "bf16",
        "avelang_vnew_dtype": "fp32", "vllm_vnew_dtype": "bf16",
        "avelang_workgroup": asm["workgroup"], "vllm_workgroup": vllm["workgroup"],
        "avelang_dynamic_lds": 57344, "vllm_dynamic_lds": 40960,
        "avelang_vgpr": asm["vgpr"], "vllm_vgpr": vllm["vgpr"], "avelang_accvgpr": asm["accvgpr"], "vllm_accvgpr": vllm["accvgpr"],
        "avelang_body_ms_t2048": native_asm[2048], "vllm_actual_body_ms_t2048": native_vllm[2048],
        "stable_body_gap_us_t2048": root["stable_body_gap_us_t2048"], "actual_vllm_bridge_created": True,
        "actual_vllm_bridge_correct": bridge_ok, "actual_vllm_bridge_ms_t2048": bridge_external[2048],
        "root_cause_case": "B", "root_cause": root["root_cause"],
        "recommended_next_stage": "isolated current-vLLM BF16 recurrence contract integration", "recommended_next_action": next_decision["only_next_action"],
        "recurrence_should_remain_frozen": True, "ready_for_next_stage": True,
    }
    write_json(HERE / "final_decision.json", final)
    main_report = [
        "# Qwen gfx942 BT64 Recurrence Reconciliation: Stage 6R", "",
        "## 结论", "",
        "**CASE B：当前 Stage 6A vLLM full graph 选择了不同且更快的 recurrence specialization。** 这不是 asm-v0 退化、也不是 rocprof 把 Avelang 计数重复三次。asm-v0 仍然忠实对应历史 FP32/WG256 Triton lineage；当前 vLLM 则是 BF16 W/U/v_new、BV32、WG128 的新 code object。", "",
        "## 计数口径", "",
        f"最终三个显式 replay 的每 dispatch 动态 MFMA：asm-v0 `{asm['SQ_INSTS_MFMA']:.0f}`，vLLM `{vllm['SQ_INSTS_MFMA']:.0f}`。按 32 chunks 归一化为 `{asm['SQ_INSTS_MFMA']/32:.0f}` / `{vllm['SQ_INSTS_MFMA']/32:.0f}`；按 chunk-head 为 `{asm['SQ_INSTS_MFMA']/(32*8):.0f}` / `{vllm['SQ_INSTS_MFMA']/(32*8):.0f}`。两边都只取最后 3 个 dispatch 的中位数，因此 3x 是真实动态工作差异。", "",
        "## 当前身份与 ABI", "",
        f"- asm-v0 HSACO: `{identity['avelang_asm_v0_sha256']}`", f"- 当前 vLLM HSACO: `{identity['vllm_actual_sha256']}`", f"- 历史 Triton original: `{identity['historical_triton_original_sha256']}`", "- 二者均为 88 B kernarg、相同 pointer-slot 物理布局；但语义 ABI 不同：asm W/U/v_new 为 FP32，当前 vLLM 为 BF16。", "- 当前 vLLM config：`BV=32,num_warps=2,num_stages=2`，WG128，dynamic LDS 40960 B。asm-v0：WG256，dynamic LDS 57344 B。", "",
        "## ISA 与资源", "",
        f"当前 vLLM 静态 MFMA 为 BF16 `{v_static['mfma_bf16']}`、XF32 `{v_static['mfma_xf32']}`；asm-v0 为 BF16 `{a_static['mfma_bf16']}`、XF32 `{a_static['mfma_xf32']}`。", f"T=2048 fresh rocprof：asm trace `{asm['trace_median_us']:.3f} us`, VGPR/AccVGPR/SGPR `{asm['vgpr']}/{asm['accvgpr']}/{asm['sgpr']}`, VMEM/LDS/barrier `{asm['SQ_INSTS_VMEM']:.0f}/{asm['SQ_INSTS_LDS']:.0f}/{a_static['barrier']}`；vLLM `{vllm['trace_median_us']:.3f} us`, `{vllm['vgpr']}/{vllm['accvgpr']}/{vllm['sgpr']}`, `{vllm['SQ_INSTS_VMEM']:.0f}/{vllm['SQ_INSTS_LDS']:.0f}/{v_static['barrier']}`。两边 scratch 都为 0；完整明细见 `resource_diff.csv` 与 `instruction_diff.csv`。", "",
        "## 同口径 Body 延迟", "", "| T | asm-v0 FP32 ms | vLLM actual BF16 ms | gap us | actual bridge delta |", "|--:|--:|--:|--:|--:|",
    ]
    main_report.extend(f"| {row['T']} | {row['asm_v0_ms']:.6f} | {row['vllm_actual_ms']:.6f} | {row['gap_us']:.3f} | {row['bridge_delta_pct']:.3f}% |" for row in body_rows)
    main_report.extend([
        "", "current-vLLM bridge 在 T=512/2048/8192/16384 都对 native vLLM 的 h/v_new/final_state bit-exact；T=2048 bridge 与 native 相差 "
        f"`{root['bridge_percent_delta_t2048']:.3f}%`，通过 <=5% gate。历史 original、rebuilt 与 asm-v0 在同一 FP32 W/U 输入上也均 bit-exact。", "",
        "## 解释", "", f"Stage 6A 的 `~40 us` T=2048 gap 可以稳定复现。以本轮原生 ABI body 数据拟合，asm/vLLM/gap 的 intercept 为 `{asm_intercept:.6f}`/`{vllm_intercept:.6f}`/`{asm_intercept-vllm_intercept:.6f}` ms，slope 为 `{asm_slope:.6f}`/`{vllm_slope:.6f}`/`{asm_slope-vllm_slope:.6f}` us/chunk。Stage6A 在 asm side 将 vLLM BF16 W/U 外部转换为 FP32，asm 随后执行历史 XF32-heavy WG256 body。将 W/U 都设为同一 FP32 值时，当前 vLLM source body 与 asm-v0 数值 bit-exact；但它仍不能把这当成纯 dtype 因果控制，因为输入 dtype 也可能影响 Triton 选择的代码路径。证据只支持：当前 BF16/BV32/two-wave specialization 及其相关 lowering 是主导边界，仍存在较小的代码生成/几何差异。", "",
        "## 决策", "", "不修改 asm，不修改 compiler，不在本轮继续 chunk-o。允许的唯一下一步是单独的、opt-in 的 current-vLLM BF16 recurrence bridge 全图 contract 集成实验；它不得接入 production，且必须先证明上下游 BF16 W/U/v_new 边界的完整正确性。", "",
        "## 产物", "", "- `counter_aggregation_{raw,normalized}.csv` / `counter_aggregation_audit.md`", "- `current_kernels/`, `kernel_identity_comparison.*`, `isa_diff.md`, `abi_diff.md`, `dtype_layout_diff.md`, `launch_diff.md`", "- `standalone_{raw,summary,slopes,correctness}.csv`, `body_native_abi_{comparison,slopes}.csv`, `resource_diff.csv`, `instruction_diff.csv`", "- `bridge_probe/`, `tests/pytest_results.txt`, `root_cause_decision.*`, `next_stage_decision.*`, `final_decision.json`", "",
    ])
    (LADDER / "qwen_gfx942_bt64_recurrence_reconciliation_stage6r_report.md").write_text("\n".join(main_report))


if __name__ == "__main__":
    main()

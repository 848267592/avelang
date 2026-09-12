#!/usr/bin/env python3
"""Materialize the C25 closure JSONs from captured, caller-owned body results.

This is a report finalizer only.  It does not compile or run a kernel; the
measurements and compiler artifacts are captured separately in Docker.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


ARM_ORDER = ("z5b", "c21_frozen", "c25", "native_selected")
LENGTHS = (2048, 4096, 8192, 16384)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(root: Path, name: str, value: Any) -> None:
    (root / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = numerator / denominator
    return {"intercept_ms": y_mean - slope * x_mean, "slope_us_per_chunk": slope * 1000.0}


def table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    inputs = args.input_dir

    formal_by_t = {
        t: read_json(inputs / f"stage6z_c25_formal_body_T{t}.json") for t in LENGTHS
    }
    correctness = read_json(inputs / "stage6z_c25_correctness.json")
    pmc_by_t = {
        t: {
            arm: read_json(inputs / "c25_pmc" / f"stage6z_c25_{arm}_T{t}_rocprof.json")
            for arm in ARM_ORDER
        }
        for t in (2048, 8192)
    }

    medians = {
        arm: [formal_by_t[t]["summary"][arm]["median_of_session_medians_ms"] for t in LENGTHS]
        for arm in ARM_ORDER
    }
    chunks = [formal_by_t[t]["chunks"] for t in LENGTHS]
    fits = {arm: linear_fit(chunks, medians[arm]) for arm in ARM_ORDER}
    gap_closed = (
        (fits["z5b"]["slope_us_per_chunk"] - fits["c25"]["slope_us_per_chunk"])
        / (fits["z5b"]["slope_us_per_chunk"] - fits["native_selected"]["slope_us_per_chunk"])
    )

    native_timeline = {
        "schema": "stage6z.c25.native-current-ready-timeline.v1",
        "conclusion": "HYBRID",
        "native_pipeline_mechanism": "HYBRID",
        "evidence_level": "A: fresh selected Triton TTGIR/LLVM/final ISA capture",
        "mechanism": {
            "stage_rotation": "TTGIR has local_alloc-backed stage buffers carried by loop iter_args; current operands are prepared in the preceding stage.",
            "intra_iteration_overlap": "The selected ISA issues next packet loads while current MFMA instructions consume already-ready LDS/register operands, then waits and publishes the next packet.",
            "not_inferred_from_num_stages_only": True,
        },
        "by_T": {
            "2048": {
                "selected": {"BK": 32, "BV": 64, "num_warps": 4, "workgroup": 256, "num_stages": 2, "lds_bytes": 12288, "hsaco_sha256": "cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d"},
                "timeline": [
                    ["CURRENT_READY", "previous rotating stage publishes current local operand"],
                    ["NEXT_LOAD", "ISA region around 219/224/234"],
                    ["CURRENT_MFMA_BEGIN", "around 222"],
                    ["CURRENT_MFMA_END", "around 244"],
                    ["NEXT_WAIT", "partial wait around 236 when needed"],
                    ["NEXT_COMMIT", "following LDS publication/rotation"],
                ],
            },
            "8192": {
                "selected": {"BK": 32, "BV": 64, "num_warps": 2, "workgroup": 128, "num_stages": 2, "lds_bytes": 12288, "hsaco_sha256": "e201dd58c83e64565f10754789ac294df53343b9c9bbbe764e8f667817066ee5"},
                "timeline": [
                    ["CURRENT_READY", "previous rotating stage publishes current local operand"],
                    ["NEXT_LOAD", "ISA region around 255/256"],
                    ["CURRENT_MFMA_BEGIN", "around 259"],
                    ["CURRENT_MFMA_END", "around 271"],
                    ["NEXT_WAIT", "vmcnt(4) around 278"],
                    ["NEXT_COMMIT", "following LDS stores and stage rotation"],
                ],
            },
        },
        "artifact_root": "codex_qwen_gfx942_c25_current_ready_next_pending/native_fresh",
    }
    write_json(root, "stage6z_c25_native_current_ready_timeline.json", native_timeline)

    synthetic = {
        "schema": "stage6z.c25.partial-wait-synthetic.v1",
        "status": "PASS",
        "test": "c25_current_ready_pending_codegen_test",
        "natural_waitcnt": True,
        "hardcoded_waitcnt": False,
        "llvm_dependency_graph": "older H BF16x8 global load feeds H LDS/MFMA; younger K BF16x8 global load is not consumed until the delayed K commit",
        "isa_order": [
            "global_load_dwordx4 H_current",
            "global_load_dwordx4 K_next",
            "s_waitcnt vmcnt(1)",
            "ds_write_b128 H_current",
            "current H MFMA x several",
            "s_waitcnt vmcnt(0)",
            "ds_write_b128 K_next",
        ],
        "outstanding_younger_packet": "K_next remains outstanding across the H publication/MFMA window.",
        "artifacts": "codex_qwen_gfx942_c25_current_ready_next_pending/synthetic",
    }
    write_json(root, "stage6z_c25_partial_wait_synthetic.json", synthetic)

    real_partial = {
        "schema": "stage6z.c25.real-h-to-k.v1",
        "status": "STAGE_ROTATION_SUCCESS_NOT_PARTIAL_WAIT",
        "why_not_partial_wait": "The real C25 form deliberately publishes older H before issuing K. H_READY_WAIT is therefore vmcnt(0), but it cannot drain younger K because K has not issued yet.",
        "isa_order": [
            [346, "H_LOAD", "global_load_dwordx4"],
            [352, "H_READY_WAIT", "s_waitcnt vmcnt(0)"],
            [353, "H_LDS_COMMIT", "ds_write_b128"],
            [358, "K_NEXT_LOAD", "global_load_dwordx4"],
            [384, "H_MFMA_BEGIN", "v_mfma..."],
            [399, "H_MFMA_END", "v_mfma..."],
            [402, "K_FINAL_WAIT", "s_waitcnt vmcnt(0)"],
            [403, "K_COMMIT", "ds_write_b128"],
        ],
        "required_partial_wait_predicate": False,
        "artifact": "codex_qwen_gfx942_c25_current_ready_next_pending/real_h_to_k_fixed/machine/final_isa.s",
    }
    write_json(root, "stage6z_c25_real_partial_wait_overlap.json", real_partial)

    plan = {
        "schema": "stage6z.c25.ready-pending-plan.v1",
        "schedule": "h_ready_then_k_pending_region_slots",
        "two_logical_stages": True,
        "phases": {
            "prologue": "IssuePacket(H current) -> CommitPacket(H current) -> READY(current H)",
            "steady": "IssuePacket(K next) -> PENDING(K next); consume READY H through current MFMA region",
            "epilogue": "wait/CommitPacket(K next) -> READY(next K) for its existing consumer",
        },
        "preserved": ["C21 physical ownership", "C24 IssuePacket/CommitPacket SSA", "BF16 ABI", "MFMA32 geometry", "K32 order", "V/scoreV/g"],
        "scope_limit": "A region-level H-to-K0 edge, not a complete native TTGIR cross-iteration memdesc rotation.",
    }
    write_json(root, "stage6z_c25_ready_pending_plan.json", plan)

    overlap = {
        "schema": "stage6z.c25.real-overlap-evidence.v1",
        "status": "PASS",
        "two_consecutive_logical_stages": [
            {"current": "H current packet", "pending": "K0 next packet", "mfmas_spanned": 4, "order": "K_NEXT_LOAD(358) < H_MFMA[384..399] < K_FINAL_WAIT(402) < K_COMMIT(403)"},
            {"current": "next K0 after commit", "pending": "existing following C21 consumer lifetime", "evidence": "region plan uses distinct ready/pending SSA state; full native cross-loop rotation was not claimed"},
        ],
        "current_ready_before_next_issue": "H packet waits and commits at 352/353, before K next issue at 358.",
        "no_global_vnew_reload": True,
        "machine_graph_changed": True,
        "hsaco_sha256": "16234db4147b44e2d84e70299eb24f0a88f2d2d82378cfc3298bc90590e91913",
    }
    write_json(root, "stage6z_c25_real_overlap_evidence.json", overlap)

    survival = {
        "schema": "stage6z.c25.pipeline-survival.v1",
        "stages": [
            {"stage": "post-region-plan MLIR", "survives": True, "evidence": "avelang.stage6z.c25.current_ready_next_pending=h_ready_then_k_pending_region_slots"},
            {"stage": "post-block MLIR", "survives": True, "evidence": "H Issue/Commit and K Issue/Commit SSA are materialized"},
            {"stage": "LLVM", "survives": True, "evidence": "H load/commit remains before K next load; artifact lowered_llvm.ll"},
            {"stage": "ISel/pre-schedule MIR", "survives": True, "evidence": "packet def/use ordering captured under llc_mir"},
            {"stage": "post-waitcnt MIR", "survives": True, "evidence": "wait analysis preserves delayed K wait"},
            {"stage": "post-RA/ISA", "survives": True, "evidence": "346 < 352 < 353 < 358 < 384..399 < 402 < 403"},
        ],
        "initial_mlir_note": "The initial MLIR printer was intentionally skipped due its known instability for this capture; no initial-MLIR equality claim is made.",
        "artifact_root": "codex_qwen_gfx942_c25_current_ready_next_pending/real_h_to_k_fixed/machine",
    }
    write_json(root, "stage6z_c25_pipeline_survival.json", survival)

    liveness = {
        "schema": "stage6z.c25.liveness.v1",
        "evidence_level": "B: exact MIR def/use and post-RA physical regions; not cycle-accurate LiveIntervals",
        "packets": [
            {"packet": "H current", "physical_range": "v[38:41] representative ISA packet", "start": "H_LOAD 346", "end": "H_LDS_COMMIT 353", "mfmas_crossed": 0},
            {"packet": "K next", "physical_range": "v[34:37] representative ISA packet", "start": "K_NEXT_LOAD 358", "end": "K_COMMIT 403", "mfmas_crossed": 4},
        ],
        "resource_identity": {"vgpr": 188, "agpr": 80, "sgpr": 36, "lds_bytes": 24576, "private_segment_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0},
        "accumulator_assessment": "No evidence of a new whole-QH/QK accumulator group whose lifetime crosses the packet window. The extra state is the current/next packet and minimal addresses, but C25 retains the C21-sized code-object allocation.",
        "warning": "Code-object VGPR/AGPR, profiler Accum_VGPR, and MIR virtual-register cardinality are distinct metrics.",
    }
    write_json(root, "stage6z_c25_liveness.json", liveness)

    write_json(root, "stage6z_c25_correctness.json", correctness)

    formal = {
        "schema": "stage6z.c25.formal-body.v1",
        "scope": "caller_owned_isolated_body_diagnostic",
        "contract": formal_by_t[2048]["contract"],
        "results_by_T": formal_by_t,
        "interpretation": "C21 frozen is a source-control arm compiled in the contemporary run, not a historical hash-guarded C21 code object. Native selected is direct selected Triton body, not public eager API.",
    }
    write_json(root, "stage6z_c25_formal_body.json", formal)

    slope = {
        "schema": "stage6z.c25.longtext-slope.v1",
        "model": "ordinary least squares: median_session_ms = intercept_ms + slope_ms_per_chunk * chunk_count",
        "points": [{"T": t, "chunks": formal_by_t[t]["chunks"]} for t in LENGTHS],
        "fits": fits,
        "slope_gap_closed": gap_closed,
        "meaning": "Negative means C25 enlarged rather than closed the Z5B-to-native slope gap.",
    }
    write_json(root, "stage6z_c25_longtext_slope.json", slope)

    machine_resources = {
        "schema": "stage6z.c25.machine-resources.v1",
        "c25_exact_lto": {
            "hsaco_sha256": "16234db4147b44e2d84e70299eb24f0a88f2d2d82378cfc3298bc90590e91913",
            "isa_sha256": "90fc3fd3ac0bb288ddb061fe6a3f9107a558c89d790b522bf967cbe2fd46f9be",
            "llvm_sha256": "e4c8701931919170c03596fdd52b0d4c20d7521e0ad5e2d6760a99c4e2a6f7ca",
            "vgpr": 188, "agpr": 80, "sgpr": 36, "lds_bytes": 24576,
            "private_segment_bytes": 0, "vgpr_spill_count": 0, "sgpr_spill_count": 0,
            "static": {"mfma": 20, "global_load": 94, "global_store": 18, "ds_read": 40, "ds_write": 100, "barrier": 18, "waitcnt": 96},
        },
        "profiler_resources": {
            str(t): {arm: {k: pmc_by_t[t][arm].get(k) for k in ("VGPR_Count", "Accum_VGPR_Count", "LDS_Block_Size", "Scratch_Size", "OccupancyPercent")} for arm in ARM_ORDER}
            for t in (2048, 8192)
        },
        "native_shape_caveat": "At T2048 selected native is WG256; at T8192 selected native is WG128. Per-CTA comparisons at T8192 are descriptive rather than same-workgroup equivalence.",
    }
    write_json(root, "stage6z_c25_machine_resources.json", machine_resources)

    pmc = {
        "schema": "stage6z.c25.pmc.v1",
        "capture": "rocprofv3 fresh process, dynamic counters normalized by CTA count supplied by the capture driver",
        "by_T": pmc_by_t,
        "caveat": "Counters are hardware dynamic counts from the capture, not values inferred from static ISA. T8192 native has a different selected WG (128).",
    }
    write_json(root, "stage6z_c25_pmc.json", pmc)

    causal = {
        "schema": "stage6z.c25.causal-delta.v1",
        "controlled_invariants": ["BF16 ABI", "BT64/BV64/BK32", "WG256 C21/C25 source ownership", "MFMA geometry and dynamic MFMA/CTA", "K32 accumulation order", "causal mask", "V/scoreV/g", "caller-owned output contract"],
        "C24_failure": "K issue preceded H issue; H's required vmcnt(0) occurred after both loads and drained K before H MFMA.",
        "C25_delta": "Commit current H before issuing K, then delay K's wait/commit across four H MFMAs.",
        "overlap_proved": True,
        "performance_result": "No benefit: C25 has the same dynamic MFMA/CTA as C21 (160) and is slower than Z5B at every formal length. The local H-to-K region proof does not eliminate the broader Z5B materialization/address costs or reproduce native full loop-stage rotation.",
        "causal_claim_limit": "C25 demonstrates dependency materialization, not a whole-kernel latency-hiding win.",
    }
    write_json(root, "stage6z_c25_causal_delta.json", causal)

    regression = {
        "schema": "stage6z.c25.regressions.v1",
        "results": [
            {"name": "c25_current_ready_pending_codegen_test", "status": "PASS"},
            {"name": "C25 correctness fresh process T64/T2048/T8192, five cases each", "status": "PASS, BF16 byte-exact to Z5B"},
            {"name": "C25 helper script syntax", "status": "PASS"},
            {"name": "exact LTO replay", "status": "PASS, no private segment or spills"},
            {"name": "git diff --check", "status": "PASS"},
        ],
        "formal_performance_executed": True,
        "stop_after_c25": True,
    }
    write_json(root, "stage6z_c25_regression_results.json", regression)

    report = []
    report += ["# Qwen gfx942 C25: Current-Ready / Next-Pending Pipeline Proof", "", "## 结论", "", "**`STOP_C25_CURRENT_READY_OVERLAP_NO_PERF`。** C25 成功把 C24 缺失的依赖顺序保留到最终 ISA：当前 H 在 next K issue 前已经 READY，next K 在四条当前 H MFMA 期间保持 PENDING，随后才 wait/commit。所有正确性 gate 通过，但 T=2048 到 T=16384 均慢于 Z5B，且长文本 slope 从 Z5B 的 %.6f 上升到 C25 的 %.6f us/chunk。因此这条假设已完整验证且止损；不启动 C26。" % (fits["z5b"]["slope_us_per_chunk"], fits["c25"]["slope_us_per_chunk"]), "", "C25 是有效的 compiler dependency/schedule proof，不是性能 baseline。Z5B 继续是 isolated chunk-o performance baseline。", "", "## Native 机制", "", "fresh native selected artifact 在 T=2048 为 BK32/BV64/WG256/2 stages，在 T=8192、4096、16384 为 BK32/BV64/WG128/2 stages。TTGIR 的 local_alloc 与 loop iter_args 证明 current operand 由旋转 stage 准备；ISA 又在 current MFMA 前后交错 next packet 的 load/wait/commit。因此分类为 **HYBRID**：stage rotation 为主，iteration 内也存在 overlap；不是仅根据 `num_stages=2` 推断。", "", "T=2048 机械窗口：next load 约 219/224/234，current MFMA 约 222/230/244，必要 partial wait 约 236。T=8192：next load 约 255/256，current MFMA 约 259/268/269/271，`vmcnt(4)` 约 278。详见 `stage6z_c25_native_current_ready_timeline.json`。", "", "## C24 到 C25", "", "C24 已有 IssuePacket/CommitPacket SSA，却先 issue K_next、后 issue H_current。H publication 的 `vmcnt(0)` 同时等待 H 与更早发出的 K，故 K 无法跨 H MFMA 保持 outstanding。", "", "C25 保留 C21 mapping、C24 packet representation、BF16 ABI、MFMA/K32 数学和所有 V/scoreV/g 路径。它只把 H 作为 current packet 在 K issue 前 publish：", "", "```text", "H_LOAD(346) -> H_READY_WAIT vmcnt(0)(352) -> H_LDS_COMMIT(353)", "-> K_NEXT_LOAD(358) -> H_MFMA x4 (384..399)", "-> K_FINAL_WAIT vmcnt(0)(402) -> K_COMMIT(403)", "```", "", "这满足 `NEXT_LOAD < CURRENT_MFMA_BEGIN < CURRENT_MFMA_END < NEXT_WAIT < NEXT_COMMIT`。real H_READY_WAIT 是 `vmcnt(0)`，但它发生在 K issue 之前，所以不会 drain K；这不是 real partial-wait，而是 Ready/Pending stage rotation。", "", "## Partial-Wait 可行性", "", "synthetic H-old/K-young proof 由 AMDGPU wait analysis 自然生成 `s_waitcnt vmcnt(1)`：`H load -> K load -> vmcnt(1) -> H LDS/MFMA -> vmcnt(0) -> K LDS`。未硬编码 waitcnt。故 gfx942/LLVM 可以表达 older-current 等待、younger-next 保持 outstanding。", "", "但 real C25 选择更稳的 current-ready 方式：先令 H ready，再 issue K。它实现 region-level H-to-K0 two-stage schedule，而非完整 native TTGIR cross-iteration memdesc rotation；报告不把两者混为一谈。", "", "## Liveness 与资源", "", "C25 exact LTO code object：VGPR=188、AGPR=80、SGPR=36、LDS=24576 B、private=0、VGPR/SGPR spill=0。H packet 的代表性 `v[38:41]` 从 346 到 353；K packet 的 `v[34:37]` 从 358 跨过 4 条 H MFMA，到 403 commit。没有证据表明 C25 新增了完整 QH/QK accumulator group；这里只报告 machine def/use 级别范围，并不伪造 cycle-accurate LiveIntervals。", "", "T=2048 dynamic PMC（每 CTA）：Z5B=MFMA/VMEM/LDS/VALU/SALU `160/672/672/7072/768`；C21=`160/304/688/6422/626`；C25=`160/304/688/6470/658`；native=`320/224/904/5912/1296`。native 的每 CTA MFMA 是两倍，说明 native CTA ownership 不同，不能据此宣称 same-CTA 工作相同。T=8192 native 选择 WG128，此时 per-CTA 同样只作描述。", "", "## 正确性", "", "T=64/2048/8192 均以 fresh process 对 Z5B 做 random、zero-V-new、caller-owned NaN prefill、structured Q/H/K、token/value pattern。所有 H/output 都 BF16 byte-exact，finite，无 caller-owned output 泄漏；prologue、steady、epilogue 均覆盖。", "", "## 正式 Body Timing", "", "口径：caller-owned preallocated output、current HIP stream、no Graph、warmup=10、repeat=50、7 个 fresh Python process sessions、balanced rotating order、HIP event。它是 isolated body diagnostic，不是 Eager public API 排名。", "", table_row(["T", "chunks", "Z5B ms", "C21 source control ms", "C25 ms", "native selected ms", "C25-Z5B us"]), table_row(["---:", "---:", "---:", "---:", "---:", "---:", "---:"])]
    for t in LENGTHS:
        d = formal_by_t[t]
        s = d["summary"]
        report.append(table_row([str(t), str(d["chunks"]), "%.6f" % s["z5b"]["median_of_session_medians_ms"], "%.6f" % s["c21_frozen"]["median_of_session_medians_ms"], "%.6f" % s["c25"]["median_of_session_medians_ms"], "%.6f" % s["native_selected"]["median_of_session_medians_ms"], "%+.3f" % d["paired_c25_minus_z5b_us"]["median_us"]]))
    report += ["", "T=8192 C25 比 Z5B 慢 %.2f%%（paired 95%% CI `[%0.3f, %0.3f] us`）；T=16384 慢 %.2f%%（CI `[%0.3f, %0.3f] us`）。四个长度均未出现正收益。" % ((medians["c25"][2] / medians["z5b"][2] - 1) * 100, *formal_by_t[8192]["paired_c25_minus_z5b_us"]["bootstrap_ci95_us"], (medians["c25"][3] / medians["z5b"][3] - 1) * 100, *formal_by_t[16384]["paired_c25_minus_z5b_us"]["bootstrap_ci95_us"]), "", "## 长文本 Slope", "", table_row(["arm", "intercept ms", "slope us/chunk"]), table_row(["---", "---:", "---:"])]
    for arm, label in (("z5b", "Z5B"), ("c21_frozen", "C21 source control"), ("c25", "C25"), ("native_selected", "native selected")):
        report.append(table_row([label, "%.6f" % fits[arm]["intercept_ms"], "%.6f" % fits[arm]["slope_us_per_chunk"]]))
    report += ["", "`slope_gap_closed = %.6f`。负值表示 C25 使 Z5B-to-native slope gap 扩大约 13.1%%，而非关闭。当前 C25 的 local overlap 没有消除 Z5B 更大的 global/materialization/address work，因此不能把机器窗口 overlap 当作端到端 latency hiding 的充分条件。" % gap_closed, "", "## 直接回答", "", "1. Native current operand 由 previous rotating stage 准备，并在 next issue 前 ready。", "2. Native 为 HYBRID：stage rotation 主导，包含 iteration 内 load/MFMA overlap。", "3. C24 先 K 后 H，H 的 `vmcnt(0)` 同时 drain K。", "4. 可以：synthetic 自然生成 `vmcnt(1)`。", "5. synthetic PASS。", "6. real H-to-K partial-wait 未采用；real 成功的是 current-ready stage rotation。", "7. PASS：H READY、K PENDING 真实进入 ISA。", "8. 显式为 H prologue ready、steady K pending/H consume、epilogue K commit。", "9. PASS：358 < 384..399 < 402 < 403。", "10. PASS：H 353 commit，早于 K 358 issue。", "11. K pending 跨 4 条 H MFMA。", "12. packet 是短 H/K packet；C25 code object 保持 VGPR188/AGPR80，spill=0。", "13. T64/T2048/T8192 全 PASS，BF16 exact。", "14. 正式数值见上表。", "15. T8192/T16384 分别慢 5.22%%/6.19%%。", "16. Z5B/C25/native slope 为 %.6f/%.6f/%.6f us/chunk。" % (fits["z5b"]["slope_us_per_chunk"], fits["c25"]["slope_us_per_chunk"], fits["native_selected"]["slope_us_per_chunk"]), "17. slope gap closed = %.6f，实际扩大。" % gap_closed, "18. overlap 已由 ISA 证明，但性能无收益；没有把收益归因给 overlap。", "19. 最终为 `STOP_C25_CURRENT_READY_OVERLAP_NO_PERF`。", "", "## 止损", "", "C25 到此关闭。按任务约束，不自动启动 C26、packet/barrier/VALU sweep、RA tuning、新 layout、新 superloop 或新 FullPhysicalRegion。", "", "## 证据文件", "", "- `stage6z_c25_native_current_ready_timeline.json`", "- `stage6z_c25_partial_wait_synthetic.json`", "- `stage6z_c25_real_partial_wait_overlap.json`", "- `stage6z_c25_ready_pending_plan.json`", "- `stage6z_c25_real_overlap_evidence.json`", "- `stage6z_c25_pipeline_survival.json`", "- `stage6z_c25_liveness.json`", "- `stage6z_c25_correctness.json`", "- `stage6z_c25_formal_body.json`", "- `stage6z_c25_longtext_slope.json`", "- `stage6z_c25_machine_resources.json`", "- `stage6z_c25_pmc.json`", "- `stage6z_c25_causal_delta.json`", "- `stage6z_c25_regression_results.json`", "", "Raw compiler and native artifacts are under `codex_qwen_gfx942_c25_current_ready_next_pending/`."]
    (root / "qwen_gfx942_c25_current_ready_next_pending.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

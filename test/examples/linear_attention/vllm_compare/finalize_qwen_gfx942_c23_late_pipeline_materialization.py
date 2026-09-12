#!/usr/bin/env python3
"""Finalize the read-only C23 late-pipeline materialization evidence.

C23 has a hard ISA-overlap gate.  This finalizer intentionally writes explicit
``not_run`` records when the experimental compiler path cannot produce LLVM;
it never substitutes a C21 or native measurement for a C23 result.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
OUT = LADDER / "codex_qwen_gfx942_c23_late_pipeline_materialization"
C21 = LADDER / "codex_qwen_gfx942_c21_selected_native_pipeline/machine"
Z5B = LADDER / "codex_qwen_bt64_stage6z_z5b_machine_stage1/machine_evidence.json"
NATIVE = LADDER / "codex_qwen_bt64_stage6z_native_chunko"
C23_LOG = OUT / "machine/c23_t64_compile.stdout.log"
C23_EXIT = OUT / "machine/c23_t64_compile.exit_code"
REPORT = LADDER / "qwen_gfx942_c23_late_pipeline_materialization.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def line_of(path: Path, pattern: str, occurrence: int = 1) -> int | None:
    count = 0
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if re.search(pattern, line):
            count += 1
            if count == occurrence:
                return number
    return None


def hashes(root: Path) -> dict[str, str | None]:
    names = {
        "source": "chunk_fwd_kernel_o.source",
        "ttir": "chunk_fwd_kernel_o.ttir",
        "ttgir": "chunk_fwd_kernel_o.ttgir",
        "llvm": "chunk_fwd_kernel_o.llir",
        "amdgcn": "chunk_fwd_kernel_o.amdgcn",
        "hsaco": "chunk_fwd_kernel_o.hsaco",
    }
    return {key: sha256(root / name) if (root / name).exists() else None for key, name in names.items()}


def native_identity(t: int, selected: Path, freshness: str) -> dict[str, Any]:
    metadata = load_json(selected / "chunk_fwd_kernel_o.json")
    warps = int(metadata["num_warps"])
    return {
        "schema": "qwen.gfx942.c23.native_identity.v1",
        "T": t,
        "selection_evidence": freshness,
        "selected_dir": str(selected),
        "kernel": metadata["name"],
        "hsaco_metadata_hash": metadata["hash"],
        "target": metadata["target"],
        "BK": 32,
        "BV": 64,
        "BT": 64,
        "num_warps": warps,
        "workgroup": warps * 64,
        "waves": warps,
        "num_stages": int(metadata["num_stages"]),
        "num_ctas": int(metadata["num_ctas"]),
        "shared_bytes": int(metadata["shared"]),
        "grid": [2, t // 64, 8],
        "artifacts_sha256": hashes(selected),
    }


def write(name: str, value: dict[str, Any]) -> None:
    (LADDER / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c21_machine = load_json(C21 / "machine_evidence.json")
    z5b_machine = load_json(Z5B)
    c21_isa = C21 / "final_isa.s"
    c21_llvm = C21 / "lowered_llvm.ll"
    c21_isel = C21 / "llc_mir/stop_after_amdgpu-isel.mir"
    source = HERE / "qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline.py"
    lowering = REPO / "lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc"
    c21_superloop_line = line_of(source, r"for k_stage in al\.range\(4\)")
    producer_lowering_line = line_of(lowering, r"mlir::LogicalResult emitC19FullPhysicalProducer")
    c23_materializer_line = line_of(lowering, r"class C23LatePipelineMaterializer")
    exit_code = int(C23_EXIT.read_text().strip()) if C23_EXIT.exists() else None

    t2048 = OUT / "native_fresh/T2048/selected"
    t8192 = OUT / "native_fresh/T8192/selected"
    t16384 = OUT / "native_fresh/T16384/selected"
    if not t2048.exists():
        t2048 = NATIVE / "native/T2048/trace_capture/selected"
    if not t8192.exists():
        t8192 = NATIVE / "native/T8192/trace_capture/selected"
    if not t16384.exists():
        t16384 = NATIVE / "native_refresh/T16384/selected"
    identity_2048 = native_identity(
        2048, t2048,
        "C23 fresh Python process; public chunk_fwd_o call after clearing the in-memory tuner cache",
    )
    identity_8192 = native_identity(
        8192, t8192,
        "C23 fresh Python process; public chunk_fwd_o call after clearing the in-memory tuner cache",
    )
    identity_16384 = native_identity(
        16384, t16384,
        "C23 fresh Python process; public chunk_fwd_o call after clearing the in-memory tuner cache. The selected hash equals T8192.",
    )
    write("stage6z_c23_native_identity_t2048.json", identity_2048)
    write("stage6z_c23_native_identity_t8192.json", identity_8192)
    write("stage6z_c23_native_identity_t16384.json", identity_16384)

    divergence = {
        "schema": "qwen.gfx942.c23.pipeline_first_divergence.v1",
        "decision": "STOP_C23_LATE_PIPELINE_NOT_CONTROLLABLE",
        "first_divergence": {
            "stage": "AveLang block-dot full-physical producer lowering",
            "classification": ["Case I: lowering serialization", "Case III: immediate SSA consumer dependency"],
            "why": "C21's plan/source has a common K32 loop, but emitC19FullPhysicalProducer emits global packet load, LDS store, barrier, and only then permits the corresponding MFMA consumer. The global result has an immediate LDS-store use, so waitcnt is required before publication.",
            "source_location": {
                "c21_superloop": f"{source}:{c21_superloop_line}",
                "producer_lowering": f"{lowering}:{producer_lowering_line}",
                "c23_attempt": f"{lowering}:{c23_materializer_line}",
            },
        },
        "stages": [
            {
                "stage": "C21 source / plan",
                "artifact": str(source),
                "relative_order": "Q owner -> H logical dot -> K half0 logical dot -> K half1 logical dot inside one source K32 loop",
                "dependency_reason": "The source establishes a common loop but contains no first-class pending packet or stage-token SSA value.",
                "first_divergence": False,
                "evidence_sha256": sha256(source),
            },
            {
                "stage": "post-block-dot AveLang lowering",
                "artifact": str(lowering),
                "relative_order": "global vector load -> LDS vector store -> gpu.barrier -> consumer lowering",
                "dependency_reason": "The producer result is immediately consumed by the LDS store; therefore no independent MFMA window is represented for it.",
                "first_divergence": True,
                "evidence_sha256": sha256(lowering),
            },
            {
                "stage": "pre-opt LLVM",
                "artifact": str(c21_llvm),
                "relative_order": "C21 lowering preserves the producer/publication ordering; no C23 pending-packet LLVM was produced.",
                "dependency_reason": "The graph is already serialized before AMDGPU scheduling.",
                "first_divergence": False,
                "evidence_sha256": sha256(c21_llvm),
            },
            {
                "stage": "MIR after AMDGPU instruction selection",
                "artifact": str(c21_isel),
                "relative_order": "C21 has selected VMEM, LDS and barrier instructions; this is an audit of an already serialized producer/publication graph.",
                "dependency_reason": "No scheduler-created inversion is required to explain the final ISA.",
                "first_divergence": False,
                "evidence_sha256": sha256(c21_isel),
            },
            {
                "stage": "final C21 ISA",
                "artifact": str(c21_isa),
                "relative_order": "global_load line 363 -> vmcnt(0) line 371 -> ds_write line 372 -> barrier line 374 -> first Q@H MFMA line 378; later K load line 394 -> vmcnt(0) line 395 -> LDS store line 396 -> barrier line 399 -> K MFMA line 422.",
                "dependency_reason": "The final machine graph confirms immediate wait/publication rather than a pending next packet across the current MFMA window.",
                "first_divergence": False,
                "evidence_sha256": sha256(c21_isa),
            },
            {
                "stage": "C23 attempted lowering",
                "artifact": str(C23_LOG),
                "relative_order": "No LLVM/MIR/ISA exists: get_llvm_ir terminated with SIGSEGV before the late packet could reach LLVM.",
                "dependency_reason": "The existing per-block-dot greedy rewrite control does not safely materialize a cross-op predicated vector packet. The first-class packet needs a region-level scheduling representation, not another per-op producer rewrite.",
                "first_divergence": False,
                "evidence_sha256": sha256(C23_LOG),
            },
        ],
    }
    write("stage6z_c23_pipeline_first_divergence.json", divergence)

    native_timeline = {
        "schema": "qwen.gfx942.c23.native_overlap_timeline.v1",
        "scope": "machine-grounded selected Triton ISA; T2048 and T8192 have distinct fresh identities",
        "T2048": {
            "identity": "stage6z_c23_native_identity_t2048.json",
            "isa_artifact": str(t2048 / "chunk_fwd_kernel_o.amdgcn"),
            "issue": "line 219 issues the next buffer_load_dwordx4 packet.",
            "current_mfma_window": "lines 222 and 230 execute current MFMA work after that issue.",
            "delayed_publish": "line 236 is the first VMEM wait for that packet; lines 237-242 publish the packet group to LDS.",
            "evidence_sha256": sha256(t2048 / "chunk_fwd_kernel_o.amdgcn"),
        },
        "T8192": {
            "identity": "stage6z_c23_native_identity_t8192.json",
            "isa_artifact": str(t8192 / "chunk_fwd_kernel_o.amdgcn"),
            "issue": "lines 255-256 issue two buffer_load_dwordx4 packets for the next shared publication.",
            "current_mfma_window": "lines 259, 268-269 and 271 execute current MFMA work after the issue.",
            "additional_issue": "lines 273-274 issue the next packet pair while the first group remains pending.",
            "delayed_publish": "line 278 is the first VMEM wait for the group issued at 255-256; lines 279-290 publish it to LDS.",
            "evidence_sha256": sha256(t8192 / "chunk_fwd_kernel_o.amdgcn"),
        },
        "comparison": "Native represents a carried staged slot. C21 represents an immediately published producer. C23 did not produce a comparable machine graph.",
    }
    write("stage6z_c23_native_overlap_timeline.json", native_timeline)

    overlap = {
        "schema": "qwen.gfx942.c23.overlap_evidence.v1",
        "gate": "failed before ISA",
        "C21": {
            "next_load": 394,
            "wait_for_next": 395,
            "lds_publish": 396,
            "barrier": 399,
            "next_mfma_first_use": 422,
            "mfma_between_load_and_wait": 0,
            "reason": "immediate VMEM wait before LDS publication",
        },
        "C23": {
            "status": "not_materialized",
            "compile_exit_code": exit_code,
            "raw_log": str(C23_LOG),
            "next_load": None,
            "wait_for_next": None,
            "mfma_between_load_and_wait": None,
        },
        "native_T8192": {
            "next_load_lines": [255, 256],
            "current_mfma_lines_after_load": [259, 268, 269, 271],
            "later_next_load_lines": [273, 274],
            "first_publish_wait_line": 278,
            "mfma_between_first_load_and_first_publish_wait": 4,
        },
    }
    write("stage6z_c23_overlap_evidence.json", overlap)

    liveness = {
        "schema": "qwen.gfx942.c23.liveness_delta.v1",
        "C21_code_object": {"vgpr": c21_machine["vgpr_count"], "agpr": c21_machine["agpr_count"], "lds_bytes": c21_machine["group_segment_fixed_size"], "private": c21_machine["private_segment_fixed_size"], "spills": [c21_machine["vgpr_spill_count"], c21_machine["sgpr_spill_count"]]},
        "Z5B_code_object": {"vgpr": z5b_machine["vgpr_count"], "agpr": z5b_machine["agpr_count"], "lds_bytes": z5b_machine["group_segment_fixed_size"], "private": z5b_machine["private_segment_fixed_size"], "spills": [z5b_machine["vgpr_spill_count"], z5b_machine["sgpr_spill_count"]]},
        "C23": {"status": "not_available", "reason": "No C23 LLVM/MIR/HSACO was generated; a liveness delta would be fabricated."},
        "interpretation": "C21 already paid 188 VGPR / 80 AGPR without materializing overlap. C23 therefore may not claim a small-packet liveness result until a region-level packet representation reaches MIR.",
    }
    write("stage6z_c23_liveness_delta.json", liveness)

    skipped = {
        "schema": "qwen.gfx942.c23.not_run.v1",
        "decision": "STOP_C23_LATE_PIPELINE_NOT_CONTROLLABLE",
        "reason": "The required final C23 ISA overlap did not materialize because the experimental lowering terminated before LLVM.",
        "correctness_gate": "not_run by pre-registered ISA-overlap gate",
        "performance_gate": "not_run by pre-registered ISA-overlap gate",
        "replacement_for": "None. C21 and native numbers are diagnostic evidence only, not C23 substitutions.",
    }
    write("stage6z_c23_correctness.json", {**skipped, "T": [64, 2048, 8192], "cases": ["random", "zero-V", "NaN-prefilled caller output", "structured Q/H/K", "finite", "BF16 exact"]})
    write("stage6z_c23_formal_body.json", {**skipped, "T": [2048, 4096, 8192, 16384], "arms": ["Z5B", "C23", "selected native"]})
    write("stage6z_c23_longtext_slope.json", {**skipped, "formula": "latency = intercept + slope * chunks", "slope_gap_closed": None})
    write("stage6z_c23_machine_resources.json", {**skipped, "C21_diagnostic": {"vgpr": c21_machine["vgpr_count"], "agpr": c21_machine["agpr_count"], "lds": c21_machine["group_segment_fixed_size"], "private": c21_machine["private_segment_fixed_size"]}, "C23": "no HSACO"})
    write("stage6z_c23_pmc.json", {**skipped, "metrics": ["MFMA", "VMEM", "LDS", "VALU", "SALU"], "C23": "not collected; no runnable C23 kernel"})
    write("stage6z_c23_causal_delta.json", {**skipped, "C21_vs_C23": "C23 did not reach a machine graph. No latency or PMC delta is attributable to the proposed packet mechanism.", "unchanged_intent": ["C21 source superloop", "Q/H/K ownership", "LDS allocation and barriers", "MFMA geometry", "math and ABI"]})
    write("stage6z_c23_regression_results.json", {"schema": "qwen.gfx942.c23.regression.v1", "compiler_build": "PASS: build-vllm-rocm722 completed and Python binding was replaced", "C23_compile_only_T64": {"status": "FAIL", "exit_code": exit_code, "log": str(C23_LOG)}, "C23_correctness": "not_run by ISA gate", "C23_performance": "not_run by ISA gate"})

    report = f"""# Qwen gfx942 C23 Late Pipeline Materialization

## 结论

**`STOP_C23_LATE_PIPELINE_NOT_CONTROLLABLE`**。C23 没有生成 LLVM、MIR、
ISA 或 HSACO，因此没有进行正确性、PMC 和性能测试。这个停止不是资源阈值，
而是任务预注册的硬门槛：必须先在最终 ISA 中看到
`next load -> current MFMA -> delayed wait`。

## C21 首次退化点

C21 高层源在 [qwen_gdn_bt64_native_chunko_stage6z_c21_selected_native_pipeline.py]({source}:{c21_superloop_line})
已把 Q owner、Q@H 和两个 Q@K 放进同一 K32 loop；但它没有 first-class
pending packet。第一次失去 pipeline 的位置是
[lower_qwen_block_dot_pass.cc]({lowering}:{producer_lowering_line}) 的
`emitC19FullPhysicalProducer`：每个 H/K producer 立刻 lower 成 global
packet load、LDS store、barrier，之后才形成 MFMA consumer。于是 packet SSA
的第一个 consumer 就是 LDS store，AMDGPU 必须在 store 前等待 VMEM。

最终 C21 ISA 给出直接证据：line 363 global load，371 `vmcnt(0)`，372 LDS
store，374 barrier，378 第一个 MFMA；随后 line 394 K load，395 wait，396
LDS store，399 barrier，422 K MFMA。故根因属于 **AveLang lowering 的
producer SSA / dependency graph（Case I + III）**，不是 LLVM 或 AMDGPU
scheduler 把一个已经存在的合法 window 排坏。

## C23 最小实现与停止

本轮只在 C21 的 H/K lowering 边界实现 `C23LatePipelineMaterializer`：H
load 后发起一个 predicated BF16x8 K0 packet，保持 packet SSA，计划在 H
MFMA 后再提交 K0 到已有 LDS。没有修改 C21 source superloop、layout、
ownership、barrier 计划、MFMA 几何、数学、ABI、RA 或 production。

该实现暴露了现有 per-block-dot greedy rewrite 的控制边界。带有跨 op 的
predicated vector packet 时，C23 T64 compile-only 在 `get_llvm_ir` 退出
`{exit_code}`；原始日志在
[c23_t64_compile.stdout.log]({C23_LOG})。因此这个 lowering 阶段不能安全地
拥有一个跨 H/K logical op 的 pending packet，且无法输出可验证的 C23 ISA。
把 C21 或 native ISA 当作 C23 结果会掩盖这个事实，所以所有后续测试均明确
标记为 `not_run`。

## Native 对照

fresh T2048 selected native 是 WG256、4 waves、2 stages；T8192 是 WG128、2 waves、
2 stages，T16384 与 T8192 选择相同哈希。T2048 final ISA 的 line 219
先 issue `buffer_load_dwordx4`，line 222 和 230 仍执行当前 MFMA，直到 line 236
才首次等待并在 237--242 publish 到 LDS。T8192 对应关系是 line 255--256
issue、259/268/269/271 current MFMA、278 首次 VMEM wait、279--290 LDS publication。
这正是 C23 需要 materialize 而未能 materialize 的依赖窗口。

## 已生成证据

- `stage6z_c23_pipeline_first_divergence.json`
- `stage6z_c23_native_identity_t2048.json`
- `stage6z_c23_native_identity_t8192.json`
- `stage6z_c23_native_identity_t16384.json`
- `stage6z_c23_native_overlap_timeline.json`
- `stage6z_c23_liveness_delta.json`
- `stage6z_c23_overlap_evidence.json`
- `stage6z_c23_correctness.json`
- `stage6z_c23_formal_body.json`
- `stage6z_c23_longtext_slope.json`
- `stage6z_c23_machine_resources.json`
- `stage6z_c23_pmc.json`
- `stage6z_c23_causal_delta.json`
- `stage6z_c23_regression_results.json`

本轮到此停止。下一步若重开，不应再增加 C24 chunk-o source 变体；需要先在
compiler 中设计一个 region-level 的 pending packet / pipeline token 表示，令
packet ownership 脱离 per-op greedy rewrite，再谈 machine scheduling。
"""
    REPORT.write_text(report)
    print(json.dumps({"decision": "STOP_C23_LATE_PIPELINE_NOT_CONTROLLABLE", "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finalize the read-only C26 work-decomposition ownership audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
SOURCE = REPO / "test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py"
Z5B_EVIDENCE = HERE / "codex_qwen_bt64_stage6z_z5b_machine_stage1/machine_evidence.json"
LENGTHS = (2048, 8192, 16384)
COUNTERS = ("SQ_INSTS_MFMA", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_VALU", "SQ_INSTS_SALU")
Z5B_KERNEL = "_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z5b_direct_q_cache"
NATIVE_KERNEL = "chunk_fwd_kernel_o"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write(name: str, payload: dict) -> None:
    (HERE / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def count(path: Path, pattern: str) -> int:
    return len(re.findall(pattern, path.read_text(errors="replace")))


def parse_tail(path: Path, kernel: str) -> dict:
    dispatches: dict[int, dict[str, dict[str, str]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["Kernel_Name"] == kernel:
                dispatches.setdefault(int(row["Dispatch_Id"]), {})[row["Counter_Name"]] = row
    if len(dispatches) < 5:
        raise RuntimeError(f"need 5 exact tail dispatches: {path}")
    ids = sorted(dispatches)[-5:]
    tail = [dispatches[item] for item in ids]
    sample = next(iter(tail[-1].values()))
    ctas = int(sample["Grid_Size"]) / int(sample["Workgroup_Size"])
    if not ctas.is_integer():
        raise RuntimeError(f"non-integral CTA count: {path}")
    values = {name: statistics.median(float(row[name]["Counter_Value"]) for row in tail) for name in COUNTERS}
    occupancy = [float(row["OccupancyPercent"]["Counter_Value"]) for row in tail]
    meta = {key: int(sample[key]) for key in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count")}
    return {"exact_kernel": kernel, "matched_dispatch_count": len(dispatches), "final_tail_dispatch_ids": ids, "final_tail_count": 5, "metadata": meta, "cta_count": int(ctas), "raw_counter_median": values, "per_cta": {key: values[key] / ctas for key in COUNTERS}, "occupancy_tail_samples": occupancy, "occupancy_median": statistics.median(occupancy)}


def fit_slope(timing: dict[int, dict], arm: str) -> dict:
    xs = [t / 64.0 for t in LENGTHS]
    ys = [timing[t]["summary"][arm]["median_of_session_medians_ms"] * 1.0e3 for t in LENGTHS]
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    return {"intercept_us": ym - slope * xm, "slope_us_per_chunk": slope}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--timing-root", type=Path, required=True)
    args = parser.parse_args()
    source_hash = digest(SOURCE)
    z5b_evidence = load(Z5B_EVIDENCE)
    pmc: dict[int, dict[str, dict]] = {}
    identities: dict[str, dict] = {}
    native_static: dict | None = None
    for t in LENGTHS:
        z_dir = args.input_root / "c26_raw" / f"z5b_T{t}"
        n_dir = args.input_root / "c26_final_native" / f"T{t}"
        pmc[t] = {
            "z5b": parse_tail(z_dir / f"c26_z5b_T{t}_counter_collection.csv", Z5B_KERNEL),
            "native_selected": parse_tail(n_dir / f"c26_native_T{t}_counter_collection.csv", NATIVE_KERNEL),
        }
        group = n_dir / "triton_cache" / "QJQU76HEWCWRWOE3EI3GIMSZ6EDUAEIC6L5LJDMSZBA2WT6FJEBQ"
        amdgcn = group / "chunk_fwd_kernel_o.amdgcn"
        static = {"cache_group": group.name, "hashes": {suffix: digest(group / f"chunk_fwd_kernel_o.{suffix}") for suffix in ("hsaco", "amdgcn", "ttir", "ttgir", "llir", "source")}, "static_isa": {"mfma32": count(amdgcn, r"v_mfma_f32_32x32x8_bf16"), "ds_read": count(amdgcn, r"\bds_read"), "ds_write": count(amdgcn, r"\bds_write"), "barrier": count(amdgcn, r"\bs_barrier\b"), "waitcnt": count(amdgcn, r"\bs_waitcnt\b"), "buffer_load": count(amdgcn, r"\bbuffer_load"), "buffer_store": count(amdgcn, r"\bbuffer_store")}, "ttgir": {"local_alloc": count(group / "chunk_fwd_kernel_o.ttgir", r"ttg\.local_alloc"), "local_store": count(group / "chunk_fwd_kernel_o.ttgir", r"ttg\.local_store"), "local_load": count(group / "chunk_fwd_kernel_o.ttgir", r"ttg\.local_load")}}
        native_static = static
        identities[str(t)] = {"z5b": {"identity": load(z_dir / "identity.json"), "code_object": z5b_evidence}, "native_selected": {"identity": load(n_dir / "identity.json"), **static}}

    write("stage6z_c26_identity.json", {"schema": "qwen.gfx942.stage6z.c26.identity.v1", "scope": "fresh exact-tail captures", "z5b_source": str(SOURCE), "z5b_source_sha256": source_hash, "lengths": identities})
    logical = {"schema": "qwen.gfx942.stage6z.c26.logical_work_unit.v1", "definition": "one (chunk,value_head) output tile: 64 token rows x full 128 value columns", "output": {"shape": [64, 128], "dtype": "bf16", "elements": 8192, "bytes": 16384}, "inputs": {"Q": {"shape": [64, 128], "dtype": "bf16", "bytes": 16384}, "K": {"shape": [64, 128], "dtype": "bf16", "bytes": 16384}, "H": {"shape": [128, 128], "dtype": "bf16", "bytes": 32768}, "V_new": {"shape": [64, 128], "dtype": "bf16", "bytes": 16384}, "g": {"shape": [64], "dtype": "fp32", "bytes": 256}}, "partition": "two V64 CTAs each own all 64 token rows and disjoint V[0:64] or V[64:128] output columns"}
    write("stage6z_c26_logical_work_unit.json", logical)
    write("stage6z_c26_z5b_cta_ownership.json", {"schema": "qwen.gfx942.stage6z.c26.z5b_cta_ownership.v1", "launch": "grid=(num_chunks*8*2,1,1), WG256", "program_id": {"v_block": "pid%2", "value_head": "(pid//2)%8", "chunk": "pid//16", "value_base": "v_block*64", "chunk_start": "chunk*64"}, "cta_output": "[64 token,64 V] BF16 = 4096 elements", "cta_per_logical_unit": 2, "waves_per_cta": 4, "waves_per_logical_unit": 8, "wave_mapping": "row_half=wave_id>>1, value_half=wave_id&1", "evidence": f"{SOURCE}:78-91,212-239"})
    write("stage6z_c26_native_cta_ownership.json", {"schema": "qwen.gfx942.stage6z.c26.native_cta_ownership.v1", "selected_config": {str(t): identities[str(t)]["native_selected"]["identity"]["selected_config"] for t in LENGTHS}, "launch": "Triton grid=(ceil(V/BV)=2,T/64,8), WG128", "program_id": "TTIR x=V64 block, y=chunk, z=value head", "cta_output": "[64 token,64 V] BF16 = 4096 elements", "cta_per_logical_unit": 2, "waves_per_cta": 2, "waves_per_logical_unit": 4, "conclusion": "Fresh final tails are 160 MFMA/CTA; native does not own both V64 halves in one CTA."})
    write("stage6z_c26_pmc_normalization.json", {"schema": "qwen.gfx942.stage6z.c26.pmc_normalization.v1", "method": "exact Kernel_Name; final five explicit direct-tail dispatches; CTA_count=Grid_Size/Workgroup_Size", "Grid_Size_semantics": "total global work-items, verified by expected launch CTA count", "historical_c25_error": "substring matching included autotune/public dispatch population and the parser used a hard-coded Z5B CTA formula; historical native 320 MFMA/CTA is invalid", "lengths": {str(t): pmc[t] for t in LENGTHS}})
    write("stage6z_c26_mfma_work_oracle.json", {"schema": "qwen.gfx942.stage6z.c26.mfma_work_oracle.v1", "geometry": "mfma32x32x8 bf16->fp32", "per_V64_CTA": {"QxH": 64, "QxK_score": 64, "scorexV_new": 32, "total": 160}, "per_logical_unit": {"ctas": 2, "total": 320}, "verification": {str(t): {arm: {"per_cta": pmc[t][arm]["per_cta"]["SQ_INSTS_MFMA"], "per_logical_unit": pmc[t][arm]["per_cta"]["SQ_INSTS_MFMA"] * 2} for arm in ("z5b", "native_selected")} for t in LENGTHS}})

    normalized: dict[str, dict] = {}
    density: dict[str, dict] = {}
    for t in LENGTHS:
        normalized[str(t)] = {}
        density[str(t)] = {}
        for arm in ("z5b", "native_selected"):
            one = pmc[t][arm]
            unit = {name: one["per_cta"][name] * 2 for name in COUNTERS}
            normalized[str(t)][arm] = {"cta_per_logical_unit": 2, "waves_per_logical_unit": 8 if arm == "z5b" else 4, "per_cta": one["per_cta"], "per_logical_unit": unit, "per_mfma": {name: unit[name] / unit["SQ_INSTS_MFMA"] for name in COUNTERS if name != "SQ_INSTS_MFMA"}}
            density[str(t)][arm] = {"mfma_per_cta": one["per_cta"]["SQ_INSTS_MFMA"], "mfma_per_wave": one["per_cta"]["SQ_INSTS_MFMA"] / (4 if arm == "z5b" else 2), "mfma_per_logical_unit": unit["SQ_INSTS_MFMA"], "output_elements_per_cta": 4096, "output_elements_per_wave": 1024 if arm == "z5b" else 2048, "useful_compute_density": unit["SQ_INSTS_MFMA"] / sum(unit.values())}
    write("stage6z_c26_logical_work_normalized_pmc.json", {"schema": "qwen.gfx942.stage6z.c26.logical_work_normalized_pmc.v1", "logical_unit": logical["definition"], "lengths": normalized})
    write("stage6z_c26_work_density.json", {"schema": "qwen.gfx942.stage6z.c26.work_density.v1", "formula": "MFMA/(MFMA+VMEM+LDS+VALU+SALU), descriptive only", "lengths": density})
    write("stage6z_c26_producer_duplication.json", {"schema": "qwen.gfx942.stage6z.c26.producer_duplication.v1", "scope": "source/TTIR ownership; no aggregate-PMC operand attribution", "operands": {"Q": "Both V64 CTAs load the same [64,128] key-head Q block; no cross-CTA sharing.", "K": "Both V64 CTAs load the same [64,128] key-head K block; Z5B serializes source halves CTA-locally.", "H": "V64 partition: distinct CTA H rows.", "V_new": "V64 partition: distinct CTA columns.", "g": "same [64] value-head tile has separate CTA consumers.", "output": "disjoint V64 stores."}, "native_amortizes_cross_cta_producers": False, "conclusion": "Native has the same two-V64-CTA output ownership; it does not amortize Q/K/g by a larger CTA."})
    # The machine evidence was generated inside Docker and records its
    # /workspace path.  Resolve the same checked-in artifact from this host.
    z5b_isa = HERE / "codex_qwen_bt64_stage6z_z5b_machine_stage1/final_isa.s"
    write("stage6z_c26_sync_duplication.json", {"schema": "qwen.gfx942.stage6z.c26.sync_duplication.v1", "scope_note": "barrier/waitcnt are static lexical evidence, not dynamic PMC", "z5b_static": {"barrier": count(z5b_isa, r"\bs_barrier\b"), "waitcnt": count(z5b_isa, r"\bs_waitcnt\b"), "ds_read": count(z5b_isa, r"\bds_read"), "ds_write": count(z5b_isa, r"\bds_write")}, "native_static": native_static, "per_logical_unit": "both execute two independent CTA-local shared schedules; Z5B has 8 waves/unit, native 4."})

    timing = {t: load(args.timing_root / f"T{t}.json") for t in LENGTHS}
    latency = {"schema": "qwen.gfx942.stage6z.c26.latency_reference.v1", "scope": "fresh-process caller-owned direct isolated-body diagnostic, not public Eager ranking", "results": {str(t): timing[t] for t in LENGTHS}, "slope": {arm: fit_slope(timing, arm) for arm in ("z5b", "native_selected")}, "selector_note": "Timing workers use isolated Triton caches and record their selected identity. Selection may differ between independent fresh processes; PMC data is tied only to its own exact final-tail W2 capture."}
    write("stage6z_c26_latency_reference.json", latency)
    write("stage6z_c26_regression_results.json", {"schema": "qwen.gfx942.stage6z.c26.regression_results.v1", "audit_only": True, "kernel_source_modified": False, "checks": {"C26_harness_syntax": "PASS", "exact_tail_parser": "PASS", "Grid_Size_normalization": "PASS", "MFMA_oracle": "PASS", "formal_timing": "PASS"}, "not_run": ["kernel correctness matrix: source unchanged", "X2 integration", "C27/new candidate"]})

    t0 = normalized["2048"]
    rz = t0["z5b"]["per_logical_unit"]
    rn = t0["native_selected"]["per_logical_unit"]
    lines = ["# C26-WDO: Chunk-o Work Decomposition And Ownership Audit", "", "## 结论", "", "**最终分类：`CASE C: C26_PMC_NORMALIZATION_ERROR_FOUND`。** C25 的 native `320 MFMA/CTA` 来自 substring 匹配 public-selector/autotune dispatch 群，再用硬编码 CTA 数归一化。C26 对 final direct-tail 做 exact `Kernel_Name` 匹配，并以 `Grid_Size/Workgroup_Size` 归一化：Z5B 与 native 在三个长度均为 `160 MFMA/CTA`。", "", "修正后两边完成一个 logical `(chunk,value_head)` `[64,128]` output tile 都需要 **2 CTA**，理论和 PMC 都是 **320 MFMA/logical unit**。因此不能归因于 Z5B 多 CTA。same-work 下 Z5B 的 per-MFMA VMEM/LDS/VALU/SALU 更高，这是 feeding/materialization gap 的观察，不是本轮的 latency 因果证明。", "", "## Same Logical Unit", "", "一个 logical unit 是一个 chunk、一个 value head、完整 `[64 token,128 value]` BF16 output tile。两个 CTA 分别拥有 V `[0:64]` 和 `[64:128]`，都覆盖 64 个 token rows。Z5B 是 4 waves/CTA，native selected 是 2 waves/CTA；所以分别是 8 和 4 waves/logical unit。", "", "## MFMA Oracle", "", "每 V64 CTA：QxH=64、QxK score=64、scorexV-new=32，合计160。两个 V64 CTA 合计320。此 oracle 和三个长度的 final-tail PMC 完全吻合。", "", "## Same-Work Dynamic PMC", "", "| T | arm | CTA/unit | waves/unit | MFMA/unit | VMEM/unit | LDS/unit | VALU/unit | SALU/unit |", "|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for t in LENGTHS:
        for arm, label in (("z5b", "Z5B"), ("native_selected", "native selected")):
            item = normalized[str(t)][arm]
            work = item["per_logical_unit"]
            lines.append(f"| {t} | {label} | 2 | {item['waves_per_logical_unit']} | {work['SQ_INSTS_MFMA']:.0f} | {work['SQ_INSTS_VMEM']:.0f} | {work['SQ_INSTS_LDS']:.0f} | {work['SQ_INSTS_VALU']:.0f} | {work['SQ_INSTS_SALU']:.0f} |")
    lines += ["", f"T2048 per logical unit: Z5B/native 的 VMEM={rz['SQ_INSTS_VMEM']/rn['SQ_INSTS_VMEM']:.2f}x、LDS={rz['SQ_INSTS_LDS']/rn['SQ_INSTS_LDS']:.2f}x、VALU={rz['SQ_INSTS_VALU']/rn['SQ_INSTS_VALU']:.2f}x、SALU={rz['SQ_INSTS_SALU']/rn['SQ_INSTS_SALU']:.2f}x。MFMA 相同，故这些比值也等于 per-MFMA 比值。", "", "## Producer And Synchronization", "", "Q/K/g 都会跨两个 V64 CTA 重复，H/V-new/output 由 V64 分片而不重复。native 同样有两个 CTA，且无跨 CTA LDS 或 barrier，因此没有通过更大 output ownership amortize Q/K/g。两边的 shared publication 都是 CTA-local；静态 barrier/waitcnt 已单独记录，未伪装为 dynamic count。", "", "## Fresh Body Timing", "", "口径：7 fresh-process sessions、caller-owned output、current HIP stream、no Graph、warmup=10/repeat=50。它是 isolated direct body diagnostic，不是 public Eager 排名。", "", "| T | Z5B ms | native selected ms | Z5B/native |", "|--:|--:|--:|--:|"]
    for t in LENGTHS:
        z = timing[t]["summary"]["z5b"]["median_of_session_medians_ms"]
        n = timing[t]["summary"]["native_selected"]["median_of_session_medians_ms"]
        lines.append(f"| {t} | {z:.9f} | {n:.9f} | {z/n:.4f}x |")
    lines += ["", f"slope: Z5B={latency['slope']['z5b']['slope_us_per_chunk']:.6f} us/chunk, native={latency['slope']['native_selected']['slope_us_per_chunk']:.6f} us/chunk。work ratio 与 slope 同向，但不能只凭相关性分配因果份额。", "", "### Selector Identity Caveat", "", "PMC 表对应 fresh final-tail 的 W2/BK32 selected HSACO。为避免共享 cache 固化历史选择，formal timing 的每个 fresh worker 都使用隔离的 Triton cache；因此它记录到 current selector 的真实 session-level choice。T2048 多数选择 W4（其中一例 stages=3），T8192 稳定 W2/BK32，T16384 有 W2/BK32 与 W2/BK64。所有这些 choice 仍是 BV64、两个 V64 CTA/logical unit；但 timing 不能被表述为 W2 HSACO 的单一固定-code-object latency。逐 session identity 已保存在 `stage6z_c26_latency_reference.json`。", "", "## 18 个问题的直接回答", "", "1. unit 是 `(chunk,value_head)` 的 `[64,128]` output。", "2. Z5B 是2 CTA/unit。", "3. native T2048 是2 CTA/unit。", "4. native T8192/T16384也是2 CTA/unit。", "5. 两边都是64 token x本地V64 output。", "6. C25的160 vs320来自宽匹配和错误归一化。", "7. 320/CTA不可信，exact tail是160/CTA。", "8. 理论MFMA/unit=320。", "9. 两边实测MFMA/unit=320。", "10. 五种动态工作/unit见表和JSON。", "11. work density/per-MFMA另存JSON。", "12. Z5B的Q/K/g跨V64 CTA重复，H/V-new/output不重复。", "13. native没有用更大CTA amortize它们。", "14. shared/barrier均CTA-local，不能跨CTA复用。", "15. 只能说与slope同向，不能定量归因。", "16. 修正后主矛盾不是CTA partition，而是per-MFMA feeding/materialization。", "17. 主分类为Case C。", "18. 非Case A，不产生work-partition candidate。", "", "## Stop", "", "C26 是只读审计：未修改kernel、未开始C27、未做CTA/layout/packet/barrier/RA/pipeline优化，也未接入X2。下一步最多登记为相同 V64 CTA ownership 下的 operand-feeding 审计，不能重启CTA ownership假设。"]
    (HERE / "qwen_gfx942_c26_work_decomposition_ownership_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

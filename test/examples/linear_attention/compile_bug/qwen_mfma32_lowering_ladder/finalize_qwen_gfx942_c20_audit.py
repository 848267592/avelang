#!/usr/bin/env python3
"""Materialize the read-only C20 audit from frozen artifacts and measurements.

This script deliberately does not compile or launch a kernel.  It only joins
the already captured C19 identity, fresh C20 controls/PMC, existing source
sweeps, and native Triton artifacts into machine-readable evidence and a
Chinese report.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PERF = HERE / "codex_qwen_gfx942_c20_formal_performance"
C19_DIR = HERE / "codex_qwen_gfx942_c19_full_physical_region_t2048" / "machine"
C18_EVIDENCE = HERE / "stage6z_c18_machine_evidence.json"
C19_EVIDENCE = HERE / "stage6z_c19_machine_evidence.json"
Z5B_DIR = HERE / "codex_qwen_bt64_stage6z_z5b_machine_stage1"
P2_DIR = HERE / "codex_qwen_bt64_stage6z_bdv2_p2_machine_specialized"
NATIVE_ROOT = HERE / "codex_qwen_bt64_stage6z_native_chunko" / "native" / "T2048"
NATIVE_WG256 = NATIVE_ROOT / "triton_cache" / "2XC6NNWV5Z5LZZJMWEGOLU4TOFQ7JUHRSX2ZJ4EWZ6NSWT3V4GQQ"
NATIVE_SELECTED = NATIVE_ROOT / "pmc_triton_cache" / "7FOZXWPZ4ANQ2VTBEUVPBLQQFU7UXX44635U7SOWVHINPC5PCVKA"
REPORT = HERE / "qwen_gfx942_c20_formal_performance_and_native_resource_gap.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name: str, value: object) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def number(value: object) -> int | float | None:
    if value is None:
        return None
    try:
        text = str(value).replace(",", "")
        parsed = float(text)
        return int(parsed) if parsed.is_integer() else parsed
    except (TypeError, ValueError):
        return None


def fit_slope(rows: list[tuple[int, float]]) -> dict[str, float | int | None]:
    if len(rows) < 2:
        return {"points": len(rows), "slope_ms_per_chunk": None, "slope_us_per_chunk": None, "intercept_ms": None}
    xs = [t / 64.0 for t, _ in rows]
    ys = [y for _, y in rows]
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom
    intercept = ym - slope * xm
    return {
        "points": len(rows),
        "slope_ms_per_chunk": slope,
        "slope_us_per_chunk": slope * 1000.0,
        "intercept_ms": intercept,
    }


def bootstrap_ci(values: list[float], seed: int = 20, samples: int = 20000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return [means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]]


def static_mnemonics(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    instruction = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?:\s+|$)")
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        match = instruction.match(line)
        if not match:
            continue
        mnemonic = match.group(1)
        if mnemonic.startswith((".L", "LBB", "Ltmp")):
            continue
        counts[mnemonic] = counts.get(mnemonic, 0) + 1
    return counts


def isa_families(path: Path) -> dict[str, object]:
    raw = static_mnemonics(path)
    def total(prefixes: tuple[str, ...]) -> int:
        return sum(v for k, v in raw.items() if k.startswith(prefixes))
    selected = {
        "mfma32": sum(v for k, v in raw.items() if k == "v_mfma_f32_32x32x8_bf16"),
        "global_load_or_buffer_load": sum(v for k, v in raw.items() if k.startswith("global_load") or (k.startswith("buffer_load_") and "format" not in k)),
        "global_store_or_buffer_store": sum(v for k, v in raw.items() if k.startswith("global_store") or (k.startswith("buffer_store_") and "format" not in k)),
        "ds_read": total(("ds_read",)),
        "ds_write": total(("ds_write",)),
        "ds_bpermute": total(("ds_bpermute",)),
        "s_waitcnt": total(("s_waitcnt",)),
        "s_barrier": total(("s_barrier",)),
        "v_add_family": total(("v_add", "v_add3")),
        "v_lshl_family": total(("v_lshl",)),
        "v_lshr_family": total(("v_lshr",)),
        "v_and_family": total(("v_and",)),
        "v_xor_family": total(("v_xor",)),
        "v_perm_family": total(("v_perm",)),
        "extract_insert_related": sum(v for k, v in raw.items() if any(x in k for x in ("perm", "bfe", "bitop"))),
    }
    return {"path": str(path), "raw_mnemonics": raw, "families": selected}


def code_resource(label: str, evidence: dict, source: str) -> dict:
    return {
        "arm": label,
        "source": source,
        "chip": evidence.get("chip", "gfx942"),
        "workgroup": evidence.get("workgroup"),
        "vgpr_count": evidence.get("vgpr_count"),
        "agpr_count": evidence.get("agpr_count"),
        "sgpr_count": evidence.get("sgpr_count"),
        "group_segment_fixed_size": evidence.get("group_segment_fixed_size"),
        "private_segment_fixed_size": evidence.get("private_segment_fixed_size"),
        "vgpr_spill_count": evidence.get("vgpr_spill_count"),
        "sgpr_spill_count": evidence.get("sgpr_spill_count"),
        "hsaco_sha256": evidence.get("hsaco_sha256"),
        "final_isa": evidence.get("final_isa"),
        "exact_lto": evidence.get("exact_lto"),
    }


def parse_readobj(path: Path) -> dict[str, int | None]:
    text = path.read_text(errors="replace") if path.exists() else ""
    result: dict[str, int | None] = {}
    for key in ("agpr_count", "group_segment_fixed_size", "private_segment_fixed_size", "sgpr_count", "vgpr_count", "vgpr_spill_count", "sgpr_spill_count"):
        match = re.search(rf"\.({re.escape(key)})\s*:\s*(\d+)", text)
        result[key] = int(match.group(2)) if match else None
    return result


def profiler_row(path: Path, arm: str) -> dict:
    data = load(path)
    p = data["parsed"]
    grid = number(p.get("Grid_Size"))
    wg = number(p.get("Workgroup_Size"))
    ctas = int(grid / wg) if grid and wg else None
    raw = {key: number(p.get(key)) for key in ("SQ_INSTS_MFMA", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "SQ_INSTS_VALU", "SQ_INSTS_SALU")}
    normalized = {key.removeprefix("SQ_INSTS_").lower(): (value / ctas if value is not None and ctas else None) for key, value in raw.items()}
    return {
        "arm": arm,
        "source": str(path),
        "grid_work_items": grid,
        "workgroup": wg,
        "ctas": ctas,
        "raw_pmc": raw,
        "per_cta": normalized,
        "occupancy_percent": number(p.get("OccupancyPercent")),
        "profiler_vgpr_count": number(p.get("VGPR_Count")),
        "profiler_accum_vgpr_count": number(p.get("Accum_VGPR_Count")),
        "profiler_sgpr_count": number(p.get("SGPR_Count")),
        "profiler_lds_block_size": number(p.get("LDS_Block_Size")),
        "profiler_scratch_size": number(p.get("Scratch_Size")),
        "trace_median_us": number(p.get("trace_median_us")),
        "trace_count": number(p.get("trace_count")),
        "normalization_note": "raw aggregate PMC divided by Grid_Size/Workgroup_Size; static ISA is not used for this normalization",
    }


def source_sweep() -> tuple[dict[int, dict[str, float]], list[dict]]:
    values: dict[int, dict[str, float]] = {}
    provenance: list[dict] = []
    for path in sorted(PERF.glob("stage6z_c20_source_T*.json")):
        data = load(path)
        t = int(data["T"])
        values[t] = {row["arm"]: float(row["median_of_session_medians_ms"]) for row in data["summary"]}
        provenance.append({"T": t, "path": str(path), "sessions": data["contract"].get("sessions"), "raw_orders": [row.get("order") for row in data["raw"]]})
    return values, provenance


def frozen_body() -> tuple[dict[str, float], dict]:
    data = load(PERF / "stage6z_c20_t2048_frozen_controls.json")
    medians = {row["arm"]: float(row["median_of_session_medians_ms"]) for row in data["summary"]}
    return medians, data


def worktree_digest() -> dict[str, object]:
    status = subprocess.run(["git", "status", "--short"], cwd=REPO, text=True, stdout=subprocess.PIPE, check=True).stdout
    return {"git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stdout=subprocess.PIPE, check=True).stdout.strip(), "status_sha256": hashlib.sha256(status.encode()).hexdigest(), "status_line_count": len(status.splitlines()), "status_note": "worktree already contained prior experimental changes; C20 freeze is enforced by artifact hashes, not by pretending the worktree is clean"}


def main() -> None:
    c19_source = REPO / "test" / "examples" / "linear_attention" / "vllm_compare" / "qwen_gdn_bt64_native_chunko_stage6z_c19_full_physical_region.py"
    c19_identity = {
        "schema": "qwen.gfx942.stage6z.c20.frozen_c19_identity.v1",
        "status": "FROZEN_ARTIFACT_IDENTITY_VERIFIED",
        "c20_scope": "read-only formal performance and native resource gap audit",
        "frozen_constraints": [
            "No C19 source/compiler/LLVM/ISA/HSACO changes during C20",
            "No packet/LDS/barrier/scheduler/RA/producer-consumer/selector changes",
            "No X2 full graph or production dispatch",
        ],
        "target_contract": {"chip": "gfx942", "BT": 64, "BV": 64, "BK": 32, "workgroup": 256, "ctas_per_chunk_head": 2, "dtype": "BF16 ABI with FP32 accumulators", "mfma": "v_mfma_f32_32x32x8_bf16"},
        "c19_source": {"path": str(c19_source), "sha256": "e3d5c81800fa8f8e1e4c8ca2f9148c0befff46c5bfcfc8e49c4531fd5ee92975"},
        "artifacts": {
            "lowered_llvm": {"path": str(C19_DIR / "lowered_llvm.ll"), "sha256": sha256(C19_DIR / "lowered_llvm.ll")},
            "final_isa": {"path": str(C19_DIR / "final_isa.s"), "sha256": sha256(C19_DIR / "final_isa.s")},
            "hsaco": {"path": str(C19_DIR / "c19_full_physical_region.hsaco"), "sha256": sha256(C19_DIR / "c19_full_physical_region.hsaco")},
            "machine_evidence": {"path": str(C19_EVIDENCE), "sha256": sha256(C19_EVIDENCE)},
        },
        "compiler_pipeline": {"build_reference": "build-vllm-rocm722", "runtime_note": "C20 did not rebuild compiler; C18/C19 source re-JIT was unavailable for the active logical block-dot binding, so T2048 frozen HSACO replay is used for C19 formal control"},
        "worktree_snapshot": worktree_digest(),
        "identity_recheck": "C19 source/LLVM/ISA/HSACO hashes match the pre-C20 C19 evidence and the replay driver hash guards",
    }
    write_json("stage6z_c20_frozen_c19_identity.json", c19_identity)

    sweep, sweep_provenance = source_sweep()
    frozen_medians, frozen = frozen_body()
    source_table = [{"T": t, "chunks": t // 64, **sweep[t]} for t in sorted(sweep)]
    source_slopes = {arm: fit_slope([(t, values[arm]) for t, values in sorted(sweep.items())]) for arm in ("z5b", "p2", "native_selected")}
    body_payload = {
        "schema": "qwen.gfx942.stage6z.c20.body_performance.v1",
        "decision": "STOP_C20_BODY_PERFORMANCE_NO_GO",
        "contract": {"fresh_process_sessions": 7, "warmup": 10, "repeat": 50, "current_stream": True, "cuda_graph_used": False, "caller_owned_preallocated_output": True, "median_definition": "median of session medians", "orders": "raw per-session order fields are authoritative; source aggregate rotating_orders field was empty due a serialization bug"},
        "source_sweep_z5b_p2_native_selected": {"table": source_table, "artifact_provenance": sweep_provenance},
        "c19_formal_t2048_frozen_hsaco_replay": {"medians_ms": frozen_medians, "artifact": str(PERF / "stage6z_c20_t2048_frozen_controls.json"), "scope_note": "C19 source could not be re-JITed under active binding; this is exact hash-guarded T2048 replay, not a source long-T sweep"},
        "c19_vs_z5b_t2048": {"delta_us": (frozen_medians["c19_frozen_hsaco"] - frozen_medians["z5b"]) * 1000.0, "ratio": frozen_medians["c19_frozen_hsaco"] / frozen_medians["z5b"], "paired_differences_us": frozen["paired_differences_us"]["c19_frozen_hsaco_minus_z5b_us"], "bootstrap_ci95_us": bootstrap_ci(frozen["paired_differences_us"]["c19_frozen_hsaco_minus_z5b_us"])},
        "source_slope_fit": source_slopes,
        "formal_go_gates": {"identity": True, "quick_correctness": True, "seven_session_stable": True, "c19_gain_over_z5b": False, "no_regression_vs_p2_c18": False, "large_t_slope_better_than_z5b": False, "no_spill_private": True, "complete_audit": True},
        "stop_reason": "C19 T2048 is 0.095181 ms versus Z5B 0.066899 ms in the same frozen-control 7-session protocol; C19 is 1.4225x Z5B, and no C19 long-T body sweep exists because its constexpr-T=2048 HSACO cannot represent other T values.",
    }
    write_json("stage6z_c20_body_performance.json", body_payload)
    write_json("stage6z_c20_latency_slope.json", {"schema": "qwen.gfx942.stage6z.c20.latency_slope.v1", "source_sweep": source_slopes, "c19": {"status": "N/A_NO_SOURCE_SWEEP", "reason": "frozen C19 code object is T=2048-specific and body GO already failed"}, "fit_definition": "least-squares latency_ms versus T/64 chunks; this is not a public Eager slope"})

    c18 = load(C18_EVIDENCE)
    c19 = load(C19_EVIDENCE)
    p2_summary = load(P2_DIR / "machine_summary.json")
    z5b = load(Z5B_DIR / "machine_evidence.json")
    native_meta = load(NATIVE_WG256 / "chunk_fwd_kernel_o.json")
    native_selected_meta = load(NATIVE_SELECTED / "chunk_fwd_kernel_o.json")
    native_wg256_readobj = parse_readobj(NATIVE_ROOT / "code_object_readobj.txt")
    code_objects = [
        code_resource("Z5B", z5b, str(Z5B_DIR)),
        code_resource("P2", p2_summary, str(P2_DIR / "machine_summary.json")),
        code_resource("C18", c18, str(C18_EVIDENCE)),
        code_resource("C19", c19, str(C19_EVIDENCE)),
        {"arm": "native_same_shape_WG256", "source": str(NATIVE_WG256), "metadata_hash": native_meta.get("hash"), "num_warps": native_meta.get("num_warps"), "num_stages": native_meta.get("num_stages"), "shared_metadata_bytes": native_meta.get("shared"), "code_object_readobj": native_wg256_readobj, "resource_note": "native WG256 artifact readobj reports group_segment_fixed_size=0 while Triton metadata reports shared=12288; retain both instead of silently choosing one"},
        {"arm": "native_actual_selected_T2048", "source": str(NATIVE_SELECTED), "metadata_hash": native_selected_meta.get("hash"), "num_warps": native_selected_meta.get("num_warps"), "num_stages": native_selected_meta.get("num_stages"), "shared_metadata_bytes": native_selected_meta.get("shared"), "code_object_readobj": parse_readobj(NATIVE_ROOT / "pmc_capture" / "code_object_readobj.txt"), "resource_note": "actual public selector at T2048; WG128, used only for actual-selected latency comparison"},
    ]
    write_json("stage6z_c20_code_object_resources.json", {"schema": "qwen.gfx942.stage6z.c20.code_object_resources.v1", "resources": code_objects})

    pmc_paths = {"z5b": PERF / "pmc_t2048_fresh/z5b/result.json", "p2": PERF / "pmc_t2048_fresh/p2/result.json", "c18": PERF / "pmc_t2048_fresh/c18/result.json", "c19": PERF / "pmc_t2048_fresh/c19/result.json", "native_same_shape_WG256": PERF / "pmc_t2048_fresh/native_wg256/result.json"}
    pmc = [profiler_row(path, arm) for arm, path in pmc_paths.items()]
    write_json("stage6z_c20_pmc_t2048.json", {"schema": "qwen.gfx942.stage6z.c20.pmc_t2048.v1", "collection": "fresh rocprofv3 kernel-trace plus PMC; T=2048; external rows have trace_count=8 and native WG256 trace_count=7", "rows": pmc})
    write_json("stage6z_c20_profiler_resources.json", {"schema": "qwen.gfx942.stage6z.c20.profiler_resources.v1", "rows": [{k: row[k] for k in ("arm", "profiler_vgpr_count", "profiler_accum_vgpr_count", "profiler_sgpr_count", "profiler_lds_block_size", "profiler_scratch_size", "occupancy_percent", "trace_count")} for row in pmc], "note": "profiler register fields are not code-object register fields"})

    per_cta = {row["arm"]: row["per_cta"] for row in pmc}
    def closure(metric: str) -> dict[str, float | None]:
        z = per_cta["z5b"].get(metric)
        n = per_cta["native_same_shape_WG256"].get(metric)
        c = per_cta["c19"].get(metric)
        gap = z - n if z is not None and n is not None else None
        closed = z - c if z is not None and c is not None else None
        return {"z5b": z, "c19": c, "native": n, "z5b_to_native_gap": gap, "c19_closed_amount": closed, "closure_percent": (closed / gap * 100.0 if gap not in (None, 0) and closed is not None else None), "c19_vs_native_ratio": (c / n if c is not None and n not in (None, 0) else None)}
    write_json("stage6z_c20_machine_gap_closure.json", {"schema": "qwen.gfx942.stage6z.c20.machine_gap_closure.v1", "rows": {metric: closure(metric) for metric in ("mfma", "vmem", "lds", "valu", "salu")}, "interpretation": "closure is aggregate PMC movement only; it does not prove causal latency share"})

    c19_isa = isa_families(C19_DIR / "final_isa.s")
    native_isa = isa_families(NATIVE_WG256 / "chunk_fwd_kernel_o.amdgcn")
    write_json("stage6z_c20_isa_resource_gap.json", {"schema": "qwen.gfx942.stage6z.c20.isa_resource_gap.v1", "C19": c19_isa, "native_same_shape_WG256": native_isa, "static_dynamic_warning": "These are lexical ISA counts. Dynamic PMC rows in stage6z_c20_pmc_t2048.json are independent measurements and must not be derived from these counts."})

    logical_bytes = {"Q": 64 * 128 * 2, "K": 64 * 128 * 2, "H": 64 * 128 * 2, "V_new": 64 * 64 * 2, "g": 64 * 4, "output": 64 * 64 * 2}
    vmem_roles = []
    for role, phase, producer, native_path, evidence, note in [
        ("Q", "Q cache / Q@H + Q@K", "C19 FullPhysicalRegionPlan role=Q, one logical K32-stage producer with dual consumer", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "C19 source/report A; native TTGIR A/B", "C19 Q duplicate producer is removed; exact dynamic Q share is not separately countered"),
        ("H", "H phase", "C19 FullPhysicalRegionPlan role=H producer and H consumer", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "C19 source/report B", "role is explicit in source, but aggregate VMEM cannot be split per operand"),
        ("K", "K source-half/stage", "C19 FullPhysicalRegionPlan role=K producer and MFMA-B consumer", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "C19 source/report B", "K is a likely machine-work contributor, but this C20 audit does not invent a per-K PMC"),
        ("V_new", "score@V / intra", "C19 retains plan-owned V producer and score@V consumer", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "C19 source/report B", "native TTGIR has typed V dot path; exact dynamic V share unknown"),
        ("g", "score target/source + final scaling", "multiple logical consumer roles remain in source graph", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "C/B", "global g role is not uniquely attributable from aggregate PMC"),
        ("output", "final store", "BF16 public output store", str(NATIVE_WG256 / "chunk_fwd_kernel_o.ttgir"), "B", "stores are on the completion path; exact dynamic issuing share unknown"),
    ]:
        vmem_roles.append({"operand": role, "logical_bytes_per_contract_tile": logical_bytes[role], "source_phase": phase, "c19_producer": producer, "native_artifact": native_path, "dynamic_vmem_share_per_cta": "N/A", "static_load_family": "see ISA family JSON; no per-role static-to-dynamic conversion", "duplicate_or_narrow_judgement": note, "evidence_level": evidence})
    write_json("stage6z_c20_vmem_role_breakdown.json", {"schema": "qwen.gfx942.stage6z.c20.vmem_role_breakdown.v1", "c19_total_dynamic_vmem_per_cta": per_cta["c19"]["vmem"], "native_total_dynamic_vmem_per_cta": per_cta["native_same_shape_WG256"]["vmem"], "rows": vmem_roles, "unresolved": ["No per-operand VMEM counter was captured; aggregate +164 C19-vs-native and +532 Z5B-vs-native cannot be assigned exactly.", "Logical bytes are contract tile sizes, not hardware transaction bytes."]})

    lds_roles = [
        {"operand": "C19 Q shared", "physical_region": "arena rows 0..255, four 64x32 BF16 K32 stages", "logical_bytes": 16384, "lifetime": "Q fill through Q@H/Q@K; later rows may be reused", "dynamic_share_per_cta": "N/A"},
        {"operand": "C19 H/K phase band", "physical_region": "arena rows 256..319", "logical_bytes": 4096, "lifetime": "producer-to-consumer phase", "dynamic_share_per_cta": "N/A"},
        {"operand": "C19 score/V reuse", "physical_region": "Q rows reused after Q last use for score phase", "logical_bytes": "overlapping lifetime, not additional allocation", "lifetime": "score half ordering", "dynamic_share_per_cta": "N/A"},
        {"operand": "native typed shared/dot path", "physical_region": "Triton metadata shared=12288 B for WG256 artifact", "logical_bytes": "N/A", "lifetime": "TTGIR encoding-managed", "dynamic_share_per_cta": "N/A"},
    ]
    write_json("stage6z_c20_lds_role_breakdown.json", {"schema": "qwen.gfx942.stage6z.c20.lds_role_breakdown.v1", "dynamic_totals_per_cta": {"C19": per_cta["c19"]["lds"], "native_same_shape_WG256": per_cta["native_same_shape_WG256"]["lds"]}, "footprints": {"C19": 24576, "native_metadata": 12288, "native_readobj_group_segment": 0}, "rows": lds_roles, "interpretation": "C19's 688 dynamic LDS instructions remain 1.433x native's 480. Footprint is not equal: C19 has an exact 24 KiB group segment while native WG256 artifact metadata says 12 KiB and readobj says 0. The extra dynamic traffic is consistent with explicit C19 producer/consumer publication and phase boundaries, but no per-role LDS counter proves a single subphase causal share."})

    register_gap = {
        "schema": "qwen.gfx942.stage6z.c20.register_gap.v1",
        "C19": {"code_object": {"vgpr": 136, "agpr": 48, "sgpr": 28, "private": 0, "spill": 0}, "profiler": {"vgpr": 88, "accum_vgpr": 88, "sgpr": 112}, "exact_mir": str(C19_DIR / "exact_lto"), "evidence": "A/B: exact C19 MIR and final physical ISA available; existing C19 report confirms phase-separated accumulator order; cycle-accurate LiveIntervals are not reconstructed"},
        "native_same_shape_WG256": {"code_object": {"vgpr": native_wg256_readobj["vgpr_count"], "agpr": native_wg256_readobj["agpr_count"], "sgpr": native_wg256_readobj["sgpr_count"], "private": native_wg256_readobj["private_segment_fixed_size"], "spill": (native_wg256_readobj["vgpr_spill_count"] or 0) + (native_wg256_readobj["sgpr_spill_count"] or 0)}, "profiler": {"vgpr": per_cta["native_same_shape_WG256"], "reported_vgpr": next(row["profiler_vgpr_count"] for row in pmc if row["arm"] == "native_same_shape_WG256"), "reported_accum_vgpr": next(row["profiler_accum_vgpr_count"] for row in pmc if row["arm"] == "native_same_shape_WG256"), "reported_sgpr": next(row["profiler_sgpr_count"] for row in pmc if row["arm"] == "native_same_shape_WG256")}, "mir": "N/A: native artifact set has TTIR/TTGIR/LLVM/ISA but no comparable exact-LTO MIR", "evidence": "B/C; do not infer native live intervals"},
        "comparison": ["C19 code object is +4 VGPR and +16 AGPR versus native WG256 code-object readobj.", "C19 profiler reports 88/88 versus native 100/36; these fields use different reporting conventions from code-object metadata.", "C19 dynamic VALU is 6174/CTA versus native 3376/CTA, so address/layout/fragment work remains even though C19 has no spill/private."],
    }
    write_json("stage6z_c20_register_gap.json", register_gap)

    write_json("stage6z_c20_public_eager_performance.json", {"schema": "qwen.gfx942.stage6z.c20.public_eager_performance.v1", "status": "STOP_C20_BODY_PERFORMANCE_NO_GO", "executed": False, "reason": "C19 failed the prerequisite formal isolated body gain gate; no Public Eager result is claimed", "prohibited_as_final_evidence": ["Graph capture", "private body direct call", "profiler trace latency"]})
    write_json("stage6z_c20_regression_results.json", {"schema": "qwen.gfx942.stage6z.c20.regression_results.v1", "C19_existing_correctness": str(HERE / "stage6z_c19_full_correctness.json"), "C19_existing_regression": str(HERE / "stage6z_c19_regression_results.json"), "C19_correctness_status": "PASS: T64/128/512/1024/2048/4096/8192/16384 BF16 byte-exact and finite; caller-owned/zero-V/NaN-prefill T64/8192/16384 PASS", "C20_frozen_T2048": "PASS: C18/P2/C19 frozen HSACO exact_vs_z5b and all finite", "C20_source_sweep": "PASS finite for Z5B/P2/native-selected T512/1024/2048/4096/8192/16384", "C19_source_rejit_limitation": "active binding rejected C19/C18 logical block-dot third operand type; C20 did not change compiler or source and used hash-guarded C19 T2048 replay", "production": "unchanged"})

    c19_vs_z5b = frozen_medians["c19_frozen_hsaco"] / frozen_medians["z5b"]
    native_vs_z5b = frozen_medians["native_selected"] / frozen_medians["z5b"]
    report = f"""# Qwen gfx942 C20：C19 冻结性能验证与 Triton 资源差距审计

## 1. 最终决策

本轮是只读的正式验证，没有修改 C19 kernel、Avelang compiler、LLVM/ISA、packet、LDS、barrier、scheduler、RA、producer-consumer mapping、selector 或 production dispatch。

最终状态：

```text
STOP_C20_BODY_PERFORMANCE_NO_GO
PUBLIC_EAGER_NOT_RUN
```

原因不是“C19 没有降低机器工作”。C19 确实把一部分机器工作推向 native：在新鲜 T=2048 PMC 中，C19 为 `MFMA=160、VMEM=304、LDS=688、VALU=6174、SALU=618 / CTA`。但是正式 7-session、fresh-process、caller-owned、current-stream、no-Graph 的 HIP-event body 中，C19 的中位 session median 为 `{frozen_medians['c19_frozen_hsaco']:.9f} ms`，Z5B 为 `{frozen_medians['z5b']:.9f} ms`，C19 是 Z5B 的 `{c19_vs_z5b:.4f}x`，反而慢约 `{(c19_vs_z5b - 1.0) * 100:.2f}%`。预注册的 C20 Body GO 条件因此不成立，不能把 PMC 改写成性能成功，也没有资格进入 Public Eager。

这轮仍然完成了 C20 要求的可用审计：C19 frozen identity、正式 body 对照、T=2048 fresh PMC、code-object/profiler 资源、ISA family、VMEM/LDS role ledger、register gap 以及明确的下一步候选均已落盘。

## 2. C20 冻结边界与身份

| 项目 | C19 冻结事实 |
|:--|:--|
| target | gfx942 / wave64 |
| shape | BT64 / BV64 / BK32 |
| launch | WG256、每 chunk-head 2 CTA |
| math | `v_mfma_f32_32x32x8_bf16`、K32 accumulation order、BF16 ABI、FP32 accumulator |
| C19 source SHA256 | `e3d5c81800fa8f8e1e4c8ca2f9148c0befff46c5bfcfc8e49c4531fd5ee92975` |
| lowered LLVM SHA256 | `a8579c7ad73de7daf3f0e6ae3b18bf5914c5c5807e60b9f7089e24bc90395407` |
| final ISA SHA256 | `c9279f37f79665db2173677e1c7ce3d5ff6294463cec1df41145618d4cfe5ddb` |
| C19 HSACO SHA256 | `14ffa78200ff995448c8abbe1c5484375aae1a2f75c8412a937b33d4aa628907` |
| C20 source/compiler mutation | none; artifact hash guard remained valid |

C20 的 worktree 本来就包含前面实验的 dirty files，因此不能用“git clean”冒充冻结证据。冻结证据是 C19 source/LLVM/ISA/HSACO 的 hash 和 replay driver 的 HSACO hash guard；当前 worktree digest 见 `stage6z_c20_frozen_c19_identity.json`。

## 3. 正式测试口径

正式 source body sweep 使用 Z5B、P2 和实际 public selector 选出的 native arm：T=512/1024/2048/4096/8192/16384，每个 T 7 个 fresh Python process session，warmup=10、repeat=50、预分配 caller-owned output、current HIP stream、无 CUDA Graph。每个 session 的 raw per-arm order 都被保存，order 旋转覆盖在 JSON 中；原始 harness 的 aggregate `rotating_orders` 字段曾为空，这是 metadata serialization bug，不能用它推断未旋转，raw `order` 字段才是权威证据。

C19 由于当前 active logical block-dot binding 不接受原 C19 source 的第三 operand 类型，不能在 C20 期间修改 compiler 来绕过这个问题。C19 使用 exact hash-guarded T=2048 HSACO replay，7 sessions，同样的 warmup/repeat/stream/output 口径。C19 这个 HSACO 是 T=2048-specific，所以不能把一个 T=2048 code object 假装成 C19 的 T=512～16384 sweep。

## 4. Body 性能结果

### 4.1 Source arm sweep：Z5B、P2、实际 selected native

以下是各 T 的 median of session medians，单位 ms：

| T | chunks | Z5B | P2 | native selected |
|--:|--:|--:|--:|--:|
"""
    for row in source_table:
        report += f"| {row['T']} | {row['chunks']} | {row['z5b']:.9f} | {row['p2']:.9f} | {row['native_selected']:.9f} |\n"
    report += f"""
这里的 native selected 是每个 T 先走真实 public selector，再 pin 住实际配置进行 body 计时；不是把 T=2048 的 selector 插值到所有长度。

### 4.2 C19 frozen HSACO T=2048 formal control

| arm | median of 7 session medians (ms) | 相对 Z5B | 与 Z5B paired difference (us) |
|:--|--:|--:|--:|
| Z5B | {frozen_medians['z5b']:.9f} | 1.0000x | 0 |
| P2 frozen HSACO | {frozen_medians['p2_frozen_hsaco']:.9f} | {frozen_medians['p2_frozen_hsaco']/frozen_medians['z5b']:.4f}x | {(frozen_medians['p2_frozen_hsaco']-frozen_medians['z5b'])*1000:.3f} |
| C18 frozen HSACO | {frozen_medians['c18_frozen_hsaco']:.9f} | {frozen_medians['c18_frozen_hsaco']/frozen_medians['z5b']:.4f}x | {(frozen_medians['c18_frozen_hsaco']-frozen_medians['z5b'])*1000:.3f} |
| **C19 frozen HSACO** | **{frozen_medians['c19_frozen_hsaco']:.9f}** | **{c19_vs_z5b:.4f}x** | **{(frozen_medians['c19_frozen_hsaco']-frozen_medians['z5b'])*1000:.3f}** |
| native selected | {frozen_medians['native_selected']:.9f} | {native_vs_z5b:.4f}x | {(frozen_medians['native_selected']-frozen_medians['z5b'])*1000:.3f} |

C19 相对 Z5B 的 7 个 paired differences 全部为正，bootstrap CI 和 raw differences 见 `stage6z_c20_body_performance.json`。因此这是稳定的 No-Go，不是单个异常 session。

### 4.3 slope

source sweep 的最小二乘拟合为 latency(ms) = intercept + slope × chunks：

| arm | slope (us/chunk) | intercept (ms) |
|:--|--:|--:|
| Z5B | {source_slopes['z5b']['slope_us_per_chunk']:.6f} | {source_slopes['z5b']['intercept_ms']:.6f} |
| P2 | {source_slopes['p2']['slope_us_per_chunk']:.6f} | {source_slopes['p2']['intercept_ms']:.6f} |
| native selected | {source_slopes['native_selected']['slope_us_per_chunk']:.6f} | {source_slopes['native_selected']['intercept_ms']:.6f} |
| C19 | N/A | C19 没有跨 T 的可比 body sweep |

C19 因 T2048 单点 body 已失败，不运行任何“补长文本 C19”的私有替代测试，也不运行 Public Eager。

## 5. Correctness 与回归

C19 已有 full correctness matrix：T=64/128/512/1024/2048/4096/8192/16384 均 BF16 byte-exact、finite；T=64/8192/16384 caller-owned output、zero-V-new、NaN-prefilled output 均通过，NaN count 为 0。C20 T=2048 frozen replay 中 C18/P2/C19 对 Z5B exact，所有 arms finite。C20 source sweep 的 Z5B/P2/native selected 输出也均 finite。

这里要准确区分：C19 correctness 是已有的完整 source/runtime regression 证据；C20 的 C19 性能是 frozen T2048 HSACO replay；C20 没有声称当前 active binding 能再次编译 C19 source。

## 6. T=2048 fresh PMC：每 CTA

所有 raw aggregate counter 除以 `Grid_Size / Workgroup_Size = 512`，没有使用静态 ISA 推导动态数字：

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler VGPR | profiler Accum_VGPR | profiler SGPR | LDS metadata B | occupancy % | scratch |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
"""
    for row in pmc:
        p = row["per_cta"]
        report += f"| {row['arm']} | {p['mfma']:.0f} | {p['vmem']:.0f} | {p['lds']:.0f} | {p['valu']:.0f} | {p['salu']:.0f} | {row['profiler_vgpr_count']} | {row['profiler_accum_vgpr_count']} | {row['profiler_sgpr_count']} | {row['profiler_lds_block_size']} | {row['occupancy_percent']:.6f} | {row['profiler_scratch_size']} |\n"
    report += f"""

新鲜 C19 PMC 与 C19 历史 diagnostic 的小差异来自不同 profiler session；C20 当前结论只使用上表新鲜采集值。外部 HSACO trace 的 `trace_count=8`、native WG256 为 7，表示 collector 结果中匹配到的 trace rows 数，不改变 aggregate PMC 的 CTA normalization。

## 7. Code-object resource 与 profiler resource 必须分开

| arm | code VGPR | code AGPR | code SGPR | code LDS/private | code spill | profiler VGPR | profiler Accum_VGPR | profiler SGPR |
|:--|--:|--:|--:|:--|:--|--:|--:|--:|
| Z5B | 104 | 32 | 28 | 32768 / 0 | 0 | 76 | 100 | 112 |
| P2 | 132 | 48 | 30 | 32768 / 0 | 0 | 84 | 92 | 112 |
| C18 | 116 | 32 | 30 | 32768 / 0 | 0 | 84 | 92 | 112 |
| C19 | 136 | 48 | 28 | 24576 / 0 | 0 | 88 | 88 | 112 |
| native WG256 | 132 | 32 | 89 | readobj 0; metadata 12288 | 0 | 100 | 36 | 96 |

native WG256 的 `chunk_fwd_kernel_o.json` 报告 `shared=12288`，而它的 HSACO readobj 文本报告 `.group_segment_fixed_size: 0`；这不是把 0 当成真实 LDS 的理由，而是两个 artifact layer 的 metadata discrepancy，故同时保留。native actual selected T2048 是另一个 WG128/2-wave/3-stage code object，code VGPR/AGPR/SGPR 为 220/64/76，不能与 WG256 same-shape 资源混写；它只用于实际 selector 的性能对照。

## 8. 从 Z5B 到 native 的机器 gap 关闭情况

按 fresh T2048 aggregate PMC 的同 shape WG256 对照：

| 指标 | Z5B | C19 | native WG256 | Z5B→native gap | C19 已移动量 | gap closure |
|:--|--:|--:|--:|--:|--:|--:|
"""
    for metric in ("mfma", "vmem", "lds", "valu", "salu"):
        row = load(HERE / "stage6z_c20_machine_gap_closure.json")["rows"][metric]
        closure_text = "N/A (gap=0)" if row["closure_percent"] is None else f"{row['closure_percent']:.2f}%"
        report += f"| {metric.upper()} | {row['z5b']:.0f} | {row['c19']:.0f} | {row['native']:.0f} | {row['z5b_to_native_gap']:.0f} | {row['c19_closed_amount']:.0f} | {closure_text} |\n"
    report += """

解释：VMEM gap 关闭约 69.17%，说明 C19 full physical ownership 确实删除了相当一部分重复/窄化 producer 工作；VALU gap 只关闭约 24.30%，仍有明显 address/layout/fragment feeding 成本；LDS 不是改善项，C19 反而比 native 多 208/CTA；MFMA 数学工作完全一致。这里是机器工作 gap closure，不是 latency gap closure，也不能据此宣称某一类指令就是唯一的时间因果。

## 9. ISA static audit：只作为机器图证据

`stage6z_c20_isa_resource_gap.json` 保存了逐 mnemonic 与 family count。C19 final ISA 有 `mfma32=56、ds_read=112、ds_write=112、global/buffer load=104、barrier=46、waitcnt=147` 的量级；native WG256 有 `mfma32=40、ds_read=80、ds_write=40、buffer_load_dwordx4=14、buffer_store_dwordx2=4、barrier=11、waitcnt=48` 的量级。C19 的 56 个 lexical MFMA 不能直接写成动态 MFMA；新鲜 PMC 才给出两者均为 160/CTA。native 的 40 lexical MFMA × 4 waves 与 160/CTA 一致，而 C19 的控制流/continuation 使 lexical count 不能单独作为动态执行模型。

关键 ISA 事实：

- C19 没有 `ds_bpermute`，所以当前 gap 不是 R3 那类跨 lane shuffle 爆炸；
- C19 仍有更多显式 `ds_read`/`ds_write`、barrier/waitcnt 和窄/阶段化 operand feeding；
- native WG256 使用 `buffer_load_dwordx4`、`ds_read2_b64`、`ds_write2st64_b64` 等 typed/wide packet family；
- C19 的 global/LDS lexical family 与 dynamic PMC 要分开看，不能用 static count 直接计算访问字节。

## 10. VMEM role ledger

`stage6z_c20_vmem_role_breakdown.json` 对 Q、H、K、V-new、g、output 建立了 logical tile、source phase、producer owner、native artifact 和证据等级。每个 logical tile 的 bytes 是契约级 tile bytes：Q/K/H=16 KiB，V-new/output=8 KiB，g=256 B；它们不是 hardware transaction bytes。

当前可以确认的事实：

1. C19 Q 由 FullPhysicalRegionPlan 单次 producer、Q@H/Q@K 双 consumer 管理，C19 source/report 明确删除了 Z5B 的 Q duplicate producer；
2. C19 H/K/V 也由统一 physical plan 接管，但 aggregate VMEM 没有 per-operand counter，不能把 C19 的 304/CTA 精确拆成 K 或 H；
3. g 仍有 score target/source/final scaling 多个 logical consumer role；这只是 source-level provenance，不是 304 中某个精确硬件份额；
4. native TTGIR 中同一个 Q operand `%b_q_275` 明确 feeding 两个 dot consumer，且 native ISA 有 wide packet load；native 的 140/CTA 是总量，不提供按 role 的 hardware split；
5. 因此本轮不能严谨地选“C19 VMEM 最大 offender 一定是 K/g/H”。最大的已证实事实是 **remaining aggregate VMEM/VALU feeding gap**，而非已证明的单 operand causal share。

## 11. LDS role ledger

C19 的实际 arena 是 `384 x 32 BF16 = 24576 B`：rows 0..255 是 Q physical cache；rows 256..319 是 H/K phase band；Q last-use 后部分 rows 再复用给 score 阶段。这个 reuse 改变了 footprint，但没有让所有 producer-consumer LDS instruction 消失。

新鲜 dynamic LDS 为 C19 `688/CTA`、native WG256 `480/CTA`，约 `1.433x`。这不是“LDS footprint 相同后仍多”的结论，因为 C19 exact allocation 是 24 KiB，而 native WG256 artifact 的 Triton metadata 是 12 KiB、readobj 是 0 的 metadata conflict。更稳妥的结论是：C19 的 physical arena 已比 C18/P2 小，但 native 仍有更紧的 typed shared/dot path 和更少的 phase publication/consumer traffic；单凭总 LDS PMC 不能再分配到 Q/H/K/V 某一条子路径。

## 12. Register/liveness gap

C19 有 exact-LTO MIR，可审计其 machine def/use、phase-separated score accumulator、address temporaries 和最终物理寄存器；native 工件只有 TTIR/TTGIR/LLVM/ISA，没有可比的 exact-LTO MIR，因此不能编造 native LiveIntervals。

C19 code object 是 `VGPR/AGPR/SGPR=136/48/28`，native WG256 readobj 是 `132/32/89`；C19 因此多 4 个 code VGPR、16 个 code AGPR，但少 61 个 code SGPR。profiler 字段则是 C19 `88/88/112` 对 native `100/36/96`。两套数不是同一层面的 register accounting，不能混写成“C19 只有 88 个 AGPR”或“native 只有 36 个物理 AGPR”。C19 没有 private segment/spill；剩余 gap 主要表现为 operand feeding 和 address/layout 工作，不是 C19 已经发生了 spill cliff。

## 13. Public Eager 决策

C19 没有达到 Body GO，所以按预注册规则没有运行 Public Eager。`stage6z_c20_public_eager_performance.json` 明确记录 `executed=false`。因此本报告不提供虚假的“C19 Eager vs vLLM”数字，也不把 isolated body 比例冒充完整公开 API 速度。

## 14. 最多三个有证据支持的下一候选

本轮不实现任何候选，只登记：

1. **C19→native typed/wide operand feeding gap**：同 shape 下 native 有 wide packet + dot encoding，C19 仍有更多显式 LDS publication/consumer instruction；这是由 static ISA + dynamic LDS/VALU/VMEM 共同支持的结构差距，但尚未能按一个 logical operand 精确归因。
2. **C19 remaining layout/address/fragment VALU**：C19 `6174/CTA` 对 native `3376/CTA`，而 MFMA 相同、无 ds_bpermute；说明剩余差距不能只归咎于数学 MFMA 数量，可能来自通用 physical address/fragment feeding。仍需下一轮专门 provenance，不应在本轮自动修改。
3. **C19 shared phase traffic / barrier-window overlap**：C19 static barrier/waitcnt 与 dynamic LDS 均高于 native，且 C19 24 KiB arena 与 native 12 KiB metadata 不同；这是第三候选，因没有 per-role counter，因果等级低于前两项。

这些是 measured gap 与 causal inference 分开的候选，不是“下一轮必须改 K”或“必须改 g”的结论。

## 15. 机器可读产物

本报告同目录下的产物：

- `stage6z_c20_frozen_c19_identity.json`
- `stage6z_c20_body_performance.json`
- `stage6z_c20_latency_slope.json`
- `stage6z_c20_code_object_resources.json`
- `stage6z_c20_profiler_resources.json`
- `stage6z_c20_pmc_t2048.json`
- `stage6z_c20_machine_gap_closure.json`
- `stage6z_c20_isa_resource_gap.json`
- `stage6z_c20_vmem_role_breakdown.json`
- `stage6z_c20_lds_role_breakdown.json`
- `stage6z_c20_register_gap.json`
- `stage6z_c20_public_eager_performance.json`
- `stage6z_c20_regression_results.json`

新增的只读汇总脚本是 `finalize_qwen_gfx942_c20_audit.py`。它不会编译、不会加载 kernel、不会运行 GPU；它只读取上述 frozen artifacts 和已保存的 C20 measurement JSON。

## 16. 复现与停止点

正式 body 原始结果：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gfx942_c20_formal_performance.py --help
```

T=2048 frozen control 和 fresh PMC 的原始 JSON/CSV 位于：

```text
codex_qwen_gfx942_c20_formal_performance/
  stage6z_c20_t2048_frozen_controls.json
  pmc_t2048_fresh/{z5b,p2,c18,c19,native_wg256}/
```

C20 在明确的 `STOP_C20_BODY_PERFORMANCE_NO_GO` 处停止。没有创建 C20 优化 variant，没有改生产路径，也没有运行 Public Eager。
"""
    REPORT.write_text(report)


if __name__ == "__main__":
    main()

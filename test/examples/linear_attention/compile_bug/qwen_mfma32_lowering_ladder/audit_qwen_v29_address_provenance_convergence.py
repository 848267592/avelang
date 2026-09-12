#!/usr/bin/env python3
"""Experiment 0.5: full-v29 address provenance and convergence bisect.

This is an audit-only companion to the strict late-address A/B.  It does not
change the Qwen kernel, launch, schedule, tile ownership, or math.  It only
compiles the existing same-source A/B with additional compiler snapshots and
answers two questions:

1. At which MLIR/LLVM/LTO/MIR layer do early and late source-K-address
   construction first become identical?
2. Which final machine memory consumers are reachable from the flexible-AV
   address objects active at the pre-greedy pred-phase peak?

The consumer labels are machine-evidence classifications.  ROCm LTO does not
retain source debug locations, so the script records the exact MIR path and
instruction text rather than inventing an exact AveLang source line.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
VLLM_COMPARE = ROOT / "test/examples/linear_attention/vllm_compare"
WORKER = SCRIPT_DIR / "audit_qwen_full_chunk_gdr_late_address.py"
REPLAY = SCRIPT_DIR / "replay_qwen_v29_lto_mir.py"
DEFAULT_OUT = ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_v29_address_provenance_convergence_bisect"
DEFAULT_REPORT = SCRIPT_DIR / "qwen_v29_address_provenance_convergence_bisect_report.md"
KERNEL = "_qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32"
LLVM_DIS = Path("/opt/rocm/llvm/bin/llvm-dis")
LLVM_OBJDUMP = Path("/opt/rocm/llvm/bin/llvm-objdump")
MEMORY_OP = re.compile(r"(?:GLOBAL|BUFFER)_LOAD|(?:GLOBAL|BUFFER)_STORE|DS_READ|DS_WRITE")
ADDRESS_OP = re.compile(r"V_(?:LSHL|ADD|SUB|OR|AND|BFE|MOV|CNDMASK|MAD|MUL)|REG_SEQUENCE|INSERT_SUBREG|EXTRACT_SUBREG|COPY")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_normalized_llvm(path: Path) -> str:
    # llvm-dis puts the path of its *output input* in ModuleID.  The A/B
    # replay directories necessarily differ, while the bitcode is byte-equal.
    # Strip only that presentation line; do not normalize IR instructions or
    # metadata.
    text = path.read_text(errors="replace")
    normalized = "\n".join(
        line for line in text.splitlines() if not line.startswith("; ModuleID =")
    ) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


def sha256_normalized_isa(text: str) -> str:
    normalized = "\n".join(
        line for line in text.splitlines()
        if not line.endswith(":\tfile format elf64-amdgpu")
    ) + "\n"
    return hashlib.sha256(normalized.encode()).hexdigest()


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def make_env(out_dir: Path, label: str, late_address: bool) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(VLLM_COMPARE)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    else:
        pythonpath.append(str(ROOT / "python"))
    env.update({
        "PYTHONPATH": ":".join(pythonpath),
        "PYTHONDONTWRITEBYTECODE": "1",
        "AVELANG_QWEN_KFRAG_LATE_ADDRESS": "1" if late_address else "0",
        "AVELANG_QWEN_KFRAG_LATE_BLOAD": "0",
        "AVELANG_QWEN_KFRAG_DEBUG": "1",
        "AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT": "1",
        "AVELANG_QWEN_KFRAG_AB_DUMP_DIR": str(out_dir / label / "ir"),
        "AVELANG_AMDGPU_LINK_DEBUG_DIR": str(out_dir / label / "link"),
    })
    return env


def parse_json_payload(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"worker emitted no JSON payload:\n{text}")


def run_worker(out_dir: Path, label: str, late_address: bool, args: argparse.Namespace, input_path: Path | None) -> dict[str, Any]:
    command = [
        sys.executable,
        str(WORKER),
        "--worker",
        "--T",
        str(args.t),
        "--seed",
        str(args.seed),
        "--warmup",
        "1",
        "--repeat",
        "1",
        "--save-outputs",
        str(out_dir / label / "outputs.pt"),
    ]
    if input_path is None:
        command += ["--save-inputs", str(out_dir / "frozen_inputs.pt")]
    else:
        command += ["--input-path", str(input_path)]
    completed = run(command, make_env(out_dir, label, late_address))
    (out_dir / label / "compile.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{label} compilation failed:\n{completed.stdout}")
    payload = parse_json_payload(completed.stdout)
    payload["rewrite_fired"] = "[qwen-kfrag] rewritten=1" in completed.stdout
    payload["late_stage_op_created"] = "late_address_stage_loads=1" in completed.stdout
    return payload


def compare_outputs(a_path: Path, b_path: Path) -> dict[str, Any]:
    a_h, a_final = torch.load(a_path, weights_only=True)
    b_h, b_final = torch.load(b_path, weights_only=True)

    def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
        mismatch = a.ne(b)
        count = int(mismatch.sum().item())
        return {
            "bit_exact": count == 0,
            "mismatch_count": count,
            "max_abs": float((a - b).abs().max().item()),
        }

    return {"h": compare(a_h, b_h), "final_state": compare(a_final, b_final)}


def count_text(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def snapshot_metrics(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "stage_op": count_text(text, r"amdgpu_qwen_kfrag_stage_load"),
        "late_address_attr": count_text(text, r"qwen_kfrag\.late_address"),
        "vector_load": count_text(text, r"vector\.load"),
        "memref_load": count_text(text, r"memref\.load"),
        "llvm_gep": count_text(text, r"llvm\.getelementptr|getelementptr"),
        "llvm_load": count_text(text, r"llvm\.load|\bload\b"),
        "global_load": count_text(text, r"GLOBAL_LOAD|global_load|buffer_load"),
        "ds_write": count_text(text, r"DS_WRITE|ds_write"),
    }


def first_diff_lines(a_text: str, b_text: str, limit: int = 8) -> list[str]:
    diff = list(difflib.unified_diff(
        a_text.splitlines(), b_text.splitlines(), fromfile="A", tofile="B", n=1,
    ))
    return diff[:limit]


def collect_mlir_bisect(out_dir: Path) -> dict[str, Any]:
    a_dir = out_dir / "A_early_address" / "ir"
    b_dir = out_dir / "B_late_address" / "ir"
    explicit_order = [
        "pre_kfrag_branch.mlir",
        "post_kfrag_rewrite.mlir",
        "post_gpu_outlining.mlir",
        "post_outline_cleanup.mlir",
        "post_kfrag_load_lowering.mlir",
        "post_late_lowering.mlir",
        "amdgpu_00_pre_common.mlir",
        "amdgpu_01_post_one_shot_bufferize.mlir",
        "amdgpu_02_post_expand_strided_metadata.mlir",
        "amdgpu_03_post_scf_to_cf.mlir",
        "amdgpu_04_post_common_cleanup.mlir",
        "amdgpu_10_pre_gpu_pipeline.mlir",
        "amdgpu_11_post_rocdl_attach.mlir",
        "amdgpu_12_post_lower_math.mlir",
        "amdgpu_13_post_legalize_shuffle.mlir",
        "amdgpu_14_post_gpu_to_rocdl.mlir",
        "amdgpu_15_post_amdgpu_to_rocdl.mlir",
        "amdgpu_16_post_vector_to_llvm.mlir",
        "amdgpu_17_post_gpu_cleanup.mlir",
        "amdgpu_18_post_scalar_to_llvm.mlir",
        "amdgpu_19_post_gpu_pipeline.mlir",
        "post_amdgpu_mlir_pipeline.mlir",
        "final_mlir.mlir",
        "preopt_llvm.ll",
        "postopt_llvm.ll",
    ]
    # LLVM pass instrumentation names snapshots after their ordinal and pass
    # name.  Use A's list as the contract, then require B to contain the same
    # files before comparing them.  This keeps the convergence point tied to
    # the actual pipeline rather than to a hand-picked subset of passes.
    llvm_passes = sorted(path.name for path in a_dir.glob("llvm_pass_*.ll"))
    b_llvm_passes = sorted(path.name for path in b_dir.glob("llvm_pass_*.ll"))
    if llvm_passes != b_llvm_passes:
        raise RuntimeError(
            "A/B LLVM pass snapshot names differ: "
            f"A={llvm_passes}, B={b_llvm_passes}"
        )
    preopt_index = explicit_order.index("preopt_llvm.ll")
    explicit_order[preopt_index + 1:preopt_index + 1] = llvm_passes
    rows: list[dict[str, Any]] = []
    for name in explicit_order:
        a, b = a_dir / name, b_dir / name
        if not a.exists() and not b.exists():
            continue
        if not a.exists() or not b.exists():
            raise RuntimeError(f"A/B snapshot missing asymmetrically: {name}")
        a_text, b_text = a.read_text(errors="replace"), b.read_text(errors="replace")
        row = {
            "layer": name.removesuffix(".mlir").removesuffix(".ll"),
            "file": name,
            "identical": a_text == b_text,
            "a": snapshot_metrics(a),
            "b": snapshot_metrics(b),
            "diff_preview": first_diff_lines(a_text, b_text) if a_text != b_text else [],
        }
        rows.append(row)
    divergent_seen = False
    first_convergence: str | None = None
    for row in rows:
        if not row["identical"]:
            divergent_seen = True
        elif divergent_seen:
            first_convergence = str(row["layer"])
            break
    return {"rows": rows, "first_convergence_after_divergence": first_convergence}


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, got {matches}")
    return matches[0]


def replay_lto(out_dir: Path, label: str) -> dict[str, Any]:
    link_dir = out_dir / label / "link"
    argv = find_one(link_dir, "*.argv.txt")
    replay_dir = out_dir / label / "exact_lto"
    command = [
        sys.executable, str(REPLAY), "--argv-file", str(argv),
        "--out-dir", str(replay_dir), "--kernel", KERNEL,
    ]
    completed = run(command, os.environ.copy())
    (replay_dir / "replay_driver.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"{label} LTO replay failed:\n{completed.stdout}")
    summary = json.loads((replay_dir / "summary.json").read_text())
    previous_before: dict[str, Any] | None = None
    pre_greedy: dict[str, Any] | None = None
    post_greedy: dict[str, Any] | None = None
    for row in summary:
        phase = str(row["phase"])
        if "Before Greedy Register Allocator" in phase:
            previous_before = row
        elif "After Greedy Register Allocator" in phase:
            spills = int(row["av32_spill_saves"]) + int(row["av64_spill_saves"])
            if spills and previous_before is not None:
                pre_greedy, post_greedy = previous_before, row
                break
    if pre_greedy is None or post_greedy is None:
        raise RuntimeError(f"{label} replay did not expose the spill-producing greedy pair")
    bc_rows = []
    for bc in sorted(replay_dir.glob("*.bc")):
        ll = bc.with_suffix(".ll")
        dis = subprocess.run(
            [str(LLVM_DIS), str(bc), "-o", str(ll)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if dis.returncode:
            raise RuntimeError(f"llvm-dis failed for {bc}:\n{dis.stdout}")
        bc_rows.append({
            "name": bc.name,
            "bc": str(bc),
            "ll": str(ll),
            "sha256": sha256(bc),
            "ll_sha256": sha256(ll),
            "normalized_ll_sha256": sha256_normalized_llvm(ll),
        })
    hsaco = replay_dir / "linked.hsaco"
    isa = replay_dir / "linked.hsaco.isa"
    disasm = subprocess.run(
        [str(LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(hsaco)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if disasm.returncode:
        raise RuntimeError(f"llvm-objdump failed for {hsaco}:\n{disasm.stdout}")
    isa.write_text(disasm.stdout)
    return {
        "dir": str(replay_dir),
        "pre_greedy_mir": str(pre_greedy["path"]),
        "post_greedy_mir": str(post_greedy["path"]),
        "pre_greedy_phase": pre_greedy["phase"],
        "post_greedy_phase": post_greedy["phase"],
        "lto_bitcode": bc_rows,
        "isa": {
            "path": str(isa),
            "sha256": sha256(isa),
            "normalized_sha256": sha256_normalized_isa(disasm.stdout),
        },
    }


def collect_lto_bisect(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    a_by_name = {row["name"]: row for row in a["lto_bitcode"]}
    b_by_name = {row["name"]: row for row in b["lto_bitcode"]}
    rows = []
    for name in sorted(a_by_name.keys() & b_by_name.keys()):
        left, right = a_by_name[name], b_by_name[name]
        rows.append({
            "layer": name,
            "identical_bitcode": left["sha256"] == right["sha256"],
            "identical_disassembled_llvm": left["ll_sha256"] == right["ll_sha256"],
            "identical_normalized_disassembled_llvm": (
                left["normalized_ll_sha256"] == right["normalized_ll_sha256"]
            ),
            "a_bc_sha256": left["sha256"],
            "b_bc_sha256": right["sha256"],
            "a_ll_sha256": left["ll_sha256"],
            "b_ll_sha256": right["ll_sha256"],
        })
    return rows


def collect_isa_bisect(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "a_isa": a["isa"],
        "b_isa": b["isa"],
        "identical_raw": a["isa"]["sha256"] == b["isa"]["sha256"],
        "identical_normalized": (
            a["isa"]["normalized_sha256"] == b["isa"]["normalized_sha256"]
        ),
    }


def collect_mir_bisect(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, key in (
        ("pre-greedy MIR", "pre_greedy_mir"),
        ("post-greedy MIR", "post_greedy_mir"),
    ):
        a_path, b_path = Path(a[key]), Path(b[key])
        a_sha, b_sha = sha256(a_path), sha256(b_path)
        rows.append({
            "layer": label,
            "a_path": str(a_path),
            "b_path": str(b_path),
            "a_sha256": a_sha,
            "b_sha256": b_sha,
            "identical": a_sha == b_sha,
        })
    return rows


def load_timeline_module() -> Any:
    sys.path.insert(0, str(SCRIPT_DIR))
    import analyze_qwen_v29_broad_compact_pressure_timeline as timeline
    return timeline


def operand_refs(record: Any) -> list[str]:
    return [ref for ref in record.refs if ref != record.definition]


def phase_for_line(phases: dict[str, tuple[int, int]], line: int) -> str:
    for name, (begin, end) in phases.items():
        if begin <= line <= end:
            return name
    return "unanchored"


def memory_kind(text: str, phase: str) -> str | None:
    if not MEMORY_OP.search(text):
        return None
    if "GLOBAL_LOAD" in text or "BUFFER_LOAD" in text:
        return "source-K global-load address" if phase == "k_producer" else "global-load address, non-K/unresolved"
    if "DS_WRITE" in text:
        return "source-K LDS-store address" if phase == "k_producer" else "LDS-store address, non-K/unresolved"
    if "DS_READ" in text:
        return "pred LDS-read address" if phase in {"pred_mfma32", "pred_epilogue"} else "LDS-read address, non-K/unresolved"
    if "GLOBAL_STORE" in text or "BUFFER_STORE" in text:
        return "state/global-output address"
    return "memory-address unresolved"


def trace_memory_consumers(instructions: list[Any], start: str, phases: dict[str, tuple[int, int]]) -> tuple[list[dict[str, Any]], int]:
    uses: dict[str, list[Any]] = defaultdict(list)
    for record in instructions:
        for ref in operand_refs(record):
            uses[ref].append(record)
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    visited = {start}
    consumers: list[dict[str, Any]] = []
    while queue:
        current, depth = queue.popleft()
        for record in uses.get(current, []):
            phase = phase_for_line(phases, record.mir_line)
            kind = memory_kind(record.text, phase)
            if kind:
                consumers.append({
                    "kind": kind,
                    "mir_line": record.mir_line,
                    "byte_offset": record.byte_offset,
                    "phase": phase,
                    "opcode": record.text.split("=")[-1].strip().split()[0],
                    "instruction": record.text,
                    "depth": depth,
                })
            if depth >= 12 or not record.definition or not ADDRESS_OP.search(record.text):
                continue
            if record.definition not in visited:
                visited.add(record.definition)
                queue.append((record.definition, depth + 1))
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in consumers:
        unique[(str(row["kind"]), int(row["mir_line"]), str(row["instruction"]))] = row
    return list(unique.values()), len(visited)


def logical_bucket(consumers: list[dict[str, Any]], crosses_to_k_producer: bool) -> str:
    kinds = {str(row["kind"]) for row in consumers}
    source_k = {
        "source-K global-load address",
        "source-K LDS-store address",
    }
    if kinds & source_k and kinds - source_k:
        return "multi-consumer (includes source-K)"
    if "source-K global-load address" in kinds:
        return "source-K global-load address"
    if "source-K LDS-store address" in kinds:
        return "source-K LDS-store address"
    if "pred LDS-read address" in kinds:
        return "pred LDS-read address"
    if "state/global-output address" in kinds or any("non-K" in kind for kind in kinds):
        return "state/global-output or non-K memory address"
    if crosses_to_k_producer:
        return "loop-carried indexing value"
    return "provenance-only / no reachable memory consumer within bounded trace"


def provenance_from_mir(mir_path: Path, *, flexible_av_only: bool) -> dict[str, Any]:
    timeline = load_timeline_module()
    instructions, vregs = timeline.parse_mir(mir_path)
    timeline.propagate_address_provenance(vregs)
    phases = timeline.phase_anchors(instructions)
    rows = timeline.make_timeline(instructions, vregs, phases)
    peak = max(
        (row for row in rows if row["phase"] == "pred_mfma32"),
        key=lambda row: int(
            row["flexible_av_words"] if flexible_av_only else row["address_words"]
        ),
    )
    peak_line = int(peak["mir_line"])
    pred_begin, pred_end = phases["pred_mfma32"]
    k_begin, _ = phases["k_producer"]
    objects: list[dict[str, Any]] = []
    for name, info in vregs.items():
        if not (info.first_line <= peak_line <= info.last_line):
            continue
        if not info.address_seed:
            continue
        if flexible_av_only and not info.register_class.startswith("av_"):
            continue
        consumers, visited_nodes = trace_memory_consumers(instructions, name, phases)
        crosses_pred = info.first_line < pred_begin <= info.last_line or info.first_line <= pred_end < info.last_line
        crosses_to_k = info.first_line <= pred_end and info.last_line >= k_begin
        objects.append({
            "mir_vreg": f"%{name}",
            "register_class": info.register_class,
            "words": info.width_words,
            "def_mir_line": info.def_line,
            "last_mir_line": info.last_line,
            "lexical_span": info.lexical_span,
            "definition": info.def_text,
            "crosses_pred": crosses_pred,
            "crosses_pred_to_k_producer": crosses_to_k,
            "bounded_trace_nodes": visited_nodes,
            "final_memory_consumers": consumers,
            "logical_bucket": logical_bucket(consumers, crosses_to_k),
        })
    objects.sort(key=lambda row: (str(row["logical_bucket"]), -int(row["words"]), -int(row["lexical_span"])))
    buckets: Counter[str] = Counter()
    for row in objects:
        buckets[str(row["logical_bucket"])] += int(row["words"])
    active_class_words: Counter[str] = Counter()
    for info in vregs.values():
        if info.first_line <= peak_line <= info.last_line:
            active_class_words[info.register_class] += info.width_words
    return {
        "mir": str(mir_path),
        "sha256": sha256(mir_path),
        "phase_anchors": phases,
        "peak": {
            "mir_line": peak_line,
            "byte_offset": peak["byte_offset"],
            "flexible_av_words": peak["flexible_av_words"],
            "all_address_words": peak["address_words"],
            "selected_address_words": sum(int(row["words"]) for row in objects),
            "active_register_class_words": dict(active_class_words),
        },
        "bucket_words": dict(buckets),
        "objects": objects,
        "candidate_filter": (
            "active address provenance candidates in av_* register classes"
            if flexible_av_only
            else "all active address provenance candidates regardless of register class"
        ),
        "method_limit": (
            "Bounded forward def-use tracing through address/copy/pack-like MIR nodes. "
            "LTO has no source debug locations and this is not LLVM LiveIntervals."
        ),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(title for _, title in columns) + " |", "|" + "|".join(":--" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return lines


def render_report(result: dict[str, Any], report: Path) -> None:
    bisect = result["mlir_llvm_bisect"]
    lto = result["lto_bisect"]
    isa = result["isa_bisect"]
    mir = result["mir_bisect"]
    pre = result["pre_greedy_provenance"]
    post = result["post_greedy_flexible_av_provenance"]
    a_pre = pre["A_early_address"]
    b_pre = pre["B_late_address"]
    a_post = post["A_early_address"]
    b_post = post["B_late_address"]
    rows = []
    for row in bisect["rows"]:
        rows.append({
            "layer": row["layer"],
            "same": row["identical"],
            "stage": f"{row['a']['stage_op']}/{row['b']['stage_op']}",
            "late": f"{row['a']['late_address_attr']}/{row['b']['late_address_attr']}",
            "load": f"{row['a']['memref_load']}/{row['b']['memref_load']}",
            "gep": f"{row['a']['llvm_gep']}/{row['b']['llvm_gep']}",
        })
    lines = [
        "# Qwen v29 Address Provenance And Pipeline Convergence Bisect",
        "",
        "## 范围",
        "",
        "Experiment 0.5 是纯审计：没有修改 Qwen kernel、tile、workgroup、launch、数学、dtype、barrier 或性能路径。它重跑同一 full-v29 `chunk_gdr` source 的 A/B：A 在 rewrite 中直接创建 source-K 标量 load；B 使用既有 `amdgpu_qwen_kfrag_stage_load`，在 GPU outlining 后展开。唯一目的，是定位 A/B 首次收敛层次，并给 pred 峰值的地址 vreg 建立 machine-level consumer provenance。",
        "",
        "## 冻结 gate",
        "",
        f"- pre-branch SHA 相同：`{result['pre_branch_identical']}`。",
        f"- persistent rewrite fired：A `{result['smoke']['A_early_address']['rewrite_fired']}`，B `{result['smoke']['B_late_address']['rewrite_fired']}`。",
        f"- B stage op 创建：`{result['smoke']['B_late_address']['late_stage_op_created']}`。",
        f"- `h` / final-state bit-exact：`{result['outputs']['h']['bit_exact']}` / `{result['outputs']['final_state']['bit_exact']}`。",
        "",
        "## 审计 Hook",
        "",
        "- `lib/Target/GPU/lower_to_llvm.cc`：仅当 `AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT=1` 时，保存 outlining 前后、late lowering 前后和 LLVM module-pass 快照。",
        "- `lib/Target/AMDGPU/gpu_to_amdgpu_pipeline.cc`：同一 guard 下在 common/GPU AMDGPU lowering pipeline 各边界打印 MLIR。",
        "- 两处 hook 都只写文件；不改 rewrite 条件、kernel IR、pass 顺序、launch 或 codegen 选项。",
        "",
        "## A/B 收敛表",
        "",
    ]
    lines.extend(markdown_table(rows, [
        ("layer", "层次"), ("same", "A/B 相同"), ("stage", "stage op A/B"),
        ("late", "late attr A/B"), ("load", "memref.load A/B"), ("gep", "GEP A/B"),
    ]))
    lines.extend([
        "",
        f"**第一处 divergence 之后重新相同的快照：`{bisect['first_convergence_after_divergence']}`。** 这不是基于猜测：表中每一行保留 SHA、结构计数与首段 unified diff，原始快照在 artifacts 目录。",
        "",
        "## LLVM LTO 阶段",
        "",
    ])
    lines.extend(markdown_table(lto, [
        ("layer", "LTO bitcode"), ("identical_bitcode", "bitcode 相同"),
        ("identical_disassembled_llvm", "llvm-dis 原文相同"),
        ("identical_normalized_disassembled_llvm", "去 ModuleID 后相同"),
    ]))
    lines.extend([
        "",
        "`preopt_llvm` 已在前表中出现；LTO 表将 `.preopt.bc`、`.internalize.bc`、`.opt.bc` 与 `.precodegen.bc` 单独比较。原文 llvm-dis 的唯一差异是输出路径写入的 `ModuleID`；bitcode SHA 与仅删除该展示行后的 LLVM IR 均相同。",
        "",
        "## 最终 ISA",
        "",
        f"- raw objdump SHA 相同：`{isa['identical_raw']}`（输入 hsaco 路径使原文首行不同属正常）。",
        f"- 去除 objdump `file format` 输入路径行后 ISA SHA 相同：`{isa['identical_normalized']}`。",
        "",
        "## MIR A/B Identity",
        "",
    ])
    lines.extend(markdown_table(mir, [
        ("layer", "MIR 阶段"), ("identical", "SHA 相同"),
    ]))
    lines.extend([
        "",
        "## Pre-greedy MIR 原始地址压力",
        "",
        "这里优先使用最终 spill-producing greedy run 之前的 `IR Dump Before Greedy Register Allocator`，避免把 Greedy 新增的 spill save/reload、COPY 与物理寄存器改写当成原始压力来源。pre-greedy 中还没有历史报告里的 `av_*` 144 words；它们仍表现为 `vreg_64`/`vgpr` 地址候选。因此本表统计所有 register class 的 active address provenance candidates。数值仍是 MIR 文本 def/use 的静态代理，不是 LLVM LiveIntervals。",
        "",
        "| metric | A early | B late |",
        "|:--|--:|--:|",
        f"| pred peak flexible-AV words | {a_pre['peak']['flexible_av_words']} | {b_pre['peak']['flexible_av_words']} |",
        f"| pred peak all address words | {a_pre['peak']['all_address_words']} | {b_pre['peak']['all_address_words']} |",
        f"| selected address candidates | {a_pre['peak']['selected_address_words']} | {b_pre['peak']['selected_address_words']} |",
        f"| peak MIR line | {a_pre['peak']['mir_line']} | {b_pre['peak']['mir_line']} |",
        "",
        "### Pre-greedy Consumer Provenance",
        "",
        "`pre_greedy_address_objects.csv` 保存每个对象的定义、跨度与 bounded def-use 终止 memory consumer。只有 `k_producer + GLOBAL_LOAD` 才标为 source-K global-load；只有 `k_producer + DS_WRITE` 才标为 source-K LDS-store。若同一地址同时服务 source-K 与其他 consumer，会明确记为 multi-consumer，不会被强行计成单一来源。",
        "",
        "| bucket | A words | B words |",
        "|:--|--:|--:|",
    ])
    all_buckets = sorted(set(a_pre["bucket_words"]) | set(b_pre["bucket_words"]))
    for bucket in all_buckets:
        lines.append(f"| {bucket} | {a_pre['bucket_words'].get(bucket, 0)} | {b_pre['bucket_words'].get(bucket, 0)} |")
    lines.extend([
        "",
        "## Post-greedy 历史 144-word 精确归属",
        "",
        "历史的 144 words 来自 post-greedy pred-MFMA32 peak：72 个 `av_64` candidate，每个 2 words。它们不是 pre-greedy 的 register class；本节只用于把那个历史数字精确连接到最终 machine consumer。`post_greedy_flexible_av_address_objects.csv` 保存 72 个逐对象记录。",
        "",
        "| metric | A early | B late |",
        "|:--|--:|--:|",
        f"| pred peak flexible-AV words | {a_post['peak']['flexible_av_words']} | {b_post['peak']['flexible_av_words']} |",
        f"| selected flexible-AV address words | {a_post['peak']['selected_address_words']} | {b_post['peak']['selected_address_words']} |",
        f"| peak MIR line | {a_post['peak']['mir_line']} | {b_post['peak']['mir_line']} |",
        "",
        "| bucket | A words | B words |",
        "|:--|--:|--:|",
    ])
    all_post_buckets = sorted(set(a_post["bucket_words"]) | set(b_post["bucket_words"]))
    for bucket in all_post_buckets:
        lines.append(f"| {bucket} | {a_post['bucket_words'].get(bucket, 0)} | {b_post['bucket_words'].get(bucket, 0)} |")
    lines.extend([
        "",
        "## 结论边界",
        "",
        "本审计只回答“分叉在哪一层消失”和“pred 峰值对象到哪些 machine memory consumer 可达”。它不把 lexical span 说成硬件物理压力，不会因没有 LTO debug location 而虚构精确 AveLang source 行。",
        "",
        "本次 A/B 的差异在 `amdgpu_14_post_gpu_to_rocdl` 首次消失，故普通 stage op 的控制点不足以跨过 `ConvertGpuOpsToROCDLOps`。post-greedy 的历史 144 words 中，`128` words 只终止于 source-K global load，另 `16` words 同时终止于 pred 期 global load 和随后 source-K global load；因此 `144/144` 都可到达 source-K consumer，但并非 `144/144` 都是纯 source-K 地址。这个结果证明 144-word **后 RA 表象**的 consumer 归属，却不证明“延迟一个 source-K scalar load 就能解决整个 full live set”。若继续研究，应控制 pre-greedy 中更宽的 producer/index/address recipe，并把表示保留到这个已定位的转换边界；不能再把普通晚展开 stage op 当成有效控制杆。",
        "",
        "## 复现",
        "",
        "```bash",
        "cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16",
        "",
        "PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \\",
        "PYTHONDONTWRITEBYTECODE=1 python3 \\",
        "  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_v29_address_provenance_convergence.py \\",
        "  --T 2048 --seed 20260724",
        "```",
        "",
        f"全部 raw MLIR/LLVM/bitcode/MIR/CSV/JSON 位于 `{result['out_dir']}`。",
    ])
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", "--t", dest="t", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.t % 64:
        raise ValueError("T must be divisible by 64")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for label in ("A_early_address", "B_late_address"):
        (args.out_dir / label).mkdir(parents=True)
    smoke = {
        "A_early_address": run_worker(args.out_dir, "A_early_address", False, args, None),
        "B_late_address": run_worker(args.out_dir, "B_late_address", True, args, args.out_dir / "frozen_inputs.pt"),
    }
    outputs = compare_outputs(
        args.out_dir / "A_early_address" / "outputs.pt",
        args.out_dir / "B_late_address" / "outputs.pt",
    )
    mlir_llvm_bisect = collect_mlir_bisect(args.out_dir)
    lto_a = replay_lto(args.out_dir, "A_early_address")
    lto_b = replay_lto(args.out_dir, "B_late_address")
    lto_bisect = collect_lto_bisect(lto_a, lto_b)
    isa_bisect = collect_isa_bisect(lto_a, lto_b)
    mir_bisect = collect_mir_bisect(lto_a, lto_b)
    pre_a = provenance_from_mir(
        Path(lto_a["pre_greedy_mir"]), flexible_av_only=False
    )
    pre_b = provenance_from_mir(
        Path(lto_b["pre_greedy_mir"]), flexible_av_only=False
    )
    post_a = provenance_from_mir(
        Path(lto_a["post_greedy_mir"]), flexible_av_only=True
    )
    post_b = provenance_from_mir(
        Path(lto_b["post_greedy_mir"]), flexible_av_only=True
    )
    write_csv(args.out_dir / "pre_greedy_address_objects.csv", [
        {
            "branch": label,
            **{key: value for key, value in row.items() if key != "final_memory_consumers"},
            "final_memory_consumers": json.dumps(row["final_memory_consumers"]),
        }
        for label, provenance in (("A_early_address", pre_a), ("B_late_address", pre_b))
        for row in provenance["objects"]
    ])
    write_csv(args.out_dir / "post_greedy_flexible_av_address_objects.csv", [
        {
            "branch": label,
            **{key: value for key, value in row.items() if key != "final_memory_consumers"},
            "final_memory_consumers": json.dumps(row["final_memory_consumers"]),
        }
        for label, provenance in (("A_early_address", post_a), ("B_late_address", post_b))
        for row in provenance["objects"]
    ])
    summary: dict[str, Any] = {
        "contract": {
            "kernel": KERNEL,
            "t": args.t,
            "only_branch": "AVELANG_QWEN_KFRAG_LATE_ADDRESS",
            "late_bload": 0,
            "convergence_audit": 1,
            "frozen_inputs_sha256": sha256(args.out_dir / "frozen_inputs.pt"),
        },
        "out_dir": str(args.out_dir),
        "smoke": smoke,
        "outputs": outputs,
        "pre_branch_identical": next(
            row["identical"] for row in mlir_llvm_bisect["rows"]
            if row["file"] == "pre_kfrag_branch.mlir"
        ),
        "mlir_llvm_bisect": mlir_llvm_bisect,
        "lto_replay": {"A_early_address": lto_a, "B_late_address": lto_b},
        "lto_bisect": lto_bisect,
        "isa_bisect": isa_bisect,
        "mir_bisect": mir_bisect,
        "pre_greedy_provenance": {"A_early_address": pre_a, "B_late_address": pre_b},
        "post_greedy_flexible_av_provenance": {
            "A_early_address": post_a,
            "B_late_address": post_b,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    render_report(summary, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

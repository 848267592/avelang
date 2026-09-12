#!/usr/bin/env python3
"""Strict full-v29 A/B for late source-K scalar-address generation.

Both branches compile the same full ``chunk_gdr`` source and the same compact
``[128,64]`` K producer rewrite.  The only compiler branch is whether the
producer's scalar BF16 source-K load is emitted while the rewrite builds its
stage loop (A) or kept as ``amdgpu_qwen_kfrag_stage_load`` until after GPU
outlining (B).  The latter is expanded back to the same scalar memref.load
immediately before its existing LDS store.

This is intentionally distinct from the older LATE_BLOAD experiment: the
four update MFMA16 B-fragment consumers remain generic vector loads in both
branches.  The audit tests whether late source-address construction reduces
the observed pred-phase 144 flexible-AV address words and final spills.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[4]
VLLM_COMPARE = ROOT / "test/examples/linear_attention/vllm_compare"
FULL_MODULE = "qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp"
KERNEL = "_qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32"
REPLAY = SCRIPT_DIR / "replay_qwen_v29_lto_mir.py"
DEFAULT_OUT = ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_full_chunk_gdr_late_address"
DEFAULT_REPORT = SCRIPT_DIR / "qwen_full_chunk_gdr_late_address_generation_report.md"
PMCS = [
    "SQ_INSTS_MFMA",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS",
    "OccupancyPercent",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_json_payload(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"worker did not emit JSON:\n{text}")


def make_inputs_cpu(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    k = (torch.randn((1, t, 4, 128), generator=generator) * 0.02).to(torch.bfloat16).contiguous()
    w = (torch.randn((1, t, 8, 128), generator=generator) * 0.02).contiguous()
    u = torch.randn((1, t, 8, 128), generator=generator).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, 8), generator=generator)) / 16.0).contiguous()
    initial_state = (torch.randn((1, 8, 128, 128), generator=generator) * 0.02).contiguous()
    chunks = t // 64
    g_chunks = g.view(1, chunks, 64, 8).transpose(2, 3).contiguous()
    g_last = g_chunks[:, :, :, -1].contiguous()
    decay = torch.exp(g_last.unsqueeze(-1) - g_chunks).contiguous()
    return k, w, u, decay, torch.exp(g_last).contiguous(), initial_state


def worker(args: argparse.Namespace) -> None:
    if args.input_path:
        cpu_tensors = torch.load(args.input_path, weights_only=True)
    else:
        cpu_tensors = make_inputs_cpu(args.t, args.seed)
        assert args.save_inputs is not None
        args.save_inputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_tensors, args.save_inputs)
    k, w, u, decay, g_last_exp, initial_state = tuple(
        tensor.to(device="cuda").contiguous() for tensor in cpu_tensors
    )
    sys.path.insert(0, str(VLLM_COMPARE))
    module = __import__(FULL_MODULE, fromlist=[KERNEL])
    raw_kernel = getattr(module, KERNEL)
    chunks = args.t // 64
    h = torch.empty((1, chunks, 8, 128, 128), dtype=torch.float32, device="cuda")
    final_state = torch.empty((1, 8, 128, 128), dtype=torch.float32, device="cuda")

    def launch() -> None:
        raw_kernel[lambda: ((32, 1, 1), (128, 1, 1))](
            k,
            w,
            u,
            decay,
            g_last_exp,
            initial_state,
            h,
            final_state,
            args.t,
            chunks,
            True,
            num_warps=2,
        )

    launch()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(args.repeat):
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if args.save_outputs:
        args.save_outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save((h.detach().cpu(), final_state.detach().cpu()), args.save_outputs)
    print(json.dumps({
        "kernel": KERNEL,
        "latency_ms_median": statistics.median(samples),
        "h_finite": bool(torch.isfinite(h).all().item()),
        "final_state_finite": bool(torch.isfinite(final_state).all().item()),
        "h_checksum_abs": float(h.abs().sum().item()),
        "final_state_checksum_abs": float(final_state.abs().sum().item()),
    }, sort_keys=True))


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


def make_env(out_dir: Path, label: str, late_address: bool, *, capture: bool) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(VLLM_COMPARE)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    else:
        pythonpath.append(str(ROOT / "python"))
    env.update({
        "PYTHONPATH": ":".join(pythonpath),
        "AVELANG_QWEN_KFRAG_LATE_ADDRESS": "1" if late_address else "0",
        # Freeze the previously disproven terminal B-load branch.
        "AVELANG_QWEN_KFRAG_LATE_BLOAD": "0",
        "AVELANG_QWEN_KFRAG_DEBUG": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if capture:
        env["AVELANG_QWEN_KFRAG_AB_DUMP_DIR"] = str(out_dir / label / "ir")
        env["AVELANG_AMDGPU_LINK_DEBUG_DIR"] = str(out_dir / label / "link")
    else:
        env.pop("AVELANG_QWEN_KFRAG_AB_DUMP_DIR", None)
        env.pop("AVELANG_AMDGPU_LINK_DEBUG_DIR", None)
    return env


def run_worker(
    out_dir: Path,
    label: str,
    late_address: bool,
    args: argparse.Namespace,
    input_path: Path | None,
) -> dict[str, object]:
    label_dir = out_dir / label
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--T",
        str(args.t),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--save-outputs",
        str(label_dir / "outputs.pt"),
    ]
    if input_path is None:
        command += ["--save-inputs", str(out_dir / "frozen_inputs.pt")]
    else:
        command += ["--input-path", str(input_path)]
    result = run(command, make_env(out_dir, label, late_address, capture=True))
    (label_dir / "smoke.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} worker failed:\n{result.stdout}")
    payload = parse_json_payload(result.stdout)
    payload["rewrite_fired"] = "[qwen-kfrag] rewritten=1" in result.stdout
    payload["late_address"] = late_address
    payload["late_address_stage_op_created"] = "late_address_stage_loads=1" in result.stdout
    return payload


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    mismatch = a.ne(b)
    count = int(mismatch.sum().item())
    result: dict[str, object] = {
        "shape_equal": tuple(a.shape) == tuple(b.shape),
        "dtype_equal": a.dtype == b.dtype,
        "bit_exact": count == 0,
        "mismatch_count": count,
        "max_abs": float((a - b).abs().max().item()),
    }
    if count:
        result["first_mismatch_flat_index"] = int(mismatch.flatten().nonzero()[0].item())
    return result


def compare_outputs(a_path: Path, b_path: Path) -> dict[str, object]:
    a_h, a_final = torch.load(a_path, weights_only=True)
    b_h, b_final = torch.load(b_path, weights_only=True)
    return {"h": compare_tensors(a_h, b_h), "final_state": compare_tensors(a_final, b_final)}


def text_count(path: Path, pattern: str) -> int:
    return len(re.findall(pattern, path.read_text(errors="replace")))


def inspect_ir(out_dir: Path) -> dict[str, object]:
    a_dir = out_dir / "A_early_address" / "ir"
    b_dir = out_dir / "B_late_address" / "ir"
    names = [
        "pre_kfrag_branch.mlir",
        "post_kfrag_rewrite.mlir",
        "post_kfrag_load_lowering.mlir",
        "final_mlir.mlir",
        "preopt_llvm.ll",
        "postopt_llvm.ll",
    ]
    snapshots: dict[str, object] = {}
    for name in names:
        a = a_dir / name
        b = b_dir / name
        if not a.exists() or not b.exists():
            raise RuntimeError(f"missing snapshot {name}")
        snapshots[name] = {
            "early_sha256": sha256(a),
            "late_sha256": sha256(b),
            "identical": a.read_bytes() == b.read_bytes(),
        }
    post_rewrite_a = a_dir / "post_kfrag_rewrite.mlir"
    post_rewrite_b = b_dir / "post_kfrag_rewrite.mlir"
    post_lower_a = a_dir / "post_kfrag_load_lowering.mlir"
    post_lower_b = b_dir / "post_kfrag_load_lowering.mlir"
    return {
        "snapshots": snapshots,
        "structural_counts": {
            "pre_persistent_ops_early": text_count(a_dir / "pre_kfrag_branch.mlir", r"amdgpu_qwen_update_kfrag_load"),
            "pre_persistent_ops_late": text_count(b_dir / "pre_kfrag_branch.mlir", r"amdgpu_qwen_update_kfrag_load"),
            "stage_op_after_rewrite_early": text_count(post_rewrite_a, r"amdgpu_qwen_kfrag_stage_load"),
            "stage_op_after_rewrite_late": text_count(post_rewrite_b, r"amdgpu_qwen_kfrag_stage_load"),
            "stage_op_after_late_lowering_early": text_count(post_lower_a, r"amdgpu_qwen_kfrag_stage_load"),
            "stage_op_after_late_lowering_late": text_count(post_lower_b, r"amdgpu_qwen_kfrag_stage_load"),
            "late_address_attr_early": text_count(post_lower_a, r"qwen_kfrag\.late_address"),
            "late_address_attr_late": text_count(post_lower_b, r"qwen_kfrag\.late_address"),
            "vector_load_early": text_count(post_lower_a, r"vector\.load"),
            "vector_load_late": text_count(post_lower_b, r"vector\.load"),
            "alloca_early": text_count(post_lower_a, r"memref\.alloca"),
            "alloca_late": text_count(post_lower_b, r"memref\.alloca"),
        },
    }


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} in {directory}, got {matches}")
    return matches[0]


def disassemble(hsaco: Path) -> Path:
    isa = hsaco.with_suffix(".isa")
    result = subprocess.run(
        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--no-show-raw-insn", str(hsaco)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    isa.write_text(result.stdout)
    return isa


def isa_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    normalized = "\n".join(line for line in text.splitlines() if not line.endswith(":\tfile format elf64-amdgpu"))
    writes = [int(value) for value in re.findall(r"v_accvgpr_write_b32\s+a(\d+)", text)]
    return {
        "sha256": sha256(path),
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "mfma16": text.count("v_mfma_f32_16x16x16_bf16"),
        "mfma32": text.count("v_mfma_f32_32x32x8_bf16"),
        "ds_read_b64": len(re.findall(r"ds_read_b64", text)),
        "ds_write": len(re.findall(r"\bds_write", text)),
        "global_or_buffer_load": len(re.findall(r"\b(?:global|buffer)_load", text)),
        "barrier": len(re.findall(r"\bs_barrier", text)),
        "accvgpr_write": len(writes),
        "accvgpr_read": len(re.findall(r"v_accvgpr_read_b32", text)),
        "high_agpr_write_ge100": sum(value >= 100 for value in writes),
        "max_explicit_agpr_write": max(writes) if writes else None,
    }


def run_lto_replay(out_dir: Path, label: str) -> Path:
    link_dir = out_dir / label / "link"
    argv = find_one(link_dir, "*.argv.txt")
    mir_dir = out_dir / label / "exact_lto_postra"
    command = [
        sys.executable,
        str(REPLAY),
        "--argv-file",
        str(argv),
        "--out-dir",
        str(mir_dir),
        "--kernel",
        KERNEL,
    ]
    result = run(command, os.environ.copy())
    (mir_dir / "replay.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} LTO replay failed:\n{result.stdout}")
    summary = json.loads((mir_dir / "summary.json").read_text())
    candidates = [row for row in summary if "After greedy" in str(row["phase"])]
    if not candidates:
        candidates = [row for row in summary if int(row["av32_spill_saves"]) + int(row["av64_spill_saves"]) > 0]
    if not candidates:
        raise RuntimeError(f"{label} replay produced no post-greedy spill section")
    return Path(str(candidates[0]["path"]))


def mir_pressure(mir_path: Path) -> dict[str, object]:
    sys.path.insert(0, str(SCRIPT_DIR))
    import analyze_qwen_v29_broad_compact_pressure_timeline as timeline

    instructions, vregs = timeline.parse_mir(mir_path)
    timeline.propagate_address_provenance(vregs)
    timeline.propagate_fragment_provenance(vregs)
    phases = timeline.phase_anchors(instructions)
    rows = timeline.make_timeline(instructions, vregs, phases)
    pred_row = max(
        (row for row in rows if row["phase"] == "pred_mfma32"),
        key=lambda row: int(row["flexible_av_words"]),
    )
    snapshot = timeline.active_snapshot(vregs, int(pred_row["mir_line"]))
    address_words = sum(
        int(row["words"])
        for row in snapshot
        if str(row["register_class"]).startswith("av_") and bool(row["address"])
    )
    spills = timeline.spill_context(instructions, vregs, phases)
    return {
        "mir": str(mir_path),
        "sha256": sha256(mir_path),
        "phases": phases,
        "pred_flexible_av_peak": {
            "mir_line": pred_row["mir_line"],
            "byte_offset": pred_row["byte_offset"],
            "flexible_av_words": pred_row["flexible_av_words"],
            "address_flexible_av_words": address_words,
            "live_object_count": len(snapshot),
        },
        "spill": {
            "av_spill_count": spills["av_spill_count"],
            "av_spill_words": spills["av_spill_words"],
            "first_av_spill": spills["first_av_spill"],
        },
    }


def parse_profile(profile_dir: Path) -> dict[str, object]:
    trace = find_one(profile_dir, "*_kernel_trace.csv")
    counters = find_one(profile_dir, "*_counter_collection.csv")
    with trace.open() as handle:
        trace_rows = [row for row in csv.DictReader(handle) if KERNEL in row.get("Kernel_Name", "")]
    if not trace_rows:
        raise RuntimeError(f"no matching kernel in {trace}")
    durations = sorted((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in trace_rows)
    with counters.open() as handle:
        counter_rows = [row for row in csv.DictReader(handle) if KERNEL in row.get("Kernel_Name", "")]
    if not counter_rows:
        raise RuntimeError(f"no matching counters in {counters}")
    result: dict[str, object] = {
        "trace_median_us": statistics.median(durations),
        "trace_count": len(trace_rows),
        "VGPR_Count": trace_rows[-1].get("VGPR_Count"),
        "Accum_VGPR_Count": trace_rows[-1].get("Accum_VGPR_Count"),
        "SGPR_Count": trace_rows[-1].get("SGPR_Count"),
        "Scratch_Size": trace_rows[-1].get("Scratch_Size"),
        "LDS_Block_Size": trace_rows[-1].get("LDS_Block_Size"),
        "Workgroup_Size": trace_rows[-1].get("Workgroup_Size_X"),
        "Grid_Size": trace_rows[-1].get("Grid_Size_X"),
    }
    for row in counter_rows:
        name = row.get("Counter_Name")
        if name in PMCS and row.get("Counter_Value"):
            result[name] = float(row["Counter_Value"])
    return result


def run_rocprof(out_dir: Path, label: str, late_address: bool, args: argparse.Namespace) -> dict[str, object]:
    profile_dir = out_dir / label / "rocprof"
    command = [
        "/opt/rocm/bin/rocprofv3",
        "--kernel-trace",
        "--pmc",
        *PMCS,
        "--kernel-include-regex",
        KERNEL,
        "-d",
        str(profile_dir),
        "-o",
        label,
        "-f",
        "csv",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--T",
        str(args.t),
        "--seed",
        str(args.seed),
        "--warmup",
        str(args.rocprof_warmup),
        "--repeat",
        str(args.rocprof_repeat),
        "--input-path",
        str(out_dir / "frozen_inputs.pt"),
    ]
    result = run(command, make_env(out_dir, label, late_address, capture=False))
    (out_dir / label / "rocprof.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} rocprof failed:\n{result.stdout}")
    return parse_profile(profile_dir)


def as_number(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def render_report(result: dict[str, object], report: Path) -> None:
    ir = result["ir"]  # type: ignore[assignment]
    pressure = result["mir_pressure"]  # type: ignore[assignment]
    isa = result["isa"]  # type: ignore[assignment]
    smoke = result["smoke"]  # type: ignore[assignment]
    outputs = result["outputs"]  # type: ignore[assignment]
    early_pressure = pressure["A_early_address"]  # type: ignore[index]
    late_pressure = pressure["B_late_address"]  # type: ignore[index]
    early_addr = int(early_pressure["pred_flexible_av_peak"]["address_flexible_av_words"])  # type: ignore[index]
    late_addr = int(late_pressure["pred_flexible_av_peak"]["address_flexible_av_words"])  # type: ignore[index]
    early_spill = int(early_pressure["spill"]["av_spill_words"])  # type: ignore[index]
    late_spill = int(late_pressure["spill"]["av_spill_words"])  # type: ignore[index]
    postopt_same = bool(ir["snapshots"]["postopt_llvm.ll"]["identical"])  # type: ignore[index]
    isa_same = isa["A_early_address"]["normalized_sha256"] == isa["B_late_address"]["normalized_sha256"]  # type: ignore[index]
    outputs_exact = bool(outputs["h"]["bit_exact"]) and bool(outputs["final_state"]["bit_exact"])  # type: ignore[index]
    lines = [
        "# Full v29 Late Scalar-Address Generation A/B",
        "",
        "## 实验目的",
        "",
        "本实验只改变 compact `[128,64]` K producer 的 source-K 全局地址何时具现化。A 在 producer rewrite 中直接创建标量 `memref.load`；B 先保留 `amdgpu_qwen_kfrag_stage_load`，在 GPU outlining 后、紧邻原有 LDS store 时展开为同一标量 BF16 `memref.load`。",
        "",
        "四个 update MFMA16 B-fragment consumer、K tile shape、8192 个 source-K global load、8192 个 LDS store、MFMA schedule、barrier、launch `(32,1,1)/(128,1,1)`、数学和输入完全冻结。`AVELANG_QWEN_KFRAG_LATE_BLOAD=0` 固定，故这不是旧 terminal B-load 实验。",
        "",
        "## 分叉完整性",
        "",
        f"- pre-branch MLIR 是否相同：`{ir['snapshots']['pre_kfrag_branch.mlir']['identical']}`。",
        f"- A/B post-rewrite stage-op 计数：`{ir['structural_counts']['stage_op_after_rewrite_early']}` / `{ir['structural_counts']['stage_op_after_rewrite_late']}`。",
        f"- A/B post-late-lowering 残留 stage-op：`{ir['structural_counts']['stage_op_after_late_lowering_early']}` / `{ir['structural_counts']['stage_op_after_late_lowering_late']}`。",
        f"- B 的 late-address 标记计数：`{ir['structural_counts']['late_address_attr_late']}`；A 为 `{ir['structural_counts']['late_address_attr_early']}`。",
        f"- post-opt LLVM 是否相同：`{postopt_same}`；规范化 ISA 是否相同：`{isa_same}`。",
        "",
        "这四项共同区分两种情况：若 stage op 根本未穿过 outlining，这是实现失败；若它按预期穿过但 post-opt LLVM/ISA 又收敛，则是后端把 timing 差异消除了，而不是 A/B 漏跑。",
        "",
        "## 语义与工作量 gate",
        "",
        f"- full `h` bit-exact：`{outputs['h']['bit_exact']}`，max abs `{outputs['h']['max_abs']}`。",
        f"- full final-state bit-exact：`{outputs['final_state']['bit_exact']}`，max abs `{outputs['final_state']['max_abs']}`。",
        f"- persistent rewrite fired：A `{smoke['A_early_address']['rewrite_fired']}`，B `{smoke['B_late_address']['rewrite_fired']}`。",
        "",
        "| ISA static metric | A early address | B late address |",
        "|:--|--:|--:|",
    ]
    for key in ("mfma16", "mfma32", "ds_read_b64", "ds_write", "global_or_buffer_load", "barrier", "accvgpr_write", "high_agpr_write_ge100", "max_explicit_agpr_write"):
        lines.append(f"| {key} | {isa['A_early_address'][key]} | {isa['B_late_address'][key]} |")
    lines.extend([
        "",
        "## Post-greedy MIR 地址压力与 spill",
        "",
        "这里的词数是 MIR 虚寄存器从首次定义到最后文本使用的静态 liveness proxy，不是 LLVM LiveIntervals；`av_*` 是 flexible AV class，不能直接等同于物理 AGPR。它仍能机械化比较同一高层图下的地址链是否在 pred 区间保持活跃。",
        "",
        "| metric | A early address | B late address | B-A |",
        "|:--|--:|--:|--:|",
        f"| pred peak flexible-AV words | {early_pressure['pred_flexible_av_peak']['flexible_av_words']} | {late_pressure['pred_flexible_av_peak']['flexible_av_words']} | {int(late_pressure['pred_flexible_av_peak']['flexible_av_words']) - int(early_pressure['pred_flexible_av_peak']['flexible_av_words'])} |",
        f"| pred peak address-marked flexible-AV words | {early_addr} | {late_addr} | {late_addr - early_addr} |",
        f"| AV spill words | {early_spill} | {late_spill} | {late_spill - early_spill} |",
        f"| first AV spill | {early_pressure['spill']['first_av_spill']} | {late_pressure['spill']['first_av_spill']} | - |",
        "",
    ])
    if "rocprof" in result:
        profile = result["rocprof"]  # type: ignore[assignment]
        lines.extend([
            "## T=2048 rocprof",
            "",
            "| metric | A early address | B late address | B-A |",
            "|:--|--:|--:|--:|",
        ])
        for key in ("trace_median_us", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count", "Scratch_Size", "LDS_Block_Size", "SQ_INSTS_MFMA", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM", "SQ_INSTS_LDS", "OccupancyPercent"):
            a = profile["A_early_address"].get(key, "N/A")
            b = profile["B_late_address"].get(key, "N/A")
            a_number, b_number = as_number(a), as_number(b)
            delta = f"{b_number - a_number:.6g}" if a_number is not None and b_number is not None else "N/A"
            lines.append(f"| {key} | {a} | {b} | {delta} |")
        lines.append("")
    lines.extend([
        "## 判定",
        "",
    ])
    if not outputs_exact:
        lines.append("**No-Go：A/B 未保持 original full-v29 语义，不能将任何资源差异解释为地址生命周期收益。**")
    elif postopt_same and isa_same:
        lines.append("**No-Go：dedicated stage op 已按约定穿过 GPU outlining 并在 late pass 展开，但后续优化将 A/B 收敛为相同 post-opt LLVM 和 ISA。** 因而没有 retained lowering difference 可以压低 144 个地址 words 或 spill；任何微小 timing 波动都不可解释为编译器收益。")
    elif late_addr < early_addr and late_spill < early_spill:
        lines.append("**Positive evidence：在同一 full high-level source/schedule 下，B 的 late scalar-address generation 同时压低了 pred 地址活跃词数和 AV spill。** 若静态 ISA 工作量与 bit-exact gate 同时成立，这构成针对 address-lifetime lowering 的强证据，但仍不能泛化为所有 Avelang lowering 的结论。")
    else:
        lines.append("**Negative/ambiguous：B 保留了不同 lowering，但没有同时压低 pred 地址词数和 spill。** 这否定“只延后 source-K 标量地址生成即可显著修复 full cliff”的假设；不能据此归因高层算法或 Avelang 的全部 backend。")
    lines.extend([
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \\",
        "PYTHONDONTWRITEBYTECODE=1 python3 \\",
        "  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_full_chunk_gdr_late_address.py \\",
        "  --T 2048 --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5",
        "```",
        "",
        f"Raw A/B evidence is under `{result['out_dir']}`. The experiment does not resolve the existing original-v29 nonzero-W-vs-reference recurrence issue; it tests bit-exact preservation relative to the same full-v29 implementation only.",
    ])
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--T", "--t", dest="t", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--save-inputs", type=Path)
    parser.add_argument("--save-outputs", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-rocprof", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
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
    ir = inspect_ir(args.out_dir)
    isa: dict[str, object] = {}
    mir_pressure: dict[str, object] = {}
    for label in ("A_early_address", "B_late_address"):
        isa[label] = isa_metrics(disassemble(find_one(args.out_dir / label / "link", "*.linked.out")))
        mir_pressure[label] = mir_pressure_fn(run_lto_replay(args.out_dir, label))
    result: dict[str, object] = {
        "contract": {
            "source_module": FULL_MODULE,
            "kernel": KERNEL,
            "t": args.t,
            "only_branch": "AVELANG_QWEN_KFRAG_LATE_ADDRESS after identical pre_kfrag_branch.mlir",
            "terminal_b_fragment_branch": "frozen AVELANG_QWEN_KFRAG_LATE_BLOAD=0",
            "frozen_inputs_sha256": sha256(args.out_dir / "frozen_inputs.pt"),
        },
        "out_dir": str(args.out_dir),
        "smoke": smoke,
        "outputs": outputs,
        "ir": ir,
        "isa": isa,
        "mir_pressure": mir_pressure,
    }
    if not args.skip_rocprof:
        result["rocprof"] = {
            "A_early_address": run_rocprof(args.out_dir, "A_early_address", False, args),
            "B_late_address": run_rocprof(args.out_dir, "B_late_address", True, args),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    render_report(result, args.report)
    print(json.dumps(result, indent=2, sort_keys=True))


# Keep the public helper name distinct from the main result dictionary.
mir_pressure_fn = mir_pressure


if __name__ == "__main__":
    main()

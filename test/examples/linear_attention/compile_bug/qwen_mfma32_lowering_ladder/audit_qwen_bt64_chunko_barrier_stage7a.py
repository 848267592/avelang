#!/usr/bin/env python3
"""Stage 7A provenance capture for the frozen Stage 6Z Z1 chunk-o.

This script does not alter Z1.  It reconstructs the same JIT AST source, saves
the pre-link LLVM IR and backend assembly, captures the final code object, and
builds a conservative source -> LLVM -> final-ISA barrier ledger.  Source line
locations are preserved through LLVM DebugLoc when available; final HSACO PCs
are then matched by ordered barrier ordinal only after count/order validation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

import _avelang_bindings as _C
from avelang.backends.amdgpu import compiler as amdgpu_compiler
from avelang.compiler import code_generator as cg
from avelang.runtime.driver import driver


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = HERE / "codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

import qwen_gdn_bt64_native_chunko_stage6z as z1  # noqa: E402
from stage2_runner import make_inputs  # noqa: E402


BT = 64
WORKGROUP = 256
KERNEL_NAME = z1._qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1.fn.__name__
DEFAULT_OUT = HERE / "codex_qwen_bt64_chunko_barrier_stage7a"


# This is a semantic ledger, not a claim that every occurrence is redundant.
# The micro-repros test the candidate-elision classifications independently.
SOURCE_SITES = {
    90: {
        "site": "A.stage_qh",
        "phase": "A inter Q/H K32 stage",
        "hazard": "CTA Q/H LDS producer -> MFMA LDS consumer (RAW)",
        "classification": "required for this shared staging schedule",
    },
    96: {
        "site": "A.pack_frag",
        "phase": "A per-lane frag_words pack",
        "hazard": "frag_words[tid] write -> same-lane load",
        "classification": "candidate source-scheduling barrier",
    },
    103: {
        "site": "A.reuse_frag",
        "phase": "A next packed fragment",
        "hazard": "same-lane frag_words reuse after MFMA",
        "classification": "candidate source-scheduling barrier",
    },
    125: {
        "site": "B.stage_qk",
        "phase": "B score Q/K K32 stage",
        "hazard": "CTA Q/K LDS producer -> owner-wave MFMA consumer (RAW)",
        "classification": "required for this shared staging schedule",
    },
    131: {
        "site": "B.pack_frag",
        "phase": "B per-lane frag_words pack",
        "hazard": "frag_words[tid] write -> same-lane load",
        "classification": "candidate source-scheduling barrier",
    },
    139: {
        "site": "B.reuse_frag",
        "phase": "B next packed fragment",
        "hazard": "same-lane frag_words reuse after MFMA",
        "classification": "candidate source-scheduling barrier",
    },
    153: {
        "site": "B.serialize_score",
        "phase": "B score half store",
        "hazard": "score store -> later score/V consume; next stage is physically disjoint",
        "classification": "candidate phase-merge barrier",
    },
    164: {
        "site": "C.stage_v",
        "phase": "C V-new transpose",
        "hazard": "CTA V-new LDS producer + earlier score store -> intra MFMA consumer (RAW)",
        "classification": "required for this shared staging schedule",
    },
    172: {
        "site": "C.pack_frag",
        "phase": "C score/V per-lane frag_words pack",
        "hazard": "frag_words[tid] write -> same-lane load",
        "classification": "candidate source-scheduling barrier",
    },
    179: {
        "site": "C.reuse_frag",
        "phase": "C next score/V packed fragment",
        "hazard": "same-lane frag_words reuse after MFMA",
        "classification": "candidate source-scheduling barrier",
    },
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_generator(src: Any) -> _C.MLIRGenerator:
    constexprs_json = cg._serialize_constexprs(src)
    jit_deps = cg._collect_jit_dependencies(src.fn)
    imports = cg._build_import_module([src.fn, *jit_deps])
    generator = _C.MLIRGenerator()
    generator.generate_from_python_ast(imports)
    for dep in jit_deps:
        generator.add_jit_dependency(dep.parse())
    for dep in jit_deps:
        dep_func = cg._get_function_def(dep.parse())
        globals_ = {}
        collect = getattr(dep, "_collect_global_constexprs", None)
        if callable(collect):
            globals_ = collect()
        generator.visit_function_def(dep_func, cg._serialize_global_constexprs(globals_), "jit")
    generator.visit_function_def(cg._get_function_def(src.fn.parse()), constexprs_json, "kernel")
    return generator


def make_ast_source(t: int) -> tuple[Any, Any, Any, tuple[torch.Tensor, ...]]:
    q, k, _, g, _, _ = make_inputs(t, 2026072701 + t, "random", True)
    torch.manual_seed(2026072801 + t)
    chunks = t // BT
    vn = torch.randn((1, t, 8, 128), device=q.device, dtype=torch.float32).to(torch.bfloat16)
    h = torch.randn((1, chunks, 8, 128, 128), device=q.device, dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty_like(vn)
    fn = z1._qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
    device = driver.active.get_current_device()
    _cache, _key, target, backend, binder = fn.device_caches[device]
    bound, specialization, options = binder(q, k, vn, h, g, out, float(128**-0.5), t, chunks, num_warps=4)
    options, signature, constexprs, globals_, attrs = fn._pack_args(
        backend, {"num_warps": 4}, bound, specialization, options
    )
    return fn.ASTSource(fn, signature, constexprs, attrs, globals_), target, options, (q, k, vn, h, g, out)


def tool(name: str) -> str:
    for candidate in (
        f"/opt/rocm/llvm/bin/{name}",
        f"/opt/rocm/bin/{name}",
        shutil.which(name),
    ):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(f"could not locate {name}")


def capture_final_hsaco(tensors: tuple[torch.Tensor, ...], out: Path) -> None:
    q, k, vn, h, g, output = tensors
    original = amdgpu_compiler.AmdgpuCompiler.compile
    captured = False

    def wrapped(self: Any, src: Any, target: Any, options: Any = None) -> bytes:
        nonlocal captured
        binary = original(self, src, target, options)
        if src.fn.fn.__name__ == KERNEL_NAME and not captured:
            out.write_bytes(binary)
            captured = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        z1.qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z1_launch_into(q, k, vn, h, g, output)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
    if not captured:
        raise RuntimeError("Stage 7A failed to capture a freshly compiled Z1 HSACO")


LLVM_BARRIER = re.compile(r"^\s*(?:tail )?call void @llvm\.amdgcn\.s\.barrier\(\)")
DEBUG_LINE = re.compile(r"!(\d+) = !DILocation\(line: (\d+)")
ASM_BARRIER = re.compile(r"^\s*s_barrier\b")
ISA_BARRIER = re.compile(r"^\s*s_barrier\b.*?//\s*([0-9A-Fa-f]+):")


def llvm_barriers(text: str) -> list[dict[str, Any]]:
    debug = {int(node): int(line) for node, line in DEBUG_LINE.findall(text)}
    rows = []
    for ordinal, (line_no, line) in enumerate(enumerate(text.splitlines(), start=1), start=1):
        match = LLVM_BARRIER.search(line)
        if not match:
            continue
        dbg_match = re.search(r"!dbg !(\d+)", line)
        dbg = int(dbg_match.group(1)) if dbg_match else None
        source_line = debug.get(dbg)
        rows.append(
            {
                "ordinal": ordinal,
                "llvm_line": line_no,
                "debug_node": dbg,
                "source_line": source_line,
                "site": SOURCE_SITES.get(source_line, {}).get("site", "unresolved"),
            }
        )
    return rows


def static_source_schedule() -> list[dict[str, Any]]:
    """Source-order expansion for this exact constexpr specialization.

    A's four K stages are unrolled.  B preserves one static K-stage body per
    source half (executed four times at runtime), while C's two score halves
    and two packed K fragments are unrolled.  This sequence is checked against
    the LLVM/pre-LTO/final-ISA count before it is used for PC attribution.
    """
    rows: list[dict[str, Any]] = []

    def add(line: int, execution: str) -> None:
        item = dict(SOURCE_SITES[line])
        item.update({"source_line": line, "execution": execution})
        rows.append(item)

    for k_stage in range(4):
        add(90, f"A k_stage={k_stage}")
        add(96, f"A k_stage={k_stage}, kt=0")
        add(103, f"A k_stage={k_stage}, kt=0")
        add(96, f"A k_stage={k_stage}, kt=1")
        add(103, f"A k_stage={k_stage}, kt=1")

    for source_half in range(2):
        add(125, f"B source_half={source_half}, dynamic k_stage body x4")
        add(131, f"B source_half={source_half}, dynamic k_stage body x4, kt=0")
        add(139, f"B source_half={source_half}, dynamic k_stage body x4, kt=0")
        add(131, f"B source_half={source_half}, dynamic k_stage body x4, kt=1")
        add(139, f"B source_half={source_half}, dynamic k_stage body x4, kt=1")
        add(153, f"B source_half={source_half} serialize score")

    add(164, "C V-new transpose")
    for source_half in range(2):
        for kt in range(2):
            add(172, f"C source_half={source_half}, kt={kt}")
            add(179, f"C source_half={source_half}, kt={kt}")
    return rows


def asm_barriers(text: str) -> list[dict[str, int]]:
    return [
        {"ordinal": ordinal, "asm_line": line_no}
        for ordinal, (line_no, line) in enumerate(enumerate(text.splitlines(), start=1), start=1)
        if ASM_BARRIER.search(line)
    ]


def isa_barriers(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = ISA_BARRIER.search(line)
        if not match:
            continue
        window = lines[max(0, index - 3) : min(len(lines), index + 4)]
        rows.append(
            {
                "ordinal": len(rows) + 1,
                "isa_line": index + 1,
                "pc": "0x" + match.group(1).lower(),
                "window": window,
            }
        )
    return rows


def classify_native() -> dict[str, Any]:
    root = HERE / "codex_qwen_bt64_stage6z_native_chunko/native/T2048/selected"
    amdgcn = (root / "chunk_fwd_kernel_o.amdgcn").read_text(errors="replace")
    ttgir = (root / "chunk_fwd_kernel_o.ttgir").read_text(errors="replace")
    return {
        "source": str(root / "chunk_fwd_kernel_o.source"),
        "amdgcn_static_barriers": sum(1 for line in amdgcn.splitlines() if ASM_BARRIER.search(line)),
        "ttgir_local_alloc": ttgir.count("ttg.local_alloc"),
        "ttgir_local_dealloc": ttgir.count("ttg.local_dealloc"),
        "phase_alignment": [
            {
                "native_phase": "K32 Q/K/H local-load and dual dot pipeline",
                "native_source": "chunk_o.py:93-113",
                "z1_phases": "A.stage_qh + B.stage_qk + all pack/reuse sites",
                "interpretation": "native local allocation/lifetime is pipeline-generated; it does not expose ten Python barrier calls",
            },
            {
                "native_phase": "decay, causal mask, BF16 score operand",
                "native_source": "chunk_o.py:115-125",
                "z1_phases": "B.serialize_score",
                "interpretation": "native keeps score as a compiler-managed local operand rather than serializing two halves through generic shared views",
            },
            {
                "native_phase": "V load and score-times-V dot, BF16 output store",
                "native_source": "chunk_o.py:127-138",
                "z1_phases": "C.stage_v + C.pack_frag + C.reuse_frag",
                "interpretation": "native deallocates source local buffers before score/V local buffers; Z1 manually stages/re-packs both through CTA shared storage",
            },
        ],
        "limitation": "Triton source has no explicit tl.barrier at these lines. Its 10 barriers are lowering/pipeline operations, so a per-source-line one-to-one mapping is not available from Python source.",
    }


def write_ledger(out: Path, llvm_rows: list[dict[str, Any]], asm_rows: list[dict[str, int]], isa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_schedule = static_source_schedule()
    ordered = len(llvm_rows) == len(isa_rows) == len(asm_rows) == len(source_schedule)
    ledger: list[dict[str, Any]] = []
    for index, isa in enumerate(isa_rows):
        llvm = llvm_rows[index] if ordered else {}
        source = source_schedule[index] if ordered else {}
        source_line = source.get("source_line")
        info = SOURCE_SITES.get(source_line, {})
        ledger.append(
            {
                "isa_ordinal": isa["ordinal"],
                "isa_pc": isa["pc"],
                "isa_line": isa["isa_line"],
                "source_line": source_line,
                "source_site": info.get("site", "unresolved"),
                "source_execution": source.get("execution", "unresolved"),
                "phase": info.get("phase", "unresolved"),
                "hazard": info.get("hazard", "unresolved"),
                "classification": info.get("classification", "unresolved"),
                "llvm_ordinal": llvm.get("ordinal"),
                "llvm_line": llvm.get("llvm_line"),
                "pre_lto_asm_ordinal": asm_rows[index]["ordinal"] if ordered and len(asm_rows) == len(isa_rows) else None,
                "mapping": "source schedule -> LLVM call -> pre-LTO asm -> final ISA ordinal match" if ordered else "no exact ordinal match; inspect PC window",
            }
        )
    write_json(out / "z1_barrier_ledger.json", ledger)
    lines = ["# Stage 7A Z1 Barrier Provenance Ledger", "", "| ISA | PC | source | execution | phase | hazard | classification | LLVM |", "|---:|:---|:---|:---|:---|:---|:---|---:|"]
    for row in ledger:
        lines.append(
            "| {isa} | `{pc}` | {source}:{line} | {execution} | {phase} | {hazard} | {kind} | {llvm} |".format(
                isa=row["isa_ordinal"],
                pc=row["isa_pc"],
                source=row["source_site"],
                line=row["source_line"],
                execution=row["source_execution"],
                phase=row["phase"],
                hazard=row["hazard"],
                kind=row["classification"],
                llvm=row["llvm_ordinal"],
            )
        )
    (out / "z1_barrier_ledger.md").write_text("\n".join(lines) + "\n")
    return ledger


def child_mlir(args: argparse.Namespace) -> None:
    src, target, options, _ = make_ast_source(args.T)
    generator = build_generator(src)
    Path(args.mlir_out).write_text(generator.get_mlir())
    print(args.mlir_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mlir-child", action="store_true")
    parser.add_argument("--mlir-out", type=Path)
    args = parser.parse_args()
    if args.mlir_child:
        if args.mlir_out is None:
            raise ValueError("--mlir-child requires --mlir-out")
        child_mlir(args)
        return
    if args.T < BT or args.T % BT:
        raise ValueError("Stage 7A supports T divisible by 64")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7A requires the gfx942 HIP runtime")

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    src, target, options, tensors = make_ast_source(args.T)
    (out / "z1_source.py").write_text(inspect.getsource(z1))
    generator = build_generator(src)

    # The MLIR printer has historically asserted for generated JIT modules.
    # Isolate it in a child so a printer failure cannot corrupt the exact LLVM
    # and HSACO audit from this process.
    mlir_path = out / "z1_initial.mlir"
    mlir = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--mlir-child", "--T", str(args.T), "--mlir-out", str(mlir_path)],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    write_json(
        out / "mlir_status.json",
        {"returncode": mlir.returncode, "path": str(mlir_path), "stdout_tail": "\n".join(mlir.stdout.splitlines()[-20:])},
    )

    # Lowering mutates the MLIR module.  Keep a separate generator for the
    # diagnostic assembly emission so this audit never asks the same module to
    # traverse the AveLang-to-LLVM pipeline twice.
    llvm_ir = generator.get_llvm_ir(target.tuple, target.chip, 2, getattr(options, "num_warps", 4))
    llvm_path = out / "z1_prelink.ll"
    llvm_path.write_text(llvm_ir)
    asm_generator = build_generator(src)
    asm = asm_generator.get_assembly(target.tuple, target.chip, 2, getattr(options, "num_warps", 4))
    asm_path = out / "z1_prelink.s"
    asm_path.write_text(asm)
    hsaco_path = out / "z1_final.hsaco"
    capture_final_hsaco(tensors, hsaco_path)
    objdump = tool("llvm-objdump")
    isa_path = out / "z1_final.isa"
    disasm = subprocess.run([objdump, "-d", str(hsaco_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    isa_path.write_text(disasm.stdout)

    llvm_rows = llvm_barriers(llvm_ir)
    asm_rows = asm_barriers(asm)
    isa_rows = isa_barriers(disasm.stdout)
    ledger = write_ledger(out, llvm_rows, asm_rows, isa_rows)
    by_site = Counter(str(row.get("source_site")) for row in ledger)
    summary = {
        "kernel": KERNEL_NAME,
        "target": {"triple": target.tuple, "chip": target.chip, "num_warps": getattr(options, "num_warps", None)},
        "counts": {"llvm": len(llvm_rows), "pre_lto_assembly": len(asm_rows), "final_isa": len(isa_rows)},
        "source_site_counts": dict(sorted(by_site.items())),
        "mlir_status": json.loads((out / "mlir_status.json").read_text()),
        "artifacts": {"llvm": str(llvm_path), "assembly": str(asm_path), "hsaco": str(hsaco_path), "isa": str(isa_path)},
        "native": classify_native(),
    }
    write_json(out / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

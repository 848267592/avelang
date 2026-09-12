#!/usr/bin/env python3
"""ISA and rocprof helper for the v28 Avelang vs Triton gap analysis.

This file intentionally does not define or modify kernels.  It only:

* dumps the already-generated Avelang v28 code object by wrapping the compiler;
* triggers the selected vLLM Triton chunk_delta_h config into a dedicated cache;
* disassembles hsaco files with llvm-objdump;
* classifies ISA mnemonics and parses rocprof CSV counter output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
VLLM_ROOT = PROJECT_ROOT / "vllm_stageb_snapshot"
DEFAULT_OUT_DIR = PROJECT_ROOT / "avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap"


def _ensure_import_paths() -> None:
    sys.path.insert(0, str(SCRIPT_DIR))
    sys.path.insert(0, str(VLLM_ROOT))


def dump_avelang_v28(args: argparse.Namespace) -> None:
    """Run the existing v28 full path once and save compiled binary bytes."""
    _ensure_import_paths()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from avelang.backends.amdgpu import compiler as amdgpu_compiler
    from bench_qwen_gdn_v28_triton64_geometry import make_inputs
    from qwen_gdn_chunked_avelang_v28_triton64_geometry import (
        MODE_FULL_V28,
        qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry,
    )

    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped: list[Path] = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        kernel_name = src.fn.fn.__name__
        if args.kernel_substr in kernel_name:
            suffix = len(dumped)
            path = out_dir / f"{kernel_name}.{suffix}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
            print(f"dumped_avelang_hsaco={path}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        k, w, u, g, initial_state = make_inputs(args.T, args.with_initial_state, seed=args.seed + args.T)
        result = qwen_gdn_chunk_gdr_avelang_v28_triton64_geometry(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            chunk_size=64,
            variant=MODE_FULL_V28,
        )
        torch.cuda.synchronize()
        checksum = sum(float(x.float().abs().mean()) for x in result)
        print(f"dump_avelang_v28_done,T={args.T},with_initial_state={args.with_initial_state},checksum={checksum:.9g}")
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile

    if not dumped:
        raise RuntimeError(f"no Avelang kernel matching {args.kernel_substr!r} was compiled/dumped")


def dump_triton_selected(args: argparse.Namespace) -> None:
    """Trigger the real vLLM Triton selected config into a dedicated cache."""
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path(args.out_dir) / "triton_cache"))
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")
    _ensure_import_paths()

    from profile_vllm_triton_chunk_delta_h import main as vllm_main

    forwarded = [
        "profile_vllm_triton_chunk_delta_h.py",
        "--T",
        str(args.T),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--force-bv",
        "32",
        "--force-num-warps",
        "2",
        "--force-num-stages",
        "2",
    ]
    if args.with_initial_state:
        forwarded += ["--with-initial-state", "--output-final-state"]
    sys.argv = forwarded
    vllm_main()

    cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
    hsacos = sorted(cache_dir.rglob("chunk_gated_delta_rule_fwd_kernel_h_blockdim64.hsaco"))
    for path in hsacos:
        print(f"triton_selected_hsaco={path}")
    if not hsacos:
        raise RuntimeError(f"no selected Triton hsaco found under {cache_dir}")


def find_objdump() -> str:
    candidates = [
        os.environ.get("LLVM_OBJDUMP"),
        "/opt/rocm/llvm/bin/llvm-objdump",
        "/opt/rocm/bin/llvm-objdump",
        shutil.which("llvm-objdump"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError("llvm-objdump not found; set LLVM_OBJDUMP")


def disassemble(args: argparse.Namespace) -> None:
    objdump = find_objdump()
    hsaco = Path(args.hsaco)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [objdump, "-d", "--no-show-raw-insn", str(hsaco)]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out.write_text(result.stdout)
    print(f"disassembled={out}")


MNEMONIC_RE = re.compile(r"^\s*(?:[0-9a-fA-F]+:\s*(?:[0-9a-fA-F]{2}\s+)*)?([A-Za-z_][A-Za-z0-9_.$]*)\b")


def iter_mnemonics(text: str) -> Iterable[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Disassembly", "file format")) or stripped.endswith(">:"):
            continue
        match = MNEMONIC_RE.match(line)
        if match:
            yield match.group(1)


def category_for(mnemonic: str) -> str:
    if mnemonic.startswith("v_mfma"):
        return "v_mfma"
    if mnemonic.startswith(("global_load", "flat_load", "buffer_load")):
        return "global_buffer_or_flat_load"
    if mnemonic.startswith(("global_store", "flat_store", "buffer_store")):
        return "global_buffer_or_flat_store"
    if mnemonic.startswith("ds_read"):
        return "ds_read"
    if mnemonic.startswith("ds_write"):
        return "ds_write"
    if mnemonic.startswith(("ds_bpermute", "ds_permute")):
        return "ds_permute"
    if mnemonic.startswith(("s_branch", "s_cbranch")):
        return "branch"
    if mnemonic.startswith("v_"):
        return "other_valu"
    if mnemonic.startswith("s_"):
        return "salu_control"
    return "other"


def analyze_isa_file(path: Path, tail: int) -> dict[str, object]:
    text = path.read_text(errors="replace")
    mnemonics = list(iter_mnemonics(text))
    category_counts = Counter(category_for(m) for m in mnemonics)
    mnemonic_counts = Counter(mnemonics)
    store_widths = Counter(m for m in mnemonics if m.startswith(("global_store", "flat_store", "buffer_store")))
    load_widths = Counter(m for m in mnemonics if m.startswith(("global_load", "flat_load", "buffer_load")))
    tail_mnemonics = mnemonics[-tail:] if tail > 0 else []
    tail_categories = Counter(category_for(m) for m in tail_mnemonics)
    return {
        "path": str(path),
        "total_instructions": len(mnemonics),
        "categories": dict(sorted(category_counts.items())),
        "top_mnemonics": dict(mnemonic_counts.most_common(40)),
        "mfma_mnemonics": dict(sorted((m, c) for m, c in mnemonic_counts.items() if m.startswith("v_mfma"))),
        "global_store_mnemonics": dict(sorted(store_widths.items())),
        "global_load_mnemonics": dict(sorted(load_widths.items())),
        "tail_instruction_count": len(tail_mnemonics),
        "tail_categories": dict(sorted(tail_categories.items())),
        "tail_store_mnemonics": dict(
            sorted((m, c) for m, c in Counter(tail_mnemonics).items() if m.startswith(("global_store", "flat_store", "buffer_store")))
        ),
    }


def analyze_isa(args: argparse.Namespace) -> None:
    rows = [analyze_isa_file(Path(path), args.tail) for path in args.isa]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"isa_analysis={out}")
    for row in rows:
        print(json.dumps(row, indent=2, sort_keys=True))


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_rocprof_counter_csv(path: Path, kernel_substr: str) -> dict[str, float | str]:
    metrics: dict[str, list[float]] = {}
    metadata: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_substr and kernel_substr not in row.get("Kernel_Name", ""):
                continue
            for key in ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count", "Accum_VGPR_Count", "SGPR_Count"):
                if key in row and row[key] != "":
                    metadata[key] = row[key]
            name = row.get("Counter_Name") or row.get("Metric")
            value = row.get("Value") or row.get("Counter_Value")
            if name is None:
                continue
            number = _to_float(value)
            if number is not None:
                metrics.setdefault(name, []).append(number)
    summary: dict[str, float | str] = dict(metadata)
    for name, values in metrics.items():
        summary[name] = statistics.median(values)
    return summary


def parse_rocprof_trace_csv(path: Path, kernel_substr: str) -> dict[str, float]:
    durations = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if kernel_substr and kernel_substr not in row.get("Kernel_Name", ""):
                continue
            start = _to_float(row.get("Start_Timestamp", ""))
            end = _to_float(row.get("End_Timestamp", ""))
            if start is not None and end is not None:
                durations.append((end - start) / 1000.0)
    if not durations:
        return {}
    return {"trace_median_us": statistics.median(durations), "trace_count": float(len(durations))}


def parse_rocprof(args: argparse.Namespace) -> None:
    result: dict[str, object] = {}
    if args.counter_csv:
        result.update(parse_rocprof_counter_csv(Path(args.counter_csv), args.kernel_substr))
    if args.trace_csv:
        result.update(parse_rocprof_trace_csv(Path(args.trace_csv), args.kernel_substr))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"rocprof_summary={out}")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dump-avelang-v28")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR / "hsaco"))
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--seed", type=int, default=280000)
    p.add_argument("--kernel-substr", default="v28_triton64_geometry")
    p.add_argument("--with-initial-state", dest="with_initial_state", action="store_true")
    p.add_argument("--without-initial-state", dest="with_initial_state", action="store_false")
    p.set_defaults(with_initial_state=True)
    p.set_defaults(func=dump_avelang_v28)

    p = sub.add_parser("dump-triton-selected")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--with-initial-state", dest="with_initial_state", action="store_true")
    p.add_argument("--without-initial-state", dest="with_initial_state", action="store_false")
    p.set_defaults(with_initial_state=True)
    p.set_defaults(func=dump_triton_selected)

    p = sub.add_parser("disassemble")
    p.add_argument("--hsaco", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=disassemble)

    p = sub.add_parser("analyze-isa")
    p.add_argument("--isa", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tail", type=int, default=250)
    p.set_defaults(func=analyze_isa)

    p = sub.add_parser("parse-rocprof")
    p.add_argument("--counter-csv")
    p.add_argument("--trace-csv")
    p.add_argument("--kernel-substr", required=True)
    p.add_argument("--out")
    p.set_defaults(func=parse_rocprof)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

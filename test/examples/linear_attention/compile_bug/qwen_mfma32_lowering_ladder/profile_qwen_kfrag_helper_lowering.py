#!/usr/bin/env python3
"""Shared profiling implementation for Qwen K-fragment L6 experiments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

from profile_qwen_mfma32_lowering_ladder import (
    PMCS,
    analyze_isa,
    disassemble,
    parse_rocprof_dir,
    run_cmd,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_qwen_mfma32_l6_kstage_update_variants.py"
OUT_DIR = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering"
HSACO_DIR = OUT_DIR / "hsaco"
KERNEL = "_qwen_mfma32_l6_kstage_update_variants_kernel"
VARIANTS = [
    "L6_baseline_current_update",
    "L6_fixed_kfrag_producer_consumer_rewrite",
    "L6_subtile16_stage_full_update_like",
]


def load_repro():
    spec = importlib.util.spec_from_file_location("l6_kfrag_repro", REPRO)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {REPRO}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_json_output(text: str):
    start = text.find("[")
    if start < 0:
        start = text.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON payload in output:\n{text}")
    return json.loads(text[start:])


def run_smoke(args: argparse.Namespace) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    for variant in VARIANTS:
        result = run_cmd(
            [
                sys.executable,
                str(REPRO),
                "--variant",
                variant,
                "--seed",
                str(args.seed),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--json",
            ]
        )
        payload = parse_json_output(result.stdout)
        rewrite_match = re.search(
            r"Qwen K-fragment producer-consumer rewrites:\s*([0-9]+)",
            result.stdout,
        )
        for row in payload:
            row["rewrite_remark_seen"] = rewrite_match is not None
            row["rewrite_count"] = (
                int(rewrite_match.group(1)) if rewrite_match is not None else 0
            )
        rows.extend(payload)
    result = run_cmd(
        [
            sys.executable,
            str(REPRO),
            "--seed",
            str(args.seed),
            "--check-rewrite-equivalence",
        ]
    )
    return rows, parse_json_output(result.stdout)


def run_rocprof(args: argparse.Namespace, variant: str) -> dict[str, object]:
    out_dir = args.out_dir / f"rocprof_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_cmd(
        [
            "/opt/rocm/bin/rocprofv3",
            "--kernel-trace",
            "--pmc",
            *PMCS,
            "--kernel-include-regex",
            KERNEL,
            "-d",
            str(out_dir),
            "-o",
            f"{variant}_counters",
            "-f",
            "csv",
            "--",
            sys.executable,
            str(REPRO),
            "--variant",
            variant,
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.rocprof_warmup),
            "--repeat",
            str(args.rocprof_repeat),
        ],
        check=False,
    )
    row = parse_rocprof_dir(out_dir, KERNEL)
    row["returncode"] = result.returncode
    row["stdout_tail"] = "\n".join(result.stdout.splitlines()[-20:])
    return row


def dump_hsaco(args: argparse.Namespace, variant: str) -> Path:
    import torch
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    repro = load_repro()
    args.hsaco_dir.mkdir(parents=True, exist_ok=True)
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped: list[Path] = []

    def wrapped_compile(self, src, target, options=None):
        binary = original_compile(self, src, target, options)
        if KERNEL in src.fn.fn.__name__ and not dumped:
            path = args.hsaco_dir / f"{variant}.{src.fn.fn.__name__}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        tensors = repro.make_inputs(args.seed)
        repro.launch_variant(variant, tensors)
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no hsaco dumped for {variant}")
    return dumped[0]


ACC_WRITE_RE = re.compile(r"\bv_accvgpr_write_b32\s+a([0-9]+)")
ACC_READ_RE = re.compile(r"\bv_accvgpr_read_b32\b[^,]*,\s*a([0-9]+)")


def analyze_high_agpr(isa_path: Path) -> dict[str, object]:
    text = isa_path.read_text(errors="replace")
    writes = [int(value) for value in ACC_WRITE_RE.findall(text)]
    reads = [int(value) for value in ACC_READ_RE.findall(text)]
    return {
        "max_accvgpr_write_index": max(writes) if writes else None,
        "max_accvgpr_read_index": max(reads) if reads else None,
        "accvgpr_write_count": len(writes),
        "accvgpr_read_count": len(reads),
        "high_accvgpr_write_count_ge100": sum(value >= 100 for value in writes),
        "high_accvgpr_read_count_ge100": sum(value >= 100 for value in reads),
        "has_a100_a131_write_region": all(value in writes for value in range(100, 132)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hsaco-dir", type=Path, default=HSACO_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([sys.executable, "-m", "py_compile", str(REPRO)])
    smoke, equivalence = run_smoke(args)
    (args.out_dir / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True))
    (args.out_dir / "rewrite_equivalence.json").write_text(
        json.dumps(equivalence, indent=2, sort_keys=True)
    )

    rocprof = {variant: run_rocprof(args, variant) for variant in VARIANTS}
    (args.out_dir / "rocprof_summary.json").write_text(
        json.dumps(rocprof, indent=2, sort_keys=True)
    )

    isa = {}
    for variant in VARIANTS:
        hsaco = dump_hsaco(args, variant)
        isa_path = disassemble(hsaco)
        row = analyze_isa(isa_path)
        row.update(analyze_high_agpr(isa_path))
        row["hsaco"] = str(hsaco)
        row["isa"] = str(isa_path)
        isa[variant] = row
    (args.out_dir / "isa_summary.json").write_text(
        json.dumps(isa, indent=2, sort_keys=True)
    )
    print(json.dumps({"smoke": smoke, "equivalence": equivalence, "rocprof": rocprof, "isa": isa}, indent=2))


if __name__ == "__main__":
    main()

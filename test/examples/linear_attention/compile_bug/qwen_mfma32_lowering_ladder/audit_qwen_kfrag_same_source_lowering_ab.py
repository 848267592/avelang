#!/usr/bin/env python3
"""Same-source A/B audit for generic versus late Qwen B-fragment lowering.

Both runs compile exactly the same R3 source kernel.  The only compile-time
switch is AVELANG_QWEN_KFRAG_LATE_BLOAD, read after pre_kfrag_branch.mlir is
snapshotted.  R3 is used because its observable sink depends on the real
pred -> v_decay -> update B-fragment dataflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from profile_qwen_kfrag_helper_lowering import (
    PMCS,
    analyze_high_agpr,
    analyze_isa,
    disassemble,
    parse_rocprof_dir,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
REPRO = SCRIPT_DIR / "repro_qwen_kfrag_full_loop_regression.py"
KERNEL = "_qwen_kfrag_full_loop_regression_kernel"
VARIANT = "R3_rewrite_plus_state_update_writeback"
DEFAULT_OUT = PROJECT_ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_kfrag_same_source_lowering_ab"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )


def parse_json(text: str) -> list[dict[str, object]]:
    start = text.rfind("\n[")
    if start >= 0:
        start += 1
    else:
        start = text.find("[")
    if start < 0:
        raise RuntimeError(f"missing JSON payload:\n{text}")
    return json.loads(text[start:])


def count_matches(path: Path, pattern: str) -> int:
    return len(re.findall(pattern, path.read_text(errors="replace")))


def make_env(out_dir: Path, label: str, late_bload: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "AVELANG_QWEN_KFRAG_LATE_BLOAD": "1" if late_bload else "0",
            "AVELANG_QWEN_KFRAG_DEBUG": "1",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR": str(out_dir / label / "ir"),
            "AVELANG_AMDGPU_LINK_DEBUG_DIR": str(out_dir / label / "link"),
        }
    )
    return env


def run_smoke(
    out_dir: Path,
    label: str,
    late_bload: bool,
    args: argparse.Namespace,
    input_path: Path | None = None,
) -> dict[str, object]:
    label_dir = out_dir / label
    sink_path = label_dir / "sink.pt"
    shared_inputs = out_dir / "frozen_inputs.pt"
    input_args = (
        ["--input-path", str(input_path)]
        if input_path is not None
        else ["--save-inputs", str(shared_inputs)]
    )
    result = run(
        [
            sys.executable,
            str(REPRO),
            "--variant", VARIANT,
            "--seed", str(args.seed),
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--save-sink", str(sink_path),
            *input_args,
            "--json",
        ],
        make_env(out_dir, label, late_bload),
    )
    (label_dir / "smoke.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} smoke failed:\n{result.stdout}")
    row = parse_json(result.stdout)[0]
    row["rewrite_debug"] = "[qwen-kfrag] rewritten=1" in result.stdout
    row["late_bload"] = late_bload
    return row


def run_profile(out_dir: Path, label: str, late_bload: bool, args: argparse.Namespace) -> dict[str, object]:
    profile_dir = out_dir / label / "rocprof"
    result = run(
        [
            "/opt/rocm/bin/rocprofv3",
            "--kernel-trace",
            "--pmc", *PMCS,
            "--kernel-include-regex", KERNEL,
            "-d", str(profile_dir),
            "-o", label,
            "-f", "csv",
            "--", sys.executable, str(REPRO),
            "--variant", VARIANT,
            "--seed", str(args.seed),
            "--warmup", str(args.rocprof_warmup),
            "--repeat", str(args.rocprof_repeat),
            "--input-path", str(out_dir / "frozen_inputs.pt"),
            "--json",
        ],
        make_env(out_dir, label, late_bload),
    )
    (out_dir / label / "rocprof.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} rocprof failed:\n{result.stdout}")
    row = parse_rocprof_dir(profile_dir, KERNEL)
    row["returncode"] = result.returncode
    return row


def find_hsaco(link_dir: Path) -> Path:
    candidates = sorted(link_dir.glob("*.linked.out"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one linked code object in {link_dir}, found {candidates}")
    return candidates[0]


def compare_sinks(generic: Path, specialized: Path) -> dict[str, object]:
    lhs = torch.load(generic, weights_only=True)
    rhs = torch.load(specialized, weights_only=True)
    different = lhs.ne(rhs)
    mismatch_count = int(different.sum().item())
    result: dict[str, object] = {
        "shape_equal": list(lhs.shape) == list(rhs.shape),
        "dtype_equal": lhs.dtype == rhs.dtype,
        "bit_exact": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "max_abs": float((lhs - rhs).abs().max().item()),
    }
    if mismatch_count:
        result["first_mismatch_flat_index"] = int(different.flatten().nonzero()[0].item())
    return result


def inspect_ir(out_dir: Path) -> dict[str, object]:
    generic = out_dir / "A_generic" / "ir"
    specialized = out_dir / "B_specialized" / "ir"
    phases = [
        "pre_kfrag_branch.mlir",
        "post_kfrag_rewrite.mlir",
        "post_kfrag_load_lowering.mlir",
        "final_mlir.mlir",
        "preopt_llvm.ll",
        "postopt_llvm.ll",
    ]
    rows: dict[str, object] = {}
    for phase in phases:
        lhs = generic / phase
        rhs = specialized / phase
        if not lhs.exists() or not rhs.exists():
            raise RuntimeError(f"missing audit snapshot {phase}")
        rows[phase] = {
            "generic_sha256": sha256(lhs),
            "specialized_sha256": sha256(rhs),
            "identical": lhs.read_bytes() == rhs.read_bytes(),
        }
    pre = generic / "pre_kfrag_branch.mlir"
    post_generic = generic / "post_kfrag_load_lowering.mlir"
    post_specialized = specialized / "post_kfrag_load_lowering.mlir"
    rows["structural_counts"] = {
        "pre_persistent_op": count_matches(pre, r"amdgpu_qwen_update_kfrag_load"),
        "generic_vector_load": count_matches(post_generic, r"vector\.load"),
        "specialized_direct_lds_attr": count_matches(post_specialized, r"qwen_kfrag\.direct_lds_b64"),
        "generic_shared_alloc": count_matches(post_generic, r"memref\.alloca"),
        "specialized_shared_alloc": count_matches(post_specialized, r"memref\.alloca"),
    }
    return rows


def inspect_isa(out_dir: Path) -> dict[str, object]:
    rows: dict[str, object] = {}
    for label in ("A_generic", "B_specialized"):
        hsaco = find_hsaco(out_dir / label / "link")
        isa = disassemble(hsaco)
        isa_lines = isa.read_text(errors="replace").splitlines()
        normalized_isa = "\n".join(
            line for line in isa_lines if not line.endswith(":\tfile format elf64-amdgpu")
        )
        row = analyze_isa(isa)
        row.update(analyze_high_agpr(isa))
        text = isa.read_text(errors="replace")
        row.update(
            {
                "hsaco": str(hsaco),
                "isa": str(isa),
                "isa_sha256": sha256(isa),
                "normalized_isa_sha256": hashlib.sha256(normalized_isa.encode()).hexdigest(),
                "mfma16_count": len(re.findall(r"v_mfma_f32_16x16x16_bf16", text)),
                "mfma32_count": len(re.findall(r"v_mfma_f32_32x32x8_bf16", text)),
                "ds_read_b64_count": len(re.findall(r"ds_read_b64", text)),
                "ds_write_count": len(re.findall(r"\bds_write", text)),
                "global_load_count": len(re.findall(r"\b(?:global|buffer)_load", text)),
            }
        )
        rows[label] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--skip-rocprof", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for label in ("A_generic", "B_specialized"):
        (args.out_dir / label).mkdir(parents=True)

    smoke = {
        "A_generic": run_smoke(args.out_dir, "A_generic", False, args),
        "B_specialized": run_smoke(
            args.out_dir,
            "B_specialized",
            True,
            args,
            args.out_dir / "frozen_inputs.pt",
        ),
    }
    result: dict[str, object] = {
        "contract": {
            "source": str(REPRO),
            "variant": VARIANT,
            "only_branch": "AVELANG_QWEN_KFRAG_LATE_BLOAD after pre_kfrag_branch.mlir",
            "frozen_inputs_sha256": sha256(args.out_dir / "frozen_inputs.pt"),
        },
        "smoke": smoke,
        "sink_comparison": compare_sinks(
            args.out_dir / "A_generic" / "sink.pt",
            args.out_dir / "B_specialized" / "sink.pt",
        ),
        "ir": inspect_ir(args.out_dir),
        "isa": inspect_isa(args.out_dir),
    }
    if not args.skip_rocprof:
        result["rocprof"] = {
            "A_generic": run_profile(args.out_dir, "A_generic", False, args),
            "B_specialized": run_profile(args.out_dir, "B_specialized", True, args),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

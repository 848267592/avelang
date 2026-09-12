#!/usr/bin/env python3
"""Strict full chunk_gdr A/B for generic versus late B-fragment lowering.

The worker imports one existing full v29 rewrite kernel. A and B receive the
same frozen inputs and issue the same raw preallocated launch. The sole
compile-time branch is AVELANG_QWEN_KFRAG_LATE_BLOAD after the audit hook has
written pre_kfrag_branch.mlir.
"""

from __future__ import annotations

import argparse
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
DEFAULT_OUT = ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_full_chunk_gdr_kfrag_lowering_ab"
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
    start = text.rfind("\n{")
    if start >= 0:
        return json.loads(text[start + 1 :])
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"worker did not emit JSON:\n{text}")
    return json.loads(text[start:])


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
        if args.save_inputs:
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
    samples = []
    for _ in range(args.repeat):
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    if args.save_outputs:
        args.save_outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save((h.detach().cpu(), final_state.detach().cpu()), args.save_outputs)
    print(
        json.dumps(
            {
                "kernel": KERNEL,
                "latency_ms_median": statistics.median(samples),
                "h_checksum_abs": float(h.abs().sum().item()),
                "final_state_checksum_abs": float(final_state.abs().sum().item()),
                "input_checksums_abs": [float(tensor.abs().sum().item()) for tensor in (k, w, u, decay, g_last_exp, initial_state)],
                "h_finite": bool(torch.isfinite(h).all().item()),
                "final_state_finite": bool(torch.isfinite(final_state).all().item()),
            },
            sort_keys=True,
        )
    )


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


def make_env(out_dir: Path, label: str, late: bool) -> dict[str, str]:
    env = os.environ.copy()
    # Preserve the caller's binding precedence.  The ROCm experiment runs
    # against the freshly built binding in /tmp; putting ROOT/python first
    # silently selects the stale workspace extension and hides the dedicated
    # intrinsic from this otherwise identical A/B.
    pythonpath = [str(VLLM_COMPARE)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    else:
        pythonpath.append(str(ROOT / "python"))
    env.update(
        {
            "PYTHONPATH": ":".join(pythonpath),
            "AVELANG_QWEN_KFRAG_LATE_BLOAD": "1" if late else "0",
            "AVELANG_QWEN_KFRAG_DEBUG": "1",
            "AVELANG_QWEN_KFRAG_AB_DUMP_DIR": str(out_dir / label / "ir"),
            "AVELANG_AMDGPU_LINK_DEBUG_DIR": str(out_dir / label / "link"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def run_worker(out_dir: Path, label: str, late: bool, args: argparse.Namespace, input_path: Path | None) -> dict[str, object]:
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
    result = run(command, make_env(out_dir, label, late))
    (label_dir / "smoke.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} worker failed:\n{result.stdout}")
    row = parse_json_payload(result.stdout)
    row["rewrite_fired"] = "[qwen-kfrag] rewritten=1" in result.stdout
    row["late_bload"] = late
    return row


def compare_outputs(a_path: Path, b_path: Path) -> dict[str, object]:
    a_h, a_final = torch.load(a_path, weights_only=True)
    b_h, b_final = torch.load(b_path, weights_only=True)

    def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
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

    return {"h": compare(a_h, b_h), "final_state": compare(a_final, b_final)}


def count(path: Path, pattern: str) -> int:
    return len(re.findall(pattern, path.read_text(errors="replace")))


def inspect_ir(out_dir: Path) -> dict[str, object]:
    a_dir = out_dir / "A_generic" / "ir"
    b_dir = out_dir / "B_specialized" / "ir"
    names = [
        "pre_kfrag_branch.mlir",
        "post_kfrag_rewrite.mlir",
        "post_kfrag_load_lowering.mlir",
        "final_mlir.mlir",
        "preopt_llvm.ll",
        "postopt_llvm.ll",
    ]
    result: dict[str, object] = {}
    for name in names:
        a = a_dir / name
        b = b_dir / name
        if not a.exists() or not b.exists():
            raise RuntimeError(f"missing snapshot {name}")
        result[name] = {
            "generic_sha256": sha256(a),
            "specialized_sha256": sha256(b),
            "identical": a.read_bytes() == b.read_bytes(),
        }
    generic_post = a_dir / "post_kfrag_load_lowering.mlir"
    special_post = b_dir / "post_kfrag_load_lowering.mlir"
    result["structural_counts"] = {
        "pre_persistent_ops": count(a_dir / "pre_kfrag_branch.mlir", r"amdgpu_qwen_update_kfrag_load"),
        "generic_vector_loads": count(generic_post, r"vector\.load"),
        "specialized_direct_loads": count(special_post, r"qwen_kfrag\.direct_lds_b64"),
        "generic_alloca": count(generic_post, r"memref\.alloca"),
        "specialized_alloca": count(special_post, r"memref\.alloca"),
    }
    return result


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


def find_hsaco(link_dir: Path) -> Path:
    matches = sorted(link_dir.glob("*.linked.out"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one code object in {link_dir}, got {matches}")
    return matches[0]


def isa_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    normalized = "\n".join(line for line in text.splitlines() if not line.endswith(":\tfile format elf64-amdgpu"))
    max_agpr = [int(value) for value in re.findall(r"v_accvgpr_write_b32\s+a(\d+)", text)]
    return {
        "sha256": sha256(path),
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "mfma16": len(re.findall(r"v_mfma_f32_16x16x16_bf16", text)),
        "mfma32": len(re.findall(r"v_mfma_f32_32x32x8_bf16", text)),
        "ds_read_b64": len(re.findall(r"ds_read_b64", text)),
        "ds_write": len(re.findall(r"\bds_write", text)),
        "global_or_buffer_load": len(re.findall(r"\b(?:global|buffer)_load", text)),
        "barrier": len(re.findall(r"\bs_barrier", text)),
        "accvgpr_write": len(re.findall(r"v_accvgpr_write_b32", text)),
        "accvgpr_read": len(re.findall(r"v_accvgpr_read_b32", text)),
        "max_explicit_agpr_write": max(max_agpr) if max_agpr else None,
        "high_agpr_write_ge100": len([value for value in max_agpr if value >= 100]),
    }


def parse_profile(profile_dir: Path) -> dict[str, object]:
    trace = next(profile_dir.glob("*_kernel_trace.csv"))
    counters = next(profile_dir.glob("*_counter_collection.csv"))
    import csv

    with trace.open() as handle:
        rows = [row for row in csv.DictReader(handle) if KERNEL in row.get("Kernel_Name", "")]
    if not rows:
        raise RuntimeError(f"no {KERNEL} rows in {trace}")
    durations = sorted((int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0 for row in rows)
    median = statistics.median(durations)
    first = rows[-1]
    with counters.open() as handle:
        counter_rows = list(csv.DictReader(handle))
    matching = [row for row in counter_rows if KERNEL in row.get("Kernel_Name", "")]
    if not matching:
        raise RuntimeError(f"no {KERNEL} counters in {counters}")
    result = {
        "trace_median_us": median,
        "trace_count": len(rows),
        "VGPR_Count": first.get("VGPR_Count"),
        "Accum_VGPR_Count": first.get("Accum_VGPR_Count"),
        "SGPR_Count": first.get("SGPR_Count"),
        "Scratch_Size": first.get("Scratch_Size"),
        "LDS_Block_Size": first.get("LDS_Block_Size"),
        "Workgroup_Size": first.get("Workgroup_Size_X"),
        "Grid_Size": first.get("Grid_Size_X"),
        "counter_csv": str(counters),
        "trace_csv": str(trace),
    }
    for counter in matching:
        counter_name = counter.get("Counter_Name")
        if counter_name in PMCS and counter.get("Counter_Value"):
            result[counter_name] = float(counter["Counter_Value"])
    return result


def run_rocprof(out_dir: Path, label: str, late: bool, args: argparse.Namespace) -> dict[str, object]:
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
    result = run(command, make_env(out_dir, label, late))
    (out_dir / label / "rocprof.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{label} rocprof failed:\n{result.stdout}")
    return parse_profile(profile_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--T", "--t", dest="t", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rocprof-warmup", type=int, default=2)
    parser.add_argument("--rocprof-repeat", type=int, default=5)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--save-inputs", type=Path)
    parser.add_argument("--save-outputs", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-rocprof", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.t % 64:
        raise ValueError("T must be divisible by 64")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    for label in ("A_generic", "B_specialized"):
        (args.out_dir / label).mkdir(parents=True)
    smoke = {
        "A_generic": run_worker(args.out_dir, "A_generic", False, args, None),
        "B_specialized": run_worker(
            args.out_dir,
            "B_specialized",
            True,
            args,
            args.out_dir / "frozen_inputs.pt",
        ),
    }
    isa: dict[str, object] = {}
    for label in ("A_generic", "B_specialized"):
        isa[label] = isa_metrics(disassemble(find_hsaco(args.out_dir / label / "link")))
    result: dict[str, object] = {
        "contract": {
            "source_module": FULL_MODULE,
            "kernel": KERNEL,
            "t": args.t,
            "only_branch": "AVELANG_QWEN_KFRAG_LATE_BLOAD after pre_kfrag_branch.mlir",
            "frozen_inputs_sha256": sha256(args.out_dir / "frozen_inputs.pt"),
        },
        "smoke": smoke,
        "outputs": compare_outputs(
            args.out_dir / "A_generic" / "outputs.pt",
            args.out_dir / "B_specialized" / "outputs.pt",
        ),
        "ir": inspect_ir(args.out_dir),
        "isa": isa,
    }
    if not args.skip_rocprof:
        result["rocprof"] = {
            "A_generic": run_rocprof(args.out_dir, "A_generic", False, args),
            "B_specialized": run_rocprof(args.out_dir, "B_specialized", True, args),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

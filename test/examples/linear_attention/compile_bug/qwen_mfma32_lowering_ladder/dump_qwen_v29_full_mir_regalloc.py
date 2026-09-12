#!/usr/bin/env python3
"""Dump exact full-v29 original/rewrite LLVM IR and pre/post-RA MIR.

This is a JIT/backend debug-chain probe: it reconstructs the same ASTSource
the AMDGPU JIT sees at T=2048, then invokes the installed ROCm llc at late
machine-code stop points. It does not change a Qwen kernel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch

from dump_l6_mir_regalloc_artifacts import build_generator, run_llc_variants
from avelang.runtime.driver import driver


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
COMPARE = ROOT / "test/examples/linear_attention/vllm_compare"
sys.path.insert(0, str(COMPARE))

import qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full as original
import qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_rewrite_exp as rewrite


OUT = ROOT / "test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc"
STACK_RE = re.compile(r"stack:\s*\[.*?\]", re.DOTALL)
FRAME_RE = re.compile(r"stackSize:\s*(\d+)")
VREG_RE = re.compile(r"%([0-9]+)(?::[A-Za-z0-9_]+)?")
MFMA_RE = re.compile(r"V_MFMA|v_mfma", re.IGNORECASE)
SPILL_RE = re.compile(r"spill|stack\.\d+", re.IGNORECASE)


def make_source(module, kernel, tensors):
    device = driver.active.get_current_device()
    _cache, _key, target, backend, binder = kernel.device_caches[device]
    bound, specialization, options = binder(*tensors, num_warps=2)
    options, signature, constexprs, globals_, attrs = kernel._pack_args(
        backend, {"num_warps": 2}, bound, specialization, options
    )
    return kernel.ASTSource(kernel, signature, constexprs, attrs, globals_), target, options


def summarize(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    frames = [int(v) for v in FRAME_RE.findall(text)]
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "vreg_count": len(set(VREG_RE.findall(text))),
        "mfma_lines": sum(1 for line in text.splitlines() if MFMA_RE.search(line)),
        "spill_mentions": len(SPILL_RE.findall(text)),
        "frame_sizes": frames,
    }


def dump_variant(name: str, module, kernel, args: argparse.Namespace) -> dict[str, object]:
    data = module._make_inputs(args.T, seed=args.seed)
    k, w, u, decay, g_last, initial = data
    chunks = args.T // 64
    h = torch.empty((1, chunks, 8, 128, 128), dtype=torch.float32, device=k.device)
    final = torch.empty((1, 8, 128, 128), dtype=torch.float32, device=k.device)
    src, target, options = make_source(
        module, kernel, (k, w, u, decay, g_last, initial, h, final, args.T, chunks, True)
    )
    out = args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    llvm_ir = build_generator(src).get_llvm_ir(target.tuple, target.chip, args.opt_level, options.num_warps)
    ll = out / "lowered_optimized.ll"
    ll.write_text(llvm_ir)
    llc = run_llc_variants(ll, out, target.chip)
    mir = {}
    for stop in ("amdgpu-isel", "greedy", "virtregrewriter", "prologepilog", "postrapseudos"):
        p = Path(llc.get(stop, {}).get("mir", ""))
        if p:
            mir[stop] = summarize(p)
    return {"target": target.tuple, "chip": target.chip, "llvm_ir": str(ll), "llc": llc, "mir": mir}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--opt-level", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "original": dump_variant("original", original, original._qwen_gdn_fused_chunk_gdr_full_bf16_kernel_v29_mfma32, args),
        "kfrag_rewrite": dump_variant("kfrag_rewrite", rewrite, rewrite._qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32, args),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

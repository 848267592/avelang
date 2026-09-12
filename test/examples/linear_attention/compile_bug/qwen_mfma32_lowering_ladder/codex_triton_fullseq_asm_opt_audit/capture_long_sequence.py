#!/usr/bin/env python3
"""Freeze the real vLLM BT64 chunk-delta-h specialization across sequence lengths.

This script intentionally calls the installed vLLM wrapper.  It is not an
Avelang implementation and its cache contents are retained as local audit
artifacts only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "triton_cache_longseq"
os.environ.setdefault("TRITON_CACHE_DIR", str(DEFAULT_CACHE))

VLLM_COMPARE = HERE.parents[4] / "vllm_compare"
sys.path.insert(0, str(VLLM_COMPARE))
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402


BT, HK, HV, KDIM, VDIM = 64, 4, 8, 128, 128
KERNEL_NAME = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    k = (torch.randn((1, t, HK, KDIM), device="cuda", dtype=torch.bfloat16) * 0.05).contiguous()
    w = (torch.randn((1, t, HV, KDIM), device="cuda") * 0.04).contiguous()
    u = (torch.randn((1, t, HV, VDIM), device="cuda") * 0.04).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn((1, t, HV), device="cuda")) / 16.0).contiguous()
    h0 = (torch.randn((1, HV, VDIM, KDIM), device="cuda") * 0.03).contiguous()
    return k, w, u, g, h0


def patch_rocm_autotune() -> None:
    autotuner = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn
    original = list(autotuner.configs)
    filtered = [config for config in original if config.num_stages != 4]
    if len(filtered) != len(original):
        autotuner.configs = filtered
        autotuner.cache.clear()


def call(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor, h0: torch.Tensor):
    return chunk_delta_h.chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, gk=None, initial_state=h0,
        output_final_state=True, chunk_size=BT, save_new_value=True, cu_seqlens=None,
    )


def time_call(fn, warmup: int, repeat: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(repeat):
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        values.append(float(start.elapsed_time(end)))
    values.sort()
    return {
        "median_ms": statistics.median(values),
        "p10_ms": values[(len(values) - 1) // 10],
        "p90_ms": values[(len(values) - 1) * 9 // 10],
    }


def selected_kernel() -> tuple[object, object]:
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = kernel.fn
    config = next(iter(autotuner.cache.values()))
    compiled = autotuner.fn.device_caches[0][0]
    candidates = [
        value for key, value in compiled.items()
        if f"'num_warps': {config.num_warps}" in key
        and f"'num_stages': {config.num_stages}" in key
        and f"('constexpr', {config.kwargs['BV']})" in key
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify selected compiled kernel: {len(candidates)} candidates")
    return config, candidates[0]


def cache_files_for_hash(cache: Path, compile_hash: str) -> list[Path]:
    groups: set[Path] = set()
    for metadata in cache.rglob("*.json"):
        try:
            if compile_hash in metadata.read_text(errors="ignore"):
                groups.add(metadata.parent)
        except OSError:
            pass
    return sorted(path for group in groups for path in group.iterdir() if path.is_file())


def write_shared_artifacts(compiled, out: Path) -> None:
    shared = out / "golden_fullseq/shared"
    shared.mkdir(parents=True, exist_ok=True)
    names = {"hsaco": "original.hsaco", "amdgcn": "original_from_triton.s", "ttir": "original.ttir", "ttgir": "original.ttgir", "llir": "original.llir"}
    for asm_key, name in names.items():
        value = compiled.asm.get(asm_key)
        if value is None:
            continue
        path = shared / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[64, 128, 512, 2048, 8192, 16384])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires HIP GPU")
    if any(t < BT or t % BT for t in args.T):
        raise ValueError("all T values must be positive multiples of 64")

    out = args.out
    cache = Path(os.environ["TRITON_CACHE_DIR"])
    patch_rocm_autotune()
    rows = []
    compiled_hashes: dict[str, dict[str, object]] = {}
    for t in args.T:
        tensors = make_inputs(t, args.seed + t)
        result = call(*tensors)
        torch.cuda.synchronize()
        config, compiled = selected_kernel()
        metadata = compiled.metadata._asdict()
        compile_hash = str(metadata["hash"])
        hsaco = compiled.asm["hsaco"]
        hsaco_hash = hashlib.sha256(hsaco).hexdigest()
        write_shared_artifacts(compiled, out)
        compiled_hashes.setdefault(compile_hash, {
            "hsaco_sha256": hsaco_hash,
            "metadata": metadata,
            "source_signature": compiled.src.signature,
            "source_constants": {str(key): value for key, value in compiled.src.constants.items()},
            "cache_files": [str(path.relative_to(cache)) for path in cache_files_for_hash(cache, compile_hash)],
        })
        timing = None if args.skip_timing else time_call(lambda: call(*tensors), args.warmup, args.repeat)
        rows.append({
            "T": t,
            "kernel_symbol": KERNEL_NAME,
            "launches_per_wrapper_call": 1,
            "grid": [4, 8, 1],
            "workgroup": [config.num_warps * 64, 1, 1],
            "dynamic_lds_bytes": int(metadata["shared"]),
            "cache_compile_hash": compile_hash,
            "hsaco_sha256": hsaco_hash,
            "selected_config": {"BV": config.kwargs["BV"], "num_warps": config.num_warps, "num_stages": config.num_stages, "num_ctas": config.num_ctas},
            "runtime_T": True,
            "outputs": {"h": list(result[0].shape), "v_new": list(result[1].shape), "final_state": list(result[2].shape)},
            "timing": timing,
        })
    manifest = {
        "cache_dir": str(cache),
        "file_count": len([path for path in cache.rglob("*") if path.is_file()]),
        "compiled_hashes": compiled_hashes,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "long_sequence_specializations.json").write_text(json.dumps({"rows": rows}, indent=2, default=str) + "\n")
    (out / "long_sequence_cache_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (out / "long_sequence_execution_graph.md").write_text(
        "# Long Sequence Execution Graph\n\n"
        "The real vLLM wrapper launches one `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` dispatch per call. "
        "`T` is runtime (`@triton.jit(do_not_specialize=[\"T\"])`); the kernel loops over `ceil(T/64)` chunks inside each program. "
        "For B=1,H=8,V=128,BV=32, the grid remains `(4,8,1)` for every tested length.\n\n"
        "| T | HSACO SHA256 | config | grid | WG | dynamic LDS | event median ms |\n|--:|:--|:--|:--|--:|--:|--:|\n" +
        "".join(f"| {row['T']} | `{row['hsaco_sha256']}` | BV={row['selected_config']['BV']}, w={row['selected_config']['num_warps']}, s={row['selected_config']['num_stages']} | `(4,8,1)` | {row['workgroup'][0]} | {row['dynamic_lds_bytes']} | {row['timing']['median_ms'] if row['timing'] else 'not measured'} |\n" for row in rows)
    )
    print(json.dumps({"rows": rows, "compile_hashes": list(compiled_hashes)}, indent=2, default=str))


if __name__ == "__main__":
    main()

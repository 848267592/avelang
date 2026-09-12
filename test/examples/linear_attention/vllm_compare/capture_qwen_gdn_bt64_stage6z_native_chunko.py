#!/usr/bin/env python3
"""Z0-only capture of the native vLLM BT64 ``chunk_fwd_kernel_o`` selection.

The script deliberately creates no Avelang kernel.  In a fresh process it
runs the native eager public API, reads the actual Triton autotuner choice,
copies only the matching cache group, and records source/IR/ISA metadata.
The optional rocprof trace is launched by the surrounding Z0 driver so its
autotune prefix can be retained separately from this functional capture.
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
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"

BT = 64
H = 8
HG = 4
K = 128
V = 128


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_dict(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {"available": False}
    return {
        "available": True,
        "kwargs": dict(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", None),
        "maxnreg": getattr(config, "maxnreg", None),
    }


def autotuner_for(kernel: Any) -> Any:
    current = kernel
    while current is not None:
        if hasattr(current, "configs") and hasattr(current, "best_config"):
            return current
        next_kernel = getattr(current, "fn", None)
        if next_kernel is current:
            break
        current = next_kernel
    raise RuntimeError("could not locate Triton Autotuner under chunk_fwd_kernel_o")


def selected_config(autotuner: Any) -> Any:
    config = getattr(autotuner, "best_config", None)
    if config is not None:
        return config
    cache = getattr(autotuner, "cache", {})
    if cache:
        return next(iter(cache.values()))
    raise RuntimeError("native chunk-o public call completed without an autotuner selection")


def cache_groups(cache_root: Path) -> list[Path]:
    return sorted(path.parent for path in cache_root.rglob("chunk_fwd_kernel_o.json"))


def candidate_summary(group: Path) -> dict[str, Any]:
    metadata_path = group / "chunk_fwd_kernel_o.json"
    ttir_path = group / "chunk_fwd_kernel_o.ttir"
    metadata = json.loads(metadata_path.read_text())
    ttir = ttir_path.read_text() if ttir_path.exists() else ""
    return {
        "group": str(group),
        "metadata": metadata,
        "ttir": ttir,
        "files": sorted(path.name for path in group.glob("chunk_fwd_kernel_o.*")),
    }


def match_selected_group(groups: list[Path], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    kwargs = config.get("kwargs", {})
    bk = int(kwargs.get("BK", -1))
    bv = int(kwargs.get("BV", -1))
    warps = config.get("num_warps")
    stages = config.get("num_stages")
    candidates: list[dict[str, Any]] = []
    for group in groups:
        item = candidate_summary(group)
        metadata = item["metadata"]
        ttir = item["ttir"]
        # ``b_A`` is always [64,64], so generic tensor-shape presence is not
        # enough to distinguish BV=32 from BV=64.  Match the reduction loop
        # and the actual V-new load produced by source line 133 instead.
        reduction_trip_count = 128 // bk if bk > 0 and 128 % bk == 0 else -1
        loop_bk_match = bool(re.search(r"scf\.for .*? to %c" + str(reduction_trip_count) + r"_i32", ttir))
        v_load_bv_match = bool(
            re.search(r"%b_v[^=]* = tt\.load [^\n]* : tensor<64x" + str(bv) + r"x!tt\.ptr<bf16>>", ttir)
        )
        score = 0
        score += 4 if metadata.get("num_warps") == warps else 0
        score += 4 if metadata.get("num_stages") == stages else 0
        score += 2 if loop_bk_match else 0
        score += 2 if v_load_bv_match else 0
        item.update(
            {
                "match_score": score,
                "loop_bk_match": loop_bk_match,
                "v_load_bv_match": v_load_bv_match,
                "reduction_trip_count": reduction_trip_count,
            }
        )
        candidates.append(item)
    if not candidates:
        raise RuntimeError("no chunk_fwd_kernel_o cache groups were produced by the public call")
    ranked = sorted(candidates, key=lambda item: (int(item["match_score"]), str(item["group"])), reverse=True)
    best = ranked[0]
    if int(best["match_score"]) < 10:
        raise RuntimeError(
            "could not map native autotuner config to a unique cache group; "
            f"best score={best['match_score']} config={config} group={best['group']}"
        )
    equally_ranked = [item for item in ranked if item["match_score"] == best["match_score"]]
    if len(equally_ranked) != 1:
        raise RuntimeError(
            "native cache matching is ambiguous; refusing to infer selected specialization: "
            + ", ".join(str(item["group"]) for item in equally_ranked)
        )
    return best, ranked


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def static_summary(group: Path) -> dict[str, Any]:
    contents = {
        suffix: (group / f"chunk_fwd_kernel_o.{suffix}").read_text(errors="replace")
        if (group / f"chunk_fwd_kernel_o.{suffix}").exists()
        else ""
        for suffix in ("ttir", "ttgir", "llir", "amdgcn")
    }
    return {
        "ttir_dot_ops": count(contents["ttir"], r"tt\.dot"),
        "ttgir_local_alloc": count(contents["ttgir"], r"ttg\.local_alloc"),
        "ttgir_local_store": count(contents["ttgir"], r"ttg\.local_store"),
        "ttgir_local_load": count(contents["ttgir"], r"ttg\.local_load"),
        "ttgir_local_dealloc": count(contents["ttgir"], r"ttg\.local_dealloc"),
        "ttgir_mfma32_layout": count(contents["ttgir"], r"instrShape = \[32, 32, 8\]"),
        "amdgcn_mfma32": count(contents["amdgcn"], r"v_mfma_f32_32x32x8_bf16"),
        "amdgcn_mfma16": count(contents["amdgcn"], r"v_mfma_f32_16x16x16_bf16"),
        "amdgcn_ds_read": count(contents["amdgcn"], r"\bds_read"),
        "amdgcn_ds_write": count(contents["amdgcn"], r"\bds_write"),
        "amdgcn_barrier": count(contents["amdgcn"], r"\bs_barrier\b"),
        "amdgcn_buffer_load": count(contents["amdgcn"], r"\bbuffer_load"),
        "amdgcn_buffer_store": count(contents["amdgcn"], r"\bbuffer_store"),
        "amdgcn_global_load": count(contents["amdgcn"], r"\bglobal_load"),
        "amdgcn_global_store": count(contents["amdgcn"], r"\bglobal_store"),
    }


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {"command": command, "returncode": result.returncode, "output": result.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "output": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--final-calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026072200)
    args = parser.parse_args()
    if args.T < BT or args.T % BT:
        raise ValueError("Z0 supports only T >= 64 and divisible by 64")
    if args.warmup < 1 or args.final_calls < 1:
        raise ValueError("warmup and final-calls must be positive")

    # This must happen before Triton/vLLM imports create a cache manager.
    os.environ["TRITON_CACHE_DIR"] = str(args.cache_dir)
    sys.path[:0] = [str(HERE), str(STAGE2)]
    import torch
    from stage2_runner import make_inputs, patch_rocm_autotune
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_full
    from vllm.model_executor.layers.fla.ops import chunk_o

    if not torch.cuda.is_available():
        raise RuntimeError("Z0 requires the gfx942 HIP runtime")
    patch_rocm_autotune()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    q, k, v, g, beta, h0 = make_inputs(args.T, args.seed + args.T, "random", True)
    call = lambda: vllm_full(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0,
        output_final_state=True,
        scale=K ** -0.5,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    for _ in range(args.final_calls):
        output, final_state = call()
        if output is None or final_state is None:
            raise AssertionError("native public API did not return output and final state")
    torch.cuda.synchronize()

    autotuner = autotuner_for(chunk_o.chunk_fwd_kernel_o)
    chosen = config_dict(selected_config(autotuner))
    all_configs = [config_dict(config) for config in getattr(autotuner, "configs", [])]
    # Preserve the runtime fact even when cache-to-config matching finds an
    # ambiguity.  Z0 must report an unresolved match rather than lose the
    # selected autotuner configuration or infer an arbitrary HSACO.
    write_json(
        args.out_dir / "runtime_selection_pre_match.json",
        {
            "T": args.T,
            "runtime_selected_config": chosen,
            "all_autotune_configs": all_configs,
            "cache_groups": [str(path) for path in cache_groups(args.cache_dir)],
        },
    )
    group, candidates = match_selected_group(cache_groups(args.cache_dir), chosen)
    source_module = Path(chunk_o.__file__).resolve()
    selected_dir = args.out_dir / "selected"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(Path(str(group["group"])), selected_dir)
    shutil.copy2(source_module, args.out_dir / "chunk_o.py")
    hsaco = selected_dir / "chunk_fwd_kernel_o.hsaco"
    metadata = group["metadata"]
    readobj = command_output(["/opt/rocm/llvm/bin/llvm-readobj", "--notes", str(hsaco)])
    (args.out_dir / "code_object_readobj.txt").write_text(str(readobj["output"]))
    selection = {
        "phase": "Z0_native_public_api_capture",
        "timing_contract": "diagnostic_only_not_a_ranking",
        "cuda_graph_used": False,
        "T": args.T,
        "shape": {"B": 1, "Hk": HG, "Hv": H, "K": K, "V": V, "BT": BT},
        "public_api": "vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule",
        "public_warmup_calls": args.warmup,
        "public_final_calls": args.final_calls,
        "runtime_selected_config": chosen,
        "all_autotune_configs": all_configs,
        "selected_cache_group": str(group["group"]),
        "selected_metadata": metadata,
        "selected_hsaco_sha256": sha256(hsaco),
        "selected_static_summary": static_summary(selected_dir),
        "output": {"shape": list(output.shape), "dtype": str(output.dtype)},
        "final_state": {"shape": list(final_state.shape), "dtype": str(final_state.dtype)},
        "selection_match": {key: value for key, value in group.items() if key != "ttir"},
        "all_cache_candidates": [
            {key: value for key, value in item.items() if key not in ("ttir",)} for item in candidates
        ],
        "readobj": {"returncode": readobj["returncode"], "command": readobj["command"]},
    }
    write_json(args.out_dir / "native_capture.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

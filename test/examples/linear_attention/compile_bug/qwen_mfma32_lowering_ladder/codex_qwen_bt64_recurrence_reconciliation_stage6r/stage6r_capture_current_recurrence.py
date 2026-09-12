#!/usr/bin/env python3
"""Stage 6R capture: current vLLM recurrence specialization versus asm-v0.

This is audit-only.  It intentionally imports the same Stage 6A full-stage
helpers so the vLLM recurrence is compiled after its real cumsum/KKT/solve/WU
producers, rather than selected from a historical Triton cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
LADDER = HERE.parent
STAGE6A = LADDER / "codex_qwen_bt64_full_graph_gap_stage6a"
REPO = LADDER.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
sys.path[:0] = [str(STAGE6A), str(COMPARE)]

import stage6a_full_graph_audit as stage6a  # noqa: E402
from qwen_gdn_bt64_gfx942_asm_v0_experimental import contract as asm_contract  # noqa: E402
from qwen_gdn_bt64_gfx942_asm_v0_experimental import qwen_gdn_bt64_gfx942_asm_v0  # noqa: E402
from stage2_runner import patch_rocm_autotune  # noqa: E402
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: E402


BT = 64
CURRENT = HERE / "current_kernels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, torch.dtype):
        return str(value)
    return repr(value)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n")


def tensor_meta(tensor: torch.Tensor | None, role: str, producer: str, cast: str = "none") -> dict[str, object] | None:
    if tensor is None:
        return None
    ptr = int(tensor.data_ptr())
    return {
        "role": role,
        "producer": producer,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "data_ptr": f"0x{ptr:x}",
        "data_ptr_alignment": ptr & -ptr if ptr else 0,
        "cast_or_layout_transform": cast,
    }


def public_object(value: object) -> object:
    """Best-effort metadata projection that cannot retain device allocations."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {str(key): public_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_object(item) for item in value]
    if hasattr(value, "_asdict"):
        return public_object(value._asdict())
    return repr(value)


def metadata_mapping(value: object) -> dict[str, object]:
    """Normalize Triton metadata without assuming a particular runtime type."""
    if isinstance(value, dict):
        return public_object(value)  # type: ignore[return-value]
    if hasattr(value, "_asdict"):
        mapped = public_object(value._asdict())
        if isinstance(mapped, dict):
            return mapped
    names = ("name", "num_warps", "num_stages", "num_ctas", "shared", "cluster_dims")
    mapped = {name: public_object(getattr(value, name)) for name in names if hasattr(value, name)}
    if mapped:
        return mapped
    return {"repr": repr(value), "raw": public_object(value)}


def compiled_candidates(root: object) -> list[object]:
    """Find Triton CompiledKernel-like objects in the autotuner device cache."""
    result: list[object] = []
    seen: set[int] = set()

    def visit(value: object) -> None:
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)
        if hasattr(value, "asm") and hasattr(value, "metadata") and hasattr(value, "src"):
            result.append(value)
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(root)
    return result


def config_payload(config: object) -> dict[str, object]:
    return {
        "kwargs": public_object(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
        "num_ctas": getattr(config, "num_ctas", None),
        "repr": repr(config),
    }


def choose_compiled_kernel() -> tuple[object, dict[str, object]]:
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = kernel.fn
    configs = list(getattr(autotuner, "cache", {}).values())
    if not configs:
        raise RuntimeError("current vLLM autotuner cache is empty after recurrence execution")
    selected = configs[-1]
    candidates = compiled_candidates(getattr(autotuner.fn, "device_caches", {}))
    if not candidates:
        raise RuntimeError("unable to locate a Triton compiled kernel in current vLLM device cache")
    selected_data = config_payload(selected)
    target_warps = selected_data["num_warps"]
    target_stages = selected_data["num_stages"]
    target_bv = str(selected_data["kwargs"].get("BV"))
    ranked: list[tuple[int, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        metadata = metadata_mapping(getattr(candidate, "metadata", {}))
        source = getattr(candidate, "src", None)
        constants = public_object(getattr(source, "constants", {}))
        score = 0
        if str(metadata.get("num_warps")) == str(target_warps):
            score += 4
        if str(metadata.get("num_stages")) == str(target_stages):
            score += 4
        if target_bv in json.dumps(constants, default=str):
            score += 2
        asm = getattr(candidate, "asm", {})
        candidate_rows.append({
            "score": score,
            "metadata": metadata,
            "source_constants": constants,
            "asm_keys": sorted(asm),
            "hsaco_bytes": len(asm.get("hsaco", b"")),
        })
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], {"selected_config": selected_data, "all_cached_configs": [config_payload(item) for item in configs], "candidate_inventory": candidate_rows}


def dump_disassembly(hsaco: Path, destination: Path) -> None:
    objdump = next((candidate for candidate in (
        Path("/opt/rocm/llvm/bin/llvm-objdump"),
        Path("/opt/rocm/bin/llvm-objdump"),
    ) if candidate.is_file()), None)
    if objdump is None:
        resolved = shutil.which("llvm-objdump")
        objdump = Path(resolved) if resolved else None
    if objdump is None:
        destination.write_text("N/A: llvm-objdump is unavailable in this runtime.\n")
        return
    command = [str(objdump), "--disassemble", "--mcpu=gfx942", str(hsaco)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    destination.write_text(result.stdout)


def write_vllm_artifact(compiled: object, details: dict[str, object], tensors: dict[str, object], result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, object]:
    output = CURRENT / "vllm"
    output.mkdir(parents=True, exist_ok=True)
    asm = getattr(compiled, "asm")
    hsaco = output / "kernel.hsaco"
    hsaco.write_bytes(asm["hsaco"])
    (output / "kernel.amdgcn").write_text(str(asm.get("amdgcn", "N/A amdgcn was not retained by Triton")))
    # These are emitted by the exact compiled Triton object selected after the
    # Stage 6A producer chain, not recovered from a historical cache entry.
    (output / "kernel.llir").write_text(str(asm.get("llir", "N/A: Triton did not retain LLIR")))
    (output / "kernel.ttir").write_text(str(asm.get("ttir", "N/A: Triton did not retain TTIR")))
    (output / "kernel.ttgir").write_text(str(asm.get("ttgir", "N/A: Triton did not retain TTGIR")))
    metadata = metadata_mapping(getattr(compiled, "metadata", {}))
    source = getattr(compiled, "src", None)
    abi = {
        "source_signature": public_object(getattr(source, "signature", {})),
        "source_constants": public_object(getattr(source, "constants", {})),
        "source_arg_names": public_object(getattr(source, "arg_names", [])),
        "metadata": metadata,
        "selection": details,
        "tensors": tensors,
        "outputs": {
            "h": tensor_meta(result[0], "recurrence h", "current vLLM recurrence"),
            "v_new": tensor_meta(result[1], "recurrence v_new", "current vLLM recurrence"),
            "final_state": tensor_meta(result[2], "recurrence final state", "current vLLM recurrence"),
        },
    }
    write_json(output / "metadata.yaml", metadata)
    write_json(output / "abi.json", abi)
    launch = {
        "symbol": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "grid_formula": "(ceil(V/BV), B*H)",
        "grid_t2048": [4, 8, 1],
        "workgroup": int(metadata.get("num_warps", 0) or 0) * 64,
        "num_warps": metadata.get("num_warps"),
        "num_stages": metadata.get("num_stages"),
        "BV": details["selected_config"]["kwargs"].get("BV"),
        "BT": BT,
        "shared_bytes": metadata.get("shared"),
        "specialization_constants": public_object(getattr(source, "constants", {})),
    }
    write_json(output / "launch.json", launch)
    (output / "sha256.txt").write_text(f"{sha256(hsaco)}  kernel.hsaco\n")
    dump_disassembly(hsaco, output / "disassembly.txt")
    return {"sha256": sha256(hsaco), "launch": launch, "metadata": metadata, "abi": abi}


def write_asm_artifact(inputs: tuple[torch.Tensor, ...], stage_values: dict[str, torch.Tensor]) -> dict[str, object]:
    output = CURRENT / "avelang"
    output.mkdir(parents=True, exist_ok=True)
    source = LADDER / "codex_qwen_asm_v0_integration/assembly/qwen_gdn_bt64_gfx942_asm_v0.hsaco"
    hsaco = output / "kernel.hsaco"
    shutil.copy2(source, hsaco)
    asm_source = source.with_suffix(".s")
    (output / "kernel.amdgcn").write_text(asm_source.read_text() if asm_source.is_file() else "N/A source assembly unavailable\n")
    (output / "kernel.llir").write_text(
        "N/A: asm-v0 is a frozen external HSACO. No Avelang/LLVM source is used to build or mutate this artifact in Stage 6R.\n"
    )
    q, k, v, g, beta, initial_state = inputs
    w = stage_values["w"].float().contiguous()
    u = stage_values["u"].float().contiguous()
    g_cumsum = stage_values["g_cumsum"]
    result = qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g_cumsum, initial_state)
    torch.cuda.synchronize()
    tensors = {
        "k": tensor_meta(k, "key", "external input"),
        "w": tensor_meta(w, "W", "Stage 6A vLLM W cast outside body", "bf16 -> fp32 before asm body"),
        "u": tensor_meta(u, "U", "Stage 6A vLLM U cast outside body", "bf16 -> fp32 before asm body"),
        "g": tensor_meta(g_cumsum, "g_cumsum", "vLLM cumsum"),
        "initial_state": tensor_meta(initial_state, "initial state", "external input"),
    }
    abi = {
        "contract": asm_contract(),
        "kernarg_order": ["k", "v_or_u", "w", "v_new", "g", "h", "h0", "ht", "T", "global_scratch", "profile_scratch"],
        "kernarg_segment_bytes": 88,
        "tensors": tensors,
        "outputs": {
            "h": tensor_meta(result[0], "recurrence h", "asm-v0"),
            "v_new": tensor_meta(result[1], "recurrence v_new", "asm-v0"),
            "final_state": tensor_meta(result[2], "recurrence final state", "asm-v0"),
        },
    }
    write_json(output / "metadata.yaml", {"target": "gfx942", "symbol": "qwen_gdn_bt64_gfx942_asm_v0", "historical_resource_contract": {"vgpr": 128, "accvgpr": 192, "sgpr": 80, "scratch": 0}})
    write_json(output / "abi.json", abi)
    write_json(output / "launch.json", asm_contract())
    (output / "sha256.txt").write_text(f"{sha256(hsaco)}  kernel.hsaco\n")
    dump_disassembly(hsaco, output / "disassembly.txt")
    return {"sha256": sha256(hsaco), "contract": asm_contract(), "abi": abi}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=2048)
    args = parser.parse_args()
    if args.T % BT:
        raise ValueError("Stage 6R requires T divisible by BT=64")
    patch_rocm_autotune()
    inputs = stage6a.fixed_inputs(args.T)
    # This runs current vLLM cumsum/KKT/solve/WU and then the real recurrence.
    vllm_values = stage6a.vllm_manual_stages(inputs)
    recurrence = (vllm_values["h_bf16"], vllm_values["v_new"], vllm_values["final_state"])
    compiled, details = choose_compiled_kernel()
    q, k, v, g, beta, initial_state = inputs
    tensors = {
        "k": tensor_meta(k, "key", "external input"),
        "w": tensor_meta(vllm_values["w"], "W", "current vLLM recompute_w_u_fwd"),
        "u": tensor_meta(vllm_values["u"], "U", "current vLLM recompute_w_u_fwd"),
        "g": tensor_meta(vllm_values["g_cumsum"], "g_cumsum", "current vLLM chunk_local_cumsum"),
        "initial_state": tensor_meta(initial_state, "initial state", "external input"),
    }
    vllm_info = write_vllm_artifact(compiled, details, tensors, recurrence)
    asm_info = write_asm_artifact(inputs, vllm_values)
    write_json(HERE / "capture_summary.json", {
        "T": args.T,
        "current_vllm": vllm_info,
        "asm_v0": asm_info,
        "stage6a_runtime_path": str(STAGE6A),
        "actual_vllm_recurrence_called_after_real_stage6a_vllm_producers": True,
    })
    print(json.dumps({"T": args.T, "vllm_sha256": vllm_info["sha256"], "asm_sha256": asm_info["sha256"], "vllm_launch": vllm_info["launch"]}, indent=2))


if __name__ == "__main__":
    main()

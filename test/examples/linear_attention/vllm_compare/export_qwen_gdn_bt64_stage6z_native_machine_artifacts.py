#!/usr/bin/env python3
"""Export selected Triton chunk-o machine artifacts for a read-only audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LADDER = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder"
sys.path.insert(0, str(LADDER))

from dump_l6_mir_regalloc_artifacts import run_llc_variants  # noqa: E402


def tool(name: str) -> str:
    for candidate in (f"/opt/rocm/llvm/bin/{name}", f"/opt/rocm/bin/{name}", name):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(f"missing tool {name}")


def run(command: list[str], output: Path) -> int:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output.write_text(result.stdout)
    return result.returncode


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def static_counts(text: str) -> dict[str, int]:
    patterns = {
        "mfma32": r"\bv_mfma_f32_32x32x8_bf16\b",
        "mfma16": r"\bv_mfma_f32_16x16x16_bf16\b",
        "global_load": r"\bglobal_load",
        "global_store": r"\bglobal_store",
        "buffer_load": r"\bbuffer_load",
        "buffer_store": r"\bbuffer_store",
        "ds_read": r"\bds_read",
        "ds_write": r"\bds_write",
        "s_waitcnt": r"\bs_waitcnt\b",
        "s_barrier": r"\bs_barrier\b",
        "v_add": r"\bv_add(?:3)?(?:_u|_i|_f|\b)",
        "v_add3": r"\bv_add3\b",
        "shift": r"\b(?:v|s)_(?:lshl|lshr|ashr)(?:_add)?(?:_u|_i|\b)",
        "bitwise": r"\b(?:v|s)_(?:and|or|xor|bfe|bfi)(?:_u|_b|\b)",
        "permute": r"\b(?:ds_bpermute|ds_permute|v_permlane|v_perm|v_readlane|v_writelane|v_mov_b32_dpp)\b",
        "copy_like": r"\b(?:v_mov_b32|s_mov_b32|v_cndmask_b32)\b",
    }
    return {name: len(re.findall(pattern, text)) for name, pattern in patterns.items()}


def isa_mnemonics(text: str) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[0-9a-fA-F]+:\s+)?([A-Za-z_][A-Za-z0-9_.$]*)\b", line)
        if match:
            mnemonic = match.group(1)
            if mnemonic not in {".text", ".amd_kernel_code_t"}:
                result.append(mnemonic)
    return result


def distance_summary(text: str) -> dict[str, object]:
    mnemonics = isa_mnemonics(text)

    def distances(first: str, second: str) -> list[int]:
        values: list[int] = []
        for index, mnemonic in enumerate(mnemonics):
            if re.search(first, mnemonic):
                for next_index in range(index + 1, len(mnemonics)):
                    if re.search(second, mnemonics[next_index]):
                        values.append(next_index - index)
                        break
        return values

    summary: dict[str, object] = {}
    for name, first, second in (
        ("global_load_to_waitcnt", r"^global_load", r"^s_waitcnt$"),
        ("waitcnt_to_ds_read", r"^s_waitcnt$", r"^ds_read"),
        ("ds_read_to_barrier", r"^ds_read", r"^s_barrier$"),
        ("barrier_to_mfma", r"^s_barrier$", r"^v_mfma"),
    ):
        values = distances(first, second)
        summary[name] = {
            "kind": "static lexical instruction distance",
            "count": len(values),
            "min": min(values) if values else None,
            "median": sorted(values)[len(values) // 2] if values else None,
            "max": max(values) if values else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = args.selected_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_copy = out / "selected_triton_cache"
    shutil.copytree(selected, cache_copy, dirs_exist_ok=True)
    for name in (
        "chunk_fwd_kernel_o.source",
        "chunk_fwd_kernel_o.ttir",
        "chunk_fwd_kernel_o.ttgir",
        "chunk_fwd_kernel_o.llir",
        "chunk_fwd_kernel_o.amdgcn",
        "chunk_fwd_kernel_o.json",
        "chunk_fwd_kernel_o.hsaco",
    ):
        shutil.copy2(selected / name, out / name)
    hsaco = out / "chunk_fwd_kernel_o.hsaco"
    run([tool("llvm-objdump"), "-d", "--no-show-raw-insn", str(hsaco)], out / "final_isa.s")
    run([tool("llvm-readobj"), "--notes", str(hsaco)], out / "readobj.txt")
    (out / "triton_pre_lto_amdgcn.s").write_text((out / "chunk_fwd_kernel_o.amdgcn").read_text(errors="replace"))
    llc_dir = out / "llc_mir"
    llc_dir.mkdir(parents=True, exist_ok=True)
    llc_summary = run_llc_variants(out / "chunk_fwd_kernel_o.llir", llc_dir, "gfx942")
    (out / "llc_summary.json").write_text(json.dumps(llc_summary, indent=2, sort_keys=True) + "\n")
    final_isa = (out / "final_isa.s").read_text(errors="replace")
    manifest = {
        "source_kind": "selected Triton cache group copied after fresh T=2048 public capture",
        "selected_dir": str(selected),
        "hsaco_sha256": sha256(hsaco),
        "static_isa": static_counts(final_isa),
        "lexical_dependency_distance": distance_summary(final_isa),
        "readobj": str(out / "readobj.txt"),
        "pre_lto_amdgcn": str(out / "triton_pre_lto_amdgcn.s"),
        "llir": str(out / "chunk_fwd_kernel_o.llir"),
        "llc_mir": str(llc_dir),
    }
    (out / "machine_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

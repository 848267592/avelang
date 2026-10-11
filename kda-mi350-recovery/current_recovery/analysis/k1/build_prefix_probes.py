#!/usr/bin/env python3
"""Build C71 diagnostic prefix copies from exact saved C26/W4 disassembly.

These are not production objects. Each copy ends with s_endpgm immediately
after a named machine completion anchor; the original objects are untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CLANG = "/opt/rocm/llvm/bin/clang"
LD = "/opt/rocm/llvm/bin/ld.lld"
OUT = HERE / "prefix_probes"
INPUTS = {
    "C26": ROOT / "test/examples/kda/artifacts/kda_native_k1_c26_objdump.s",
    "W4_A": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_5c0a24b34425219f.isa",
    "W4_B": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_34419742ff3240a0.isa",
}


def extract(path: Path):
    out = []
    for source_line, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if not line.startswith("\t"):
            continue
        code = line.split("//", 1)[0].rstrip()
        if code.strip():
            op = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s", code)
            if op:
                out.append({"source_line": source_line, "op": op.group(1), "code": code})
    return out


def find_anchor(ops, name, predicate, ordinal=0):
    found = [i for i, x in enumerate(ops) if predicate(x)]
    if len(found) <= ordinal:
        raise RuntimeError(f"{name}: found {len(found)}, need ordinal {ordinal}")
    return found[ordinal]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"diagnostic_only": True, "families": {}, "launch_attempted": False}
    cuts = {
        "frontend_to_first_bf16": lambda ops: find_anchor(ops, "first_bf16", lambda x: x["op"] == "v_mfma_f32_32x32x8_bf16"),
        "bf16_to_first_fp32": lambda ops: find_anchor(ops, "first_fp32", lambda x: x["op"] == "v_mfma_f32_32x32x2_f32"),
        "fp32_to_first_store": lambda ops: find_anchor(ops, "first_store_after_fp32", lambda x: x["op"].startswith("buffer_store"), ordinal=0),
    }
    for family, input_path in INPUTS.items():
        ops = extract(input_path)
        family_rec = {
            "input": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "op_records": len(ops),
            "cuts": {},
        }
        for cut_name, locate in cuts.items():
            idx = locate(ops)
            # Only retain the prefix and an explicit termination. This is a
            # diagnostic prefix timing object, not a correctness candidate.
            body = [x["code"] for x in ops[: idx + 1]] + ["\ts_endpgm"]
            source = (
                ".text\n.p2align 8\n"
                ".globl _c71_diagnostic_kernel\n"
                ".type _c71_diagnostic_kernel,@function\n"
                "_c71_diagnostic_kernel:\n"
                + "\n".join(body) + "\n"
            )
            source_path = OUT / f"{family}_{cut_name}.s"
            object_path = OUT / f"{family}_{cut_name}.o"
            hsaco_path = OUT / f"{family}_{cut_name}.hsaco"
            source_path.write_text(source)
            proc = subprocess.run(
                [CLANG, "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx942", "-x", "assembler", "-c", str(source_path), "-o", str(object_path)],
                text=True, capture_output=True,
            )
            link_proc = None
            if proc.returncode == 0:
                link_proc = subprocess.run([LD, "-shared", str(object_path), "-o", str(hsaco_path)], text=True, capture_output=True)
            family_rec["cuts"][cut_name] = {
                "anchor_source_line": ops[idx]["source_line"],
                "anchor_op": ops[idx]["op"],
                "anchor_code": ops[idx]["code"],
                "prefix_instruction_records": idx + 1,
                "source": str(source_path),
                "object": str(object_path),
                "hsaco": str(hsaco_path),
                "assemble_returncode": proc.returncode,
                "assemble_stderr": proc.stderr[-3000:],
                "link_returncode": None if link_proc is None else link_proc.returncode,
                "link_stderr": None if link_proc is None else link_proc.stderr[-3000:],
                "launch_attempted": False,
                "diagnostic_identity": "not an exact C69 production object after truncation/reassembly",
            }
        manifest["families"][family] = family_rec
    (HERE / "prefix_probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(HERE / "prefix_probe_manifest.json")


if __name__ == "__main__":
    main()

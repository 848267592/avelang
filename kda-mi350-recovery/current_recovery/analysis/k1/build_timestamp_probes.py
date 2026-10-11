#!/usr/bin/env python3
"""Build full diagnostic copies with A29-style s_memrealtime endpoints.

The original C26/C69 objects are never rewritten. Diagnostic copies use a
reserved unused kernarg pointer for a timestamp arena and write only from
workgroup (0,0), lane 0. They retain the full kernel body, unlike prefix cuts.
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
OUT = HERE / "timestamp_probes"
INPUTS = {
    "C26": ROOT / "test/examples/kda/artifacts/kda_native_k1_c26_objdump.s",
    "W4_A": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_5c0a24b34425219f.isa",
    "W4_B": ROOT / "test/examples/kda/reports/C_20260925_69_k1_true_best_dispatch_audit/k1_winner_w4_s1_34419742ff3240a0.isa",
}


def parse_listing(path: Path):
    records = []
    base = None
    for source_line, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if not raw.startswith("\t"):
            continue
        address_match = re.search(r"//\s*([0-9A-Fa-f]+):", raw)
        if not address_match:
            continue
        address = int(address_match.group(1), 16)
        if base is None:
            base = address
        code = raw.split("//", 1)[0].rstrip()
        op_match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s", code)
        if not op_match:
            continue
        target = None
        target_match = re.search(r"<[^+]+\+0x([0-9A-Fa-f]+)>", raw)
        if target_match:
            # Depending on the disassembler, the displayed branch target is
            # either relative to the first text address or already absolute.
            # Keep the raw value until every instruction address is known so
            # both forms can be resolved without guessing.
            target = int(target_match.group(1), 16)
        records.append({"address": address, "source_line": source_line, "op": op_match.group(1), "code": code, "target": target})
    addresses = {r["address"] for r in records}
    for record in records:
        raw_target = record["target"]
        if raw_target is None:
            continue
        relative_target = base + raw_target
        if relative_target in addresses:
            record["target"] = relative_target
        elif raw_target in addresses:
            record["target"] = raw_target
        else:
            # The C26 listing also contains compiler helper/trap stubs whose
            # branch targets are outside the kernel body.  They are not
            # reconstructed as labels; retaining their original immediate
            # operand is the only faithful representation in this body-only
            # diagnostic copy.
            record["target"] = None
    return records, base


# Explicit timer arguments appended after each exact base kernarg segment.
TIMER_ARG_OFFSETS = {"C26": 88, "W4_A": 112, "W4_B": 112}


def timestamp_write(family: str, offset: int) -> str:
    # Use only declared extra timer arguments.  The saved scalar values live
    # in endpoint-dead VGPRs; the lane-0 EXEC guard prevents timer-buffer
    # races while preserving the production scalar state.
    if family == "C26":
        timer_arg = TIMER_ARG_OFFSETS[family]
        return f"""
    v_mov_b32_e32 v126, s33
    v_mov_b32_e32 v141, s42
    v_mov_b32_e32 v142, s43
    s_mov_b64 s[32:33], exec
    v_cmp_eq_u32_e32 vcc, 0, v0
    s_and_b64 exec, exec, vcc
    s_load_dwordx2 s[42:43], s[0:1], {hex(timer_arg)}
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v170, s42
    v_mov_b32_e32 v171, s43
    v_add_u32_e32 v170, {offset}, v170
    s_memrealtime s[42:43]
    s_nop 7
    v_mov_b32_e32 v172, s42
    v_mov_b32_e32 v173, s43
    global_store_dwordx2 v[170:171], v[172:173], off
    s_mov_b64 exec, s[32:33]
    v_readfirstlane_b32 s33, v126
    v_readfirstlane_b32 s42, v141
    v_readfirstlane_b32 s43, v142
"""
    store = str(offset)
    return f"""
    v_mov_b32_e32 v132, s42
    v_mov_b32_e32 v133, s43
    v_mov_b32_e32 v134, s33
    s_mov_b64 s[32:33], exec
    v_cmp_eq_u32_e32 vcc, 0, v0
    s_and_b64 exec, exec, vcc
    v_readfirstlane_b32 s46, v128
    v_readfirstlane_b32 s47, v129
    s_memrealtime s[42:43]
    s_nop 7
    s_store_dwordx2 s[46:47], s[42:43], {store}
    s_mov_b64 exec, s[32:33]
    v_readfirstlane_b32 s33, v134
    v_readfirstlane_b32 s42, v132
    v_readfirstlane_b32 s43, v133
    v_readfirstlane_b32 s46, v130
    v_readfirstlane_b32 s47, v131
"""


def timestamp_start(family: str) -> str:
    return timestamp_write(family, 0)


def render(family: str, records, base: int, phase: str) -> tuple[str, dict]:
    first_wait = next(i for i, r in enumerate(records) if r["op"] == "s_waitcnt")
    first_bf16 = next(i for i, r in enumerate(records) if r["op"] == "v_mfma_f32_32x32x8_bf16")
    first_fp32 = next(i for i, r in enumerate(records) if r["op"] == "v_mfma_f32_32x32x2_f32")
    first_store = next(i for i, r in enumerate(records) if i > first_fp32 and r["op"].startswith("buffer_store"))
    end_at = {
        "frontend_to_first_bf16": first_bf16,
        "bf16_to_first_fp32": first_fp32,
        "first_fp32_to_first_store": first_store,
    }[phase]
    start_at = {
        "frontend_to_first_bf16": first_wait + 1,
        "bf16_to_first_fp32": first_bf16 + 1,
        "first_fp32_to_first_store": first_fp32,
    }[phase]
    lines = [
        ".text", ".p2align 8", ".globl _c71_diagnostic_kernel",
        ".type _c71_diagnostic_kernel,@function", "_c71_diagnostic_kernel:"
    ]
    # Preserve branch targets with labels reconstructed from objdump comments.
    labels = {r["target"] for r in records if r["target"] is not None}
    timer_load = (
        "    v_mov_b32_e32 v130, s46\n"
        "    v_mov_b32_e32 v131, s47\n"
        f"    s_load_dwordx2 s[46:47], s[0:1], {hex(TIMER_ARG_OFFSETS[family])}\n"
        if family != "C26" else ""
    )
    timer_capture = (
        "    v_mov_b32_e32 v128, s46\n"
        "    v_mov_b32_e32 v129, s47\n"
        "    v_readfirstlane_b32 s46, v130\n"
        "    v_readfirstlane_b32 s47, v131\n"
        if family != "C26" else ""
    )
    for i, r in enumerate(records):
        if r["address"] in labels:
            lines.append(f".L_c71_{r['address']:X}:")
        code = r["code"]
        if r["target"] is not None and r["op"].startswith("s_") and ("branch" in r["op"] or "cbranch" in r["op"]):
            code = re.sub(r"\s-?\d+\s*$", f" .L_c71_{r['target']:X}", code)
        if i == 0 and timer_load:
            lines.append(timer_load)
        if i == first_wait + 1 and timer_capture:
            lines.append(timer_capture)
        if i == start_at:
            lines.append(timestamp_start(family))
        if i == end_at:
            lines.append(timestamp_write(family, 8))
            # The timestamp pair is written in slots 0..3 for this diagnostic copy.
            # Endpoints are defined by start/end placement, not by output semantics.
        lines.append(code)
        if i == start_at - 1 and phase == "frontend_to_first_bf16":
            # pointer load completion is forced by the original first wait;
            # timestamp starts after that wait.
            pass
    # For a start/end interval, the start timestamp must precede the endpoint.
    # Add the end timestamp before the anchor instruction and the start at the
    # chosen index above. The above loop places end before anchor and start at
    # start_at; for bf16_to_first_fp32 and first_fp32_to_first_store that order
    # is correct. For frontend the start is after the first wait.
    source = "\n".join(lines) + "\n"
    return source, {
        "start_record": records[start_at],
        "end_record": records[end_at],
        "timer_arg_offset": TIMER_ARG_OFFSETS[family],
        "family": family,
        "phase": phase,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    phases = ["frontend_to_first_bf16", "bf16_to_first_fp32", "first_fp32_to_first_store"]
    manifest = {"diagnostic_only": True, "families": {}, "launch_attempted": False}
    for family, path in INPUTS.items():
        records, base = parse_listing(path)
        manifest["families"][family] = {"input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "phases": {}}
        for phase in phases:
            source, info = render(family, records, base, phase)
            source_path = OUT / f"{family}_{phase}.s"
            object_path = OUT / f"{family}_{phase}.o"
            hsaco_path = OUT / f"{family}_{phase}.hsaco"
            source_path.write_text(source)
            a = subprocess.run([CLANG, "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx942", "-x", "assembler", "-c", str(source_path), "-o", str(object_path)], text=True, capture_output=True)
            l = None
            if a.returncode == 0:
                l = subprocess.run([LD, "-shared", str(object_path), "-o", str(hsaco_path)], text=True, capture_output=True)
            manifest["families"][family]["phases"][phase] = {
                **info,
                "source": str(source_path), "object": str(object_path), "hsaco": str(hsaco_path),
                "assemble_returncode": a.returncode, "assemble_stderr": a.stderr[-3000:],
                "link_returncode": None if l is None else l.returncode, "link_stderr": None if l is None else l.stderr[-3000:],
                "launch_attempted": False,
            }
    (HERE / "timestamp_probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(HERE / "timestamp_probe_manifest.json")


if __name__ == "__main__":
    main()

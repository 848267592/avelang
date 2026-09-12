#!/usr/bin/env python3
"""Classify exact AMDGPU post-greedy spill virtual registers for Qwen v29.

The input is the kernel-specific machine dump emitted by
``replay_qwen_v29_lto_mir.py``.  LLVM debug locations are intentionally
discarded by the ROCm LTO driver, so LLVM/AveLang origin is reported only when
the machine opcode makes it mechanically defensible; the script never
inventes a source-level mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


SPILL = re.compile(r"SI_SPILL_(AV32|AV64)_SAVE\s+%([0-9]+):([A-Za-z0-9_]+)")
SPILL_SLOT = re.compile(
    r"SI_SPILL_(AV32|AV64)_SAVE\s+%([0-9]+):([A-Za-z0-9_]+),\s+%stack\.([0-9]+)"
)
VREG = re.compile(r"%([0-9]+)(?:\.sub[0-9]+)?:([A-Za-z0-9_]+)")
DEF = re.compile(r"^\s*\d+B\s+.*?%([0-9]+)(?:\.sub[0-9]+)?:([A-Za-z0-9_]+)\s*=")
BB = re.compile(r"^bb\.([0-9]+)")
FRAME_OBJECT = re.compile(
    r"fi#([0-9]+): size=([0-9]+), align=([0-9]+), at location \[SP\]"
)


def category(definition: str, uses: list[str]) -> tuple[str, str]:
    text = definition + "\n" + "\n".join(uses)
    if any(token in text for token in (
        "V_LSHL_ADD_U64", "V_LSHL_ADD_U32", "V_LSHLREV_B32", "V_LSHLREV_B64",
        "V_LSHL_OR_B32", "V_ADD_U32", "V_ADD3_U32", "V_SUB_", "V_OR_B32",
        "V_OR3_B32", "V_AND_B32",
    )):
        return "scalar/vector address formation", "machine address arithmetic"
    if any(token in text for token in ("REG_SEQUENCE", "INSERT_SUBREG", "EXTRACT_SUBREG")):
        return "fragment REG_SEQUENCE/copy", "subregister packing/extraction"
    if "V_MFMA_F32_16X16X16BF16" in text:
        return "K fragment materialization", "MFMA16 B-fragment consumer"
    if "V_MFMA_F32_32X32X8BF16" in text:
        return "pred MFMA32 accumulator/epilogue", "MFMA32 pred consumer"
    if "COPY" in definition:
        if any("V_MFMA" in use for use in uses):
            return "fragment REG_SEQUENCE/copy", "COPY feeding MFMA"
        return "fragment REG_SEQUENCE/copy", "COPY-defined temporary"
    if any(token in text for token in ("DS_READ", "DS_WRITE", "BUFFER_LOAD", "GLOBAL_LOAD")):
        return "state/v_decay/update", "memory value used by update region"
    return "unclassified", "no stable machine-only origin"


def parse_frame_objects(lines: list[str]) -> dict[int, dict[str, int]]:
    """Return the post-greedy frame objects named by spill pseudo-ops.

    LLVM's frame-object listing exposes object extent/alignment, but not the
    final colored offsets in this particular save-temps dump.  Keeping those
    facts separate prevents the common, incorrect ``spill_words * 4`` frame
    size claim.
    """

    frame_objects: dict[int, dict[str, int]] = {}
    for line in lines:
        match = FRAME_OBJECT.search(line)
        if not match:
            continue
        index, size, alignment = (int(value) for value in match.groups())
        frame_objects[index] = {"size": size, "alignment": alignment}
    return frame_objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--llvm-ir", type=Path)
    args = parser.parse_args()
    lines = args.mir.read_text(errors="replace").splitlines()
    frame_objects = parse_frame_objects(lines)
    defs: dict[str, tuple[int, str, str]] = {}
    uses: dict[str, list[tuple[int, str]]] = defaultdict(list)
    blocks: dict[int, str] = {}
    current_block = "entry"
    mfma32 = []
    mfma16 = []
    branch_lines = []
    for index, line in enumerate(lines, start=1):
        block = BB.match(line)
        if block:
            current_block = block.group(1)
        blocks[index] = current_block
        if "V_MFMA_F32_32X32X8BF16" in line:
            mfma32.append(index)
        if "V_MFMA_F32_16X16X16BF16" in line:
            mfma16.append(index)
        if "S_BRANCH" in line or "S_CBRANCH" in line:
            branch_lines.append(index)
        definition = DEF.match(line)
        if definition:
            defs[definition.group(1)] = (index, definition.group(2), line)
        for match in VREG.finditer(line):
            uses[match.group(1)].append((index, line))

    rows = []
    for index, line in enumerate(lines, start=1):
        spill = SPILL.search(line)
        if not spill:
            continue
        width_kind, vreg, register_class = spill.groups()
        slot_match = SPILL_SLOT.search(line)
        stack_slot = int(slot_match.group(4)) if slot_match else None
        frame = frame_objects.get(stack_slot, {}) if stack_slot is not None else {}
        def_line, def_class, definition = defs.get(vreg, (None, register_class, "<definition not retained in dump>"))
        use_rows = [(use_line, use) for use_line, use in uses[vreg] if use_line != index]
        last_use = max((use_line for use_line, _ in use_rows), default=index)
        use_text = [use for _, use in use_rows]
        family, reason = category(definition, use_text)
        first_line = def_line or index
        rows.append(
            {
                "vreg": f"%{vreg}",
                "register_class": def_class,
                "spill_opcode": f"SI_SPILL_{width_kind}_SAVE",
                "spill_words": 1 if width_kind == "AV32" else 2,
                "stack_slot": stack_slot,
                "frame_object_size": frame.get("size"),
                "frame_object_alignment": frame.get("alignment"),
                "spill_mir_line": index,
                "def_mir_line": def_line,
                "def_block": blocks.get(def_line, "unknown") if def_line else "unknown",
                "def_instruction": definition,
                "use_count": len(use_rows),
                "last_use_mir_line": last_use,
                "last_use_block": blocks.get(last_use, "unknown"),
                "use_opcodes": " | ".join(use_text),
                "crosses_loop_backedge_heuristic": any(first_line < branch < last_use for branch in branch_lines),
                "crosses_mfma32": any(first_line < mfma < last_use for mfma in mfma32),
                "crosses_mfma16": any(first_line < mfma < last_use for mfma in mfma16),
                "category": family,
                "classification_reason": reason,
                "llvm_origin": "unavailable: ROCm LTO discarded debug locations",
                "avelang_origin": "unavailable without retained source debug metadata",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spill_vregs.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out_dir / "spill_vregs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    by_category = Counter()
    for row in rows:
        by_category[row["category"]] += int(row["spill_words"])
    total = sum(by_category.values())
    summary = {
        "spill_records": len(rows),
        "spill_words": total,
        "by_category_words": dict(by_category),
        "crosses_mfma32_words": sum(r["spill_words"] for r in rows if r["crosses_mfma32"]),
        "crosses_mfma16_words": sum(r["spill_words"] for r in rows if r["crosses_mfma16"]),
        "both_mfma_regions_words": sum(
            r["spill_words"] for r in rows if r["crosses_mfma32"] and r["crosses_mfma16"]
        ),
        "frame_objects": len(frame_objects),
        "frame_object_extent_sum": sum(frame["size"] for frame in frame_objects.values()),
        "frame_object_alignment_max": max(
            (frame["alignment"] for frame in frame_objects.values()), default=0
        ),
        "frame_coloring_note": (
            "The post-greedy dump has no final stack offsets. The object extent "
            "sum is not the private-frame size: final frame coloring/reuse occurs "
            "later in prologepilog. Read private_segment_fixed_size from the final "
            "code object."
        ),
        "mapping_limit": "MIR has no retained debug locations; source producer fields are intentionally unavailable.",
    }
    (args.out_dir / "spill_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    markdown = ["# Qwen v29 Spill VReg Classification", "", f"- Spill records: `{len(rows)}`", f"- Spill words: `{total}`", "", "| category | spill words |", "|:--|--:|"]
    markdown.extend(f"| {name} | {count} |" for name, count in sorted(by_category.items()))
    markdown.extend(
        [
            "",
            f"- Crosses MFMA32 interval: `{summary['crosses_mfma32_words']}` words",
            f"- Crosses MFMA16 interval: `{summary['crosses_mfma16_words']}` words",
            f"- Crosses both intervals: `{summary['both_mfma_regions_words']}` words",
            f"- Frame objects named by spill saves: `{summary['frame_objects']}`",
            f"- Sum of individual frame-object extents: `{summary['frame_object_extent_sum']}` bytes",
            "- Final private-frame size must be read from the code object: LLVM may color/reuse stack slots.",
            "",
            "The detailed per-vreg CSV/JSON intentionally labels LLVM/AveLang source origin as unavailable when LTO stripped debug locations.",
        ]
    )
    (args.out_dir / "spill_summary.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare broad/compact full-v29 post-greedy MIR virtual-live pressure.

This is an evidence tool, not a new lowering experiment.  It consumes the
exact LTO replay artifacts already checked into ``rocprof_outputs`` and emits
a deterministic, line-level *static liveness proxy* for the two full kernels:

* original broad-K full v29: no private scratch, profiler AccVGPR=264;
* compact-K producer/consumer full rewrite: 736 B scratch,
  profiler AccVGPR=384.

LLVM's emitted post-greedy dump does not retain LiveIntervals or source debug
locations.  Therefore this tool deliberately does not pretend that a virtual
register's lexical first/last MIR occurrence is a hardware live interval.  It
is still useful for two mechanically checkable questions:

1. where Greedy RA first inserts an AV spill; and
2. which virtual register classes/categories are lexically simultaneously
   present around each source-phase anchor.

The generated JSON/CSV retains enough provenance to replay every conclusion
against the original MIR line and byte offset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


INSTRUCTION = re.compile(r"^\s*(\d+)B\s+(.*)$")
VREG = re.compile(r"%(\d+)(?:\.[A-Za-z0-9_]+)?(?::([A-Za-z0-9_]+))?")
DEF = re.compile(
    r"(?:^|\s)(?:undef\s+|dead\s+|early-clobber\s+|renamable\s+|killed\s+)*"
    r"%(\d+)(?:\.[A-Za-z0-9_]+)?(?::([A-Za-z0-9_]+))?\s*="
)
SPILL = re.compile(r"SI_SPILL_(AV32|AV64)_SAVE\s+%(\d+):([A-Za-z0-9_]+)")
MFMA32 = "V_MFMA_F32_32X32X8BF16"
MFMA16 = "V_MFMA_F32_16X16X16BF16"

ADDRESS_OPS = (
    "V_LSHL_ADD_U64",
    "V_LSHLREV_B64",
    "V_ADD_CO",
    "V_ADDC",
    "V_ADD_U32",
    "V_ADD3_U32",
    "V_OR_B32",
    "V_OR3_B32",
    "V_LSHL_OR_B32",
)
PACK_OPS = ("REG_SEQUENCE", "INSERT_SUBREG", "EXTRACT_SUBREG")


@dataclass
class InstructionRecord:
    mir_line: int
    byte_offset: int
    text: str
    refs: list[str]
    definition: str | None


@dataclass
class VRegInfo:
    name: str
    register_class: str = "unknown"
    first_line: int = 0
    last_line: int = 0
    def_line: int | None = None
    def_text: str = "<definition unavailable>"
    use_lines: list[int] = field(default_factory=list)
    use_texts: list[str] = field(default_factory=list)
    direct_mfma16_b: bool = False
    direct_mfma16_a: bool = False
    direct_mfma32_operand: bool = False
    pred_accumulator: bool = False
    update_accumulator: bool = False
    address_seed: bool = False
    pack_seed: bool = False
    spilled_words: int = 0
    spill_lines: list[int] = field(default_factory=list)

    @property
    def width_words(self) -> int:
        match = re.search(r"_(\d+)", self.register_class)
        return max(1, int(match.group(1)) // 32) if match else 1

    @property
    def register_bank(self) -> str:
        # ``av_*`` is an allocatable flexible AMDGPU class in this dump, not a
        # proof of a physical AGPR.  Keep it separate instead of inflating the
        # AGPR column.
        if self.register_class.startswith("areg_"):
            return "agpr_class"
        if self.register_class.startswith(("vreg_", "vgpr_")):
            return "vgpr_class"
        if self.register_class.startswith("av_"):
            return "flexible_av_class"
        return "other"

    @property
    def lexical_span(self) -> int:
        return max(0, self.last_line - self.first_line)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opcode(text: str) -> str:
    """Return a stable opcode-ish marker without trying to parse all MIR."""
    for token in (*PACK_OPS, "V_MFMA", "DS_READ", "DS_WRITE", "GLOBAL_LOAD", "BUFFER_LOAD", *ADDRESS_OPS):
        if token in text:
            return token
    fields = text.replace("=", " = ").split()
    if "=" in fields:
        index = fields.index("=")
        return fields[index + 1] if index + 1 < len(fields) else "unknown"
    return fields[0] if fields else "unknown"


def parse_mir(path: Path) -> tuple[list[InstructionRecord], dict[str, VRegInfo]]:
    instructions: list[InstructionRecord] = []
    vregs: dict[str, VRegInfo] = {}

    for mir_line, raw in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = INSTRUCTION.match(raw)
        if not match:
            continue
        byte_offset, text = int(match.group(1)), match.group(2)
        refs = [ref.group(1) for ref in VREG.finditer(text)]
        definition_match = DEF.search(text)
        definition = definition_match.group(1) if definition_match else None
        record = InstructionRecord(mir_line, byte_offset, text, refs, definition)
        instructions.append(record)

        typed_refs = list(VREG.finditer(text))
        for ref in typed_refs:
            name, register_class = ref.group(1), ref.group(2)
            info = vregs.setdefault(name, VRegInfo(name=name))
            if register_class and info.register_class == "unknown":
                info.register_class = register_class
            if not info.first_line:
                info.first_line = mir_line
            info.last_line = mir_line
            info.use_lines.append(mir_line)
            info.use_texts.append(text)

        if definition:
            info = vregs[definition]
            if info.def_line is None:
                info.def_line = mir_line
                info.def_text = text
            if definition_match and definition_match.group(2) and info.register_class == "unknown":
                info.register_class = definition_match.group(2)

        # MFMA source order is destination, A, B, (optional accumulator).
        # The B operand is the update K-fragment of interest.
        if MFMA16 in text and len(refs) >= 3:
            vregs[refs[1]].direct_mfma16_a = True
            vregs[refs[2]].direct_mfma16_b = True
            if definition:
                vregs[definition].update_accumulator = True
        if MFMA32 in text:
            for ref in refs[1:]:
                vregs[ref].direct_mfma32_operand = True
            if definition:
                vregs[definition].pred_accumulator = True
        if any(marker in text for marker in ADDRESS_OPS):
            if definition:
                vregs[definition].address_seed = True
        if any(marker in text for marker in PACK_OPS):
            if definition:
                vregs[definition].pack_seed = True
        spill = SPILL.search(text)
        if spill:
            words = 1 if spill.group(1) == "AV32" else 2
            info = vregs.setdefault(spill.group(2), VRegInfo(name=spill.group(2)))
            if info.register_class == "unknown":
                info.register_class = spill.group(3)
            info.spilled_words += words
            info.spill_lines.append(mir_line)

    return instructions, vregs


def propagate_address_provenance(vregs: dict[str, VRegInfo]) -> None:
    """Track address arithmetic through local copy/pack nodes.

    The producer rewrite leaves a visible ``V_LSHL_ADD_U64 -> COPY av_64``
    chain in the compact dump.  Mark both operands of the arithmetic and the
    immediate copy descendants: an address component such as ``%6842`` is
    still relevant even when its defining opcode is the preceding ``V_SUB``.
    """

    inputs: dict[str, list[str]] = {}
    children: dict[str, list[str]] = defaultdict(list)
    for name, info in vregs.items():
        refs = [ref.group(1) for ref in VREG.finditer(info.def_text)]
        refs = [ref for ref in refs if ref != name]
        inputs[name] = refs
        for parent in refs:
            children[parent].append(name)
        if any(marker in info.def_text for marker in ADDRESS_OPS):
            info.address_seed = True
            for parent in refs:
                if parent in vregs:
                    vregs[parent].address_seed = True

    worklist = [name for name, info in vregs.items() if info.address_seed]
    seen = set(worklist)
    while worklist:
        parent = worklist.pop()
        for child in children.get(parent, []):
            child_info = vregs[child]
            if child in seen or not any(marker in child_info.def_text for marker in ("COPY", *PACK_OPS)):
                continue
            child_info.address_seed = True
            seen.add(child)
            worklist.append(child)


def propagate_fragment_provenance(vregs: dict[str, VRegInfo]) -> None:
    """Mark only local pack/copy ancestry of direct MFMA16 B operands.

    We intentionally stop at generic arithmetic.  The question here is
    whether pack/extract/copy temporaries persist, not to invent a full data
    dependence graph from textual MIR.
    """

    by_def_refs: dict[str, list[str]] = {}
    for name, info in vregs.items():
        refs = [ref.group(1) for ref in VREG.finditer(info.def_text)]
        by_def_refs[name] = [ref for ref in refs if ref != name]

    worklist = [name for name, info in vregs.items() if info.direct_mfma16_b]
    seen = set(worklist)
    while worklist:
        name = worklist.pop()
        info = vregs[name]
        info.pack_seed = info.pack_seed or info.direct_mfma16_b
        # A DS read is the fragment producer boundary.  Following its LDS
        # address would mislabel every address temporary as a fragment.
        if not any(marker in info.def_text for marker in (*PACK_OPS, "COPY")):
            continue
        for parent in by_def_refs.get(name, []):
            parent_info = vregs.get(parent)
            if parent_info is None or parent in seen:
                continue
            # Do not pull the entire kernel through arbitrary SSA arithmetic.
            if any(marker in parent_info.def_text for marker in (*PACK_OPS, "COPY")):
                parent_info.pack_seed = True
                seen.add(parent)
                worklist.append(parent)


def phase_anchors(instructions: list[InstructionRecord]) -> dict[str, tuple[int, int]]:
    mfma32 = [item.mir_line for item in instructions if MFMA32 in item.text]
    mfma16 = [item.mir_line for item in instructions if MFMA16 in item.text]
    if not mfma32 or not mfma16:
        raise ValueError("required MFMA32/MFMA16 anchors are absent")
    pred_first, pred_last = min(mfma32), max(mfma32)
    update_first, update_last = min(mfma16), max(mfma16)
    barriers = [item.mir_line for item in instructions if "S_BARRIER" in item.text]
    after_pred = next((line for line in barriers if line > pred_last), pred_last)
    before_update = max((line for line in barriers if line < update_first), default=update_first - 1)
    last_line = instructions[-1].mir_line
    return {
        "prelude": (instructions[0].mir_line, pred_first - 1),
        "pred_mfma32": (pred_first, pred_last),
        "pred_epilogue": (pred_last + 1, after_pred),
        "k_producer": (after_pred + 1, before_update),
        "update_prologue": (before_update + 1, update_first - 1),
        "update_mfma16": (update_first, update_last),
        "state_writeback": (update_last + 1, last_line),
    }


def category_flags(info: VRegInfo) -> dict[str, bool]:
    return {
        "address": info.address_seed,
        "fragment": info.pack_seed,
        "pred_acc": info.pred_accumulator,
        "update_acc": info.update_accumulator,
    }


PRESSURE_COLUMNS = (
    "vgpr_class_words",
    "agpr_class_words",
    "flexible_av_words",
    "address_words",
    "fragment_words",
    "pred_acc_words",
    "update_acc_words",
    "live_vregs",
)


def make_timeline(
    instructions: list[InstructionRecord], vregs: dict[str, VRegInfo], phases: dict[str, tuple[int, int]]
) -> list[dict[str, int | str]]:
    starts: dict[int, list[VRegInfo]] = defaultdict(list)
    ends: dict[int, list[VRegInfo]] = defaultdict(list)
    for info in vregs.values():
        if not info.first_line:
            continue
        starts[info.first_line].append(info)
        ends[info.last_line + 1].append(info)
    active: dict[str, VRegInfo] = {}
    phase_lookup: dict[int, str] = {}
    for phase, (begin, end) in phases.items():
        phase_lookup.update({line: phase for line in range(begin, end + 1)})

    rows: list[dict[str, int | str]] = []
    for instruction in instructions:
        line = instruction.mir_line
        for info in ends.get(line, []):
            active.pop(info.name, None)
        for info in starts.get(line, []):
            active[info.name] = info
        metrics: Counter[str] = Counter()
        for info in active.values():
            words = info.width_words
            bank = info.register_bank
            if bank == "vgpr_class":
                metrics["vgpr_class_words"] += words
            elif bank == "agpr_class":
                metrics["agpr_class_words"] += words
            elif bank == "flexible_av_class":
                metrics["flexible_av_words"] += words
            flags = category_flags(info)
            for key in ("address", "fragment", "pred_acc", "update_acc"):
                if flags[key]:
                    metrics[f"{key}_words"] += words
        row: dict[str, int | str] = {
            "mir_line": line,
            "byte_offset": instruction.byte_offset,
            "phase": phase_lookup.get(line, "unanchored"),
            "opcode": opcode(instruction.text),
            "live_vregs": len(active),
        }
        row.update({column: metrics[column] for column in PRESSURE_COLUMNS if column != "live_vregs"})
        rows.append(row)
    return rows


def phase_summary(rows: list[dict[str, int | str]], phases: dict[str, tuple[int, int]]) -> list[dict[str, int | str]]:
    summary: list[dict[str, int | str]] = []
    for phase, (begin, end) in phases.items():
        candidates = [row for row in rows if begin <= int(row["mir_line"]) <= end]
        if not candidates:
            continue
        result: dict[str, int | str] = {"phase": phase, "begin_mir_line": begin, "end_mir_line": end}
        for column in PRESSURE_COLUMNS:
            peak = max(candidates, key=lambda row: int(row[column]))
            result[f"peak_{column}"] = peak[column]
            result[f"peak_{column}_mir_line"] = peak["mir_line"]
            result[f"peak_{column}_opcode"] = peak["opcode"]
        summary.append(result)
    return summary


def phase_for_line(phases: dict[str, tuple[int, int]], line: int) -> str:
    for phase, (begin, end) in phases.items():
        if begin <= line <= end:
            return phase
    return "unanchored"


def spill_context(
    instructions: list[InstructionRecord],
    vregs: dict[str, VRegInfo],
    phases: dict[str, tuple[int, int]],
) -> dict[str, object]:
    spill_records = [item for item in instructions if SPILL.search(item.text)]
    av_spills = [item for item in spill_records if "SI_SPILL_AV" in item.text]
    first = av_spills[0] if av_spills else None
    spills: list[dict[str, object]] = []
    for item in av_spills:
        match = SPILL.search(item.text)
        assert match is not None
        info = vregs[match.group(2)]
        flags = category_flags(info)
        spills.append(
            {
                "mir_line": item.mir_line,
                "byte_offset": item.byte_offset,
                "vreg": f"%{info.name}",
                "register_class": info.register_class,
                "spill_words": 1 if match.group(1) == "AV32" else 2,
                "phase": phase_for_line(phases, item.mir_line),
                "def_mir_line": info.def_line,
                "last_mir_line": info.last_line,
                "lexical_span": info.lexical_span,
                "definition": info.def_text,
                "flags": [name for name, enabled in flags.items() if enabled],
                "crosses_pred_mfma32": info.first_line < phases["pred_mfma32"][0] <= info.last_line,
                "crosses_update_mfma16": info.first_line < phases["update_mfma16"][0] <= info.last_line,
            }
        )
    return {
        "first_av_spill": spills[0] if spills else None,
        "first_av_spill_context": [
            {
                "mir_line": item.mir_line,
                "byte_offset": item.byte_offset,
                "text": item.text,
            }
            for item in instructions[max(0, instructions.index(first) - 5) : instructions.index(first) + 6]
        ]
        if first
        else [],
        "av_spill_count": len(av_spills),
        "av_spill_words": sum(item["spill_words"] for item in spills),
        "spills": spills,
    }


def long_lived_delta(vregs: dict[str, VRegInfo], phases: dict[str, tuple[int, int]]) -> list[dict[str, object]]:
    pred_begin, pred_end = phases["pred_mfma32"]
    update_begin, update_end = phases["update_mfma16"]
    rows: list[dict[str, object]] = []
    for info in vregs.values():
        crosses_pred_update = info.first_line < pred_end and info.last_line > update_begin
        crosses_pred_to_k_producer = info.first_line < pred_begin and info.last_line > phases["k_producer"][0]
        is_interesting = info.address_seed or info.pack_seed or info.spilled_words
        if not is_interesting or (not crosses_pred_update and not crosses_pred_to_k_producer and info.lexical_span < 100):
            continue
        flags = category_flags(info)
        rows.append(
            {
                "vreg": f"%{info.name}",
                "register_class": info.register_class,
                "spill_words": info.spilled_words,
                "first_mir_line": info.first_line,
                "last_mir_line": info.last_line,
                "lexical_span": info.lexical_span,
                "def_mir_line": info.def_line,
                "definition_opcode": opcode(info.def_text),
                "address": flags["address"],
                "fragment": flags["fragment"],
                "pred_acc": flags["pred_acc"],
                "update_acc": flags["update_acc"],
                "crosses_pred_update": crosses_pred_update,
                "crosses_pred_to_k_producer": crosses_pred_to_k_producer,
                "definition": info.def_text,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            bool(row["crosses_pred_update"]),
            bool(row["crosses_pred_to_k_producer"]),
            int(row["spill_words"]),
            int(row["lexical_span"]),
        ),
        reverse=True,
    )


def active_snapshot(vregs: dict[str, VRegInfo], line: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for info in vregs.values():
        if not (info.first_line <= line <= info.last_line):
            continue
        flags = category_flags(info)
        if info.register_bank != "flexible_av_class" and not any(flags.values()):
            continue
        rows.append(
            {
                "vreg": f"%{info.name}",
                "register_class": info.register_class,
                "words": info.width_words,
                "first_mir_line": info.first_line,
                "last_mir_line": info.last_line,
                "lexical_span": info.lexical_span,
                "definition_opcode": opcode(info.def_text),
                "address": flags["address"],
                "fragment": flags["fragment"],
                "pred_acc": flags["pred_acc"],
                "update_acc": flags["update_acc"],
                "definition": info.def_text,
            }
        )
    return sorted(rows, key=lambda row: (int(row["words"]), int(row["lexical_span"])), reverse=True)


def first_update_b_fragment(
    instructions: list[InstructionRecord], vregs: dict[str, VRegInfo]
) -> dict[str, object]:
    item = next(instruction for instruction in instructions if MFMA16 in instruction.text)
    if len(item.refs) < 3:
        raise ValueError("first MFMA16 has no textual B operand")
    b = vregs[item.refs[2]]
    return {
        "mfma_mir_line": item.mir_line,
        "mfma_byte_offset": item.byte_offset,
        "b_vreg": f"%{b.name}",
        "b_class": b.register_class,
        "b_def_mir_line": b.def_line,
        "b_last_mir_line": b.last_line,
        "b_lexical_span": b.lexical_span,
        "b_definition": b.def_text,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(title for _, title in columns) + " |"]
    lines.append("|" + "|".join(":--" if not title.endswith("peak") else "--:" for _, title in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return lines


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--broad-mir",
        type=Path,
        default=root / "rocprof_outputs/qwen_v29_full_mir_regalloc/original_exact_lto/kernel_section_07.mir",
    )
    parser.add_argument(
        "--compact-mir",
        type=Path,
        default=root / "rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_07.mir",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        # The historical LTO directory was copied from Docker as ``nobody``
        # ownership.  Keep new derived evidence beside this analysis instead
        # of mutating the immutable capture tree.
        default=root
        / "compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_v29_broad_compact_pressure_timeline",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    broad_insns, broad_vregs = parse_mir(args.broad_mir)
    compact_insns, compact_vregs = parse_mir(args.compact_mir)
    propagate_address_provenance(broad_vregs)
    propagate_address_provenance(compact_vregs)
    propagate_fragment_provenance(broad_vregs)
    propagate_fragment_provenance(compact_vregs)
    broad_phases = phase_anchors(broad_insns)
    compact_phases = phase_anchors(compact_insns)
    broad_timeline = make_timeline(broad_insns, broad_vregs, broad_phases)
    compact_timeline = make_timeline(compact_insns, compact_vregs, compact_phases)
    broad_summary = phase_summary(broad_timeline, broad_phases)
    compact_summary = phase_summary(compact_timeline, compact_phases)
    compact_spills = spill_context(compact_insns, compact_vregs, compact_phases)
    long_lived = long_lived_delta(compact_vregs, compact_phases)

    compact_pred_row = max(
        (row for row in compact_timeline if row["phase"] == "pred_mfma32"),
        key=lambda row: int(row["flexible_av_words"]),
    )
    compact_pred_snapshot = active_snapshot(compact_vregs, int(compact_pred_row["mir_line"]))
    compact_pred_address_flexible_words = sum(
        int(row["words"])
        for row in compact_pred_snapshot
        if str(row["register_class"]).startswith("av_") and bool(row["address"])
    )
    compact_pred_ds_read_flexible_words = sum(
        int(row["words"])
        for row in compact_pred_snapshot
        if str(row["register_class"]).startswith("av_") and str(row["definition_opcode"]) == "DS_READ"
    )
    first_spill_snapshot = active_snapshot(
        compact_vregs,
        int(compact_spills["first_av_spill"]["mir_line"]),
    )
    broad_first_fragment = first_update_b_fragment(broad_insns, broad_vregs)
    compact_first_fragment = first_update_b_fragment(compact_insns, compact_vregs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "broad_timeline.csv", broad_timeline)
    write_csv(args.out_dir / "compact_timeline.csv", compact_timeline)
    write_csv(args.out_dir / "broad_phase_summary.csv", broad_summary)
    write_csv(args.out_dir / "compact_phase_summary.csv", compact_summary)
    write_csv(args.out_dir / "compact_long_lived_spilled_vregs.csv", long_lived)
    write_csv(args.out_dir / "compact_pred_flexible_av_peak_live_vregs.csv", compact_pred_snapshot)
    write_csv(args.out_dir / "compact_first_spill_live_vregs.csv", first_spill_snapshot)
    (args.out_dir / "compact_spill_context.json").write_text(json.dumps(compact_spills, indent=2) + "\n")

    phase_pairs = []
    broad_by_phase = {str(row["phase"]): row for row in broad_summary}
    compact_by_phase = {str(row["phase"]): row for row in compact_summary}
    for phase in broad_phases:
        broad = broad_by_phase.get(phase, {})
        compact = compact_by_phase.get(phase, {})
        row: dict[str, object] = {"phase": phase}
        for metric in PRESSURE_COLUMNS:
            row[f"broad_peak_{metric}"] = broad.get(f"peak_{metric}", "N/A")
            row[f"compact_peak_{metric}"] = compact.get(f"peak_{metric}", "N/A")
            if isinstance(row[f"broad_peak_{metric}"], int) and isinstance(row[f"compact_peak_{metric}"], int):
                row[f"delta_peak_{metric}"] = int(row[f"compact_peak_{metric}"]) - int(row[f"broad_peak_{metric}"])
        phase_pairs.append(row)
    write_csv(args.out_dir / "phase_comparison.csv", phase_pairs)

    summary = {
        "method": {
            "mir_stage": "IR Dump After Greedy Register Allocator (greedy)",
            "liveness": "lexical first-to-last virtual-register occurrence proxy; not LLVM LiveIntervals",
            "agpr_note": "areg_* is counted as AGPR-class; av_* is kept as flexible AV class rather than assumed AGPR",
            "source_mapping": "not available because ROCm LTO dump has no retained debug locations",
        },
        "inputs": {
            "broad": {"path": str(args.broad_mir), "sha256": sha256(args.broad_mir), "phases": broad_phases},
            "compact": {"path": str(args.compact_mir), "sha256": sha256(args.compact_mir), "phases": compact_phases},
        },
        "known_resource_tuple": {
            "broad": {"accvgpr": 264, "scratch_bytes": 0},
            "compact": {"accvgpr": 384, "scratch_bytes": 736, "spill_words": compact_spills["av_spill_words"]},
        },
        "first_compact_ra_danger": compact_spills["first_av_spill"],
        "compact_pred_flexible_av_peak": {
            "mir_line": compact_pred_row["mir_line"],
            "byte_offset": compact_pred_row["byte_offset"],
            "flexible_av_words": compact_pred_row["flexible_av_words"],
            "address_flexible_av_words": compact_pred_address_flexible_words,
            "ds_read_flexible_av_words": compact_pred_ds_read_flexible_words,
            "live_objects": compact_pred_snapshot,
        },
        "compact_first_spill_live_objects": first_spill_snapshot,
        "first_update_b_fragment": {
            "broad": broad_first_fragment,
            "compact": compact_first_fragment,
        },
        "phase_comparison": phase_pairs,
        "compact_long_lived_spilled_vregs": long_lived,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.report:
        first = compact_spills["first_av_spill"]
        report = [
            "# Qwen v29 Broad/Compact MIR 压力时间线审计",
            "",
            "## 范围与结论边界",
            "",
            "这是 Experiment 0：没有改 kernel 源码、compiler pass、launch 或数值路径。它比较两份 exact full-v29 的 "
            "`IR Dump After Greedy Register Allocator`：原始 broad-K 路径与 compact producer/consumer rewrite。",
            "",
            "已知的最终资源事实不由本脚本重新估算：broad 为 `AccVGPR=264`、`scratch=0`；"
            "compact 为 `AccVGPR=384`、`scratch=736 B`。compact 的 post-greedy MIR 中有 190 个 AV spill words "
            "（70 个 AV32 save + 60 个 AV64 save）。",
            "",
            "MIR 没有保留 LiveIntervals 或 source debug location。因此本文的逐指令压力是“虚寄存器在 MIR 文本中首次定义到最后出现”的"
            "静态代理，而不是硬件物理寄存器压力的直接读数。`areg_*` 计作 AGPR-class；`av_*` 是可分配的 flexible class，"
            "单独显示，绝不冒充确定的物理 AGPR。",
            "",
            "phase 标签完全由机器锚点得到：第一/最后一条 MFMA32、两类 MFMA 之间的 barrier、第一/最后一条 MFMA16。"
            "它们是可复现的 disassembly 区间，不是因 LTO 缺失 debug location 而无法证明的精确 AveLang 源代码行映射。",
            "",
            "## 分阶段峰值",
            "",
        ]
        report.extend(
            table(
                phase_pairs,
                [
                    ("phase", "phase"),
                    ("broad_peak_vgpr_class_words", "broad VGPR-class peak"),
                    ("compact_peak_vgpr_class_words", "compact VGPR-class peak"),
                    ("broad_peak_agpr_class_words", "broad AGPR-class peak"),
                    ("compact_peak_agpr_class_words", "compact AGPR-class peak"),
                    ("broad_peak_flexible_av_words", "broad flexible-AV peak"),
                    ("compact_peak_flexible_av_words", "compact flexible-AV peak"),
                    ("broad_peak_address_words", "broad address peak"),
                    ("compact_peak_address_words", "compact address peak"),
                    ("broad_peak_fragment_words", "broad fragment peak"),
                    ("compact_peak_fragment_words", "compact fragment peak"),
                    ("broad_peak_pred_acc_words", "broad pred-acc peak"),
                    ("compact_peak_pred_acc_words", "compact pred-acc peak"),
                    ("broad_peak_update_acc_words", "broad update-acc peak"),
                    ("compact_peak_update_acc_words", "compact update-acc peak"),
                ],
            )
        )
        report.extend(
            [
                "",
                "最强的差异出现在 update 之前：compact 的 flexible-AV 词数在 `pred_mfma32` 为 178，"
                "而 broad 只有 8；`pred_epilogue` 与 `k_producer` 都是 128 对 0。"
                "同一张表还显示 pred accumulator（16 对 16）、update accumulator（4 对 4）和 update 阶段 fragment 代理（256 对 256）没有差异。"
                "这不支持“MFMA16 geometry 或最终 B fragment 本身变大”作为该 broad/compact 差异的解释。完整逐行数据在 "
                "`phase_comparison.csv`、`broad_timeline.csv`、`compact_timeline.csv`。",
                "",
                "## 首个可见的 RA 危险点",
                "",
                f"compact 的第一条 AV spill 在 MIR line `{first['mir_line']}` / byte `{first['byte_offset']}`："
                f"`{first['vreg']}`（`{first['register_class']}`，{first['spill_words']} words），属于 `{first['phase']}`。",
                "",
                f"它的文本区间为 `{first['def_mir_line']}` 到 `{first['last_mir_line']}`，跨度仅 `{first['lexical_span']}` 行；"
                f"定义是 `{first['definition']}`。",
                "",
                "它位于第一条 pred MFMA32 之前，且自身是短寿命地址分量。因此不能把它误解成“该 vreg 跨过 pred/update”。"
                "这份 post-greedy dump 只能严格证明：Greedy 在 compact prelude 已经开始插 AV spill；它不能单独给出物理峰值的因果归属。",
                "",
                "## compact 的长寿命地址/fragment 候选",
                "",
            ]
        )
        long_table = long_lived[:24]
        if long_table:
            report.extend(
                table(
                    long_table,
                    [
                        ("vreg", "vreg"),
                        ("register_class", "class"),
                        ("spill_words", "spill words"),
                        ("first_mir_line", "first line"),
                        ("last_mir_line", "last line"),
                        ("lexical_span", "span"),
                        ("definition_opcode", "def opcode"),
                        ("address", "address"),
                        ("fragment", "fragment"),
                        ("crosses_pred_update", "crosses pred/update"),
                    ],
                )
            )
        else:
            report.append("No address/fragment/spilled vreg passed the long-interval filter.")
        report.extend(
            [
                "",
                "## pred 阶段 flexible-AV 峰值",
                "",
                f"compact 的 pred-MFMA32 flexible-AV 文本峰值在 MIR line `{compact_pred_row['mir_line']}` / "
                f"byte `{compact_pred_row['byte_offset']}`，为 `{compact_pred_row['flexible_av_words']}` words。"
                f"其中 `{compact_pred_address_flexible_words}` words 是地址标记链，而短寿命 `DS_READ` pred 输入只有 "
                f"`{compact_pred_ds_read_flexible_words}` words。"
                "活跃对象完整清单在 `compact_pred_flexible_av_peak_live_vregs.csv`；下面列出跨度最长的对象。",
                "",
            ]
        )
        report.extend(
            table(
                compact_pred_snapshot[:20],
                [
                    ("vreg", "vreg"),
                    ("register_class", "class"),
                    ("words", "words"),
                    ("first_mir_line", "first"),
                    ("last_mir_line", "last"),
                    ("lexical_span", "span"),
                    ("definition_opcode", "def opcode"),
                    ("address", "address"),
                    ("fragment", "fragment"),
                ],
            )
        )
        report.extend(
            [
                "",
                "## 对最终 B fragment 的负控制",
                "",
                f"broad 的第一条 update MFMA16 B operand 是 `{broad_first_fragment['b_vreg']}`，在 MIR line "
                f"`{broad_first_fragment['b_def_mir_line']}` 定义、`{broad_first_fragment['mfma_mir_line']}` 使用；"
                f"compact 对应为 `{compact_first_fragment['b_vreg']}`，在 `{compact_first_fragment['b_def_mir_line']}` 定义、"
                f"`{compact_first_fragment['mfma_mir_line']}` 使用。两者都是一行 def-use 的 `DS_READ_B64` handoff。"
                "这与既有 strict terminal-load A/B 的“post-opt LLVM 和 ISA 完全相同、资源不变”结果一致："
                "只替换最终 generic/direct LDS load 不能修复 full-v29 cliff。",
                "",
                "## 可行动的中间结论",
                "",
                "机器文本已直接显示 compact 独有的一批 `V_LSHL_ADD_U64 -> COPY av_64` 链：例如 `%7449` 在 line 1720 创建，"
                "跨越整个 pred MFMA32 区间，直到 line 3996 才重新被 COPY 后进入后续 `V_LSHL_ADD_U64` / global load / LDS write 链。"
                "这类对象使 compact 在 pred 阶段多出 170 flexible-AV words，并在 pred epilogue/K producer 仍多出 128 words。"
                "所以 Experiment 1 应只测试“将这些 address tuple 延后到紧邻其 global/LDS consumer 创建”，并保留 MFMA、LDS write 数、"
                "tile ownership、barrier 和数学不变。它必须预注册 post-opt LLVM、pre-RA MIR、def-use 距离与峰值 flexible-AV 的下降 gate。",
                "",
                "这仍不是“已证明 Avelang backend 必然错误”的铁证：broad/compact 的上层 producer graph 本来不同，"
                "且本审计使用静态代理。它是对下一条严格 single-variable late-address 实验的具体、可复查定位。",
                "",
                "## 复现",
                "",
                "```bash",
                "python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/analyze_qwen_v29_broad_compact_pressure_timeline.py \\",
                "  --report test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_v29_broad_compact_pressure_timeline_report.md",
                "```",
                "",
                "派生产物写入 `compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_v29_broad_compact_pressure_timeline/`："
                "逐指令 CSV、phase 比较、首个 spill 上下文、pred flexible-AV peak live-set 与完整 JSON 都保留在该目录。",
            ]
        )
        args.report.write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

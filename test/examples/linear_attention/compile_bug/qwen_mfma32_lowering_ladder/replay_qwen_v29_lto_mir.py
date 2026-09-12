#!/usr/bin/env python3
"""Replay a captured Avelang AMDGPU LTO command with exact MIR dumps.

The Avelang backend writes a replayable linker argv file when
``AVELANG_AMDGPU_LINK_DEBUG_DIR`` is set.  ROCm's final register allocation
happens inside ld.lld's full-LTO plugin, so this script appends the plugin
flags needed to preserve post-LTO bitcode and print machine IR immediately
before/after greedy RA and after virtreg rewriting.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


SPILL = re.compile(r"SI_SPILL_(AV32|AV64)_SAVE\s+%([0-9]+):([A-Za-z0-9_]+)")


def infer_link_cwd(argv_file: Path) -> Path:
    """Recover the project cwd used when the backend captured relative inputs."""
    for candidate in (argv_file.parent, *argv_file.parents):
        if (candidate / "CMakeLists.txt").is_file():
            return candidate
    return argv_file.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--argv-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--kernel", required=True, help="Exact AMDGPU kernel symbol to extract")
    args = parser.parse_args()

    args.argv_file = args.argv_file.resolve()
    args.out_dir = args.out_dir.resolve()

    command = args.argv_file.read_text().splitlines()
    if not command:
        raise RuntimeError(f"empty linker argv file: {args.argv_file}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "linked.hsaco"
    for index, value in enumerate(command[:-1]):
        if value == "-o":
            command[index + 1] = str(output)
            break
    else:
        raise RuntimeError("captured linker argv has no -o output argument")

    for plugin_option in (
        "save-temps",
        "-print-before=greedy",
        "-print-after=greedy",
        "-print-after=virtregrewriter",
        "-print-after=prologepilog",
    ):
        command.extend(("-Xlinker", f"-plugin-opt={plugin_option}"))

    (args.out_dir / "replay_argv.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(
        command,
        cwd=infer_link_cwd(args.argv_file),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (args.out_dir / "link.stdout").write_text(result.stdout)
    (args.out_dir / "link.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"LTO replay failed with exit status {result.returncode}; see {args.out_dir / 'link.stderr'}")

    header = re.compile(r"^# Machine code for function " + re.escape(args.kernel) + r":")
    sections: list[dict[str, object]] = []
    current: list[str] | None = None
    phase = "unknown"
    current_phase = "unknown"
    for line in result.stderr.splitlines():
        if line.startswith("# *** IR Dump"):
            if current is not None:
                sections.append({"text": current, "phase": current_phase})
                current = None
            phase = line.removeprefix("# *** ").removesuffix(" ***:")
        elif header.match(line):
            if current is not None:
                sections.append({"text": current, "phase": current_phase})
            current = [line]
            current_phase = phase
        elif current is not None:
            current.append(line)
    if current is not None:
        sections.append({"text": current, "phase": current_phase})

    summary = []
    for index, section in enumerate(sections):
        text = "\n".join(section["text"])
        path = args.out_dir / f"kernel_section_{index:02d}.mir"
        path.write_text(text + "\n")
        spills = SPILL.findall(text)
        summary.append(
            {
                "path": str(path),
                "phase": section["phase"],
                "lines": len(section["text"]),
                "av32_spill_saves": sum(kind == "AV32" for kind, _, _ in spills),
                "av64_spill_saves": sum(kind == "AV64" for kind, _, _ in spills),
                "spill_virtual_registers": [f"%{number}:{kind}" for _, number, kind in spills],
                "no_vregs": "NoVRegs" in text.splitlines()[0],
            }
        )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

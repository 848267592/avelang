#!/usr/bin/env python3
"""Write per-length launch/ABI manifests while sharing one verified HSACO file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SHARED = ROOT / "golden_fullseq/shared"


def main() -> None:
    original = SHARED / "original.hsaco"; rebuilt = SHARED / "rebuilt.hsaco"
    hashes = {"original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "rebuilt_sha256": hashlib.sha256(rebuilt.read_bytes()).hexdigest()}
    kernarg = {"size": 88, "alignment": 8, "parameters": ["k", "v", "w", "v_new", "g", "h", "h0", "ht", "T", "global_scratch=null", "profile_scratch=null"]}
    for t in (2048, 8192, 16384):
        out = ROOT / f"golden_fullseq/t{t}"; out.mkdir(parents=True, exist_ok=True)
        for name in ("original.hsaco", "original_from_triton.s", "rebuilt.hsaco", "original.disasm", "rebuilt.disasm"):
            link = out / name
            if link.exists() or link.is_symlink(): link.unlink()
            link.symlink_to(Path("../shared") / name)
        (out / "launch.json").write_text(json.dumps({"T": t, "grid": [4, 8, 1], "workgroup": [256, 1, 1], "dynamic_lds_bytes": 57344, "runtime_T": True}, indent=2) + "\n")
        (out / "kernarg_layout.json").write_text(json.dumps(kernarg, indent=2) + "\n")
        (out / "hashes.sha256").write_text("".join(f"{value}  {key}\n" for key, value in hashes.items()))
        (out / "metadata.yaml").write_text("---\namdhsa.kernels:\n  - .name: chunk_gated_delta_rule_fwd_kernel_h_blockdim64\n    .kernarg_segment_size: 88\n    .max_flat_workgroup_size: 256\n    .wavefront_size: 64\n    .private_segment_fixed_size: 0\namdhsa.target: amdgcn-amd-amdhsa--gfx942\n")
        (out / "build.sh").write_text("#!/usr/bin/env sh\nexec ../build_golden.sh\n")


if __name__ == "__main__": main()

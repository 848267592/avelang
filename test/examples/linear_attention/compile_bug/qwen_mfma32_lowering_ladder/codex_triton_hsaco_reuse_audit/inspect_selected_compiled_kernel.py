#!/usr/bin/env python3
"""Emit the actual compiled object selected by the vLLM Triton autotuner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "vllm_stageb_snapshot"))
from vllm.model_executor.layers.fla.ops import chunk_delta_h  # noqa: E402


def main() -> None:
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = kernel.fn
    # Match the known-good vLLM profiling wrapper, which excludes ROCm's
    # failing num_stages=4 choices before autotuning.
    autotuner.configs = [config for config in autotuner.configs if config.num_stages != 4]
    b, t, hg, h, kdim, vdim = 1, 64, 4, 8, 128, 128
    rand = lambda shape, dtype: torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    k = rand((b, t, hg, kdim), torch.bfloat16)
    w = rand((b, t, h, kdim), torch.float32)
    v = rand((b, t, h, vdim), torch.float32)
    g = rand((b, t, h), torch.float32)
    h0 = rand((b, h, vdim, kdim), torch.float32)
    chunk_delta_h.chunk_gated_delta_rule_fwd_h(k, w, v, g, None, h0, True, 64, True, None)
    torch.cuda.synchronize()
    selected = next(value for value in autotuner.cache.values()
                    if value.kwargs.get("BV") == 32 and value.num_warps == 4 and value.num_stages == 2)
    compiled = autotuner.fn.device_caches[0][0]
    selected_kernel = next(
        value for key, value in compiled.items()
        if "'num_warps': 4" in key and "'num_stages': 2" in key and "('constexpr', 32)" in key
    )
    metadata = selected_kernel.metadata
    if isinstance(metadata, dict):
        metadata_payload = dict(metadata)
    elif hasattr(metadata, "_asdict"):
        metadata_payload = dict(metadata._asdict())
    else:
        metadata_payload = {name: getattr(metadata, name) for name in dir(metadata)
                            if not name.startswith("_") and not callable(getattr(metadata, name))}
    payload = {
        "selected_config": {
            "BV": selected.kwargs["BV"], "num_warps": selected.num_warps,
            "num_stages": selected.num_stages, "num_ctas": selected.num_ctas,
        },
        "metadata": metadata_payload,
        "metadata_repr": repr(metadata),
        "metadata_dir": [name for name in dir(metadata) if not name.startswith("_")],
        "asm_keys": sorted(selected_kernel.asm.keys()),
        "hsaco_bytes": len(selected_kernel.asm["hsaco"]),
        "source_signature": selected_kernel.src.signature,
        "source_constants": {str(key): value for key, value in selected_kernel.src.constants.items()},
    }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

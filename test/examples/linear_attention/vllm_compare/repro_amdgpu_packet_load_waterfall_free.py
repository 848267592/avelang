#!/usr/bin/env python3
"""Same-source raw-buffer packet-load A/B microrepro for gfx942.

Each of 64 lanes loads one distinct contiguous 16-byte packet.  The source
always uses the legacy public raw-buffer call shape ``(rsrc, 0, byte_offset,
aux)``; the compiler-only environment selector decides whether that dynamic
offset remains a scalar ``soffset`` or is moved to the vector-address operand.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

import avelang
import avelang.language as al


PACKETS = 64
WORDS_PER_PACKET = 4
WORKGROUP = 64


@avelang.jit
def _packet_load_x4_copy_kernel(
    src_ptr: al.Pointer(al.i32),
    dst_ptr: al.Pointer(al.i32),
    packets: al.constexpr,
):
    src = al.make_tensor(src_ptr, al.i32, al.make_layout((packets * WORDS_PER_PACKET,), (1,)))
    dst = al.make_tensor(dst_ptr, al.i32, al.make_layout((packets * WORDS_PER_PACKET,), (1,)))
    tid = al.thread_id(0)
    packet = al.block_id(0) * WORKGROUP + tid
    zero = al.convert(0, al.i32)
    src_rsrc = al.amdgpu.make_rsrc(src, packets * WORDS_PER_PACKET * 4)

    if packet < packets:
        byte_offset = al.convert(packet * 16, al.i32)
        words = al.amdgpu.raw_buffer_load_x4(src_rsrc, zero, byte_offset, 0)
        for word in al.range(WORDS_PER_PACKET):
            dst[packet * WORDS_PER_PACKET + word] = words[word]


def run(mode: str) -> dict[str, object]:
    if mode not in ("current_raw", "waterfall_free"):
        raise ValueError(f"unsupported mode: {mode}")
    os.environ["AVELANG_STAGE6Z_PACKET_LOAD_LOWERING"] = mode
    src = torch.arange(PACKETS * WORDS_PER_PACKET, device="cuda", dtype=torch.int32)
    dst = torch.full_like(src, -1)
    _packet_load_x4_copy_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](src, dst, PACKETS)
    torch.cuda.synchronize()
    return {
        "mode": mode,
        "packets": PACKETS,
        "packet_bytes": 16,
        "workgroup": WORKGROUP,
        "byte_exact": bool(torch.equal(src, dst)),
        "checksum": int(dst.to(torch.int64).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("current_raw", "waterfall_free"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.mode), sort_keys=True))


if __name__ == "__main__":
    main()

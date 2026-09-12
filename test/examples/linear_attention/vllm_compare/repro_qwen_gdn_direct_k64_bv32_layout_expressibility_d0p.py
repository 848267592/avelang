#!/usr/bin/env python3
"""Compile gates for the D0-P typed LDS-layout feasibility audit.

These are intentionally tiny and do not benchmark a Qwen recurrence.  Each
case isolates one public-source capability before an update-suffix arm is
allowed to depend on it:

* ``packed_local_load``: BF16x8 raw load -> packed LDS store -> MFMA fragment;
* ``lane_shuffle``: public ``al.shuffle`` -> AMDGPU DS bpermute lowering;
* ``register_transpose_fragment``: 8x8 register transpose -> packed LDS
  row -> MFMA fragment;
* ``scaled_register_transpose_fragment``: the same transpose after BF16
  V-new is scaled through FP32 and repacked;
* ``noncontiguous_fragment``: LDS gather -> local BF16 fragment -> MFMA.

The last case is expected to be the gate: it must lower without
``builtin.unrealized_conversion_cast`` before a D0-P blocked/swizzled update
kernel can be considered expressible in ordinary AveLang source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


WORKGROUP = 128
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _d0p_packed_local_load_kernel(x_ptr: al.Pointer(al.bf16), out_ptr: al.Pointer(al.f32)):
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((WORKGROUP,), (1,)))
    tid = al.thread_id(0)
    row = tid >> 2
    group = tid & 3
    rsrc = al.amdgpu.make_rsrc(x, 32 * 32 * 2)
    zero = al.convert(0, al.i32)
    offset = al.convert((row * 32 + group * 8) * 2, al.i32)
    packed = al.amdgpu.raw_buffer_load_x4(rsrc, zero, offset, 0)
    stage = al.make_shared((32, 32), al.bf16)
    words = al.view(stage, al.u32, al.make_layout((32, 4, 4), (16, 4, 1)))
    words[row, group] = packed
    al.syncthreads()

    lane_col = tid & 31
    lane_group = tid >> 5
    acc = al.full((16,), 0.0, al.f32)
    for pack in al.range(2):
        word = pack * 2 + lane_group
        value = words[lane_col, word]
        frag = al.view(value, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[0], frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[1], frag[1], acc)
    out[tid] = acc[0]


@avelang.jit
def _d0p_lane_shuffle_kernel(x_ptr: al.Pointer(al.bf16), out_ptr: al.Pointer(al.u32)):
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((WORKGROUP, 8), (8, 1)))
    out = al.make_tensor(out_ptr, al.u32, al.make_layout((WORKGROUP,), (1,)))
    tid = al.thread_id(0)
    rsrc = al.amdgpu.make_rsrc(x, WORKGROUP * 8 * 2)
    zero = al.convert(0, al.i32)
    packed = al.amdgpu.raw_buffer_load_x4(rsrc, zero, al.convert(tid * 16, al.i32), 0)
    lane = tid & 63
    partner = lane ^ 1
    out[tid] = al.shuffle(packed[0], partner, 64)


@avelang.jit
def _d0p_register_transpose_fragment_kernel(
    x_ptr: al.Pointer(al.bf16), out_ptr: al.Pointer(al.f32)
):
    """Transpose sixteen source 8x8 blocks into row-contiguous LDS tiles.

    A source lane loads BF16x8 from one token.  The eight lanes of its
    subgroup exchange one packed BF16 pair at a time, so the destination lane
    owns one feature row across eight tokens.  The destination stores are
    contiguous and the later MFMA operand load is therefore an ordinary
    packed shared view, not an LDS gather.
    """

    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((WORKGROUP,), (1,)))
    tid = al.thread_id(0)
    lane_in_wave = tid & 63
    lane8 = tid & 7
    subgroup = tid >> 3
    feature_block = subgroup & 3
    token_block = subgroup >> 2
    rsrc = al.amdgpu.make_rsrc(x, 32 * 32 * 2)
    zero = al.convert(0, al.i32)
    source_token = token_block * 8 + lane8
    source_offset = al.convert((source_token * 32 + feature_block * 8) * 2, al.i32)
    packed = al.amdgpu.raw_buffer_load_x4(rsrc, zero, source_offset, 0)
    stage = al.make_shared((32, 32), al.bf16)

    # Every output lane selects its feature from the eight source token lanes.
    # ``shuffle`` is width-64, so each source index stays within its own wave.
    source_pair_index = lane8 >> 1
    source_word = packed[0]
    if source_pair_index == 1:
        source_word = packed[1]
    if source_pair_index == 2:
        source_word = packed[2]
    if source_pair_index == 3:
        source_word = packed[3]
    source_lane_base = lane_in_wave - lane8
    for token_in_block in al.range(8):
        pair_word = al.shuffle(source_word, source_lane_base + token_in_block, 64)
        pair = al.view(pair_word, al.Tensor((2,), al.bf16))
        value = pair[0]
        if (lane8 & 1) == 1:
            value = pair[1]
        stage[feature_block * 8 + lane8, token_block * 8 + token_in_block] = value
    al.syncthreads()

    words = al.view(stage, al.u32, al.make_layout((32, 4, 4), (16, 4, 1)))
    lane_col = tid & 31
    lane_group = tid >> 5
    acc = al.full((16,), 0.0, al.f32)
    for pack in al.range(2):
        word = pack * 2 + lane_group
        value = words[lane_col, word]
        frag = al.view(value, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[0], frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[1], frag[1], acc)
    out[tid] = acc[0]


@avelang.jit
def _d0p_scaled_register_transpose_fragment_kernel(
    x_ptr: al.Pointer(al.bf16), g_ptr: al.Pointer(al.f32), out_ptr: al.Pointer(al.f32)
):
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    g = al.make_tensor(g_ptr, al.f32, al.make_layout((32,), (1,)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((WORKGROUP,), (1,)))
    tid = al.thread_id(0)
    lane_in_wave = tid & 63
    lane8 = tid & 7
    subgroup = tid >> 3
    feature_block = subgroup & 3
    token_block = subgroup >> 2
    rsrc = al.amdgpu.make_rsrc(x, 32 * 32 * 2)
    zero = al.convert(0, al.i32)
    source_token = token_block * 8 + lane8
    source_offset = al.convert((source_token * 32 + feature_block * 8) * 2, al.i32)
    packed = al.amdgpu.raw_buffer_load_x4(rsrc, zero, source_offset, 0)
    source_pair_index = lane8 >> 1
    source_word = packed[0]
    if source_pair_index == 1:
        source_word = packed[1]
    if source_pair_index == 2:
        source_word = packed[2]
    if source_pair_index == 3:
        source_word = packed[3]
    source_pair = al.view(source_word, al.Tensor((2,), al.bf16))
    source_value = source_pair[0]
    if (lane8 & 1) == 1:
        source_value = source_pair[1]
    source_scaled = al.convert(al.convert(source_value, al.f32) * al.exp(g[31] - g[source_token]), al.f32)

    stage = al.make_shared((32, 32), al.bf16)
    source_lane_base = lane_in_wave - lane8
    for token_in_block in al.range(8):
        value = al.shuffle(source_scaled, source_lane_base + token_in_block, 64)
        stage[feature_block * 8 + lane8, token_block * 8 + token_in_block] = al.convert(value, al.bf16)
    al.syncthreads()

    words = al.view(stage, al.u32, al.make_layout((32, 4, 4), (16, 4, 1)))
    lane_col = tid & 31
    lane_group = tid >> 5
    acc = al.full((16,), 0.0, al.f32)
    for pack in al.range(2):
        word = pack * 2 + lane_group
        value = words[lane_col, word]
        frag = al.view(value, al.Tensor((2, 4, 1), al.bf16))
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[0], frag[0], acc)
        acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[1], frag[1], acc)
    out[tid] = acc[0]


@avelang.jit
def _d0p_noncontiguous_fragment_kernel(x_ptr: al.Pointer(al.bf16), out_ptr: al.Pointer(al.f32)):
    x = al.make_tensor(x_ptr, al.bf16, al.make_layout((32, 32), (32, 1)))
    out = al.make_tensor(out_ptr, al.f32, al.make_layout((WORKGROUP,), (1,)))
    tid = al.thread_id(0)
    row = tid >> 2
    group = tid & 3
    rsrc = al.amdgpu.make_rsrc(x, 32 * 32 * 2)
    zero = al.convert(0, al.i32)
    packed = al.amdgpu.raw_buffer_load_x4(rsrc, zero, al.convert((row * 32 + group * 8) * 2, al.i32), 0)
    stage = al.make_shared((32, 32), al.bf16)
    words = al.view(stage, al.u32, al.make_layout((32, 4, 4), (16, 4, 1)))
    words[row, group] = packed
    al.syncthreads()

    lane_col = tid & 31
    lane_group = tid >> 5
    gathered = al.make_local((8,), al.bf16)
    for item in al.range(8):
        # This intentionally selects eight strided LDS elements, the missing
        # producer-to-MFMA fragment bridge identified by C0.5S.
        gathered[item] = stage[(item * 4 + lane_group) & 31, lane_col]
    frag = al.view(gathered, al.Tensor((2, 4, 1), al.bf16))
    acc = al.full((16,), 0.0, al.f32)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[0], frag[0], acc)
    acc = al.amdgpu.mfma_32x32x8_bf16_f32(frag[1], frag[1], acc)
    out[tid] = acc[0]


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_name: str) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    target = _HSACO_DUMP_DIR / f"{kernel_name}.hsaco"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    original_compile = amdgpu_compiler.AmdgpuCompiler.compile
    dumped = False

    def wrapped_compile(self, src, target_info, options=None):
        nonlocal dumped
        binary = original_compile(self, src, target_info, options)
        if not dumped and kernel_name in src.fn.fn.__name__:
            target.write_bytes(binary)
            dumped = True
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"failed to dump {kernel_name}")


def _launch(case: str) -> None:
    if case == "packed_local_load":
        x = torch.randn((32, 32), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        out = torch.empty((WORKGROUP,), device="cuda", dtype=torch.float32)
        kernel = _d0p_packed_local_load_kernel
    elif case == "lane_shuffle":
        x = torch.randn((WORKGROUP, 8), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        out = torch.empty((WORKGROUP,), device="cuda", dtype=torch.uint32)
        kernel = _d0p_lane_shuffle_kernel
    elif case == "register_transpose_fragment":
        x = torch.randn((32, 32), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        out = torch.empty((WORKGROUP,), device="cuda", dtype=torch.float32)
        kernel = _d0p_register_transpose_fragment_kernel
    elif case == "scaled_register_transpose_fragment":
        x = torch.randn((32, 32), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        g = torch.randn((32,), device="cuda", dtype=torch.float32)
        out = torch.empty((WORKGROUP,), device="cuda", dtype=torch.float32)
        kernel = _d0p_scaled_register_transpose_fragment_kernel
    elif case == "noncontiguous_fragment":
        x = torch.randn((32, 32), device="cuda", dtype=torch.float32).to(torch.bfloat16)
        out = torch.empty((WORKGROUP,), device="cuda", dtype=torch.float32)
        kernel = _d0p_noncontiguous_fragment_kernel
    else:
        raise ValueError(case)

    def launch() -> None:
        if case == "scaled_register_transpose_fragment":
            kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](x, g, out, num_warps=2)
        else:
            kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](x, out, num_warps=2)

    _maybe_dump_hsaco(launch, kernel.fn.__name__)
    launch()
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "packed_local_load",
            "lane_shuffle",
            "register_transpose_fragment",
            "scaled_register_transpose_fragment",
            "noncontiguous_fragment",
        ],
        default="all",
    )
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    cases = (
        [args.case]
        if args.case != "all"
        else [
            "packed_local_load",
            "lane_shuffle",
            "register_transpose_fragment",
            "scaled_register_transpose_fragment",
            "noncontiguous_fragment",
        ]
    )
    rows = []
    for case in cases:
        try:
            _launch(case)
            rows.append({"case": case, "status": "pass"})
        except Exception as error:  # The expected non-contiguous gate remains evidence.
            rows.append({"case": case, "status": "fail", "error": str(error)})
    if args.json:
        print(json.dumps(rows, sort_keys=True))
    else:
        for row in rows:
            print(row)
    if any(row["status"] == "fail" for row in rows) and args.case != "all":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Minimal compile/correctness probe for the selected T=8192 Q@H contract.

This is deliberately not a chunk-o implementation.  It only asks whether the
opt-in C16 first-class physical operand path can accept the native selected
contract: gfx942, wave64, WG128, two waves, and a Q@H 64x32 by 32x64 block.
AVELANG_C16_WG128_QH selects the experimental two-wave producer mapping; the
default environment remains the verified WG256 C16 path.
"""

from __future__ import annotations

import os

import torch

import avelang
import avelang.language as al


WORKGROUP = 128
TILE = 64


@avelang.jit
def qh_parity_wg128_kernel(
    source_ptr: al.Pointer(al.bf16),
    debug_ptr: al.Pointer(al.bf16),
    raw_ptr: al.Pointer(al.f32),
    g_ptr: al.Pointer(al.f32),
):
    source = al.make_tensor(
        source_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
    )
    debug = al.make_tensor(
        debug_ptr, al.bf16, al.make_layout((64, 32), (32, 1))
    )
    raw = al.make_tensor(
        raw_ptr, al.f32, al.make_layout((WORKGROUP, 32), (32, 1))
    )
    g = al.make_tensor(
        g_ptr, al.f32, al.make_layout((1, TILE, 8), (TILE * 8, 8, 1))
    )
    a_stage = al.make_shared((TILE, TILE), al.bf16)
    b_stage = al.make_shared((TILE, TILE), al.bf16)
    tid = al.thread_id(0)
    zero = al.convert(0, al.bf16)
    one = al.convert(1, al.bf16)

    # 128 threads x 32 rounds covers the complete 64x64 shared stage.
    for rep in al.range(32):
        linear = tid + rep * WORKGROUP
        row = linear >> 6
        col = linear & 63
        a_stage[row, col] = zero
        b_stage[row, col] = zero
        if row < 32 and (col == row or col == row + 32):
            b_stage[row, col] = one
    al.syncthreads()

    acc = al.full((16,), 0.0, al.f32)
    zero_i32 = al.convert(0, al.i32)
    zero_f32 = al.convert(0.0, al.f32)
    result = al.amdgpu.block_dot_bf16_f32_operand(
        a_stage,
        b_stage,
        source,
        debug,
        g,
        tid,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_i32,
        zero_f32,
        acc,
        acc,
    )
    for r in al.range(32):
        raw[tid, r] = result[r]


def main() -> None:
    os.environ["AVELANG_C16_REAL_TILE_ROLE"] = "Q"
    os.environ["AVELANG_BLOCK_DOT_LOWERING"] = "specialized"
    source = torch.arange(64 * 32, device="cuda", dtype=torch.float32).reshape(64, 32)
    source = source.to(torch.bfloat16)
    # NaN prefill makes the diagnostic readback coverage observable.  An
    # empty_like buffer would make unwritten elements look valid by chance.
    debug = torch.full_like(
        source, torch.tensor(float("nan"), device=source.device)
    )
    raw = torch.empty((WORKGROUP, 32), device="cuda", dtype=torch.float32)
    g = torch.zeros((1, TILE, 8), device="cuda", dtype=torch.float32)
    qh_parity_wg128_kernel[lambda: ((1, 1, 1), (WORKGROUP, 1, 1))](
        source, debug, raw, g, num_warps=2
    )
    torch.cuda.synchronize()
    debug_written = int(torch.isfinite(debug).sum().item())
    debug_total = int(debug.numel())
    raw_finite = bool(torch.isfinite(raw).all().item())
    print(
        f"Q@H WG128 probe: debug_written={debug_written}/{debug_total} "
        f"raw_finite={raw_finite}"
    )
    if debug_written != debug_total:
        raise AssertionError(
            "Q@H WG128 physical readback is incomplete: "
            f"{debug_written}/{debug_total} elements. The current C16 "
            "lowering still assumes its WG256 physical contract."
        )
    if not torch.equal(debug, source):
        raise AssertionError("Q@H WG128 debug readback is not BF16-exact")
    if not raw_finite:
        raise AssertionError("Q@H WG128 raw output is not finite")


if __name__ == "__main__":
    main()

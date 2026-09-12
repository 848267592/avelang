"""C0.5S: source-only BF16x8-to-LDS vector-store control.

This is deliberately an operand-staging control, not a production Qwen
kernel.  C0.5 added a compiler lowering that turns typed BF16x8 producer
loads into packed LDS writes.  This file asks the narrower source-language
question: can ordinary AveLang source request the same packed LDS producer
store without a block-dot or compiler-lowering branch?

Both arms use identical input/output tensors, raw BF16x8 global loads, one
BT64/WG128 CTA per token block, LDS allocation, barriers, and global copies.
Only the shared producer store differs:

* ``scalar`` views BF16x8 then emits eight BF16 stores;
* ``packed`` stores the four raw u32 words through an explicitly typed shared
  u32 view.

The control intentionally stops after LDS staging.  The direct-K64 MFMA32
consumer needs token-contiguous fragments, whereas raw global BF16x8 loads
are V/K-contiguous.  Expressing that transpose as a packed scatter followed
by a typed MFMA fragment is the exact public-source gap being audited; it is
not hidden by a compiler lowering in this test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Callable

import torch

import avelang
import avelang.language as al


BT = 64
BV = 32
KDIM = 64
WORKGROUP = 128
_HSACO_DUMP_DIR: Path | None = None


@avelang.jit
def _qwen_direct_k64_source_scalar_lds_c05s_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_out_ptr: al.Pointer(al.bf16),
    k_out_ptr: al.Pointer(al.bf16),
    num_tokens: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((num_tokens, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((num_tokens, KDIM), (KDIM, 1)))
    v_out = al.make_tensor(v_out_ptr, al.bf16, al.make_layout((num_tokens, BV), (BV, 1)))
    k_out = al.make_tensor(k_out_ptr, al.bf16, al.make_layout((num_tokens, KDIM), (KDIM, 1)))

    tid = al.thread_id(0)
    chunk_idx = al.block_id(0)
    chunk_start = chunk_idx * BT
    load_token = tid >> 2
    row_group = tid & 3
    row_base = row_group * 8
    zero = al.convert(0, al.i32)

    v_rsrc = al.amdgpu.make_rsrc(v, num_tokens * BV * 2)
    k_rsrc = al.amdgpu.make_rsrc(k, num_tokens * KDIM * 2)
    v_stage = al.make_shared((BT, BV), al.bf16)
    k_stage = al.make_shared((BT, KDIM), al.bf16)

    for token_half in al.range(2):
        token = chunk_start + token_half * 32 + load_token
        v_offset = al.convert((token * BV + row_base) * 2, al.i32)
        packed_v = al.amdgpu.raw_buffer_load_x4(v_rsrc, zero, v_offset, 0)
        v_frag = al.view(packed_v, al.Tensor((8,), al.bf16))
        for element in al.range(8):
            v_stage[token_half * 32 + load_token, row_base + element] = v_frag[element]

        for col_half in al.range(2):
            k_offset = al.convert((token * KDIM + col_half * 32 + row_base) * 2, al.i32)
            packed_k = al.amdgpu.raw_buffer_load_x4(k_rsrc, zero, k_offset, 0)
            k_frag = al.view(packed_k, al.Tensor((8,), al.bf16))
            for element in al.range(8):
                k_stage[token_half * 32 + load_token, col_half * 32 + row_base + element] = k_frag[element]

    al.syncthreads()

    for rep_v in al.range(16):
        linear_v = tid + rep_v * WORKGROUP
        token_v = linear_v // BV
        col_v = linear_v - token_v * BV
        v_out[chunk_start + token_v, col_v] = v_stage[token_v, col_v]
    for rep_k in al.range(32):
        linear_k = tid + rep_k * WORKGROUP
        token_k = linear_k // KDIM
        col_k = linear_k - token_k * KDIM
        k_out[chunk_start + token_k, col_k] = k_stage[token_k, col_k]


@avelang.jit
def _qwen_direct_k64_source_packed_lds_c05s_kernel(
    v_ptr: al.Pointer(al.bf16),
    k_ptr: al.Pointer(al.bf16),
    v_out_ptr: al.Pointer(al.bf16),
    k_out_ptr: al.Pointer(al.bf16),
    num_tokens: al.constexpr,
):
    v = al.make_tensor(v_ptr, al.bf16, al.make_layout((num_tokens, BV), (BV, 1)))
    k = al.make_tensor(k_ptr, al.bf16, al.make_layout((num_tokens, KDIM), (KDIM, 1)))
    v_out = al.make_tensor(v_out_ptr, al.bf16, al.make_layout((num_tokens, BV), (BV, 1)))
    k_out = al.make_tensor(k_out_ptr, al.bf16, al.make_layout((num_tokens, KDIM), (KDIM, 1)))

    tid = al.thread_id(0)
    chunk_idx = al.block_id(0)
    chunk_start = chunk_idx * BT
    load_token = tid >> 2
    row_group = tid & 3
    row_base = row_group * 8
    zero = al.convert(0, al.i32)

    v_rsrc = al.amdgpu.make_rsrc(v, num_tokens * BV * 2)
    k_rsrc = al.amdgpu.make_rsrc(k, num_tokens * KDIM * 2)
    v_stage = al.make_shared((BT, BV), al.bf16)
    k_stage = al.make_shared((BT, KDIM), al.bf16)
    v_words = al.view(v_stage, al.u32, al.make_layout((BT, 4, 4), (16, 4, 1)))
    k_words = al.view(k_stage, al.u32, al.make_layout((BT, 8, 4), (32, 4, 1)))

    for token_half in al.range(2):
        token = chunk_start + token_half * 32 + load_token
        v_offset = al.convert((token * BV + row_base) * 2, al.i32)
        packed_v = al.amdgpu.raw_buffer_load_x4(v_rsrc, zero, v_offset, 0)
        v_words[token_half * 32 + load_token, row_group] = packed_v

        for col_half in al.range(2):
            k_offset = al.convert((token * KDIM + col_half * 32 + row_base) * 2, al.i32)
            packed_k = al.amdgpu.raw_buffer_load_x4(k_rsrc, zero, k_offset, 0)
            k_words[token_half * 32 + load_token, col_half * 4 + row_group] = packed_k

    al.syncthreads()

    for rep_v in al.range(16):
        linear_v = tid + rep_v * WORKGROUP
        token_v = linear_v // BV
        col_v = linear_v - token_v * BV
        v_out[chunk_start + token_v, col_v] = v_stage[token_v, col_v]
    for rep_k in al.range(32):
        linear_k = tid + rep_k * WORKGROUP
        token_k = linear_k // KDIM
        col_k = linear_k - token_k * KDIM
        k_out[chunk_start + token_k, col_k] = k_stage[token_k, col_k]


def _maybe_dump_hsaco(launch: Callable[[], None], kernel_name: str) -> None:
    if _HSACO_DUMP_DIR is None:
        return
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    dump_dir = _HSACO_DUMP_DIR
    dump_dir.mkdir(parents=True, exist_ok=True)
    target = dump_dir / f"{kernel_name}.hsaco"
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
            print(f"dumped_hsaco: {target}")
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped_compile
    try:
        launch()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original_compile
    if not dumped:
        raise RuntimeError(f"no {kernel_name} kernel matched HSACO dump")


def _make_inputs(tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    v = torch.randn((tokens, BV), device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    k = torch.randn((tokens, KDIM), device="cuda", dtype=torch.float32, generator=generator).to(torch.bfloat16)
    return v.contiguous(), k.contiguous()


def _kernel_for(arm: str):
    if arm == "scalar":
        return _qwen_direct_k64_source_scalar_lds_c05s_kernel
    if arm == "packed":
        return _qwen_direct_k64_source_packed_lds_c05s_kernel
    raise ValueError(f"unknown arm: {arm}")


def _launch(arm: str, v: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = v.shape[0]
    v_out = torch.empty_like(v)
    k_out = torch.empty_like(k)
    kernel = _kernel_for(arm)

    def launch() -> None:
        kernel[lambda: ((tokens // BT, 1, 1), (WORKGROUP, 1, 1))](
            v,
            k,
            v_out,
            k_out,
            tokens,
            num_warps=2,
        )

    _maybe_dump_hsaco(launch, kernel.fn.__name__)
    launch()
    return v_out, k_out


def _digest(*tensors: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    for tensor in tensors:
        hasher.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_case(
    tokens: int,
    *,
    seed: int,
    warmup: int,
    repeat: int,
    arm_order: tuple[str, str] = ("scalar", "packed"),
) -> dict[str, object]:
    if tokens % BT:
        raise ValueError("tokens must be divisible by 64")
    v, k = _make_inputs(tokens, seed)
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    rows: dict[str, dict[str, object]] = {}

    if set(arm_order) != {"scalar", "packed"}:
        raise ValueError(f"arm_order must contain scalar and packed once, got {arm_order}")
    for arm in arm_order:
        v_out, k_out = _launch(arm, v, k)
        torch.cuda.synchronize()
        kernel = _kernel_for(arm)

        def launch() -> None:
            kernel[lambda: ((tokens // BT, 1, 1), (WORKGROUP, 1, 1))](
                v,
                k,
                v_out,
                k_out,
                tokens,
                num_warps=2,
            )

        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch()
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))

        outputs[arm] = (v_out, k_out)
        rows[arm] = {
            "T": tokens,
            "arm": arm,
            "input_equal": bool(torch.equal(v_out, v) and torch.equal(k_out, k)),
            "finite": bool(torch.isfinite(v_out).all() and torch.isfinite(k_out).all()),
            "sha256": _digest(v_out, k_out),
            "median_ms": statistics.median(timings),
        }

    rows["scalar"]["cross_arm_equal"] = bool(
        torch.equal(outputs["scalar"][0], outputs["packed"][0])
        and torch.equal(outputs["scalar"][1], outputs["packed"][1])
    )
    rows["packed"]["cross_arm_equal"] = rows["scalar"]["cross_arm_equal"]
    return {"scalar": rows["scalar"], "packed": rows["packed"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, nargs="+", default=[64, 512, 2048])
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dump-hsaco-dir", type=Path, default=None)
    args = parser.parse_args()

    global _HSACO_DUMP_DIR
    _HSACO_DUMP_DIR = args.dump_hsaco_dir
    rows = [
        run_case(tokens, seed=args.seed + tokens, warmup=args.warmup, repeat=args.repeat)
        for tokens in args.T
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        for arm in ("scalar", "packed"):
            print("source_lds_c05s," + ",".join(f"{key}={value}" for key, value in row[arm].items()))


if __name__ == "__main__":
    main()

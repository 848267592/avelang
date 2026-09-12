#!/usr/bin/env python3
"""Prepare and dispatch exactly one native-BT64 Stage 3 kernel for rocprof."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
sys.path.insert(0, str(COMPARE))

from qwen_gdn_bt64_native_wu_chunko_mfma_v1 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_from_v24_mfma_v1,
    qwen_gdn_w_u_bt64_from_v24_mfma_v1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_bt64_layout  # noqa: E402
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)


KERNELS = {
    "wu_w": "_qwen_gdn_w_bf16_kernel_bt64_from_v24_mfma_v1",
    "wu_u": "_qwen_gdn_u_bf16_kernel_bt64_from_v24_mfma_v1",
    "chunk_o": "_qwen_gdn_chunk_o_bf16_kernel_bt64_from_v24_mfma_v1",
}


def make_inputs(t: int, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    k = (torch.randn((1, t, 4, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    v = (torch.randn((1, t, 8, 128), device="cuda") * 0.05).to(torch.bfloat16).contiguous()
    g = (torch.randn((1, t, 8), device="cuda") * 0.01).float().contiguous()
    beta = (0.5 + torch.rand((1, t, 8), device="cuda")).float().contiguous()
    return q, k, v, g, beta


def prepare(stage: str, t: int, seed: int):
    q, k, v, g, beta = make_inputs(t, seed)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a)
    if stage == "wu_w":
        return lambda: qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)[0]
    if stage == "wu_u":
        return lambda: qwen_gdn_w_u_bt64_from_v24_mfma_v1(k, v, g_cumsum, beta, a_solved)[1]
    # Keep preparation independent of native kernels so --dump-hsaco observes
    # their actual first JIT compilation rather than a cache hit.
    v_new = (torch.randn((1, t, 8, 128), device="cuda") * 0.03).float().contiguous()
    h_bf16 = (torch.randn((1, t // 64, 8, 128, 128), device="cuda") * 0.01).to(torch.bfloat16).contiguous()
    return lambda: qwen_gdn_chunk_o_bt64_from_v24_mfma_v1(q, k, v_new, h_bf16, g_cumsum)


def dump_hsaco(fn, kernel_substr: str, out_dir: Path) -> Path:
    from avelang.backends.amdgpu import compiler as amdgpu_compiler

    out_dir.mkdir(parents=True, exist_ok=True)
    dumped: list[Path] = []
    original = amdgpu_compiler.AmdgpuCompiler.compile

    def wrapped(self, src, target, options=None):
        binary = original(self, src, target, options)
        if kernel_substr in src.fn.fn.__name__ and not dumped:
            path = out_dir / f"{src.fn.fn.__name__}.hsaco"
            path.write_bytes(binary)
            dumped.append(path)
        return binary

    amdgpu_compiler.AmdgpuCompiler.compile = wrapped
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        amdgpu_compiler.AmdgpuCompiler.compile = original
    if not dumped:
        raise RuntimeError(f"failed to capture {kernel_substr} HSACO")
    return dumped[0]


def dump_isa(hsaco: Path) -> Path:
    """Write a reproducible gfx942 disassembly beside a captured code object."""
    objdump = Path("/opt/rocm/llvm/bin/llvm-objdump")
    if not objdump.is_file():
        raise RuntimeError(f"missing ROCm objdump: {objdump}")
    result = subprocess.run(
        [str(objdump), "-d", "--no-show-raw-insn", str(hsaco)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    isa = hsaco.with_suffix(".isa")
    isa.write_text(result.stdout)
    return isa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(KERNELS), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dump-hsaco", type=Path)
    parser.add_argument("--dump-isa", action="store_true", help="disassemble the HSACO captured by --dump-hsaco")
    args = parser.parse_args()
    fn = prepare(args.stage, args.T, args.seed)
    if args.dump_hsaco:
        hsaco = dump_hsaco(fn, KERNELS[args.stage], args.dump_hsaco)
        print(f"hsaco={hsaco}", flush=True)
        if args.dump_isa:
            print(f"isa={dump_isa(hsaco)}", flush=True)
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        fn()
    torch.cuda.synchronize()
    print(f"profiled stage={args.stage} kernel={KERNELS[args.stage]} T={args.T}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dispatch one Stage 4 kernel family for rocprof or HSACO capture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
COMPARE = REPO / "test/examples/linear_attention/vllm_compare"
STAGE2 = REPO / "test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2"
sys.path[:0] = [str(COMPARE), str(STAGE2)]

from qwen_gdn_bt64_nonrecurrence_mfma_v2 import (  # noqa: E402
    qwen_gdn_chunk_o_bt64_mfma_v2_s0,
    qwen_gdn_kkt_bt64_mfma_v2_s0,
    qwen_gdn_w_u_bt64_mfma_v2_s1,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import (  # noqa: E402
    qwen_gdn_solve_avelang_v18_bt64_layout,
)
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (  # noqa: E402
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
)
from stage2_runner import make_inputs, patch_rocm_autotune  # noqa: E402


KERNELS = {
    "kkt": "_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0",
    "wu_w": "_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0",
    "wu_u": "_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0",
    "chunk_o": "_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0",
}


def prepare(stage: str, t: int, seed: int):
    patch_rocm_autotune()
    q, k, v, g, beta, h0 = make_inputs(t, seed, "random", True)
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=64)
    if stage == "kkt":
        return lambda: qwen_gdn_kkt_bt64_mfma_v2_s0(k, g_cumsum, beta)
    if stage in ("wu_w", "wu_u"):
        a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=64)
        a_solved = qwen_gdn_solve_avelang_v18_bt64_layout(a)
        index = 0 if stage == "wu_w" else 1
        return lambda: qwen_gdn_w_u_bt64_mfma_v2_s1(
            k, v, g_cumsum, beta, a_solved
        )[index]
    torch.manual_seed(seed + 1)
    v_new = torch.randn((1, t, 8, 128), device="cuda", dtype=torch.float32).contiguous()
    h_bf16 = torch.randn((1, t // 64, 8, 128, 128), device="cuda", dtype=torch.bfloat16).contiguous()
    return lambda: qwen_gdn_chunk_o_bt64_mfma_v2_s0(
        q, k, v_new, h_bf16, g_cumsum
    )


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
    result = subprocess.run(
        ["/opt/rocm/llvm/bin/llvm-objdump", "-d", "--no-show-raw-insn", str(hsaco)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    path = hsaco.with_suffix(".isa")
    path.write_text(result.stdout)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(KERNELS), required=True)
    parser.add_argument("--T", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--dump-hsaco", type=Path)
    args = parser.parse_args()
    fn = prepare(args.stage, args.T, args.seed)
    if args.dump_hsaco:
        hsaco = dump_hsaco(fn, KERNELS[args.stage], args.dump_hsaco)
        print(f"hsaco={hsaco}")
        print(f"isa={dump_isa(hsaco)}")
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(args.repeat):
        fn()
    torch.cuda.synchronize()
    print(f"profiled stage={args.stage} kernel={KERNELS[args.stage]} T={args.T}")


if __name__ == "__main__":
    main()

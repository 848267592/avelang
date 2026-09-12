#!/usr/bin/env python3
"""
请进入/使用 Docker 容器 ac739c57a0bf。

复现路径：
/workspace/project/avelang/test/examples/linear_attention/vllm_compare

失败复现：
docker exec -it -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf python repro_v8_vk_launch_failure_mi300.py --block-v 8 --block-k 64

成功对照：
docker exec -it -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf python repro_v8_vk_launch_failure_mi300.py --block-v 4 --block-k 64

现象：
MI300X 上 max_threads_per_block=1024，但 v8 vk chunk_gdr 在 block_v*block_k=512 时 HIP unspecified launch failure；256 work-items 正常。
B=1,T=512,Hk=4,Hv=8,K=128,V=128,BF16, layout=[B,T,H,D].

- ok:   --block-v 4  --block-k 64   # 256 work-items
- fail: --block-v 8  --block-k 64   # 512 work-items
- fail: --block-v 16 --block-k 32   # 512 work-items
- fail: --block-v 16 --block-k 64   # 1024 work-items
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v8_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v8_vllm_layout,
)


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f = x.float()
    return (x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(args: argparse.Namespace):
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    k = torch.randn(
        args.B,
        args.T,
        args.Hk,
        args.K,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    k = l2norm(k)
    v = torch.randn(
        args.B,
        args.T,
        args.Hv,
        args.V,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    g = torch.nn.functional.logsigmoid(
        torch.randn(args.B, args.T, args.Hv, device="cuda", dtype=torch.float32, generator=generator)
    )
    g = (g / 16.0).contiguous()
    beta = torch.sigmoid(
        torch.randn(args.B, args.T, args.Hv, device="cuda", dtype=torch.float32, generator=generator)
    ).contiguous()
    initial_state = None
    if not args.no_initial_state:
        initial_state = (
            torch.randn(
                args.B,
                args.Hv,
                args.V,
                args.K,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            )
            * 0.01
        ).contiguous()
    return k, v, g, beta, initial_state


def print_env(args: argparse.Namespace) -> None:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"torch={torch.__version__}")
    print(f"hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"gcnArchName={getattr(props, 'gcnArchName', 'unknown')}")
    print(f"max_threads_per_block={props.max_threads_per_block}")
    print(
        "shape="
        f"B={args.B},T={args.T},Hk={args.Hk},Hv={args.Hv},K={args.K},V={args.V},chunk={args.chunk}"
    )
    print(f"vk_config=block_v={args.block_v},block_k={args.block_k},work_items={args.block_v * args.block_k}")
    print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")


def run_one(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    print_env(args)
    k, v, g, beta, initial_state = make_inputs(args)

    print("building v6 oracle stages up to w/u ...")
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=args.chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=args.chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=args.chunk)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=args.chunk,
        prefer_optimized=True,
    )
    sync()

    print("launching v8 vk chunk_gdr ...")
    h, vn, final_state = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=args.chunk,
        use_parallel_chunk_gdr=True,
        prefer_optimized=True,
        parallel_mode="vk",
        block_v=args.block_v,
        block_k=args.block_k,
    )
    sync()
    print("launch_status=ok")
    print(f"h_shape={tuple(h.shape)} vn_shape={tuple(vn.shape)} final_state_shape={tuple(final_state.shape)}")
    print(f"sanity={float(final_state.float().abs().max().item()):.9g}")


def run_matrix(args: argparse.Namespace) -> None:
    cases = [(4, 64), (8, 64), (16, 32), (16, 64)]
    for block_v, block_k in cases:
        cmd = [
            sys.executable,
            __file__,
            "--B",
            str(args.B),
            "--T",
            str(args.T),
            "--Hk",
            str(args.Hk),
            "--Hv",
            str(args.Hv),
            "--K",
            str(args.K),
            "--V",
            str(args.V),
            "--chunk",
            str(args.chunk),
            "--block-v",
            str(block_v),
            "--block-k",
            str(block_k),
            "--seed",
            str(args.seed),
        ]
        if args.no_initial_state:
            cmd.append("--no-initial-state")
        print("=" * 100)
        print(f"running child: block_v={block_v}, block_k={block_k}, work_items={block_v * block_k}")
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        print(result.stdout)
        print(f"child_returncode={result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--Hk", type=int, default=4)
    parser.add_argument("--Hv", type=int, default=8)
    parser.add_argument("--K", type=int, default=128)
    parser.add_argument("--V", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--block-v", type=int, default=8)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="Run safe and failing configs in isolated child processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix:
        run_matrix(args)
    else:
        run_one(args)


if __name__ == "__main__":
    main()

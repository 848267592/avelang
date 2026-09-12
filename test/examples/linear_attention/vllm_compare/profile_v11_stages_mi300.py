import sys
import argparse
import inspect
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
AVELANG_DIR = str(Path(__file__).resolve().parent)

sys.path.insert(0, AVELANG_DIR)

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)

from qwen_gdn_chunked_avelang_v9_vllm_layout_fixed import (
    qwen_gdn_chunk_o_avelang_v9_vllm_layout,
)

from qwen_gdn_chunked_avelang_v10_vllm_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v10_vllm_layout,
    qwen_gdn_chunked_avelang_v10_vllm_layout,
)

from qwen_gdn_chunked_avelang_v11_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v11_mfma_layout,
    qwen_gdn_chunked_avelang_v11_mfma_layout,
)


def sync():
    torch.cuda.synchronize()


def l2norm(x, eps=1e-6):
    x_f = x.float()
    y = x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)
    return y.to(x.dtype).contiguous()


def make_inputs(T, with_initial_state=True, dtype=torch.bfloat16, seed=1234):
    torch.manual_seed(seed)
    device = "cuda"

    B, Hk, Hv, K, V = 1, 4, 8, 128, 128

    q = torch.randn(B, T, Hk, K, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, T, Hk, K, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, T, Hv, V, device=device, dtype=dtype).contiguous()

    q = l2norm(q)
    k = l2norm(k)

    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, Hv, device=device, dtype=torch.float32)
    )
    g = (g / 16.0).contiguous()

    beta = torch.sigmoid(
        torch.randn(B, T, Hv, device=device, dtype=torch.float32)
    ).contiguous()

    if with_initial_state:
        initial_state = (
            torch.randn(B, Hv, V, K, device=device, dtype=torch.float32).contiguous()
            * 0.01
        )
    else:
        initial_state = None

    return q, k, v, g, beta, initial_state


def bench_one(name, fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        sync()
        times.append(start.elapsed_time(end))

    t = torch.tensor(times)
    result = {
        "name": name,
        "mean": t.mean().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
    }
    return result


def call_v11_gdr(k, w, u, g_cumsum, initial_state, chunk_size):
    kwargs = dict(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_mfma_chunk_gdr=True,
        prefer_optimized=True,
        block_v=16,
        block_k=64,
    )
    sig = inspect.signature(qwen_gdn_chunk_gdr_avelang_v11_mfma_layout)
    if "use_clean_kernel" in sig.parameters:
        kwargs["use_clean_kernel"] = True
    if "use_runtime_chunk_loop" in sig.parameters:
        kwargs["use_runtime_chunk_loop"] = False
    return qwen_gdn_chunk_gdr_avelang_v11_mfma_layout(**kwargs)


def call_v10_gdr(k, w, u, g_cumsum, initial_state, chunk_size):
    return qwen_gdn_chunk_gdr_avelang_v10_vllm_layout(
        k,
        w,
        u,
        g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk_size,
        use_parallel_chunk_gdr=True,
        prefer_optimized=True,
        parallel_mode="chunk_vk",
        block_v=4,
        block_k=64,
    )


def call_chunk_o(q, k, vn, h, g_cumsum, scale, chunk_size):
    return qwen_gdn_chunk_o_avelang_v9_vllm_layout(
        q,
        k,
        vn,
        h,
        g_cumsum,
        scale=scale,
        chunk_size=chunk_size,
        use_parallel_chunk_o=True,
        chunk_o_parallel_mode="vk",
        chunk_o_block_v=4,
        chunk_o_block_k=16,
    )


def full_v11(q, k, v, g, beta, initial_state, scale, chunk_size):
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
        prefer_optimized=True,
        use_parallel_chunk_o=True,
        chunk_o_block_v=4,
        chunk_o_block_k=16,
    )
    sig = inspect.signature(qwen_gdn_chunked_avelang_v11_mfma_layout)
    if "use_clean_kernel" in sig.parameters:
        kwargs["use_clean_kernel"] = True
    if "use_runtime_chunk_loop" in sig.parameters:
        kwargs["use_runtime_chunk_loop"] = False
    return qwen_gdn_chunked_avelang_v11_mfma_layout(**kwargs)


def full_v10(q, k, v, g, beta, initial_state, scale, chunk_size):
    return qwen_gdn_chunked_avelang_v10_vllm_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )


def run(T, with_initial_state, warmup, repeat):
    chunk_size = 16
    scale = 128 ** -0.5

    print("=" * 120)
    print(f"T={T}, init={with_initial_state}, chunk={chunk_size}")

    q, k, v, g, beta, initial_state = make_inputs(T, with_initial_state=with_initial_state)

    # 先构造一次中间结果，供 chunk_gdr / chunk_o stage 使用。
    # 这里不是 benchmark，只是准备依赖。
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk_size, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk_size)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(
        k,
        v,
        g_cumsum,
        beta,
        a_solved,
        chunk_size=chunk_size,
        prefer_optimized=True,
    )

    h10, vn10, final10 = call_v10_gdr(k, w, u, g_cumsum, initial_state, chunk_size)
    h11, vn11, final11 = call_v11_gdr(k, w, u, g_cumsum, initial_state, chunk_size)

    sync()

    stages = []

    stages.append(
        bench_one(
            "cumsum_v6",
            lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "KKT_v6",
            lambda: qwen_gdn_kkt_avelang_v6_standalone(
                k,
                g_cumsum,
                beta,
                chunk_size=chunk_size,
                prefer_optimized=True,
            ),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "solve_v6",
            lambda: qwen_gdn_solve_avelang_v6_standalone(
                a,
                chunk_size=chunk_size,
            ),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "w_u_v6",
            lambda: qwen_gdn_w_u_avelang_v6_standalone(
                k,
                v,
                g_cumsum,
                beta,
                a_solved,
                chunk_size=chunk_size,
                prefer_optimized=True,
            ),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "chunk_gdr_v10",
            lambda: call_v10_gdr(k, w, u, g_cumsum, initial_state, chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "chunk_gdr_v11_clean",
            lambda: call_v11_gdr(k, w, u, g_cumsum, initial_state, chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "chunk_o_v10_hvn",
            lambda: call_chunk_o(q, k, vn10, h10, g_cumsum, scale, chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "chunk_o_v11_hvn",
            lambda: call_chunk_o(q, k, vn11, h11, g_cumsum, scale, chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "full_v10",
            lambda: full_v10(q, k, v, g, beta, initial_state, scale, chunk_size),
            warmup,
            repeat,
        )
    )

    stages.append(
        bench_one(
            "full_v11_clean",
            lambda: full_v11(q, k, v, g, beta, initial_state, scale, chunk_size),
            warmup,
            repeat,
        )
    )

    full11_ms = [s for s in stages if s["name"] == "full_v11_clean"][0]["median"]

    print("-" * 120)
    print("stage summary")
    for s in stages:
        pct = s["median"] / full11_ms * 100.0
        print(
            f"{s['name']:22s} "
            f"median={s['median']:9.4f} ms "
            f"mean={s['mean']:9.4f} ms "
            f"min={s['min']:9.4f} ms "
            f"max={s['max']:9.4f} ms "
            f"pct_of_full_v11={pct:7.2f}%"
        )

    gdr10 = [s for s in stages if s["name"] == "chunk_gdr_v10"][0]["median"]
    gdr11 = [s for s in stages if s["name"] == "chunk_gdr_v11_clean"][0]["median"]
    full10 = [s for s in stages if s["name"] == "full_v10"][0]["median"]
    full11 = [s for s in stages if s["name"] == "full_v11_clean"][0]["median"]

    print("-" * 120)
    print(f"chunk_gdr v11_vs_v10 = {gdr10 / gdr11:.3f}x  (>1 means v11 chunk_gdr faster)")
    print(f"full      v11_vs_v10 = {full10 / full11:.3f}x  (>1 means v11 full faster)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--Ts", nargs="+", type=int, default=None)
    parser.add_argument("--with-initial-state", action="store_true")
    parser.add_argument("--without-initial-state", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("device:", torch.cuda.get_device_name(torch.cuda.current_device()))

    Ts = args.Ts if args.Ts is not None else [args.T]

    if args.with_initial_state:
        inits = [True]
    elif args.without_initial_state:
        inits = [False]
    else:
        inits = [False, True]

    for T in Ts:
        for init in inits:
            run(T, init, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
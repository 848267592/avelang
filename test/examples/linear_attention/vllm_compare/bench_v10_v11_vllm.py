import sys
import time
import inspect
import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VLLM_ROOT = str(PROJECT_ROOT / "vllm_stageb_snapshot")
AVELANG_DIR = str(Path(__file__).resolve().parent)

sys.path.insert(0, VLLM_ROOT)
sys.path.insert(0, AVELANG_DIR)

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule as vllm_chunk_gdn,
)
from vllm.model_executor.layers.fla.ops import chunk_delta_h as vllm_chunk_delta_h

from qwen_gdn_chunked_avelang_v10_vllm_layout_fixed import (
    qwen_gdn_chunked_avelang_v10_vllm_layout as avelang_v10_chunk_gdn,
)

from qwen_gdn_chunked_avelang_v11_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v11_mfma_layout as avelang_v11_chunk_gdn,
)


def patch_vllm_rocm_autotune_configs():
    if getattr(torch.version, "hip", None) is None:
        return

    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    configs = getattr(autotuner, "configs", None)
    if not configs:
        return

    filtered = [config for config in configs if getattr(config, "num_stages", None) != 4]
    if len(filtered) != len(configs):
        autotuner.configs = filtered
        autotuner.cache.clear()
        print(
            "patched vLLM ROCm autotune configs: "
            f"{len(configs)} -> {len(filtered)} "
            "(disabled num_stages=4 for chunk_delta_h)"
        )


patch_vllm_rocm_autotune_configs()


def sync():
    torch.cuda.synchronize()


def l2norm(x, eps=1e-6):
    x_f = x.float()
    y = x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + eps)
    return y.to(x.dtype).contiguous()


def make_inputs(B, T, Hk, Hv, K, V, dtype=torch.bfloat16, seed=1234, with_initial_state=True):
    torch.manual_seed(seed)
    device = "cuda"

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


def call_vllm(q, k, v, g, beta, initial_state, scale):
    return vllm_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )


def call_v10(q, k, v, g, beta, initial_state, scale, chunk_size):
    return avelang_v10_chunk_gdn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=chunk_size,
    )


def call_v11(q, k, v, g, beta, initial_state, scale, chunk_size):
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
    )

    sig = inspect.signature(avelang_v11_chunk_gdn)
    if "use_clean_kernel" in sig.parameters:
        kwargs["use_clean_kernel"] = True
    if "use_mfma_chunk_gdr" in sig.parameters:
        kwargs["use_mfma_chunk_gdr"] = True
    if "use_runtime_chunk_loop" in sig.parameters:
        kwargs["use_runtime_chunk_loop"] = False

    return avelang_v11_chunk_gdn(**kwargs)


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def first_call_ms(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def bench_one(fn, warmup=20, repeat=100):
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
    return {
        "mean": t.mean().item(),
        "median": t.median().item(),
        "min": t.min().item(),
        "max": t.max().item(),
    }


def run_case(T, with_initial_state, warmup, repeat):
    B, Hk, Hv, K, V = 1, 4, 8, 128, 128
    chunk_size = 16
    dtype = torch.bfloat16
    scale = K ** -0.5

    print("=" * 120)
    print(f"T={T}, init={with_initial_state}, B={B}, Hk={Hk}, Hv={Hv}, K={K}, V={V}, chunk={chunk_size}")

    q, k, v, g, beta, initial_state = make_inputs(
        B=B,
        T=T,
        Hk=Hk,
        Hv=Hv,
        K=K,
        V=V,
        dtype=dtype,
        with_initial_state=with_initial_state,
    )

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v10 = lambda: call_v10(q, k, v, g, beta, initial_state, scale, chunk_size)
    f_v11 = lambda: call_v11(q, k, v, g, beta, initial_state, scale, chunk_size)

    (o_ref, s_ref), first_vllm = first_call_ms(f_vllm)
    (o10, s10), first_v10 = first_call_ms(f_v10)
    (o11, s11), first_v11 = first_call_ms(f_v11)

    err10_o = max_abs(o10, o_ref)
    err10_s = max_abs(s10, s_ref)
    err11_o = max_abs(o11, o_ref)
    err11_s = max_abs(s11, s_ref)

    print(f"v10 err: out={err10_o:.6e}, state={err10_s:.6e}")
    print(f"v11 err: out={err11_o:.6e}, state={err11_s:.6e}")

    if max(err10_o, err10_s, err11_o, err11_s) > 1.0:
        raise RuntimeError("Correctness exploded; stop benchmark.")

    tv = bench_one(f_vllm, warmup=warmup, repeat=repeat)
    t10 = bench_one(f_v10, warmup=warmup, repeat=repeat)
    t11 = bench_one(f_v11, warmup=warmup, repeat=repeat)

    v11_vs_v10 = t10["median"] / t11["median"]
    v11_vs_vllm = tv["median"] / t11["median"]
    v10_vs_vllm = tv["median"] / t10["median"]

    print(f"first-call ms: vLLM={first_vllm:.3f}, v10={first_v10:.3f}, v11={first_v11:.3f}")
    print("vLLM steady:", tv)
    print("v10  steady:", t10)
    print("v11  steady:", t11)
    print(f"v11_vs_v10  = {v11_vs_v10:.3f}x  (>1 means v11 faster than v10)")
    print(f"v11_vs_vLLM = {v11_vs_vllm:.3f}x  (>1 means v11 faster than vLLM/Triton)")
    print(f"v10_vs_vLLM = {v10_vs_vllm:.3f}x")

    return {
        "T": T,
        "init": with_initial_state,
        "vllm_ms": tv["median"],
        "v10_ms": t10["median"],
        "v11_ms": t11["median"],
        "v11_vs_v10": v11_vs_v10,
        "v11_vs_vllm": v11_vs_vllm,
        "v10_vs_vllm": v10_vs_vllm,
        "err10_o": err10_o,
        "err10_s": err10_s,
        "err11_o": err11_o,
        "err11_s": err11_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ts", nargs="+", type=int, default=[64, 512, 1024, 2048, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--no-init-only", action="store_true")
    args = parser.parse_args()

    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("device:", torch.cuda.get_device_name(torch.cuda.current_device()))

    results = []
    for T in args.Ts:
        if args.init_only:
            inits = [True]
        elif args.no_init_only:
            inits = [False]
        else:
            inits = [False, True]

        for init in inits:
            results.append(run_case(T, init, args.warmup, args.repeat))

    print("=" * 120)
    print("summary")
    for r in results:
        print(
            f"T={r['T']:5d} init={str(r['init']):5s} "
            f"vLLM={r['vllm_ms']:.4f} ms "
            f"v10={r['v10_ms']:.4f} ms "
            f"v11={r['v11_ms']:.4f} ms "
            f"v11/v10={r['v11_vs_v10']:.3f}x "
            f"v11/vLLM={r['v11_vs_vllm']:.3f}x "
            f"v10/vLLM={r['v10_vs_vllm']:.3f}x "
            f"err10=({r['err10_o']:.2e},{r['err10_s']:.2e}) "
            f"err11=({r['err11_o']:.2e},{r['err11_s']:.2e})"
        )


if __name__ == "__main__":
    main()
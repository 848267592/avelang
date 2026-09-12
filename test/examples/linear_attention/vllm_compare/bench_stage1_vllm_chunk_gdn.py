import sys
import time
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

from qwen_gdn_chunked_avelang_v8_vllm_layout_fixed import (
    qwen_gdn_chunked_avelang_v8_vllm_layout as avelang_chunk_gdn,
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


def make_inputs(B, T, Hk, Hv, K, V, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    device = "cuda"

    q = torch.randn(B, T, Hk, K, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, T, Hk, K, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, T, Hv, V, device=device, dtype=dtype).contiguous()

    # 第一阶段：提前 normalize，关闭两边 in-kernel l2norm
    q = l2norm(q)
    k = l2norm(k)

    # g 用 log gate，小一点，避免 exp 数值过激
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, Hv, device=device, dtype=torch.float32)
    )
    g = (g / 16.0).contiguous()

    beta = torch.sigmoid(
        torch.randn(B, T, Hv, device=device, dtype=torch.float32)
    ).contiguous()

    # vLLM state layout: [B, Hv, V, K]
    initial_state = (
        torch.randn(B, Hv, V, K, device=device, dtype=torch.float32).contiguous()
        * 0.01
    )

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


def call_avelang(q, k, v, g, beta, initial_state, scale, chunk_size):
    return avelang_chunk_gdn(
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


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def check_correctness(q, k, v, g, beta, initial_state, scale, chunk_size):
    with torch.no_grad():
        o_vllm, s_vllm = call_vllm(q, k, v, g, beta, initial_state, scale)
        sync()

        o_av, s_av = call_avelang(q, k, v, g, beta, initial_state, scale, chunk_size)
        sync()

    print("output shapes:", tuple(o_vllm.shape), tuple(o_av.shape))
    print("state  shapes:", tuple(s_vllm.shape), tuple(s_av.shape))
    print("output dtype:", o_vllm.dtype, o_av.dtype)
    print("state  dtype:", s_vllm.dtype, s_av.dtype)

    out_err = max_err(o_av, o_vllm)
    state_err = max_err(s_av, s_vllm)

    print("output max abs err:", out_err)
    print("state  max abs err:", state_err)

    return out_err, state_err


def bench_one(fn, warmup=20, repeat=100):
    # 预热，排除第一次 JIT / autotune / cache 开销
    for _ in range(warmup):
        fn()
    sync()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

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


def run_case(B, T, Hk, Hv, K, V, chunk_size, dtype=torch.bfloat16):
    print("=" * 100)
    print(
        f"B={B}, T={T}, Hk={Hk}, Hv={Hv}, K={K}, V={V}, "
        f"chunk_size={chunk_size}, dtype={dtype}"
    )

    q, k, v, g, beta, initial_state = make_inputs(
        B=B,
        T=T,
        Hk=Hk,
        Hv=Hv,
        K=K,
        V=V,
        dtype=dtype,
        seed=1234,
    )

    scale = K ** -0.5

    out_err, state_err = check_correctness(
        q, k, v, g, beta, initial_state, scale, chunk_size
    )

    # 如果 correctness 完全炸了，不测性能
    if out_err > 1.0 or state_err > 1.0:
        raise RuntimeError("Correctness error too large. Stop before benchmark.")

    tv = bench_one(
        lambda: call_vllm(q, k, v, g, beta, initial_state, scale),
        warmup=20,
        repeat=100,
    )

    ta = bench_one(
        lambda: call_avelang(q, k, v, g, beta, initial_state, scale, chunk_size),
        warmup=20,
        repeat=100,
    )

    speedup = tv["median"] / ta["median"]

    print("vLLM ms:", tv)
    print("Avelang ms:", ta)
    print("speedup median:", speedup)

    return {
        "B": B,
        "T": T,
        "Hk": Hk,
        "Hv": Hv,
        "K": K,
        "V": V,
        "chunk_size": chunk_size,
        "dtype": str(dtype),
        "output_err": out_err,
        "state_err": state_err,
        "vllm_median_ms": tv["median"],
        "avelang_median_ms": ta["median"],
        "speedup": speedup,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    print("torch:", torch.__version__)
    print("hip:", getattr(torch.version, "hip", None))
    print("device:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Primary target: vLLM Qwen3Next TP4 per-rank linear attention shape.
    cases = [
        dict(B=1, T=64, Hk=4, Hv=8, K=128, V=128, chunk_size=4),
        dict(B=1, T=512, Hk=4, Hv=8, K=128, V=128, chunk_size=4),
        dict(B=1, T=1024, Hk=4, Hv=8, K=128, V=128, chunk_size=4),
        dict(B=1, T=2048, Hk=4, Hv=8, K=128, V=128, chunk_size=4),

        # Legacy/dev toy shape only; do not use as the primary conclusion.
        dict(B=1, T=512, Hk=4, Hv=8, K=64, V=64, chunk_size=8),
    ]

    results = []
    for case in cases:
        results.append(run_case(**case))

    print("=" * 100)
    print("summary")
    for r in results:
        print(
            f"B={r['B']} T={r['T']} Hk={r['Hk']} Hv={r['Hv']} "
            f"K={r['K']} V={r['V']} chunk={r['chunk_size']} "
            f"vLLM={r['vllm_median_ms']:.4f} ms "
            f"Avelang={r['avelang_median_ms']:.4f} ms "
            f"speedup={r['speedup']:.3f}x "
            f"out_err={r['output_err']:.3e} state_err={r['state_err']:.3e}"
        )


if __name__ == "__main__":
    main()

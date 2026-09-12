#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
VLLM_ROOT = str(PROJECT_ROOT / "vllm_stageb_snapshot")
AVELANG_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, VLLM_ROOT)
sys.path.insert(0, AVELANG_DIR)

from vllm.model_executor.layers.fla.ops import chunk_delta_h as vllm_chunk_delta_h
from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule as vllm_chunk_gdn
from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v13_mfma_layout_fixed import qwen_gdn_chunk_o_avelang_v13_mfma_layout
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout,
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_chunked_avelang_v17_predecay_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v18_bt64_layout_fixed import qwen_gdn_solve_avelang_v18_layout
from qwen_gdn_chunked_avelang_v19_bt32_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout,
    qwen_gdn_chunked_avelang_v19_bt32_mfma_layout,
    qwen_gdn_w_u_avelang_v19_bt32_mfma_layout,
)
from qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout,
    qwen_gdn_kkt_avelang_v20_bt32_mfma_layout,
    qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout,
)


def patch_vllm_rocm_autotune_configs() -> None:
    if getattr(torch.version, "hip", None) is None:
        return
    kernel = vllm_chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    autotuner = getattr(kernel, "fn", None)
    configs = getattr(autotuner, "configs", None)
    if configs:
        filtered = [c for c in configs if getattr(c, "num_stages", None) != 4]
        if len(filtered) != len(configs):
            autotuner.configs = filtered
            autotuner.cache.clear()
            print(f"patched vLLM ROCm autotune configs: {len(configs)} -> {len(filtered)}")


def sync() -> None:
    torch.cuda.synchronize()


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(t: int, seed: int = 1234, with_initial_state: bool = True):
    torch.manual_seed(seed)
    b, hk, hv, kdim, vdim = 1, 4, 8, 128, 128
    q = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(b, t, hk, kdim, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(b, t, hv, vdim, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(b, t, hv, device="cuda", dtype=torch.float32)).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(b, hv, vdim, kdim, device="cuda", dtype=torch.float32) * 0.01).contiguous()
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


def time_fn(fn, warmup: int, repeat: int) -> float:
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
    return statistics.median(times)


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def stage_breakdown_v17(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int):
    chunk = 16
    cumsum = lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    g_cumsum = cumsum()
    kkt = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a = kkt()
    solve = lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    a_solved = solve()
    wu = lambda: qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    w, u = wu()
    decay = lambda: qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)
    gdr_decay, gdr_g_last_exp = decay()
    gdr = lambda: qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
        k, w, u, g_cumsum, gdr_decay, gdr_g_last_exp, initial_state=initial_state, chunk_size=chunk
    )
    h, vn, _ = gdr()
    chunk_o = lambda: qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "gdr_decay": time_fn(decay, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def stage_breakdown_bt32(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int, version: str):
    chunk = 32
    cumsum = lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    g_cumsum = cumsum()
    if version == "v19":
        kkt = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
        full = lambda: qwen_gdn_chunked_avelang_v19_bt32_mfma_layout(
            q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=chunk
        )
    elif version == "v20":
        kkt = lambda: qwen_gdn_kkt_avelang_v20_bt32_mfma_layout(k, g_cumsum, beta, chunk_size=chunk)
        full = lambda: qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout(
            q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=chunk
        )
    else:
        raise ValueError(version)
    a = kkt()
    solve = lambda: qwen_gdn_solve_avelang_v18_layout(a, chunk_size=chunk)
    a_solved = solve()
    if version == "v19":
        wu = lambda: qwen_gdn_w_u_avelang_v19_bt32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    else:
        wu = lambda: qwen_gdn_w_u_avelang_v20_bt32_32x32_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    w, u = wu()
    gdr = lambda: qwen_gdn_chunk_gdr_avelang_v19_bt32_mfma_layout(k, w, u, g_cumsum, initial_state=initial_state, chunk_size=chunk)
    h, vn, _ = gdr()
    chunk_o = lambda: qwen_gdn_chunk_o_avelang_v13_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    return {
        "full": time_fn(full, warmup, repeat),
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "gdr_decay": 0.0,
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def run_t(t: int, warmup: int, repeat: int) -> None:
    q, k, v, g, beta, initial_state = make_inputs(t, seed=20000 + t)
    scale = 128 ** -0.5
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,layout=[B,T,H,D],BT32")

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v17 = lambda: qwen_gdn_chunked_avelang_v17_predecay_mfma_layout(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=16
    )
    f_v19 = lambda: qwen_gdn_chunked_avelang_v19_bt32_mfma_layout(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=32
    )
    f_v20 = lambda: qwen_gdn_chunked_avelang_v20_bt32_native_mfma_layout(
        q, k, v, g, beta, initial_state=initial_state, scale=scale, chunk_size=32
    )

    out_v17, fs_v17 = f_v17()
    out_v19, fs_v19 = f_v19()
    out_v20, fs_v20 = f_v20()
    out_vllm, fs_vllm = f_vllm()
    sync()

    full_vllm = time_fn(f_vllm, warmup, repeat)
    full_v17 = time_fn(f_v17, warmup, repeat)
    full_v19 = time_fn(f_v19, warmup, repeat)
    full_v20 = time_fn(f_v20, warmup, repeat)
    print(
        "full_latency_ms,"
        f"T={t},vllm={full_vllm:.6f},v17_predecay={full_v17:.6f},v19_bt32={full_v19:.6f},v20_bt32={full_v20:.6f},"
        f"speedup_v20_vs_v19={full_v19 / full_v20:.4f},speedup_v20_vs_v17={full_v17 / full_v20:.4f},"
        f"slowdown_v20_vs_vllm={full_v20 / full_vllm:.4f}"
    )
    print(
        "accuracy,"
        f"T={t},output_vs_v19={max_err(out_v20, out_v19):.9g},final_state_vs_v19={max_err(fs_v20, fs_v19):.9g},"
        f"output_vs_v17={max_err(out_v20, out_v17):.9g},final_state_vs_v17={max_err(fs_v20, fs_v17):.9g},"
        f"output_vs_vllm={max_err(out_v20, out_vllm):.9g},final_state_vs_vllm={max_err(fs_v20, fs_vllm):.9g}"
    )

    s17 = stage_breakdown_v17(q, k, v, g, beta, initial_state, scale, warmup, repeat)
    s19 = stage_breakdown_bt32(q, k, v, g, beta, initial_state, scale, warmup, repeat, "v19")
    s20 = stage_breakdown_bt32(q, k, v, g, beta, initial_state, scale, warmup, repeat, "v20")
    for name, stages in (("v17_predecay", s17), ("v19_bt32", s19), ("v20_bt32", s20)):
        print(
            "stage_breakdown_ms,"
            f"T={t},version={name},"
            f"cumsum={stages['cumsum']:.6f},KKT={stages['KKT']:.6f},solve={stages['solve']:.6f},"
            f"w_u={stages['w_u']:.6f},gdr_decay={stages['gdr_decay']:.6f},"
            f"chunk_gdr={stages['chunk_gdr']:.6f},chunk_o={stages['chunk_o']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")
    patch_vllm_rocm_autotune_configs()
    for t in args.T:
        run_t(t, args.warmup, args.repeat)


if __name__ == "__main__":
    main()

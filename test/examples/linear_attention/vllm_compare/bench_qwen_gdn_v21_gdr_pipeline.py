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
from qwen_gdn_chunked_avelang_v17_mfma_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout,
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_chunked_avelang_v17_predecay_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v21_4wave_pipeline_mfma_layout,
    qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout,
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


def stage_breakdown(q, k, v, g, beta, initial_state, scale, warmup: int, repeat: int, *, version: str):
    chunk = 16
    cumsum = lambda: qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    g_cumsum = cumsum()
    kkt = lambda: qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a = kkt()
    solve = lambda: qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    a_solved = solve()
    wu = lambda: qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)
    w, u = wu()
    gdr_decay_fn = lambda: qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)
    gdr_decay, gdr_g_last_exp = gdr_decay_fn()

    def gdr():
        if version == "v17_predecay":
            return qwen_gdn_chunk_gdr_avelang_v17_4wave_predecay_mfma_layout(
                k,
                w,
                u,
                g_cumsum,
                gdr_decay,
                gdr_g_last_exp,
                initial_state=initial_state,
                chunk_size=chunk,
            )
        if version == "v21_w":
            variant = "w"
        elif version == "v21_wk":
            variant = "wk"
        else:
            raise ValueError(version)
        return qwen_gdn_chunk_gdr_avelang_v21_4wave_pipeline_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            gdr_decay,
            gdr_g_last_exp,
            initial_state=initial_state,
            chunk_size=chunk,
            variant=variant,
        )

    h, vn, _ = gdr()
    chunk_o = lambda: qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)
    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu, warmup, repeat),
        "gdr_decay": time_fn(gdr_decay_fn, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def run_t(t: int, warmup: int, repeat: int) -> dict:
    q, k, v, g, beta, initial_state = make_inputs(t, seed=21000 + t)
    scale = 128 ** -0.5
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,layout=[B,T,H,D],chunk=16")

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v17 = lambda: qwen_gdn_chunked_avelang_v17_predecay_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )
    f_v21_w = lambda: qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
        variant="w",
    )
    f_v21_wk = lambda: qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
        variant="wk",
    )

    out_vllm, fs_vllm = f_vllm()
    out_v17, fs_v17 = f_v17()
    out_w, fs_w = f_v21_w()
    out_wk, fs_wk = f_v21_wk()
    sync()

    stage_warmup = max(2, warmup // 4)
    stage_repeat = max(5, repeat // 4)
    vllm_ms = time_fn(f_vllm, warmup, repeat)
    v17_ms = time_fn(f_v17, warmup, repeat)
    v21_w_ms = time_fn(f_v21_w, warmup, repeat)
    v21_wk_ms = time_fn(f_v21_wk, warmup, repeat)
    stages_v17 = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v17_predecay")
    stages_w = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v21_w")
    stages_wk = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v21_wk")

    result = {
        "T": t,
        "vllm_ms": vllm_ms,
        "v17_predecay_ms": v17_ms,
        "v21_w_ms": v21_w_ms,
        "v21_wk_ms": v21_wk_ms,
        "stages_v17": stages_v17,
        "stages_w": stages_w,
        "stages_wk": stages_wk,
        "w_out_err": max_err(out_w, out_v17),
        "w_fs_err": max_err(fs_w, fs_v17),
        "wk_out_err": max_err(out_wk, out_v17),
        "wk_fs_err": max_err(fs_wk, fs_v17),
        "v17_vllm_out_err": max_err(out_v17, out_vllm),
        "v17_vllm_fs_err": max_err(fs_v17, fs_vllm),
    }
    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, nargs="*", default=[512, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    patch_vllm_rocm_autotune_configs()
    rows = [run_t(t, args.warmup, args.repeat) for t in args.T]

    print("summary")
    for r in rows:
        s17 = r["stages_v17"]
        sw = r["stages_w"]
        swk = r["stages_wk"]
        print(
            f"T={r['T']} "
            f"vLLM={r['vllm_ms']:.4f}ms "
            f"v17_predecay={r['v17_predecay_ms']:.4f}ms "
            f"v21_w={r['v21_w_ms']:.4f}ms "
            f"v21_wk={r['v21_wk_ms']:.4f}ms "
            f"full_speedup_w_vs_v17={r['v17_predecay_ms'] / r['v21_w_ms']:.4f} "
            f"full_speedup_wk_vs_v17={r['v17_predecay_ms'] / r['v21_wk_ms']:.4f} "
            f"chunk_gdr_v17={s17['chunk_gdr']:.4f}ms "
            f"chunk_gdr_w={sw['chunk_gdr']:.4f}ms "
            f"chunk_gdr_wk={swk['chunk_gdr']:.4f}ms "
            f"chunk_gdr_speedup_w={s17['chunk_gdr'] / sw['chunk_gdr']:.4f} "
            f"chunk_gdr_speedup_wk={s17['chunk_gdr'] / swk['chunk_gdr']:.4f} "
            f"slowdown_v17_vs_vllm={r['v17_predecay_ms'] / r['vllm_ms']:.4f} "
            f"slowdown_w_vs_vllm={r['v21_w_ms'] / r['vllm_ms']:.4f} "
            f"slowdown_wk_vs_vllm={r['v21_wk_ms'] / r['vllm_ms']:.4f} "
            f"v17_stages=cumsum:{s17['cumsum']:.4f},KKT:{s17['KKT']:.4f},solve:{s17['solve']:.4f},"
            f"w_u:{s17['w_u']:.4f},gdr_decay:{s17['gdr_decay']:.4f},chunk_gdr:{s17['chunk_gdr']:.4f},chunk_o:{s17['chunk_o']:.4f} "
            f"v21_w_stages=cumsum:{sw['cumsum']:.4f},KKT:{sw['KKT']:.4f},solve:{sw['solve']:.4f},"
            f"w_u:{sw['w_u']:.4f},gdr_decay:{sw['gdr_decay']:.4f},chunk_gdr:{sw['chunk_gdr']:.4f},chunk_o:{sw['chunk_o']:.4f} "
            f"v21_wk_stages=cumsum:{swk['cumsum']:.4f},KKT:{swk['KKT']:.4f},solve:{swk['solve']:.4f},"
            f"w_u:{swk['w_u']:.4f},gdr_decay:{swk['gdr_decay']:.4f},chunk_gdr:{swk['chunk_gdr']:.4f},chunk_o:{swk['chunk_o']:.4f} "
            f"out_err_w={r['w_out_err']:.6g} fs_err_w={r['w_fs_err']:.6g} "
            f"out_err_wk={r['wk_out_err']:.6g} fs_err_wk={r['wk_fs_err']:.6g} "
            f"out_err_vllm={r['v17_vllm_out_err']:.6g} fs_err_vllm={r['v17_vllm_fs_err']:.6g}"
        )


if __name__ == "__main__":
    main()

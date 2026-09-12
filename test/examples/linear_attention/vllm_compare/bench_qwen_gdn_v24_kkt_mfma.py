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
    qwen_gdn_chunk_o_avelang_v14_mfma_layout,
    qwen_gdn_gdr_decay_avelang_v17,
    qwen_gdn_w_u_avelang_v14_mfma_layout,
)
from qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed import (
    qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout,
    qwen_gdn_chunked_avelang_v23_gdr_distributed_layout,
)
from qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed import (
    qwen_gdn_chunked_avelang_v24_kkt_mfma_layout,
    qwen_gdn_kkt_avelang_v24_bt16_mfma_layout,
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

    def cumsum():
        return qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)

    g_cumsum = cumsum()

    if version == "v23":
        def kkt():
            return qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    elif version == "v24":
        def kkt():
            return qwen_gdn_kkt_avelang_v24_bt16_mfma_layout(k, g_cumsum, beta, chunk_size=chunk)
    else:
        raise ValueError(version)

    a = kkt()

    def solve():
        return qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)

    a_solved = solve()

    def wu_fn():
        return qwen_gdn_w_u_avelang_v14_mfma_layout(k, v, g_cumsum, beta, a_solved, chunk_size=chunk)

    w, u = wu_fn()

    def gdr_decay_fn():
        return qwen_gdn_gdr_decay_avelang_v17(g_cumsum, chunk_size=chunk)

    gdr_decay, gdr_g_last_exp = gdr_decay_fn()

    def gdr():
        return qwen_gdn_chunk_gdr_avelang_v23_4wave_distributed_mfma_layout(
            k,
            w,
            u,
            g_cumsum,
            gdr_decay,
            gdr_g_last_exp,
            initial_state=initial_state,
            chunk_size=chunk,
        )

    h, vn, _ = gdr()

    def chunk_o():
        return qwen_gdn_chunk_o_avelang_v14_mfma_layout(q, k, vn, h, g_cumsum, scale=scale, chunk_size=chunk)

    return {
        "cumsum": time_fn(cumsum, warmup, repeat),
        "KKT": time_fn(kkt, warmup, repeat),
        "solve": time_fn(solve, warmup, repeat),
        "w_u": time_fn(wu_fn, warmup, repeat),
        "gdr_decay": time_fn(gdr_decay_fn, warmup, repeat),
        "chunk_gdr": time_fn(gdr, warmup, repeat),
        "chunk_o": time_fn(chunk_o, warmup, repeat),
    }


def run_t(t: int, warmup: int, repeat: int):
    q, k, v, g, beta, initial_state = make_inputs(t, seed=24000 + t)
    scale = 128 ** -0.5
    print(f"target_shape,B=1,T={t},Hk=4,Hv=8,K=128,V=128,dtype=BF16,layout=[B,T,H,D],chunk=16")

    f_vllm = lambda: call_vllm(q, k, v, g, beta, initial_state, scale)
    f_v23 = lambda: qwen_gdn_chunked_avelang_v23_gdr_distributed_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )
    f_v24 = lambda: qwen_gdn_chunked_avelang_v24_kkt_mfma_layout(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        scale=scale,
        chunk_size=16,
    )

    out_vllm, fs_vllm = f_vllm()
    out_v23, fs_v23 = f_v23()
    out_v24, fs_v24 = f_v24()
    sync()

    stage_warmup = max(2, warmup // 4)
    stage_repeat = max(5, repeat // 4)
    vllm_ms = time_fn(f_vllm, warmup, repeat)
    v23_ms = time_fn(f_v23, warmup, repeat)
    v24_ms = time_fn(f_v24, warmup, repeat)
    stages_v23 = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v23")
    stages_v24 = stage_breakdown(q, k, v, g, beta, initial_state, scale, stage_warmup, stage_repeat, version="v24")

    result = {
        "T": t,
        "vllm_ms": vllm_ms,
        "v23_ms": v23_ms,
        "v24_ms": v24_ms,
        "kkt_speedup_v24_vs_v23": stages_v23["KKT"] / stages_v24["KKT"],
        "full_speedup_v24_vs_v23": v23_ms / v24_ms,
        "slowdown_v24_vs_vllm": v24_ms / vllm_ms,
        "v24_vs_v23_out_err": max_err(out_v24, out_v23),
        "v24_vs_v23_fs_err": max_err(fs_v24, fs_v23),
        "v24_vs_vllm_out_err": max_err(out_v24, out_vllm),
        "v24_vs_vllm_fs_err": max_err(fs_v24, fs_vllm),
        "stages_v23": stages_v23,
        "stages_v24": stages_v24,
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
        s23 = r["stages_v23"]
        s24 = r["stages_v24"]
        print(
            f"T={r['T']} vLLM={r['vllm_ms']:.4f}ms v23={r['v23_ms']:.4f}ms v24={r['v24_ms']:.4f}ms "
            f"full_speedup_v24_vs_v23={r['full_speedup_v24_vs_v23']:.4f} "
            f"kkt_speedup_v24_vs_v23={r['kkt_speedup_v24_vs_v23']:.4f} "
            f"slowdown_v24_vs_vllm={r['slowdown_v24_vs_vllm']:.4f} "
            f"v23_cumsum={s23['cumsum']:.4f} v23_KKT={s23['KKT']:.4f} v23_solve={s23['solve']:.4f} "
            f"v23_w_u={s23['w_u']:.4f} v23_gdr_decay={s23['gdr_decay']:.4f} "
            f"v23_chunk_gdr={s23['chunk_gdr']:.4f} v23_chunk_o={s23['chunk_o']:.4f} "
            f"v24_cumsum={s24['cumsum']:.4f} v24_KKT={s24['KKT']:.4f} v24_solve={s24['solve']:.4f} "
            f"v24_w_u={s24['w_u']:.4f} v24_gdr_decay={s24['gdr_decay']:.4f} "
            f"v24_chunk_gdr={s24['chunk_gdr']:.4f} v24_chunk_o={s24['chunk_o']:.4f} "
            f"out_err_v23={r['v24_vs_v23_out_err']:.6g} fs_err_v23={r['v24_vs_v23_fs_err']:.6g} "
            f"out_err_vllm={r['v24_vs_vllm_out_err']:.6g} fs_err_vllm={r['v24_vs_vllm_fs_err']:.6g}"
        )


if __name__ == "__main__":
    main()

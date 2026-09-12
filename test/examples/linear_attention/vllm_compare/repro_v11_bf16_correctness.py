#!/usr/bin/env python3
"""BF16-operand correctness repro for v11 MFMA chunk_gdr.

This reference matches the current v11 arithmetic policy:
- pred uses BF16 W/state operands with FP32 accumulation
- update uses BF16 v_decay/K operands with FP32 accumulation
- state decay and final state remain FP32

It is intentionally different from the v10 FP32 oracle.
"""

from __future__ import annotations

import argparse
import torch

from qwen_gdn_chunked_avelang_v6_vllm_layout_fixed import (
    qwen_gdn_chunk_cumsum_avelang_v6_standalone,
    qwen_gdn_kkt_avelang_v6_standalone,
    qwen_gdn_solve_avelang_v6_standalone,
    qwen_gdn_w_u_avelang_v6_standalone,
)
from qwen_gdn_chunked_avelang_v11_mfma_layout_fixed import qwen_gdn_chunk_gdr_avelang_v11_mfma_layout


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def make_inputs(T: int, with_initial_state: bool, seed: int):
    torch.manual_seed(seed)
    B, Hk, Hv, K, V = 1, 4, 8, 128, 128
    k = l2norm(torch.randn(B, T, Hk, K, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(B, T, Hv, V, device="cuda", dtype=torch.bfloat16).contiguous()
    g = (torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device="cuda")) / 16.0).contiguous()
    beta = torch.sigmoid(torch.randn(B, T, Hv, device="cuda")).contiguous()
    initial_state = None
    if with_initial_state:
        initial_state = (torch.randn(B, Hv, V, K, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    return k, v, g, beta, initial_state


def build_w_u(k, v, g, beta, chunk: int):
    g_cumsum = qwen_gdn_chunk_cumsum_avelang_v6_standalone(g, chunk_size=chunk)
    a = qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=chunk, prefer_optimized=True)
    a_solved = qwen_gdn_solve_avelang_v6_standalone(a, chunk_size=chunk)
    w, u = qwen_gdn_w_u_avelang_v6_standalone(k, v, g_cumsum, beta, a_solved, chunk_size=chunk, prefer_optimized=True)
    return g_cumsum, w, u


def bf16_chunk_gdr_ref(k, w, u, g, initial_state, chunk: int):
    B, T, Hk, K = k.shape
    _, _, Hv, V = u.shape
    num_chunks = T // chunk
    repeat = Hv // Hk
    h = torch.empty((B, num_chunks, Hv, V, K), device="cuda", dtype=torch.float32)
    vn = torch.empty_like(u)
    final_state = torch.empty((B, Hv, V, K), device="cuda", dtype=torch.float32)

    for hv in range(Hv):
        kh = hv // repeat
        if initial_state is None:
            state = torch.zeros((V, K), device="cuda", dtype=torch.float32)
        else:
            state = initial_state[0, hv].clone()
        for c in range(num_chunks):
            start = c * chunk
            end = start + chunk
            h[0, c, hv] = state
            w_b = w[0, start:end, hv].to(torch.bfloat16).float()
            state_b = state.to(torch.bfloat16).float()
            pred = w_b @ state_b.T
            vn_chunk = u[0, start:end, hv] - pred
            vn[0, start:end, hv] = vn_chunk
            g_last = g[0, end - 1, hv]
            state = state * torch.exp(g_last)
            decay = torch.exp(g_last - g[0, start:end, hv])
            v_decay = (vn_chunk * decay[:, None]).to(torch.bfloat16).float()
            k_b = k[0, start:end, kh].to(torch.bfloat16).float()
            state = state + v_decay.T @ k_b
        final_state[0, hv] = state
    return h, vn, final_state


def check(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> bool:
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.float().abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    print(f"{name}_ok={bool(ok)},max_abs={max_abs:.9g},max_rel={max_rel:.9g}")
    return bool(ok)


def run_case(T: int, with_initial_state: bool, seed: int, atol: float, rtol: float, use_runtime_chunk_loop: bool = False) -> bool:
    chunk = 16
    print(f"case,T={T},with_initial_state={with_initial_state},chunk={chunk},runtime_loop={use_runtime_chunk_loop}")
    k, v, g, beta, initial_state = make_inputs(T, with_initial_state, seed)
    g_cumsum, w, u = build_w_u(k, v, g, beta, chunk)
    h11, vn11, final11 = qwen_gdn_chunk_gdr_avelang_v11_mfma_layout(
        k, w, u, g_cumsum,
        initial_state=initial_state,
        chunk_size=chunk,
        use_mfma_chunk_gdr=True,
        prefer_optimized=True,
        block_v=16,
        block_k=64,
        use_runtime_chunk_loop=use_runtime_chunk_loop,
    )
    h_ref, vn_ref, final_ref = bf16_chunk_gdr_ref(k, w, u, g_cumsum, initial_state, chunk)
    ok_h = check("h", h11, h_ref, atol, rtol)
    ok_vn = check("vn", vn11, vn_ref, atol, rtol)
    ok_final = check("final_state", final11, final_ref, atol, rtol)
    ok = ok_h and ok_vn and ok_final
    print(f"case_status={'ok' if ok else 'failed'}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--runtime-loop", action="store_true", help="Use experimental runtime chunk loop v11 kernel.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA/HIP GPU")
    if args.T % 16 != 0:
        raise ValueError("T must be divisible by 16 for the narrow v11 path")
    if args.both:
        ok0 = run_case(args.T, False, args.seed, args.atol, args.rtol, args.runtime_loop)
        ok1 = run_case(args.T, True, args.seed + 1, args.atol, args.rtol, args.runtime_loop)
        if not (ok0 and ok1):
            raise SystemExit(1)
    else:
        ok = run_case(args.T, not args.no_initial_state, args.seed, args.atol, args.rtol, args.runtime_loop)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

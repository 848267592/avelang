#!/usr/bin/env python3
"""Minimal MI300X smoke test for SGLang's ordinary Triton KDA kernels.

This intentionally bypasses the attention dispatcher and never imports or
selects FlashKDA.  It uses random tensors only; no model weights or backward
pass are involved.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains NaN or Inf")
    print(f"{name}_shape={tuple(value.shape)} finite=True")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print("framework=sglang ordinary_triton_kda")
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"device_capability={torch.cuda.get_device_capability(0)}")
    print("flashkda=False")
    print("training=False backward=False model_weights=False")

    # One 64-token KDA chunk, one query/value head, and MI300X-friendly
    # dimensions.  The state pool and index mirror the inference cache API.
    batch, tokens, heads, key_dim, value_dim = 1, 64, 1, 128, 128
    bf16 = torch.bfloat16
    scale = key_dim**-0.5
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.long)
    cache_indices = torch.tensor([0], device=device, dtype=torch.long)
    A_log = torch.zeros((heads,), device=device, dtype=torch.float32)
    dt_bias = torch.zeros((heads * key_dim,), device=device, dtype=torch.float32)

    q = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    k = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    v = torch.randn((batch, tokens, heads, value_dim), device=device, dtype=bf16)
    raw_g = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    beta = torch.sigmoid(torch.randn((batch, tokens, heads), device=device, dtype=bf16))
    prefill_state = torch.zeros(
        (1, heads, value_dim, key_dim), device=device, dtype=torch.float32
    )

    # This is the SGLang Triton wrapper's extend path.  A_log/dt_bias make the
    # wrapper execute the real per-key KDA gate path rather than a GDN fallback.
    kernel = TritonKDAKernel()
    with torch.inference_mode():
        prefill_out = kernel.extend(
            q=q,
            k=k,
            v=v,
            g=raw_g,
            beta=beta,
            ssm_states=prefill_state,
            cache_indices=cache_indices,
            query_start_loc=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
            beta_is_raw=False,
            return_intermediate_states=False,
        )
        torch.cuda.synchronize()
    if isinstance(prefill_out, tuple):
        prefill_out = prefill_out[0]
    expected_prefill = (batch, tokens, heads, value_dim)
    if tuple(prefill_out.shape) != expected_prefill:
        raise RuntimeError(
            f"unexpected SGLang prefill shape {tuple(prefill_out.shape)}; "
            f"expected {expected_prefill}"
        )
    _check_finite("sglang_prefill", prefill_out)
    print(
        "sglang_prefill_path="
        "TritonKDAKernel.extend -> sglang/kernels/ops/attention/fla/kda.py:chunk_kda"
    )

    # Call the packed recurrent Triton function directly.  Setting
    # use_qk_l2norm_in_kernel=False deliberately avoids SGLang's optional
    # custom packed-decode extension, leaving the @triton.jit kernel as the
    # implementation under test.
    mixed_qkv = torch.randn(
        (batch, 2 * heads * key_dim + heads * value_dim),
        device=device,
        dtype=bf16,
    ).contiguous()
    decode_a = torch.randn((batch, heads * key_dim), device=device, dtype=bf16)
    decode_b = torch.randn((batch, heads), device=device, dtype=bf16)
    decode_state = torch.zeros_like(prefill_state)
    decode_out = torch.empty(
        (batch, 1, heads, value_dim), device=device, dtype=bf16
    )
    with torch.inference_mode():
        decode_out, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=decode_a,
            b=decode_b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=decode_state,
            out=decode_out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=False,
            lower_bound=None,
        )
        torch.cuda.synchronize()
    expected_decode = (batch, 1, heads, value_dim)
    if tuple(decode_out.shape) != expected_decode:
        raise RuntimeError(
            f"unexpected SGLang decode shape {tuple(decode_out.shape)}; "
            f"expected {expected_decode}"
        )
    _check_finite("sglang_decode", decode_out)
    print(
        "sglang_decode_path="
        "sglang/kernels/ops/attention/fla/fused_recurrent.py:"
        "fused_recurrent_kda_packed_decode_kernel"
    )
    print("sglang_status=PASS")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Minimal MI300X smoke test for vLLM's AMD ordinary Triton KDA kernels.

The calls go straight to the Kimi-K3 AMD vendored Triton implementation.  No
model weights, training, backward pass, NVIDIA path, or FlashKDA dispatcher is
used.
"""

from __future__ import annotations

import torch

from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
    chunk_kda_with_fused_gate,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)


def _check_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all().item():
        raise RuntimeError(f"{name} contains NaN or Inf")
    print(f"{name}_shape={tuple(value.shape)} finite=True")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print("framework=vllm kimi_k3_amd ordinary_triton_kda")
    print(f"torch={torch.__version__} hip={torch.version.hip}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"device_capability={torch.cuda.get_device_capability(0)}")
    print("amd_path=True nvidia_path=False")
    print("flashkda=False training=False backward=False model_weights=False")

    batch, tokens, heads, key_dim, value_dim = 1, 64, 1, 128, 128
    bf16 = torch.bfloat16
    scale = key_dim**-0.5
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.long)
    A_log = torch.zeros((heads,), device=device, dtype=torch.float32)
    dt_bias = torch.zeros((heads * key_dim,), device=device, dtype=torch.float32)

    q = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    k = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    v = torch.randn((batch, tokens, heads, value_dim), device=device, dtype=bf16)
    raw_g = torch.randn((batch, tokens, heads, key_dim), device=device, dtype=bf16)
    raw_beta = torch.randn((batch, tokens, heads), device=device, dtype=bf16)
    prefill_state = torch.zeros(
        (1, heads, value_dim, key_dim), device=device, dtype=torch.float32
    )

    # Direct AMD KDA prefill wrapper: gate activation, chunk solve, recurrent
    # state update, and output are all the ordinary Triton path.
    with torch.inference_mode():
        prefill_out, prefill_final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            g_bias=dt_bias,
            scale=scale,
            initial_state=prefill_state,
            output_final_state=True,
            lower_bound=None,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        torch.cuda.synchronize()
    expected_prefill = (batch, tokens, heads, value_dim)
    if tuple(prefill_out.shape) != expected_prefill:
        raise RuntimeError(
            f"unexpected vLLM prefill shape {tuple(prefill_out.shape)}; "
            f"expected {expected_prefill}"
        )
    _check_finite("vllm_prefill", prefill_out)
    if prefill_final_state is not None:
        _check_finite("vllm_prefill_final_state", prefill_final_state)
    print(
        "vllm_prefill_path="
        "vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py:"
        "chunk_kda_with_fused_gate -> @triton.jit chunk kernels"
    )

    # Packed one-token recurrent decode is the AMD vendored Triton kernel.  It
    # updates a cache slot in place, matching inference-time KDA semantics.
    mixed_qkv = torch.randn(
        (batch, 2 * heads * key_dim + heads * value_dim),
        device=device,
        dtype=bf16,
    ).contiguous()
    decode_raw_g = torch.randn(
        (1, batch, heads, key_dim), device=device, dtype=bf16
    ).contiguous()
    decode_raw_beta = torch.randn(
        (1, batch, heads), device=device, dtype=bf16
    ).contiguous()
    decode_state = torch.zeros_like(prefill_state)
    state_indices = torch.tensor([0], device=device, dtype=torch.long)
    with torch.inference_mode():
        decode_out, _ = fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            raw_g=decode_raw_g,
            raw_beta=decode_raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=None,
            initial_state=decode_state,
            state_indices=state_indices,
            scale=scale,
        )
        torch.cuda.synchronize()
    expected_decode = (1, batch, heads, value_dim)
    if tuple(decode_out.shape) != expected_decode:
        raise RuntimeError(
            f"unexpected vLLM decode shape {tuple(decode_out.shape)}; "
            f"expected {expected_decode}"
        )
    _check_finite("vllm_decode", decode_out)
    print(
        "vllm_decode_path="
        "vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py:"
        "fused_recurrent_kda_packed_decode_kernel"
    )
    print("vllm_status=PASS")


if __name__ == "__main__":
    main()

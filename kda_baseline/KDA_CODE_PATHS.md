# Inference KDA reference code paths

Stage 1 source snapshot (2026-09-10):

| project | upstream URL | checkout | `git rev-parse HEAD` | `git describe --tags --always` |
|---|---|---|---|---|
| SGLang | https://github.com/sgl-project/sglang | `third_party/sglang` | `908226fea2df861769e2720161a75649ae4c6f92` | `908226f` |
| vLLM | https://github.com/vllm-project/vllm | `third_party/vllm` | `40e6042ec83eb8f2971f21043a5da40496bd188a` | `40e6042` |

Both checkouts are shallow `main` snapshots with clean worktrees. No model
weights were downloaded.

## Scope

This map is for forward-only inference KDA: ordinary Triton KDA kernels,
including chunked prefill/extend and recurrent decode. FlashKDA, FlashInfer,
training, backward, and NVIDIA-only paths are intentionally excluded.

## SGLang Triton KDA

### Dispatcher and wrapper

- Dispatcher: `third_party/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py:46-199`, `KDAKernelDispatcher`.
- With `LinearAttnKernelBackend("triton")`, the dispatcher selects
  `TritonKDAKernel` for decode (`:72-73`) and extend/prefill (`:135-136`).
- Runtime callers are `KDAAttnBackend.forward_decode` (`:544`, calling
  `packed_decode` or `decode`) and `KDAAttnBackend.forward_extend` (`:794`,
  calling `extend`).
- Wrapper: `third_party/sglang/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py:23`.

### Prefill / extend path

```text
KDAAttnBackend.forward_extend
  -> KDAKernelDispatcher.extend
  -> TritonKDAKernel.extend (:219-254)
  -> sglang.kernels.ops.attention.fla.kda.chunk_kda (:1202)
  -> chunk_kda_fwd (:1083)
  -> chunk KDA Triton kernels
```

Core kernels in `third_party/sglang/python/sglang/kernels/ops/attention/fla/kda.py`:

- `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` (`:223`)
- `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra` (`:332`)
- `_recompute_w_u_fwd_kernel` (`:520`)
- `chunk_gla_fwd_kernel_o` (`:761`)
- `kda_gate_chunk_cumsum_vector_kernel` (`:934`)
- The recurrent chunk-state kernel is imported from
  `fla/chunk_delta_h.py` as `chunk_gated_delta_rule_fwd_h`.

### Decode path

Packed one-token decode:

```text
KDAAttnBackend.forward_decode
  -> KDAKernelDispatcher.packed_decode
  -> TritonKDAKernel.packed_decode (:33-123)
  -> fla.fused_recurrent.fused_recurrent_kda_packed_decode (:508)
  -> fused_recurrent_kda_packed_decode_kernel (:406-507)
```

General/varlen decode:

```text
KDAAttnBackend.forward_decode
  -> KDAKernelDispatcher.decode
  -> TritonKDAKernel.decode (:125-157)
  -> fla.fused_sigmoid_gating_recurrent.fused_sigmoid_gating_delta_rule_update (:349)
  -> fused_sigmoid_gating_delta_rule_update_kernel (:10)
```

Both are Triton `@triton.jit` forward kernels with `is_kda=True`; neither is
FlashKDA. The optional ReplaySSM branch in `kda_triton.py` is excluded from
the baseline smoke test.

## vLLM AMD Kimi-K3 KDA

### AMD dispatcher / model wrapper

- AMD model entry: `third_party/vllm/vllm/models/kimi_k3/amd/kda.py:62`,
  `KimiK3DeltaAttention`.
- AMD attention metadata backend:
  `third_party/vllm/vllm/models/kimi_k3/amd/kda_metadata.py:102`,
  `KimiK3ROCmKDABackend`.
- The layer imports AMD KDA operators from
  `vllm.models.kimi_k3.amd.ops.third_party.kda` (`kda.py:50-53`).

### Prefill / extend path (ordinary Triton)

```text
KimiK3DeltaAttention._forward (:312)
  -> chunk_kda_prefill (:557)
  -> kda_prefill.chunk_kda_prefill (:27-179)
  -> chunk_kda_with_fused_gate (:786 in third_party/kda/chunk.py)
  -> Triton chunk kernels
```

For this baseline, `kda_prefill_backend=triton` or
`use_fused_chunk=False` must be used. The `fused` branch in
`amd/ops/kda_chunk.py` is an AMD custom HIP/C++ path and is not the Triton
reference.

Core AMD-vendored Triton files:

- `third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py`
  - `recompute_w_u_fwd_kernel` (`:53`)
  - `chunk_gla_fwd_kernel_o` (`:256`)
  - `kda_gate_chunk_cumsum_vector_kernel` (`:419`)
  - wrappers `chunk_kda_fwd` (`:662`), `chunk_kda_with_fused_gate` (`:703`,
    `:786`)
- `third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py`
  - `chunk_kda_fwd_kernel_inter_solve_fused` (`:40`)
  - `chunk_kda_fwd_kernel_intra_sub_chunk` (`:458`)
- The chunk state/output recurrence is imported from vLLM's vendored FLA
  `vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py`.

### Decode / recurrent path (ordinary Triton)

```text
KimiK3DeltaAttention._forward
  -> fused_recurrent_kda (:448 or :521 for varlen decode)
  -> amd/ops/third_party/kda/fused_recurrent.py:fused_recurrent_kda (:388)
  -> fused_recurrent_kda_fwd (:291)
  -> fused_recurrent_kda_fwd_kernel (:133)
```

Pure one-token packed decode:

```text
KimiK3DeltaAttention._forward (:611)
  -> fused_recurrent_kda_packed_decode (:537)
  -> fused_recurrent_kda_packed_decode_kernel (:446)
```

The gate/beta activation kernel is `_kda_gate_beta_fwd_kernel` (`:23`) when
the wrapper materializes gate/beta. The recurrent file is genuine
`@triton.jit` code and uses the AMD vLLM KDA path, not
`vllm/models/kimi_k3/nvidia`.

### Important backend distinction

The AMD layer also contains `ops.fused_kda_decode` (`amd/kda.py:362`) and the
`fused_kda_chunk` path. Those are AMD custom HIP/C++ kernels, not FlashKDA,
but they are deliberately not used for the ordinary Triton reference. The
smoke script will call the AMD-vendored Triton wrappers directly so the
executed path is unambiguous.

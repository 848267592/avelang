# Stage 3 KDA smoke results

Date: 2026-09-10 (UTC)

The tests use random tensors only (`B=1`, `T=64`, `H=1`, `K=V=128`), run in
`torch.inference_mode()`, and do not load model weights or execute backward.
Neither script imports FlashKDA.

## SGLang

- Container: `ljd_sglang_kda` (running, `unless-stopped`)
- Device: `AMD Instinct MI300X`, capability `(9, 4)`; container ROCm reports
  `gfx942`.
- Prefill: PASS, output `(1, 64, 1, 128)`, finite.
- Decode: PASS, packed output `(1, 1, 1, 128)`, finite.
- Prefill path: `TritonKDAKernel.extend` ->
  `sglang/kernels/ops/attention/fla/kda.py:chunk_kda`.
- Prefill kernels: `chunk_kda_scaled_dot_kkt_fwd_kernel_*`,
  `recompute_w_u_fwd_kernel`, `chunk_gla_fwd_kernel_o`, and
  `kda_gate_chunk_cumsum_vector_kernel` in
  `third_party/sglang/python/sglang/kernels/ops/attention/fla/`.
- Decode path/kernel:
  `third_party/sglang/python/sglang/kernels/ops/attention/fla/fused_recurrent.py:`
  `fused_recurrent_kda_packed_decode_kernel`.
- Log: `smoke_sglang_kda.log`.

## vLLM AMD Kimi K3

- Container: `ljd_vllm_kda` (running, `unless-stopped`)
- AMD path confirmed; NVIDIA path and FlashKDA were not used.
- Device capability `(9, 4)`; container ROCm reports `gfx942` and MI300X SKU.
- Prefill: PASS, output `(1, 64, 1, 128)`, final state `(1, 1, 128, 128)`,
  all finite.
- Decode: PASS, packed output `(1, 1, 1, 128)`, finite.
- Prefill path/kernel:
  `third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py:`
  `chunk_kda_with_fused_gate` and its `@triton.jit` chunk kernels.
- Decode path/kernel:
  `third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py:`
  `fused_recurrent_kda_packed_decode_kernel`.
- Log: `smoke_vllm_kda.log`.

## Scope and known issues

- This is an operator/kernel smoke test only; no correctness or performance
  benchmark was started.
- When the local vLLM checkout is placed first on `PYTHONPATH`, the image emits
  warnings that optional compiled `vllm._C*` modules and the commit-hash module
  are unavailable.  The direct AMD Triton KDA modules import and execute
  successfully, so this does not block the requested operator smoke test.
- vLLM's `torch.cuda.get_device_name()` string is empty in this image, but the
  HIP capability `(9,4)` and `rocm-smi` MI300X/gfx942 output are present.

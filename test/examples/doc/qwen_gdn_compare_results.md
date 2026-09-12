# Qwen GDN Baseline Compare Results

- Date: 2026-06-02T08:28:20+00:00
- Avelang git revision: 1e953d6
- Qwen official FlashQLA revision: 0.1.0 (package version; git revision unavailable)
- PyTorch: 2.10.0+rocm7.2.2.git40d237bf
- ROCm/HIP: 7.2.53211
- GPU: AMD Instinct MI210
- HIP_VISIBLE_DEVICES: not set
- warmup: 5
- repeat: 20
- seed: 2027

## Method

- All measured implementations use the same tensors for a given shape and dtype policy.
- Timing uses `torch.cuda.Event(enable_timing=True)` and `torch.cuda.synchronize()`.
- First-call Avelang JIT and `torch.compile` compile cost are excluded by correctness/warmup calls before timing.
- FP32 correctness is checked against `qwen_gdn_forward_ref` on FP32 inputs.
- BF16 v6 correctness is checked against `qwen_gdn_forward_ref(q_bf16.float(), k_bf16.float(), v_bf16.float())`.
- Native BF16 PyTorch eager/compile reference latency is marked N/A when direct BF16 execution fails.

## Qwen Official Optimized Forward

- exists: True
- runnable_on_mi210: False
- path: `/workspace/workspace/qwen_GDN/FlashQLA`
- reason: Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard.

The local official implementation is FlashQLA. Its forward entry is `chunk_gated_delta_rule_fwd`; the scanned source has an import-time `sm90 only` guard when running outside NVIDIA Hopper, so it is not a runnable AMD MI210 baseline in this environment.

## Results

| implementation | GPU | dtype_policy | B | T | Hk | Hv | K | V | chunk_size | initial_state | latency_ms | speedup_vs_pytorch_eager | speedup_vs_torch_compile | correctness_status | notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | 1.927634 | 1.000000 | N/A | reference | PyTorch eager qwen_gdn_forward_ref. |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | 0.669243 | 2.880318 | 1.000000 | pass max_abs=1.192e-07 max_rel=6.381e-06 worst=final_state | torch.compile wrapper around qwen_gdn_forward_ref. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | fp32 | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | 0.313754 | 6.143784 | 2.133022 | pass max_abs=1.192e-07 max_rel=9.822e-06 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | fp32 | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | N/A | N/A | N/A | reference uses q/k/v .float() for BF16 correctness | Native BF16 qwen_gdn_forward_ref is not a fair performance baseline: RuntimeError: expected scalar type Float but found BFloat16 |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | N/A | N/A | N/A | skipped | Skipped: BF16 PyTorch reference is not a fair native performance baseline. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | 0.318306 | N/A | N/A | pass max_abs=1.341e-07 max_rel=4.208e-05 worst=final_state | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 16 | 1 | 2 | 4 | 4 | 4 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | 2.905678 | 1.000000 | N/A | reference | PyTorch eager qwen_gdn_forward_ref. |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | 1.113533 | 2.609421 | 1.000000 | pass max_abs=1.788e-07 max_rel=6.197e-05 worst=chunk_states | torch.compile wrapper around qwen_gdn_forward_ref. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | fp32 | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | 0.334986 | 8.674039 | 3.324124 | pass max_abs=3.576e-07 max_rel=3.423e-04 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | fp32 | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | N/A | N/A | N/A | reference uses q/k/v .float() for BF16 correctness | Native BF16 qwen_gdn_forward_ref is not a fair performance baseline: RuntimeError: expected scalar type Float but found BFloat16 |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | N/A | N/A | N/A | skipped | Skipped: BF16 PyTorch reference is not a fair native performance baseline. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | 0.328410 | N/A | N/A | pass max_abs=2.831e-07 max_rel=1.674e-03 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 64 | 2 | 4 | 8 | 8 | 8 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | 4.349125 | 1.000000 | N/A | reference | PyTorch eager qwen_gdn_forward_ref. |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | 1.838809 | 2.365186 | 1.000000 | pass max_abs=2.384e-07 max_rel=1.765e-03 worst=chunk_states | torch.compile wrapper around qwen_gdn_forward_ref. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | fp32 | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | 0.332642 | 13.074508 | 5.527899 | pass max_abs=3.576e-07 max_rel=3.015e-03 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | fp32 | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | N/A | N/A | N/A | reference uses q/k/v .float() for BF16 correctness | Native BF16 qwen_gdn_forward_ref is not a fair performance baseline: RuntimeError: expected scalar type Float but found BFloat16 |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | N/A | N/A | N/A | skipped | Skipped: BF16 PyTorch reference is not a fair native performance baseline. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | 0.336826 | N/A | N/A | pass max_abs=3.576e-07 max_rel=3.673e-02 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 128 | 2 | 4 | 16 | 64 | 8 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | 4.934424 | 1.000000 | N/A | reference | PyTorch eager qwen_gdn_forward_ref. |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | fp32 | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | 2.037682 | 2.421587 | 1.000000 | pass max_abs=3.576e-07 max_rel=1.395e+00 worst=chunk_states | torch.compile wrapper around qwen_gdn_forward_ref. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | fp32 | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | 0.969101 | 5.091756 | 2.102653 | pass max_abs=3.576e-07 max_rel=1.047e+00 worst=chunk_states | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | fp32 | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |
| pytorch_eager_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | N/A | N/A | N/A | reference uses q/k/v .float() for BF16 correctness | Native BF16 qwen_gdn_forward_ref is not a fair performance baseline: RuntimeError: expected scalar type Float but found BFloat16 |
| torch_compile_qwen_gdn_forward_ref | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | N/A | N/A | N/A | skipped | Skipped: BF16 PyTorch reference is not a fair native performance baseline. |
| avelang_dsl_standalone_v6 | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | 0.993589 | N/A | N/A | pass max_abs=3.576e-07 max_rel=1.257e-02 worst=final_state | Standalone v6 from qwen_gdn_chunked_avelang_v6_standalone.py. |
| qwen_official_flashqla_forward | AMD Instinct MI210 | bf16_qkv_fp32_ref | 1 | 256 | 4 | 8 | 32 | 64 | 16 | none | N/A | N/A | N/A | skipped | Local FlashQLA forward exists but has an import-time NVIDIA Hopper sm90-only guard. path=/workspace/workspace/qwen_GDN/FlashQLA rev=0.1.0 (package version; git revision unavailable) |

## v6 Speedup Summary

- small/fp32: vs eager 6.143784x, vs torch.compile 2.133022x
- medium/fp32: vs eager 8.674039x, vs torch.compile 3.324124x
- large_v/fp32: vs eager 13.074508x, vs torch.compile 5.527899x
- larger_tv/fp32: vs eager 5.091756x, vs torch.compile 2.102653x

## Existing rocprofv3 v6 Bottleneck Data

These rows come from the previous rocprofv3 run on 2026-06-02 in the same ROCm container on AMD MI210. They are kernel-time percentages, not end-to-end wall-clock percentages.

| shape | dtype | kernel | avg_us | kernel_time_pct |
|---|---|---|---:|---:|
| larger_debug | bf16 | `_qwen_gdn_chunk_gdr_bf16_kernel_v6_standalone` | 111.438 | 62.54% |
| larger_debug | bf16 | `_qwen_gdn_chunk_o_bf16_kernel_v6_standalone` | 34.087 | 19.13% |
| larger_debug | bf16 | `_qwen_gdn_w_u_bf16_kernel_v6_standalone` | 13.507 | 7.58% |
| larger_debug | fp32 | `_qwen_gdn_chunk_gdr_fp32_kernel_v6_standalone` | 94.837 | 60.21% |
| larger_debug | fp32 | `_qwen_gdn_chunk_o_fp32_opt_kernel_v6_standalone` | 33.240 | 21.10% |
| larger_debug | fp32 | `_qwen_gdn_w_u_fp32_opt_kernel_v6_standalone` | 10.909 | 6.93% |

## Fairness Conclusion

- The fairest runnable baseline on AMD MI210 is PyTorch eager `qwen_gdn_forward_ref` FP32 vs Avelang standalone v6 FP32, plus `torch.compile` FP32 when it succeeds.
- BF16 v6 can be correctness-checked against an FP32-cast reference, but native BF16 PyTorch reference latency is not a fair baseline if direct BF16 eager fails.
- Qwen official FlashQLA exists locally, but is Hopper sm90-only by source guard and is not a runnable MI210 baseline.
- For the next kernel optimization, rocprofv3 points first at `chunk_gdr`, then `chunk_o`.

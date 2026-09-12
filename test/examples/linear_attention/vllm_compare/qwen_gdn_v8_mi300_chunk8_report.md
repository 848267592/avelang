# Qwen GDN v8 MI300 TP4 Benchmark Report

Date: 2026-06-12
Machine root: `/home/jiandongliu`
GPU: AMD Instinct MI300X / `gfx942`
Runtime: torch `2.10.0+rocm7.2.2.git40d237bf`, HIP `7.2.53211`

Note: this filename still contains `chunk8` for historical reasons. The corrected TP4 target sweep below found `chunk_size=4`.

## Target Shape

The previous report used the legacy/dev toy shape `K=64,V=64` as the main conclusion. That was wrong for the current target.

Primary target is vLLM Qwen3Next TP4 per-rank linear attention:

```text
B=1
T in {64, 512, 1024, 2048}
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
initial_state=[B,Hv,V,K]
```

Parameter classes:

- Model structure parameters: `Hk/Hv/K/V`. Primary benchmark fixes them to `4/8/128/128`.
- Workload parameters: `B/T`. This run fixes `B=1` and tests `T=64,512,1024,2048`.
- Avelang tuning parameters: `chunk_size`, `block_v`, `block_k`.

Legacy/dev shape:

```text
Hk=4,Hv=8,K=64,V=64
```

This legacy shape remains useful for quick development checks only. It is not used as the primary conclusion.

Local shape sources checked:

- `/home/jiandongliu/project/vllm_stageb_snapshot/vllm/transformers_utils/configs/qwen3_next.py`
  - `linear_num_key_heads=16`
  - `linear_num_value_heads=32`
  - `linear_key_head_dim=128`
  - `linear_value_head_dim=128`
- `/home/jiandongliu/project/vllm_stageb_snapshot/vllm/model_executor/models/qwen3_next.py`
  - Qwen3Next splits linear-attention heads by tensor parallel rank.
  - TP4 per-rank target is therefore `Hk=4,Hv=8,K=128,V=128`.
- `/home/jiandongliu/workspace/qwen_GDN/FlashQLA/benchmark/bench_gated_delta_rule.py`
  - Uses `HEAD_DIM=128`.

## Code Changes

Changed files:

- `qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py`
  - Changed v8 full/chunk_gdr default `chunk_size` to `4`.
  - Changed default `block_v` to `4`.
  - Kept `block_k=None` API behavior and changed auto-selection so `head_dim_k >= 64` selects `block_k=64`.
  - Extended validation to allow `block_k=64/128` and workgroup products up to `1024` for MI300 probing.
- `bench_qwen_gdn_v8_parallel.py`
  - Default benchmark shape is now `B=1,T=512,Hk=4,Hv=8,K=128,V=128`.
  - Default tuning is now `chunk=4, block_v=4, block_k=64`.
  - `--sweep-vk` range includes larger MI300 probes.
- `bench_stage1_vllm_chunk_gdn.py`
  - Primary built-in cases now use `K=128,V=128`.
  - The old `K=64,V=64` case is explicitly marked as legacy/dev.
  - Paths are resolved from the current project instead of old `/home/jiandongliu`.

Validation after the changes:

```text
docker exec ac739c57a0bf ... pytest -q test_qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py
5 passed in 20.64s
```

Default-parameter smoke benchmark confirms the CLI now uses the corrected shape/tuning:

```text
shape,B=1,T=512,Hk=4,Hv=8,K=128,V=128,chunk=4,initial_state=True
stage_median_ms,v8_vk_chunk_gdr,0.962549984
full_median_ms,avelang_v8_vk,1.98134398
full_median_ms,vllm,0.316991001
```

## T=512 Chunk Sweep

Shape: `B=1,T=512,Hk=4,Hv=8,K=128,V=128,BF16`

Fixed initial tuning for chunk sweep: `block_v=4, block_k=64`.

| chunk | full ms | vLLM ms | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o | output err | final err | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1.978258 | 0.319515 | 0.027000 | 0.039940 | 0.027962 | 0.108000 | 0.963912 | 0.904984 | 4.825e-4 | 4.927e-3 | ok |
| 8 | 2.072838 | 0.305734 | 0.031407 | 0.049674 | 0.027641 | 0.165926 | 0.934027 | 0.970882 | 4.825e-4 | 4.927e-3 | ok |
| 16 | 2.384021 | 0.312224 | 0.027882 | 0.060090 | 0.044466 | 0.283341 | 0.912475 | 1.164530 | 4.825e-4 | 4.927e-3 | ok |
| 32 | 4.544229 | 0.313786 | 0.027521 | 0.088531 | 1.404326 | 0.517609 | 0.910913 | 1.669440 | 4.825e-4 | 4.927e-3 | ok |
| 64 | 17.756212 | 0.311623 | 0.031046 | 0.145737 | 12.158018 | 0.991753 | 0.906226 | 3.620858 | 4.825e-4 | 4.927e-3 | ok |

Best full-forward chunk at T=512:

```text
chunk_size=4
```

Larger chunks slightly reduce `chunk_gdr`, but `solve`, `w_u`, and `chunk_o` dominate and full forward becomes worse.

## T=512 block_v/block_k Sweep

Shape: `B=1,T=512,Hk=4,Hv=8,K=128,V=128,BF16`

Fixed chunk: `chunk_size=4`.

Requested candidate grid:

```text
block_v in {4, 8, 16}
block_k in {16, 32, 64}
```

| block_v | block_k | work-items | full ms | vLLM ms | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o | output err | final err | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 16 | 64 | 2.684547 | 0.321438 | 0.028162 | 0.039659 | 0.027721 | 0.106839 | 1.645243 | 0.923571 | 4.825e-4 | 4.927e-3 | ok |
| 4 | 32 | 128 | 2.277102 | 0.312624 | 0.027762 | 0.039698 | 0.027521 | 0.106398 | 1.258710 | 0.913877 | 4.825e-4 | 4.927e-3 | ok |
| 4 | 64 | 256 | 1.979140 | 0.310702 | 0.027841 | 0.040661 | 0.027641 | 0.107720 | 0.964873 | 0.906586 | 4.825e-4 | 4.927e-3 | ok |
| 8 | 16 | 128 | 2.637838 | 0.314147 | 0.027761 | 0.039058 | 0.027321 | 0.106719 | 1.613717 | 0.912595 | 4.825e-4 | 4.927e-3 | ok |
| 8 | 32 | 256 | 2.325574 | 0.316110 | 0.027641 | 0.040140 | 0.028923 | 0.106999 | 1.310226 | 0.901339 | 4.825e-4 | 4.927e-3 | ok |
| 8 | 64 | 512 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | HIP launch failure |
| 16 | 16 | 256 | 2.722044 | 0.311382 | 0.027921 | 0.039619 | 0.027601 | 0.107400 | 1.706054 | 0.904383 | 4.825e-4 | 4.927e-3 | ok |
| 16 | 32 | 512 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | HIP launch failure |
| 16 | 64 | 1024 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | HIP launch failure |

Best T=512 configuration:

```text
chunk_size=4
block_v=4
block_k=64
```

This is the best measured full-forward result and the best measured `chunk_gdr` result among successful requested candidates.

## Validation Across T

Fixed Avelang tuning from T=512:

```text
chunk_size=4
block_v=4
block_k=64
```

| T | best chunk | best block_v | best block_k | Avelang ms | vLLM ms | vLLM/Avelang ratio | output err | final err | bottleneck stage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 4 | 4 | 64 | 0.484760 | 0.316430 | 0.653 | 4.157e-4 | 3.705e-3 | chunk_o |
| 512 | 4 | 4 | 64 | 1.979140 | 0.310702 | 0.157 | 4.825e-4 | 4.927e-3 | chunk_gdr |
| 1024 | 4 | 4 | 64 | 3.641689 | 0.310181 | 0.085 | 5.314e-4 | 3.768e-3 | chunk_gdr |
| 2048 | 4 | 4 | 64 | 7.158192 | 0.365142 | 0.051 | 6.897e-4 | 3.442e-3 | chunk_gdr |

Interpretation of ratio:

```text
vLLM/Avelang < 1 means Avelang is slower.
```

Equivalent slowdowns:

```text
T=64:   Avelang is about 1.53x slower than vLLM
T=512:  Avelang is about 6.37x slower than vLLM
T=1024: Avelang is about 11.74x slower than vLLM
T=2048: Avelang is about 19.60x slower than vLLM
```

Internal correctness against the v6 Avelang oracle stayed at fp32-noise level:

```text
full_error_vs_v6 output <= 2.24e-08
full_error_vs_v6 final_state <= 1.79e-07
```

The larger error versus vLLM is consistent with backend numeric-order differences in BF16/FP32 accumulation and is stable across candidates.

## Failure Logs

The current v8 validation allows probing workgroup products up to `1024`, but this kernel/runtime path fails for the requested candidates with `block_v * block_k > 256`.

Failed log files:

```text
/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_v8_tp4_mi300_logs/block_chunk4_bv8_bk64.log
/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_v8_tp4_mi300_logs/block_chunk4_bv16_bk32.log
/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_v8_tp4_mi300_logs/block_chunk4_bv16_bk64.log
```

Failure traceback for `block_v=8, block_k=64`:

```text
STDERR:
Traceback (most recent call last):
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 515, in <module>
    main()
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 242, in main
    h8_vk, vn8_vk, final8_vk = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py", line 531, in qwen_gdn_chunk_gdr_avelang_v8_vllm_layout
    _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk[lambda: ((grid_size, 1, 1), (block_v, block_k, 1))](
  File "/opt/avelang/python/avelang/runtime/jit.py", line 509, in <lambda>
    return lambda *args, **kwargs: self.run(dims, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/avelang/python/avelang/runtime/jit.py", line 821, in run
    kernel.run(
  File "/opt/avelang/python/avelang/backends/amdgpu/driver.py", line 390, in __call__
    return self.launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: ave-lang Error [HIP]: unspecified launch failure
```

Failure traceback for `block_v=16, block_k=32`:

```text
STDERR:
Traceback (most recent call last):
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 515, in <module>
    main()
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 242, in main
    h8_vk, vn8_vk, final8_vk = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py", line 531, in qwen_gdn_chunk_gdr_avelang_v8_vllm_layout
    _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk[lambda: ((grid_size, 1, 1), (block_v, block_k, 1))](
  File "/opt/avelang/python/avelang/runtime/jit.py", line 509, in <lambda>
    return lambda *args, **kwargs: self.run(dims, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/avelang/python/avelang/runtime/jit.py", line 821, in run
    kernel.run(
  File "/opt/avelang/python/avelang/backends/amdgpu/driver.py", line 390, in __call__
    return self.launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: ave-lang Error [HIP]: unspecified launch failure
```

Failure traceback for `block_v=16, block_k=64`:

```text
STDERR:
Traceback (most recent call last):
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 515, in <module>
    main()
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py", line 242, in main
    h8_vk, vn8_vk, final8_vk = qwen_gdn_chunk_gdr_avelang_v8_vllm_layout(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py", line 531, in qwen_gdn_chunk_gdr_avelang_v8_vllm_layout
    _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk[lambda: ((grid_size, 1, 1), (block_v, block_k, 1))](
  File "/opt/avelang/python/avelang/runtime/jit.py", line 509, in <lambda>
    return lambda *args, **kwargs: self.run(dims, *args, **kwargs)
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/avelang/python/avelang/runtime/jit.py", line 821, in run
    kernel.run(
  File "/opt/avelang/python/avelang/backends/amdgpu/driver.py", line 390, in __call__
    return self.launch(gridX, gridY, gridZ, blockX, blockY, blockZ, stream, function, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: ave-lang Error [HIP]: unspecified launch failure
```

Conclusion from failures:

- MI300 hardware can support larger workgroups, but this generated Avelang v8 vk kernel path is not currently stable above 256 work-items.
- Increasing workgroup size alone is not a valid next optimization step until the runtime/compiler launch failure is understood.

## Current Bottleneck

At the corrected TP4 `K=128,V=128` shape, v8 does improve `chunk_gdr` over v6/v7:

```text
T=512, chunk=4, block_v=4, block_k=64
v6 chunk_gdr: 1.776919 ms
v7 chunk_gdr: 1.841816 ms
v8 chunk_gdr: 0.963871 ms
```

That is about:

```text
1.84x faster than v6 chunk_gdr
1.91x faster than v7 chunk_gdr
```

But full forward is still much slower than vLLM. The main bottleneck is now shared between:

```text
chunk_gdr
chunk_o
```

For T=512:

```text
chunk_gdr: 0.964873 ms
chunk_o:   0.906586 ms
full:      1.979140 ms
```

For T=2048:

```text
chunk_gdr: 3.659275 ms
chunk_o:   3.173513 ms
full:      7.158192 ms
```

## Bottom Line

Corrected primary target:

```text
vLLM Qwen3Next TP4 per-rank
B=1,Hk=4,Hv=8,K=128,V=128,BF16
```

Best measured Avelang v8 tuning:

```text
chunk_size=4
block_v=4
block_k=64
```

Avelang v8 is correct versus the v6 oracle and close to vLLM numerically, but it is still slower than vLLM on the real target shape:

```text
T=512:  Avelang 1.979 ms vs vLLM 0.311 ms
T=2048: Avelang 7.158 ms vs vLLM 0.365 ms
```

Next optimization direction:

1. Do not treat larger workgroup products as automatically better; `>256` currently fails in this kernel path.
2. Continue optimizing `chunk_gdr`, but also start targeting `chunk_o`, because it is now comparable to `chunk_gdr`.
3. The likely next meaningful step is reducing memory traffic and kernel boundaries around `w/u -> chunk_gdr -> chunk_o`, not another scalar precompute or blind block-size increase.

## Minimal Repro Script

Added a focused repro:

```text
/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_v8_vk_launch_failure_mi300.py
```

Safe config:

```bash
docker exec -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare \
  -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf \
  python repro_v8_vk_launch_failure_mi300.py --block-v 4 --block-k 64
```

Observed safe output:

```text
device=AMD Instinct MI300X
gcnArchName=gfx942:sramecc+:xnack-
max_threads_per_block=1024
shape=B=1,T=512,Hk=4,Hv=8,K=128,V=128,chunk=4
vk_config=block_v=4,block_k=64,work_items=256
launch_status=ok
sanity=0.886256695
```

Failing config:

```bash
docker exec -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare \
  -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf \
  python repro_v8_vk_launch_failure_mi300.py --block-v 8 --block-k 64
```

Observed failure:

```text
device=AMD Instinct MI300X
gcnArchName=gfx942:sramecc+:xnack-
max_threads_per_block=1024
shape=B=1,T=512,Hk=4,Hv=8,K=128,V=128,chunk=4
vk_config=block_v=8,block_k=64,work_items=512
launching v8 vk chunk_gdr ...
RuntimeError: ave-lang Error [HIP]: unspecified launch failure
```

This repro isolates the issue to the v8 vk chunk_gdr launch with a 2D workgroup `(block_v, block_k, 1)`. The same shape and same data preparation succeed at 256 work-items and fail at 512 work-items.

## vLLM Benchmark Sanity Check

A follow-up check was run because vLLM latency looked very flat from `T=64` to `T=2048`.

Checked items:

1. Timing synchronization:
   - `bench_stage1_vllm_chunk_gdn.py::bench_one` does warmup, then `torch.cuda.synchronize()` before timing.
   - Each measured iteration records CUDA/HIP events, calls the function, records end event, then calls `torch.cuda.synchronize()` before reading elapsed time.
2. Real vLLM linear attention path:
   - `call_vllm` calls `vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule`.
   - Runtime monkey-patching confirmed calls to:
     - `chunk_local_cumsum`
     - `chunk_scaled_dot_kkt_fwd`
     - `solve_tril`
     - `recompute_w_u_fwd`
     - `chunk_gated_delta_rule_fwd_h`
     - `chunk_fwd_o`
3. Output/final_state materialization:
   - `output_final_state=True` is passed.
   - The diagnostic read `o.float().abs().sum()` and `final_state.float().abs().sum()` successfully.
4. T=2048 shape:

```text
input_shapes q=(1, 2048, 4, 128) k=(1, 2048, 4, 128) v=(1, 2048, 8, 128)
g=(1, 2048, 8) beta=(1, 2048, 8) initial_state=(1, 8, 128, 128)
output_shapes o=(1, 2048, 8, 128) final_state=(1, 8, 128, 128)
chunk_fwd_o h_shape=(1, 32, 8, 128, 128)
```

5. Autotune/cache behavior:
   - vLLM uses Triton autotune and the benchmark warms up before measuring.
   - Several kernels use `do_not_specialize=["T"]`; autotune keys include structure/chunk parameters such as `H/K/V/BT`, not the runtime `T` value.
   - On AMD, `use_cuda_graph` is false in this code path.

Diagnostic timing:

```text
timing_plain T=64   median=0.312184 ms
timing_plain T=512  median=0.313025 ms
timing_plain T=2048 median=0.358373 ms

timing_with_checksum T=64   median=0.389178 ms
timing_with_checksum T=512  median=0.400635 ms
timing_with_checksum T=2048 median=0.419263 ms
```

Conclusion: the vLLM benchmark is not obviously empty or using the wrong shape. It is genuinely calling the vLLM FLA linear-attention kernels on `[1,T,H,D]` inputs. The flat trend is still surprising, but the current evidence points more toward highly optimized/fixed-overhead Triton kernels plus warm autotune/cache behavior than a benchmark no-op.

## Fair Timing Cross-check: vLLM vs Avelang v8

A follow-up check applied the same plain timing and checksum/materialization timing to both vLLM and Avelang v8.

Avelang v8 tuning:

```text
chunk_size=4, block_v=4, block_k=64
```

Both backends were called with the same tensors and same shapes. For each T, the diagnostic printed output/final_state shapes, max errors, checksums, and then timed both backends through the same `bench_one` event/synchronize wrapper.

Plain timing:

| T | vLLM median ms | Avelang v8 median ms | vLLM/Avelang |
|---:|---:|---:|---:|
| 64 | 0.303251 | 0.480874 | 0.631 |
| 512 | 0.304172 | 1.972770 | 0.154 |
| 2048 | 0.397391 | 7.154908 | 0.056 |

Checksum/materialization timing:

| T | vLLM median ms | Avelang v8 median ms | vLLM/Avelang |
|---:|---:|---:|---:|
| 64 | 0.387936 | 0.519812 | 0.746 |
| 512 | 0.400195 | 2.016596 | 0.198 |
| 2048 | 0.458080 | 7.189400 | 0.064 |

Representative shape/checksum check for T=2048:

```text
vLLM output=(1,2048,8,128), final_state=(1,8,128,128)
Avelang output=(1,2048,8,128), final_state=(1,8,128,128)
output max_abs_err=6.48e-4
final_state max_abs_err=4.51e-3
vLLM checksum output/state=21999.121/16118.549
Avelang checksum output/state=21999.227/16118.431
```

Conclusion: the two backends are being timed with the same event/synchronize method. The checksum test adds extra reduction work and is not the primary latency metric, but it confirms that both output and final_state are materialized and readable. The large performance gap remains after applying the same materialization sanity check to Avelang v8.

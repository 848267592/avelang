# Qwen GDN v9 chunk_o VK Optimization Report

Date: 2026-06-13
Machine root: `/home/jiandongliu`
GPU: AMD Instinct MI300X / `gfx942`
Runtime in Docker: torch `2.10.0+rocm7.2.2.git40d237bf`, HIP `7.2.53211`

## Primary Target

This report uses the corrected vLLM Qwen3Next TP4 per-rank shape:

```text
B=1
T in {64,512,1024,2048}
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
chunk_size=4
initial_state=[B,Hv,V,K]
```

The old `K=64,V=64` shape is legacy/dev only and is not used for the main conclusion.

## Code Added

New files:

- `qwen_gdn_chunked_avelang_v9_vllm_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v9_vllm_layout_fixed.py`
- `bench_qwen_gdn_v9_chunk_o_vk.py`
- `qwen_gdn_v9_chunk_o_vk_report.md`

v9 is self-contained for the optimized kernels. It does not import v8 kernels. It contains:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v9_vk   # copied/renamed from tested v8 vk chunk_gdr
_qwen_gdn_chunk_o_bf16_kernel_v9_vk     # new chunk_o VxK parallel kernel
```

The v6 chunk_o fallback is preserved through:

```text
use_parallel_chunk_o=False
```

## New chunk_o Mapping

New kernel:

```python
_qwen_gdn_chunk_o_bf16_kernel_v9_vk
```

Mapping:

```text
block_id    -> flattened (B, T, Hv, V block)
thread_id.x -> value_idx inside V block
thread_id.y -> K reduction lane
workgroup   -> (chunk_o_block_v, chunk_o_block_k, 1)
```

Grid:

```text
num_v_blocks = ceil(V / chunk_o_block_v)
grid_size = B * T * Hv * num_v_blocks
```

The kernel parallelizes both K reductions:

```text
inter = sum_k q_scaled[k] * exp(g[token]) * h[chunk, head, value, k]
dot_d = sum_k q_scaled[k] * k[source_token, k]
```

First version note: `dot_d` is currently repeated per V lane. This keeps correctness simple. A follow-up can share `dot_d` across V lanes because it does not depend on `value_idx`.

## Correctness

Command:

```bash
docker exec -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare \
  -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf \
  pytest -q test_qwen_gdn_chunked_avelang_v9_vllm_layout_fixed.py -s
```

Result:

```text
4 passed in 8.01s
```

Tested chunk_o stable candidates on the primary TP4 shape:

```text
block_v=4,  block_k=64
block_v=8,  block_k=32
block_v=16, block_k=16
```

Full forward with v9 chunk_o also matches the v6 chunk_o fallback path.

## T=512 Sweep

Fixed base config:

```text
B=1,T=512,Hk=4,Hv=8,K=128,V=128,chunk=4
chunk_gdr block_v=4, block_k=64
```

Sweep command:

```bash
docker exec -w /workspace/project/avelang/test/examples/linear_attention/vllm_compare \
  -e HIP_VISIBLE_DEVICES=0 ac739c57a0bf \
  python bench_qwen_gdn_v9_chunk_o_vk.py \
    --sweep-chunk-o --warmup 3 --repeat 10 --include-vllm \
    --log-dir qwen_gdn_v9_chunk_o_vk_logs
```

Stable candidates:

| chunk_o block_v | chunk_o block_k | workgroup | chunk_o v6 ms | chunk_o v9 ms | full prev ms | full v9 ms | output max abs vs v6 | final_state err vs vLLM | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 16 | 32 | 0.904944 | 0.249771 | 1.979901 | 1.305058 | 4.10e-8 | 3.705e-3 | ok |
| 2 | 32 | 64 | 0.919165 | 0.225334 | 1.989234 | 1.283786 | 3.73e-8 | 3.705e-3 | ok |
| 2 | 64 | 128 | 0.913437 | 0.349239 | 1.976976 | 1.411056 | 3.73e-8 | 3.705e-3 | ok |
| 4 | 16 | 64 | 0.913076 | 0.175301 | 1.979019 | 1.230508 | 4.10e-8 | 3.705e-3 | ok |
| 4 | 32 | 128 | 0.908229 | 0.241639 | 1.974853 | 1.296886 | 3.73e-8 | 3.705e-3 | ok |
| 4 | 64 | 256 | 0.908469 | 0.363099 | 1.982745 | 1.423434 | 3.73e-8 | 3.705e-3 | ok |
| 8 | 16 | 128 | 0.907307 | 0.197052 | 1.971448 | 1.249335 | 4.10e-8 | 3.705e-3 | ok |
| 8 | 32 | 256 | 0.908590 | 0.276251 | 1.974732 | 1.321042 | 3.73e-8 | 3.705e-3 | ok |
| 16 | 16 | 256 | 0.906106 | 0.240597 | 1.972249 | 1.293001 | 4.10e-8 | 3.705e-3 | ok |

Extra 512-work-item probes:

| chunk_o block_v | chunk_o block_k | workgroup | status |
|---:|---:|---:|---|
| 8 | 64 | 512 | launch_failed: `ave-lang Error [HIP]: unspecified launch failure` |
| 16 | 32 | 512 | launch_failed: `ave-lang Error [HIP]: unspecified launch failure` |

Failure traceback pattern:

```text
RuntimeError: ave-lang Error [HIP]: unspecified launch failure
  File ".../qwen_gdn_chunked_avelang_v9_vllm_layout_fixed.py", line 590,
    _qwen_gdn_chunk_o_bf16_kernel_v9_vk[lambda: ((grid_size, 1, 1), (...))](...)
```

This mirrors the earlier v8 chunk_gdr >256-work-item failure. MI300 reports `max_threads_per_block=1024`, but the current Avelang kernel/runtime path is only stable at <=256 work-items for these kernels.

## Best T=512 Config

Best measured stable config:

```text
chunk_o_block_v=4
chunk_o_block_k=16
workgroup_size=64
```

Repeat=50 confirmation:

```text
chunk_o_v6:     0.911434 ms
chunk_o_v9_vk:  0.173618 ms
chunk_o speedup: 5.25x

previous v8-style full with v6 chunk_o: 1.984827 ms
Avelang v9 full:                    1.235475 ms
full speedup:                       1.61x

vLLM full:                          0.308258 ms
vLLM/Avelang v9:                    0.250x
```

Errors:

```text
output max_abs vs v6 chunk_o: 4.10e-8
output max_rel vs v6 chunk_o: 6.52e-3
output max_abs vs vLLM:       5.16e-4
final_state max_abs vs vLLM:  3.70e-3
```

## Validation Across T

Fixed config:

```text
chunk_size=4
chunk_gdr block_v=4, block_k=64
chunk_o block_v=4, block_k=16
```

| T | chunk_o v6 ms | chunk_o v9 ms | previous full ms | v9 full ms | vLLM ms | v9/full speedup vs previous | vLLM/Avelang v9 | bottleneck |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.210753 | 0.066699 | 0.478871 | 0.317712 | 0.316189 | 1.51x | 0.995x | roughly tied: fixed overhead / chunk_gdr |
| 512 | 0.913316 | 0.174659 | 1.980261 | 1.234793 | 0.320356 | 1.60x | 0.259x | chunk_gdr |
| 1024 | 1.610392 | 0.300527 | 3.634678 | 2.306466 | 0.316791 | 1.58x | 0.137x | chunk_gdr |
| 2048 | 3.169868 | 0.634703 | 7.140405 | 4.575156 | 0.368147 | 1.56x | 0.080x | chunk_gdr |

Shape prints from the benchmark confirm the primary target, for example T=2048:

```text
q=(1,2048,4,128)
k=(1,2048,4,128)
v=(1,2048,8,128)
vn=(1,2048,8,128)
h=(1,512,8,128,128)
out=(1,2048,8,128)
```

## Performance Analysis

v9 successfully moves `chunk_o` out of the top bottleneck list:

```text
T=512:
chunk_gdr:   0.964392 ms
chunk_o_v6:  0.911434 ms
chunk_o_v9:  0.173618 ms
```

Before v9, `chunk_gdr` and `chunk_o` were comparable. After v9, `chunk_o` is much smaller and the bottleneck is again clearly `chunk_gdr`.

At longer T:

```text
T=2048:
chunk_gdr:   3.699535 ms
chunk_o_v9:  0.634703 ms
full:        4.575156 ms
```

So the next optimization target should be `chunk_gdr`, not another first-order chunk_o rewrite. The current v9 chunk_o still repeats q·k dot per V lane, but because chunk_o is no longer the dominant stage, sharing dot_d across V lanes is a secondary optimization unless it can be done very cheaply.

## Bottom Line

v9 result on the corrected Qwen3Next TP4 target:

```text
Best chunk_o config: block_v=4, block_k=16
chunk_o speedup vs v6 chunk_o: about 5.25x at T=512
full forward speedup vs previous v8-style path: about 1.6x across T=512..2048
```

Avelang v9 is nearly tied with vLLM at T=64, but still slower at real longer prefill lengths:

```text
T=512:  Avelang v9 1.235 ms vs vLLM 0.320 ms
T=1024: Avelang v9 2.306 ms vs vLLM 0.317 ms
T=2048: Avelang v9 4.575 ms vs vLLM 0.368 ms
```

Next bottleneck:

```text
chunk_gdr
```

The compiler/runtime >256-work-item launch failure still blocks larger workgroup experiments and should remain a separate compiler bug track.

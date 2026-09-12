# Qwen GDN v11 Integrated Update MFMA Report

## Target

Primary target remains the vLLM Qwen3Next TP4 per-rank operator shape:

```text
B=1
T in {512,1024,2048}
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
initial_state=[B,Hv,V,K]
chunk_size=16 for v11 MFMA paths
```

This is a single-GPU per-rank operator benchmark. It is not an end-to-end TP4 multi-GPU inference claim.

## Changed Files

```text
qwen_gdn_chunked_avelang_v11_mfma_layout_fixed.py
test_qwen_gdn_chunked_avelang_v11_mfma_update.py
bench_qwen_gdn_v11_mfma_update.py
qwen_gdn_v11_mfma_update_report.md
```

## Implementation

Added opt-in kernel:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update
```

The existing scalar fallback remains intact:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_12_clean_scalar_update
```

Wrapper parameter added and propagated through chunk-gdr/full/public wrappers:

```python
use_update_mfma: bool = False
```

When `use_update_mfma=True`, v11 selects the new integrated update-MFMA kernel. Otherwise it keeps the clean scalar-update kernel.

The new kernel keeps pred MFMA unchanged and replaces only the state update with:

```text
delta_H[BV,128] = v_decay[BT,BV]^T @ k_chunk[BT,128]
state = state * exp(g_last) + delta_H
```

The MFMA update follows the already-passing standalone staged-delta prototype. Each 16-wide K tile uses a fresh `al.full((4,), 0.0, al.f32)` accumulator.

## Correctness

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
HIP_LAUNCH_BLOCKING=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_qwen_gdn_chunked_avelang_v11_mfma_update.py -s
```

Result:

```text
16 passed in 26.93s
```

Coverage:

```text
T=16,32,64,512
with initial_state
without initial_state
chunk_gdr-only: h, vn, final_state
full forward: output, final_state
oracle: v11 clean scalar_update, use_update_mfma=False
```

Worst printed differences from the run:

```text
chunk_gdr h max_abs <= 1.14738941e-06
chunk_gdr vn max_abs <= 0.000192642212
chunk_gdr final_state max_abs <= 5.01051545e-07
full output max_abs <= 3.65730375e-06
full final_state max_abs <= 5.01051545e-07
```

## Benchmark

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python bench_qwen_gdn_v11_mfma_update.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v10 scalar chunk=4 ms | v11 scalar_update chunk=16 ms | v11 update_mfma chunk=16 ms | speedup vs v11 scalar | speedup vs v10 | vLLM / v11 update_mfma |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3035 | 1.3200 | 1.4986 | 0.9998 | 1.4989x | 1.3202x | 0.3035x |
| 1024 | 0.3069 | 2.5012 | 2.8129 | 1.8146 | 1.5501x | 1.3783x | 0.1691x |
| 2048 | 0.3597 | 4.9695 | 5.6472 | 3.6219 | 1.5592x | 1.3721x | 0.0993x |

Interpretation:

```text
v11 update MFMA is faster than both v11 scalar_update and v10 scalar chunk=4.
vLLM is still substantially faster, especially at long T.
```

Equivalent slowdown vs vLLM:

```text
T=512:  v11 update_mfma is ~3.29x slower than vLLM
T=1024: v11 update_mfma is ~5.91x slower than vLLM
T=2048: v11 update_mfma is ~10.07x slower than vLLM
```

### Correctness vs Scalar/VLLM During Benchmark

| T | update vs scalar output max_abs | update vs scalar final_state max_abs | update vs vLLM output max_abs | update vs vLLM final_state max_abs |
|---:|---:|---:|---:|---:|
| 512 | 6.45407e-07 | 1.19209e-07 | 6.62010e-04 | 5.12862e-03 |
| 1024 | 5.61085e-05 | 1.93715e-07 | 6.67073e-04 | 5.16295e-03 |
| 2048 | 5.61085e-05 | 2.77758e-05 | 6.67073e-04 | 4.81206e-03 |

The update-MFMA path matches the v11 scalar-update oracle closely. The larger vLLM delta is expected from the existing backend arithmetic/layout policy differences, not from the update-MFMA integration.

## Stage Breakdown

### T=512

| stage | v11 scalar_update ms | v11 update_mfma ms |
|---|---:|---:|
| cumsum | 0.02916 | 0.02812 |
| KKT | 0.06021 | 0.06057 |
| solve | 0.04406 | 0.04386 |
| w_u | 0.28446 | 0.28354 |
| chunk_gdr | 0.87626 | 0.37407 |
| chunk_o | 0.34828 | 0.33221 |

`chunk_gdr` speedup: 2.34x

### T=1024

| stage | v11 scalar_update ms | v11 update_mfma ms |
|---|---:|---:|
| cumsum | 0.03000 | 0.03021 |
| KKT | 0.08064 | 0.07944 |
| solve | 0.04639 | 0.04547 |
| w_u | 0.49946 | 0.49898 |
| chunk_gdr | 1.70537 | 0.71334 |
| chunk_o | 0.65734 | 0.66563 |

`chunk_gdr` speedup: 2.39x

### T=2048

| stage | v11 scalar_update ms | v11 update_mfma ms |
|---|---:|---:|
| cumsum | 0.03097 | 0.02888 |
| KKT | 0.11549 | 0.11537 |
| solve | 0.04731 | 0.04783 |
| w_u | 0.96163 | 0.96087 |
| chunk_gdr | 3.36067 | 1.36591 |
| chunk_o | 1.24124 | 1.23291 |

`chunk_gdr` speedup: 2.46x

## Current Bottleneck

After integrated update MFMA, `chunk_gdr` is much lower but not gone. At T=2048:

```text
w_u       ~0.961 ms
chunk_gdr ~1.366 ms
chunk_o   ~1.233 ms
```

So the bottleneck is now shared across:

```text
chunk_gdr + chunk_o + w_u
```

`chunk_gdr` is still the single largest stage, but only narrowly. The next optimization should not ignore `chunk_o` or `w_u` anymore.

## rocprof Counters

Command:

```bash
cd /workspace/project/avelang
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex state128_13_clean_mfma_update \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v11_update_mfma \
  -o v11_update_mfma_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v11_mfma_update.py --T 512 --warmup 2 --repeat 5
```

Output:

```text
test/examples/linear_attention/rocprof_outputs/qwen_profile_v11_update_mfma/v11_update_mfma_counters_counter_collection.csv
```

Filtered kernel:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma_state128_13_clean_mfma_update
```

Counter summary, median across 16 dispatches:

| field/counter | value |
|---|---:|
| Grid_Size | 4096 |
| Workgroup_Size | 64 |
| LDS_Block_Size | 20992 |
| Scratch_Size | 0 |
| VGPR_Count | 104 |
| Accum_VGPR_Count | 32 |
| SGPR_Count | 112 |
| OccupancyPercent | 0.6289 |
| SQ_INSTS_MFMA | 81920 |
| SQ_INSTS_VALU | 2611840 |
| SQ_INSTS_SALU | 326144 |
| SQ_INSTS_VMEM | 227328 |
| SQ_INSTS_LDS | 532480 |

Conclusion:

```text
MFMA is definitely being generated and executed: SQ_INSTS_MFMA = 81920.
No scratch spill: Scratch_Size = 0.
Occupancy is very low: ~0.63%.
VGPR/accum pressure is high: VGPR=104, Accum_VGPR=32.
VALU/SALU count is still high, so the kernel is not purely tensor-core dominated.
```

## Next Step

Now that update MFMA works, the most useful next step is not another scalar update tweak. The likely directions are:

```text
1. Reduce VGPR/accumulator pressure in v11 update_mfma.
2. Reduce pred/update shared-memory staging overhead.
3. Consider fusing or retiming chunk_o because chunk_o is now comparable to chunk_gdr at long T.
4. Revisit w_u if chunk_gdr/chunk_o are further reduced.
```

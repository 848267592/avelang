# Qwen GDN Small Fix Grouped-V4 Report

## Summary

Track A produced a clean archived v29 pred-only best variant:

- `qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best.py`

It keeps only correctness-preserving `baseline` and `grouped_v4_epilogue` public variants. The grouped-v4 epilogue improves v29 pred-only at long T, but it does not transfer naturally to v23/v24 production kernels because v23 already uses distributed VN/V-decay where each thread handles one element, not a repeated scalar contiguous-V epilogue loop.

Recommendation: keep grouped-v4 as the clean best v29 pred-only source variant, but do not create a production v24 local grouped-v4 copy.

## Original Pattern

The slow v29 pred-only epilogue stores `32 x 32` VN values per token tile:

```python
for rep_out in al.range(8):
    linear_o = tid + rep_out * WORKGROUP
    token_off_o = linear_o // BV
    local_v_o = linear_o - token_off_o * BV
    pred_o = pred_partial[0, token_off_o, local_v_o] + pred_partial[1, token_off_o, local_v_o]
    vn[0, token_idx_o, value_head_idx, global_v_o] = u[...] - pred_o
```

This is `128 threads * 8 reps = 1024` scalar output elements per tile.

## Modified Pattern

Grouped-v4 processes four contiguous V elements per thread iteration:

```python
for rep4 in al.range(2):
    linear4 = tid + rep4 * WORKGROUP
    token_off4 = linear4 // 8
    group_v4 = linear4 - token_off4 * 8
    local_v4 = group_v4 * 4
    base_offset = token_idx4 * (8 * 128) + value_head_idx * 128 + value_base + local_v4
```

It still emits scalar stores, but reduces loop/address overhead.

## Correctness

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_smallfix_grouped_v4.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

| T | variant | max_abs | mean_abs |
|---:|:---|---:|---:|
| 512 | original_v29 | `4.10895050e-02` | `6.08365797e-03` |
| 512 | baseline | `4.10895050e-02` | `6.08365797e-03` |
| 512 | grouped_v4_epilogue | `4.10895050e-02` | `6.08365797e-03` |
| 1024 | original_v29 | `3.61988842e-02` | `6.09028060e-03` |
| 1024 | baseline | `3.61988842e-02` | `6.09028060e-03` |
| 1024 | grouped_v4_epilogue | `3.61988842e-02` | `6.09028060e-03` |
| 2048 | original_v29 | `4.37299609e-02` | `6.07558433e-03` |
| 2048 | baseline | `4.37299609e-02` | `6.07558433e-03` |
| 2048 | grouped_v4_epilogue | `4.37299609e-02` | `6.07558433e-03` |

Errors are BF16-level and match original v29.

## Benchmark

| T | original_v29 | baseline copy | grouped_v4_epilogue | speedup vs original |
|---:|---:|---:|---:|---:|
| 512 | `0.101892` | `0.110845` | `0.118937` | `0.8567x` |
| 1024 | `0.265454` | `0.266376` | `0.244584` | `1.0853x` |
| 2048 | `0.484118` | `0.494174` | `0.462367` | `1.0470x` |

Grouped-v4 is useful only at medium/long T. It is above the 3% threshold at T=2048, so counters/ISA were collected.

## rocprof T=2048

| variant | trace_us | WG | Grid | LDS bytes | Scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped_v4_best | `468.837` | 128 | 4096 | 24576 | 0 | 48 | 216 | 112 | 32768 | 3796160 | 279808 | 198656 | 247808 | 0.6360 |

ISA evidence:

- `v_mfma_f32_32x32x8_bf16`: present, static count 8.
- `v_mfma_f32_16x16x16_bf16`: absent.
- `global_store_dword`: static count 8.
- No `buffer_store_dwordx4`.

## v23/v24 Transfer Check

Inspected:

- `qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`

The production v23/v24 chunk_gdr path already uses distributed VN/V-decay:

```python
linear_vn = wave_id * 64 + lane
token_offset = linear_vn // 16
value_offset = linear_vn - token_offset * 16
vn[0, token_idx, value_head_idx, global_v] = v_new
v_decay_t[value_offset, token_offset] = ...
```

There is no `128 threads * 8 reps` contiguous-V scalar epilogue analogous to v29. Each 256-thread CTA covers the `16 x 16` VN tile directly. Grouping four V values per thread would change thread ownership and update staging, which is not a low-risk local micro-optimization.

Therefore no `qwen_gdn_chunked_avelang_v24_local_grouped_v4_opt.py` was created.

## v23/v24 Baseline Context

Existing v24 benchmark:

| T | vLLM | v23 | v24 | v24 KKT | v24 chunk_gdr | v24 chunk_o |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | `0.3075` | `0.3043` | `0.2916` | `0.0300` | `0.1187` | `0.0490` |
| 1024 | `0.3069` | `0.4071` | `0.3964` | `0.0289` | `0.1966` | `0.0725` |
| 2048 | `0.3643` | `0.6403` | `0.5873` | `0.0305` | `0.3490` | `0.1083` |

This confirms v24 production bottleneck is not a v29-like VN epilogue loop.

## Raw Store x4

Skipped. The standalone probe proved `raw_buffer_store_x4` works for `i32` x4 copy, but the Qwen VN epilogue needs safe packing of four independent FP32 arithmetic results. That should be a separate source/API probe, not a forced production-path change.

## Recommendation

Keep `qwen_gdn_chunked_avelang_v29_mfma32_grouped_v4_best.py` as the archived best v29 pred-only source variant. Do not apply grouped-v4 to v23/v24 production code. The small fix is low-impact and does not change the main optimization direction.

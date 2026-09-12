# Qwen GDN v16 MFMA Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v14 MFMA path

## Changed Files

- `qwen_gdn_chunked_avelang_v16_mfma_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v16_mfma_layout_fixed.py`
- `bench_qwen_gdn_v16_mfma.py`
- `qwen_gdn_v16_mfma_report.md`

## Implementation Summary

v16 keeps the v14 standalone MFMA `w_u` kernels and the v14 MFMA `chunk_o` kernel.  The only production-stage change is `chunk_gdr`.

New kernel:

- `_qwen_gdn_chunk_gdr_bf16_kernel_v16_2wave_mfma`
- Launch: 64 blocks, 128 threads per block
- `wave0` owns `K=0:64`
- `wave1` owns `K=64:128`
- Both waves share one `[BV,128]` FP32 state tile
- `pred = W @ state.T` is split into two K-half partials and reduced through shared memory
- `vn` and `v_decay_t` are written by wave0 after the partial reduction
- `update = v_decay.T @ K` is split by K-half, so both waves update disjoint state columns
- State decay is folded into update writeback:
  `state[v,k] = state[v,k] * exp(g_last) + delta[v,k]`

Optional changes were not included in this first v16 pass:

- No precomputed `gdr_decay`
- No BF16 `w` output from `w_u`

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_chunked_avelang_v16_mfma_layout_fixed.py -s --tb=short
```

Result:

```text
16 passed in 32.32s
```

Coverage:

- chunk_gdr-only v16 vs v14: `T=16,32,64,512`, with and without initial_state
- full forward v16 vs v14: `T=16,32,64,512`, with and without initial_state
- Tolerance: `atol=1e-3`, `rtol=1e-3`

Worst printed absolute errors:

- `h`: `1.4424324e-05`
- `vn`: `6.29425049e-05`
- `final_state`: `6.85453415e-06`
- `output`: `5.81517816e-05`

## Benchmark

Command:

```bash
python bench_qwen_gdn_v16_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v14 ms | v16 ms | v16 full speedup vs v14 | v16 slowdown vs vLLM |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.3099 | 0.5450 | 0.3367 | 1.6188x | 1.0864x |
| 1024 | 0.3081 | 0.9029 | 0.4902 | 1.8421x | 1.5910x |
| 2048 | 0.3618 | 1.6563 | 0.8499 | 1.9488x | 2.3495x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 512 | v14 | 0.0286 | 0.0601 | 0.0439 | 0.0513 | 0.3747 | 0.0492 |
| 512 | v16 | 0.0282 | 0.0606 | 0.0444 | 0.0502 | 0.1680 | 0.0489 |
| 1024 | v14 | 0.0286 | 0.0785 | 0.0452 | 0.0596 | 0.7109 | 0.0711 |
| 1024 | v16 | 0.0280 | 0.0789 | 0.0452 | 0.0610 | 0.3014 | 0.0713 |
| 2048 | v14 | 0.0282 | 0.1147 | 0.0478 | 0.0828 | 1.3610 | 0.1078 |
| 2048 | v16 | 0.0280 | 0.1146 | 0.0480 | 0.0813 | 0.5608 | 0.1080 |

### chunk_gdr Speedup

| T | v14 chunk_gdr ms | v16 chunk_gdr ms | speedup |
|---:|---:|---:|---:|
| 512 | 0.3747 | 0.1680 | 2.2298x |
| 1024 | 0.7109 | 0.3014 | 2.3583x |
| 2048 | 1.3610 | 0.5608 | 2.4270x |

### Accuracy vs v14 / vLLM

| T | output max_abs vs v14 | final_state max_abs vs v14 | output max_abs vs vLLM | final_state max_abs vs vLLM |
|---:|---:|---:|---:|---:|
| 512 | 1.86265e-09 | 0 | 6.14453e-04 | 5.12862e-03 |
| 1024 | 5.81518e-05 | 2.87965e-06 | 7.10934e-04 | 5.16295e-03 |
| 2048 | 5.81518e-05 | 9.68575e-08 | 7.10934e-04 | 4.81206e-03 |

## rocprof

Command:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex "chunk_gdr" \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v16_chunk_gdr \
  -o v16_chunk_gdr_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v16_mfma.py --T 2048 --warmup 2 --repeat 5
```

Median kernel counters:

| Metric | v14 chunk_gdr | v16 chunk_gdr |
|:---|---:|---:|
| Workgroup_Size | 64 | 128 |
| Grid_Size | 4096 | 8192 |
| LDS_Block_Size | 20992 | 23040 |
| Scratch_Size | 0 | 0 |
| VGPR_Count | 104 | 76 |
| Accum_VGPR_Count | 32 | 100 |
| SGPR_Count | 112 | 112 |
| SQ_INSTS_MFMA | 327680 | 327680 |
| SQ_INSTS_VALU | 10408576 | 5639552 |
| SQ_INSTS_SALU | 1296896 | 716928 |
| SQ_INSTS_VMEM | 897024 | 905216 |
| SQ_INSTS_LDS | 2123776 | 1746944 |
| OccupancyPercent | 0.6417 | 1.2782 |
| median trace us | 1322.344 | 523.958 |

Notes:

- `Workgroup_Size` doubled from 64 to 128 as intended.
- `Scratch_Size` remains zero.
- MFMA count is unchanged at the dispatch level, but pred/update work is split across two waves.
- VALU and SALU instructions drop substantially because v16 removes the separate scalar state-decay loop and halves per-wave K work.
- LDS drops despite the new `pred_partial` buffer, because update/pred staging and synchronization are cheaper overall.

## Conclusion

v16 is a clear improvement over v14.  The 2-wave cooperative `chunk_gdr` kernel reduces T=2048 `chunk_gdr` latency from `1.3610 ms` to `0.5608 ms`, a `2.4270x` speedup.  Full latency drops from `1.6563 ms` to `0.8499 ms`, a `1.9488x` speedup.

Current bottleneck after v16:

- `chunk_gdr` is still the largest individual stage at long T, but it is much closer to the rest of the pipeline.
- At T=2048, non-`chunk_gdr` stages sum to roughly `0.3808 ms`, while `chunk_gdr` is `0.5608 ms`.
- v16 is still `2.3495x` slower than vLLM at T=2048.

Recommended next action:

- Keep v16 as the new Avelang baseline candidate.
- Next, try low-risk chunk_gdr follow-ups:
  precompute `gdr_decay/g_last_exp`, then consider BF16 `w` output from `w_u` to avoid FP32 global write plus BF16 conversion in chunk_gdr.
- Do not return to v15 fused chunk_o work unless chunk_o becomes a bottleneck again.

# Qwen GDN v24 KKT MFMA Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/output/final_state`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v23 distributed chunk_gdr production path

## Changed Files

- `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`
- `test_qwen_gdn_v24_kkt_mfma.py`
- `bench_qwen_gdn_v24_kkt_mfma.py`
- `qwen_gdn_v24_kkt_mfma_report.md`

## v23 Bottleneck Ranking

At `T=2048`, v23 stage latency from the v24 benchmark run:

| stage | v23 ms |
|:---|---:|
| chunk_gdr | 0.3490 |
| KKT | 0.1151 |
| chunk_o | 0.1078 |
| w_u | 0.0816 |
| solve | 0.0475 |
| gdr_decay | 0.0304 |
| cumsum | 0.0280 |

KKT was the second largest remaining stage after v23's chunk_gdr improvement.

## KKT Audit

Current v23 calls `qwen_gdn_kkt_avelang_v6_standalone(k, g_cumsum, beta, chunk_size=16, prefer_optimized=True)`.

Inputs:

- `k`: BF16 `[B,T,Hk,K]`, target `[1,T,4,128]`
- `g`: FP32 `[B,T,Hv]`, target `[1,T,8]`; this is chunk-local cumulative `g_cumsum`
- `beta`: FP32 `[B,T,Hv]`, target `[1,T,8]`

Output:

- `a`: FP32 `[B,T,Hv,chunk_size]`, target `[1,T,8,16]`
- Layout strides: `(T*Hv*BT, Hv*BT, BT, 1)`
- `a[0, token_idx, value_head_idx, source_offset]` stores one chunk-local row entry.

Formula for each token `t`, source offset `s`, value head `h`, and key head `hk = h // 2`:

```text
chunk_start = t - (t % BT)
source_token = chunk_start + s

if s < (t % BT):
    a[t,h,s] = beta[t,h] * dot(k[t,hk,:], k[source_token,hk,:])
               * exp(g[t,h] - g[source_token,h])
else:
    a[t,h,s] = 0
```

This is a strict lower-triangular chunk-local token-token matrix, not a triangular solve.  The solve stage consumes this `a` layout directly.

v6 KKT is scalar:

- Launch shape: `grid = B*T*Hv`, `block = (1,1,1)`
- At `T=2048`: 16384 single-thread programs
- MFMA: none
- Each program serially computes up to 15 source positions, each with a serial `K=128` dot product
- This explains the remaining `~0.115 ms` at `T=2048`

v20 BT32 showed the useful direction: compute the chunk-local token-token dot matrix with MFMA, then apply strict causal mask, decay, and `beta`.  v24 reuses that idea for BT16 with the proven `mfma_16x16x16_bf16_f32` score pattern from chunk_o.

## Implementation Summary

Added native BT16 KKT:

- Kernel: `_qwen_gdn_kkt_bf16_kernel_v24_bt16_mfma`
- Wrapper: `qwen_gdn_kkt_avelang_v24_bt16_mfma_layout`
- Launch: `num_chunks * 8` CTAs, 64 threads per CTA
- Per CTA computes one `[16,16]` KKT tile for one `(chunk, value_head)`
- Stages `K_chunk[16,128]` in LDS
- Computes `K_chunk @ K_chunk.T` with 8 `mfma_16x16x16_bf16_f32` calls per CTA
- Applies strict lower triangle, `beta[t]`, and `exp(g[t]-g[s])` during writeback

Full v24 path keeps all other v23 stages unchanged:

- cumsum: v6
- KKT: v24 native MFMA
- solve: v6
- w_u: v14/v23 BT16 MFMA
- gdr_decay: v17
- chunk_gdr: v23 distributed
- chunk_o: v14/v23 MFMA

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v24_kkt_mfma.py -s --tb=short --disable-warnings
```

Result:

```text
16 passed in 26.43s
```

Coverage:

- KKT-only v24 vs v6: `T=16,32,64,512,1024`
- solve-after-KKT v24 vs v6: `T=16,64,512`
- full forward v24 vs v23: `T=16,32,64,512`, with and without initial_state

Worst printed errors:

- `KKT max_abs`: `5.96046448e-08`
- `a_solved max_abs`: `5.96046448e-08`
- `output max_abs`: `9.45031643e-05`
- `final_state max_abs`: `2.62439251e-04`

## Benchmark

Command:

```bash
python bench_qwen_gdn_v24_kkt_mfma.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v23 ms | v24 ms | full speedup vs v23 | slowdown vs vLLM |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.2995 | 0.3008 | 0.2887 | 1.0418x | 0.9638x |
| 1024 | 0.3099 | 0.3994 | 0.3889 | 1.0268x | 1.2550x |
| 2048 | 0.3614 | 0.6356 | 0.5845 | 1.0874x | 1.6172x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | gdr_decay | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 512 | v23 | 0.0286 | 0.0602 | 0.0441 | 0.0505 | 0.0290 | 0.1193 | 0.0486 |
| 512 | v24 | 0.0284 | 0.0288 | 0.0444 | 0.0501 | 0.0284 | 0.1186 | 0.0484 |
| 1024 | v23 | 0.0289 | 0.0781 | 0.0449 | 0.0606 | 0.0308 | 0.1956 | 0.0714 |
| 1024 | v24 | 0.0284 | 0.0299 | 0.0448 | 0.0599 | 0.0298 | 0.1955 | 0.0707 |
| 2048 | v23 | 0.0280 | 0.1151 | 0.0475 | 0.0816 | 0.0304 | 0.3490 | 0.1078 |
| 2048 | v24 | 0.0282 | 0.0293 | 0.0469 | 0.0820 | 0.0291 | 0.3490 | 0.1076 |

### KKT Speedup

| T | v23 KKT | v24 KKT | speedup |
|---:|---:|---:|---:|
| 512 | 0.0602 | 0.0288 | 2.0861x |
| 1024 | 0.0781 | 0.0299 | 2.6139x |
| 2048 | 0.1151 | 0.0293 | 3.9263x |

### Accuracy

| T | output max_abs vs v23 | final_state max_abs vs v23 | output max_abs vs vLLM | final_state max_abs vs vLLM |
|---:|---:|---:|---:|---:|
| 512 | 1.74223e-04 | 7.29650e-04 | 5.85493e-04 | 4.52912e-03 |
| 1024 | 2.64829e-05 | 3.30806e-06 | 6.70791e-04 | 4.70567e-03 |
| 2048 | 1.07171e-04 | 5.57449e-04 | 6.36213e-04 | 4.12196e-03 |

## rocprof

Command profiled v23/v24 KKT dispatches at `T=2048`:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex kkt \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v24_kkt_mfma \
  -o v24_kkt_mfma_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v24_kkt_mfma.py --T 2048 --warmup 2 --repeat 5
```

Median kernel counters for the two Avelang KKT kernels:

| Metric | v23/v6 KKT | v24 KKT MFMA |
|:---|---:|---:|
| Workgroup_Size | 1 | 64 |
| Grid_Size | 16384 | 65536 |
| LDS_Block_Size | 0 | 4096 |
| Scratch_Size | 0 | 0 |
| VGPR_Count | 52 | 52 |
| Accum_VGPR_Count | 132 | 4 |
| SGPR_Count | 32 | 32 |
| SQ_INSTS_MFMA | 0 | 8192 |
| SQ_INSTS_VALU | 51174400 | 410624 |
| SQ_INSTS_SALU | 3875840 | 41984 |
| SQ_INSTS_VMEM | 2796544 | 49152 |
| SQ_INSTS_LDS | 0 | 36864 |
| OccupancyPercent | 18.4860 | 2.6040 |
| median trace us | 89.914 | 4.787 |

Notes:

- v24 KKT does use MFMA.
- Workgroup size increases from scalar `1` to `64`.
- Scratch remains zero.
- VALU drops by roughly `124.6x`, VMEM by roughly `56.9x`, and SALU by roughly `92.3x`.
- Rocprof `Grid_Size` is work-item count; the v24 launch is 1024 CTAs times 64 threads = 65536 work-items.
- KKT trace time drops by roughly `18.8x`; benchmarked stage latency drops by `3.93x` at `T=2048`.

## Conclusion

v24 KKT should replace the v23 KKT stage for the BT16 production path.

At `T=2048`:

- v23 full: `0.6356 ms`
- v24 full: `0.5845 ms`
- v23 KKT: `0.1151 ms`
- v24 KKT: `0.0293 ms`

The optimization removes the old single-thread KKT bottleneck.  The new bottleneck ranking after v24 is:

1. `chunk_gdr`: `~0.349 ms`
2. `chunk_o`: `~0.108 ms`
3. `w_u`: `~0.082 ms`
4. `solve`: `~0.047 ms`
5. `KKT`: `~0.029 ms`

Recommended next direction:

- v24 becomes the new BT16 baseline.
- The next meaningful target is no longer KKT; it is either reducing remaining chunk_gdr materialization/coordination overhead or revisiting chunk_o/w_u only if the chunk_gdr path plateaus.

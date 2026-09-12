# Qwen GDN v21 GDR Pipeline Report

## Target

- Shape: `B=1`, `Hk=4`, `Hv=8`, `K=128`, `V=128`, layout `[B,T,H,D]`
- Dtypes: BF16 `q/k/v`, FP32 `g/beta/w/u/output/final_state`
- `chunk_size=16`, `BT=16`, `BV=16`
- Baseline: v17 predecay BT16

## Changed Files

- `qwen_gdn_chunked_avelang_v21_gdr_pipeline_layout_fixed.py`
- `test_qwen_gdn_v21_gdr_pipeline.py`
- `bench_qwen_gdn_v21_gdr_pipeline.py`
- `qwen_gdn_v21_gdr_pipeline_report.md`

## v17 Chunk_GDR Audit

v17 predecay chunk_gdr uses `_qwen_gdn_chunk_gdr_bf16_kernel_v17_4wave_mfma`.

Per block:

- Grid: `8 value blocks * 8 value heads = 64` blocks
- Workgroup: `256` threads, four wavefronts
- Each block owns `state[BV=16, K=128]`
- `wave0`: K `0:32`
- `wave1`: K `32:64`
- `wave2`: K `64:96`
- `wave3`: K `96:128`

Per chunk steps:

1. Write old `state` to global `h[chunk]`.
2. Stage current old state to `state_bf16[BV,128]`.
3. Stage current W to `w_bf16[BT,128]`.
4. Stage current K transposed to `k_all_t[128,BT]`.
5. Compute four K-quarter pred partials:
   `pred_partial[wave, BT, BV] = W[:,Kq] @ state[:,Kq].T`.
6. Wave0 reduces four partials, computes `vn = u - pred`, writes global `vn`, and writes BF16 `v_decay_t`.
7. Four waves update disjoint K quarters:
   `state[:,Kq] = state[:,Kq] * g_last_exp + v_decay_t @ K[:,Kq]`.
8. After all chunks, write `final_state`.

Shared buffers in v17:

- `state[BV,128]` FP32
- `state_bf16[BV,128]` BF16
- `w_bf16[BT,128]` BF16
- `v_decay_t[BV,BT]` BF16
- `k_all_t[128,BT]` BF16
- `pred_partial[4,BT,BV]` FP32

Global reads per chunk:

- Old recurrent `state` comes from shared memory, initialized from `initial_state` once per block.
- `w[1,T,8,128]` FP32, converted to BF16 for pred MFMA.
- `k[1,T,4,128]` BF16, staged transposed for update MFMA.
- `u[1,T,8,128]` FP32, read by wave0 for `vn`.
- `gdr_decay[1,num_chunks,8,16]` FP32 and `gdr_g_last_exp[1,num_chunks,8]` FP32 from the predecay kernel.
- `h` and `vn` are global outputs consumed later by chunk_o.

Double-buffer candidates:

- `w_bf16`: safe to preload for chunk `i+1`; it has no state dependency.
- `k_all_t`: safe to preload for chunk `i+1`; it has no state dependency.
- `gdr_decay` / `g_last_exp`: safe to preload, but small enough that launch/LDS pressure may dominate.
- `u`: safe to preload, but only wave0 consumes the `[BT,BV]` slice.

Not safe to move across chunks:

- `state` itself.
- `state_bf16` for the next chunk, because it must be produced from state after the current chunk update.
- `h[chunk+1]`, because it is the updated state after chunk `chunk`.

Barrier positions in v17:

- After state initialization.
- After staging state/W/K.
- After pred partial writes.
- After wave0 writes `v_decay_t`.
- After update before the next chunk.

Latest known v17 chunk_gdr rocprof at `T=2048`:

- Workgroup_Size: `256`
- Scratch_Size: `0`
- VGPR_Count: `40`
- Accum_VGPR_Count: `136`
- LDS_Block_Size: `25088`
- SQ_INSTS_MFMA: `327680`
- SQ_INSTS_VALU: `6437120`
- SQ_INSTS_VMEM: `921600`
- SQ_INSTS_LDS: `2371584`
- OccupancyPercent: `2.5372`
- median trace: about `415 us`

## v21 Implementation

v21 keeps v17 math and predecay semantics exactly. Two conservative variants were implemented:

- `v21_w`: double-buffer `w_bf16[2,BT,128]`; current K remains single-buffered as in v17.
- `v21_wk`: double-buffer both `w_bf16[2,BT,128]` and transposed `k_all_t[2,128,BT]`.

Pipeline schedule:

1. Preload chunk 0 input buffer.
2. For each chunk, use current buffer for pred/update.
3. After `v_decay_t` is produced and before update, preload chunk `i+1` into the alternate buffer if it exists.
4. Keep conservative barriers around preload, pred partial reduction, `v_decay_t`, and update.

Full double-buffer of W/K/U/decay was not implemented in the first pass; W and K are the main VMEM/LDS traffic in chunk_gdr, while U/decay are smaller and would increase LDS pressure.

## Correctness

Command:

```bash
python -m pytest -q test_qwen_gdn_v21_gdr_pipeline.py -s --tb=short
```

Result:

```text
32 passed in 32.61s
```

Coverage:

- chunk_gdr-only v21 vs v17 predecay: `T=16,32,64,512`, with and without initial_state
- full forward v21 vs v17 predecay: `T=16,32,64,512`, with and without initial_state
- variants: `v21_w`, `v21_wk`

Worst printed errors:

- `h`: `0`
- `vn`: `0`
- `final_state`: `0`
- `output`: `0`

## Benchmark

Command:

```bash
python bench_qwen_gdn_v21_gdr_pipeline.py --T 512 1024 2048 --warmup 10 --repeat 30
```

### Full Latency

| T | vLLM ms | v17 predecay ms | v21 W ms | v21 W+K ms | W speedup vs v17 | W+K speedup vs v17 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.3126 | 0.3251 | 0.3245 | 0.3202 | 1.0018x | 1.0153x |
| 1024 | 0.3087 | 0.4359 | 0.4488 | 0.4448 | 0.9712x | 0.9800x |
| 2048 | 0.3617 | 0.6950 | 0.7213 | 0.7005 | 0.9635x | 0.9922x |

### Stage Breakdown

| T | version | cumsum | KKT | solve | w_u | gdr_decay | chunk_gdr | chunk_o |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 512 | v17 predecay | 0.0286 | 0.0598 | 0.0443 | 0.0502 | 0.0299 | 0.1370 | 0.0486 |
| 512 | v21 W | 0.0283 | 0.0602 | 0.0445 | 0.0496 | 0.0292 | 0.1377 | 0.0494 |
| 512 | v21 W+K | 0.0282 | 0.0599 | 0.0453 | 0.0502 | 0.0295 | 0.1334 | 0.0486 |
| 1024 | v17 predecay | 0.0338 | 0.0811 | 0.0468 | 0.0615 | 0.0336 | 0.2289 | 0.0724 |
| 1024 | v21 W | 0.0315 | 0.0800 | 0.0464 | 0.0602 | 0.0312 | 0.2429 | 0.0715 |
| 1024 | v21 W+K | 0.0296 | 0.0794 | 0.0465 | 0.0603 | 0.0298 | 0.2296 | 0.0716 |
| 2048 | v17 predecay | 0.0280 | 0.1167 | 0.0495 | 0.0836 | 0.0300 | 0.4099 | 0.1078 |
| 2048 | v21 W | 0.0292 | 0.1161 | 0.0494 | 0.0822 | 0.0304 | 0.4364 | 0.1074 |
| 2048 | v21 W+K | 0.0291 | 0.1156 | 0.0490 | 0.0821 | 0.0304 | 0.4127 | 0.1080 |

### chunk_gdr Speedup

| T | v17 chunk_gdr | v21 W chunk_gdr | v21 W+K chunk_gdr | W speedup | W+K speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.1370 | 0.1377 | 0.1334 | 0.9951x | 1.0267x |
| 1024 | 0.2289 | 0.2429 | 0.2296 | 0.9423x | 0.9970x |
| 2048 | 0.4099 | 0.4364 | 0.4127 | 0.9392x | 0.9932x |

Accuracy vs v17 predecay:

- `v21_w` output/final_state max_abs: `0`
- `v21_wk` output/final_state max_abs: `0`

## rocprof

Command:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex chunk_gdr \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v21_gdr_pipeline \
  -o v21_gdr_pipeline_counters \
  -f csv \
  -- python test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v21_gdr_pipeline.py --T 2048 --warmup 2 --repeat 5
```

Median counters at `T=2048`:

| Metric | v17 predecay | v21 W | v21 W+K |
|:---|---:|---:|---:|
| Workgroup_Size | 256 | 256 | 256 |
| Grid_Size | 16384 | 16384 | 16384 |
| LDS_Block_Size | 25088 | 29184 | 33280 |
| Scratch_Size | 0 | 0 | 0 |
| VGPR_Count | 40 | 68 | 72 |
| Accum_VGPR_Count | 136 | 132 | 192 |
| SGPR_Count | 112 | 112 | 112 |
| SQ_INSTS_MFMA | 327680 | 327680 | 327680 |
| SQ_INSTS_VALU | 5576960 | 6149440 | 6473280 |
| SQ_INSTS_SALU | 830976 | 964800 | 1029824 |
| SQ_INSTS_VMEM | 921600 | 921600 | 921600 |
| SQ_INSTS_LDS | 2371584 | 2404352 | 2404352 |
| OccupancyPercent | 2.5272 | 2.5128 | 2.5259 |
| median trace us | 366.084 | 394.947 | 372.894 |

Answers:

1. LDS_Block_Size increased: `25088 -> 29184` for W-only and `25088 -> 33280` for W+K.
2. Occupancy did not meaningfully improve. W-only slightly decreased; W+K is effectively equal to v17.
3. VMEM count stayed the same at `921600`; double-buffering changed staging order but did not reduce global memory traffic.
4. Median trace did not improve. W-only regressed by about `28.9 us`; W+K regressed by about `6.8 us`.
5. Double-buffering mostly increased resource pressure. Without async copy or independent work to overlap on the same wavefronts, the next-chunk preload remains on the critical path.
6. Best measured variant remains v17 predecay. v21 W+K is close but not faster at long T; v21 W-only is worse.

## Conclusion

v21 is a correct but mostly negative result.

At `T=2048`:

- v17 predecay full: `0.6950 ms`
- v21 W full: `0.7213 ms`
- v21 W+K full: `0.7005 ms`
- v17 chunk_gdr: `0.4099 ms`
- v21 W chunk_gdr: `0.4364 ms`
- v21 W+K chunk_gdr: `0.4127 ms`

The conservative software pipeline did not hide chunk-local W/K load latency. It kept VMEM constant while increasing LDS allocation, VALU/SALU instructions, and register pressure. Because Avelang is not issuing true async copies here, the preload of chunk `i+1` still executes on the same critical path as chunk `i`.

Recommended next direction:

- Keep v17 predecay as the best BT16 baseline.
- Do not continue simple double-buffer staging as-is.
- If continuing BT16 chunk_gdr, focus on reducing coordination overhead rather than reordering loads: partial-reduction LDS traffic, pred/update barrier count, and wave0-only `vn/v_decay` serialization.
- A more radical path would need real async copy or a mapping that gives different waves independent useful work while state update is in flight; the current four waves all remain tied to the same state recurrence.

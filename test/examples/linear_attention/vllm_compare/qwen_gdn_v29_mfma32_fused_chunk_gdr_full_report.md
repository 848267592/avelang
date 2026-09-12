# Qwen GDN v29 MFMA32 Fused Chunk_GDR Full Report

## Summary

Implemented a real fused pred-update chunk_gdr candidate:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full.py`
- `bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full.py`

The candidate removes full global `vn` materialization and performs state update in-kernel.  It is **not production-ready**:

- update-only isolation with `w=0` is correct (`final_state max_abs <= 3.1e-5`);
- full pred+update recurrence does not pass correctness;
- T=2048 latency is `0.8364 ms`, slower than v24 full `0.6251 ms`;
- rocprof shows high resource pressure: `LDS=61440`, `VGPR=128`, `AccVGPR=264`, `trace=833.9 us`.

Decision: stop this v29 fused full candidate.  The useful result is diagnostic: avoiding global `vn` is still promising, but adding full update with the current source schedule drives resource pressure too high, and MFMA32 pred error is amplified by recurrence.

## Implementation

Starting point:

- Reused v29 fused skeleton pred schedule:
  - `BT=64`, `BV=32`
  - workgroup `128`
  - two waves split K halves `0:64` and `64:128`
  - pred uses `mfma_32x32x8_bf16_f32`

Math copied from v23/v24 chunk_gdr:

```text
pred = W @ state.T
u_corr = u - pred
v_decay = u_corr * decay
delta_H = v_decay.T @ K
state = state * g_last_exp + delta_H
```

Original skeleton only staged corrected values:

```python
u_corr_t_bf16[local_v, token_off] = bf16(u - pred)
```

New fused full candidate stages decay-corrected values and updates state:

```python
v_decay_t_bf16[local_v, token_base + token_off] = bf16((u - pred) * decay)
state[out_v, out_k] = state[out_v, out_k] * g_last_exp + update_acc
```

Important implementation note: pred uses MFMA32, but update uses the proven v23-style `mfma_16x16x16_bf16_f32` pattern over four 16-token subtiles.  I tried native MFMA32 update first, but it failed update correctness; the v23-style update isolates the fused dataflow correctness risk.

The wrapper returns:

- `h`
- `final_state`

It intentionally does not return full global `vn`.

## Correctness

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full.py \
  --T 512 1024 2048 --warmup 3 --repeat 10
```

Full recurrence correctness failed:

| T | h max_abs | h mean_abs | final_state max_abs | final_state mean_abs |
|---:|---:|---:|---:|---:|
| 512 | `2.092025e+04` | `6.048572e+02` | `5.842288e+04` | `8.180729e+03` |
| 1024 | `7.032799e+07` | `8.733711e+05` | `2.073074e+08` | `2.414688e+07` |
| 2048 | `7.766104e+14` | `4.217769e+12` | `2.108693e+15` | `2.374940e+14` |

Update isolation with `w=0` passed:

| T | w=0 final_state max_abs | w=0 final_state mean_abs |
|---:|---:|---:|
| 512 | `1.525879e-05` | `7.375062e-07` |
| 1024 | `2.288818e-05` | `1.079752e-06` |
| 2048 | `3.051758e-05` | `1.412535e-06` |

Interpretation:

- the fused update dataflow and v23-style update indexing are correct;
- the full recurrent path diverges when MFMA32 pred is fed back into state;
- previous pred-only error was BF16-level for a fixed small state, but recurrence amplifies it.

## Benchmark

Same command as correctness above.

| T | v24 full | v29 grouped pred-only | v29 fused skeleton | v29 fused chunk_gdr full |
|---:|---:|---:|---:|---:|
| 512 | `0.312364` | `0.121241` | `0.077074` | `0.234408` |
| 1024 | `0.437550` | `0.250973` | `0.137845` | `0.445062` |
| 2048 | `0.625129` | `0.503909` | `0.240296` | `0.836383` |

vLLM was not re-profiled in this script.  The recent v24 report context on the same environment had vLLM around `0.364 ms` at T=2048, so this v29 candidate is not competitive.

## rocprof

Command:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex fused_chunk_gdr_full \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_fused_chunk_gdr_full \
  -o v29_fused_chunk_gdr_full_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full.py \
       --T 2048 --warmup 2 --repeat 5 --no-check-ref
```

| metric | v29 fused chunk_gdr full |
|:---|---:|
| trace median us | `833.9205` |
| Workgroup_Size | `128` |
| Grid_Size | `4096` |
| LDS_Block_Size | `61440` |
| Scratch_Size | `0` |
| VGPR_Count | `128` |
| Accum_VGPR_Count | `264` |
| SGPR_Count | `112` |
| SQ_INSTS_MFMA | `294912` |
| SQ_INSTS_VALU | `4977280` |
| SQ_INSTS_SALU | `810496` |
| SQ_INSTS_VMEM | `399360` |
| SQ_INSTS_LDS | `1242304` |
| OccupancyPercent | `0.6459` |

Compared with the previous fused skeleton (`trace ~205.8 us`, `AccVGPR=192`, `VALU=1.55M`, `SALU=32K`, `VMEM=168K`), full update brings the kernel back into the slow/resource-heavy region.

## ISA Evidence

HSACO dump command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full.py \
  --T 64 --warmup 1 --repeat 1 --no-check-ref --dump-hsaco-dir /tmp/v29_full_hsaco
```

Static ISA grep:

| pattern | count |
|:---|---:|
| `v_mfma_f32_32x32x8_bf16` | `16` |
| `v_mfma_f32_16x16x16_bf16` | `64` |
| `v_mfma` | `80` |
| `global_load` | `192` |
| `global_store` | `64` |
| `buffer_load` | `6` |
| `buffer_store` | `6` |
| `ds_read` | `257` |
| `ds_write` | `235` |
| `v_lshl` | `873` |
| `v_add` | `192` |
| `v_or` | `269` |
| `v_bfe` | `113` |
| `s_barrier` | `13` |
| `s_waitcnt` | `318` |

MFMA32 is present in the pred path.  MFMA16 is also present because update deliberately uses the v23-style 16x16 kernel pattern.

## Diagnosis

The stop condition is met.

1. Full correctness cannot pass with the current MFMA32 pred recurrence.
2. Reintroducing correct update makes latency worse than v24.
3. Counters return close to or worse than the original slow region:
   - AccVGPR rises to `264`;
   - LDS block reaches `61440`;
   - VALU rises to `4.98M`;
   - trace is `833.9 us`.
4. Global `vn` materialization is removed, so the remaining cost is not the old full-VN epilogue.

This suggests the current source-level v29 fused full path is not the right production route.  The likely blockers are:

- MFMA32 pred numerical behavior is not stable enough for recurrent state feedback under this handmade layout/schedule;
- using v23-style 16x16 update inside a BT64/BV32 kernel creates too much LDS/register pressure;
- a correct high-performance version probably needs a backend/block-dot lowering closer to Triton, or a native MFMA32 update schedule whose mapping is proven by one-hot tests.

## Final Decision

Do not continue this exact v29 fused full source path.

Recommended next step:

- return to v24/v23 production optimization for near-term wins, or
- escalate the v29 MFMA32 fused design to backend/block-dot lowering work.

If v29 is revisited, the next prerequisite is not another epilogue tweak.  It is a deterministic MFMA32 update one-hot mapping and recurrence-stability test before writing another full chunk_gdr.

## Files

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_mfma32_fused_chunk_gdr_full.py`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v29_mfma32_fused_chunk_gdr_full_report.md`

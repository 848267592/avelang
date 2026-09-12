# Qwen GDN v29 MFMA32 Fused Pred-Update Prototype Report

## Summary

Track B created a profiling-only fused pred-update skeleton:

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_pred_update_proto.py`
- `bench_qwen_gdn_v29_mfma32_fused_pred_update.py`

The skeleton uses the original v29 k-split MFMA32 pred schedule, computes `u_corr = u - pred`, stages all corrected values into shared memory as `[V, token]` BF16, and avoids full global `vn` materialization. It does not implement the full state update yet.

At `T=2048`:

- original v29 pred-only: `0.482537 ms`
- grouped_v4 pred-only: `0.462788 ms`
- fused_no_vn_store upper bound: `0.209131 ms`
- fused_pred_update_skeleton: `0.238193 ms`

This is much closer to the no-VN-store upper bound than to original v29, so source-level fusion appears viable enough to continue. The next step should be a partial/full update that consumes the staged `u_corr` without writing global `vn`.

## Existing Update Code Studied

Relevant files:

- `qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py`
- `qwen_gdn_chunked_avelang_v28_triton64_geometry.py`

v23/v24 production chunk_gdr computes:

```python
pred_value = pred_partial[0,t,v] + pred_partial[1,t,v] + pred_partial[2,t,v] + pred_partial[3,t,v]
v_new = u[0, token_idx, value_head_idx, global_v] - pred_value
vn[0, token_idx, value_head_idx, global_v] = v_new
v_decay_t[value_offset, token_offset] = bf16(v_new * decay)
```

Then update consumes `v_decay_t` via MFMA:

```python
delta_H = v_decay_t[BV,BT] @ k_chunk[BT,K]
state[:, Ktile] = state[:, Ktile] * g_last_exp + delta_H
```

The important layout is V-major/token-minor shared staging:

```python
v_decay_t[value_offset, token_offset]
```

## What Was Fused

The prototype computes:

```python
pred_skel = pred_partial[0, token, v] + pred_partial[1, token, v]
u_corr_t_bf16[v, token] = bf16(u_flat[offset] - pred_skel)
```

This avoids:

- full global `vn` store
- future global `vn` reload

It preserves:

- original v29 k-split 32x32 MFMA pred schedule
- full `u` load dependency
- full corrected-value shared staging in update-friendly `[V, token]` form

## What Was Not Fused

The full update MFMA was not implemented in this pass. The current file is a skeleton/profiling prototype only. Full update needs the BT64/BV32 state update geometry and decay handling to be wired correctly; doing that in one jump risks reintroducing the global-VN path or silently changing math.

No `update_only` kernel reading global `vn` was written.

## Correctness Status

Correctness-preserving rows in the benchmark:

- `original_v29`
- `grouped_v4_best`

Profiling-only rows:

- `fused_no_vn_store_upper_bound`
- `fused_pred_update_skeleton`

`fused_pred_update_skeleton` intentionally does not output full `vn`, so it is not compared against the torch full-VN reference.

## Benchmark

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v29_mfma32_fused_pred_update.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

| T | original_v29 | grouped_v4_best | fused_no_vn_store | fused_pred_update_skeleton |
|---:|---:|---:|---:|---:|
| 512 | `0.105657` | `0.118676` | `0.068702` | `0.074771` |
| 1024 | `0.239937` | `0.240056` | `0.115932` | `0.135301` |
| 2048 | `0.482537` | `0.462788` | `0.209131` | `0.238193` |

At `T=2048`, skeleton is `2.03x` faster than original v29 pred-only and only `13.9%` slower than the no-VN-store upper bound.

## rocprof T=2048

| variant | trace_us | WG | Grid | LDS bytes | Scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grouped_v4_best | `468.837` | 128 | 4096 | 24576 | 0 | 48 | 216 | 112 | 32768 | 3796160 | 279808 | 198656 | 247808 | 0.6360 |
| fused_pred_update_skeleton | `205.846` | 128 | 4096 | 26624 | 0 | 72 | 192 | 112 | 32768 | 1548992 | 32128 | 167936 | 299008 | 0.6239 |

Interpretation:

- MFMA dynamic count is unchanged.
- VALU drops by about `2.45x`.
- SALU drops by about `8.7x`.
- VMEM drops from `198656` to `167936`.
- LDS increases because full `u_corr` is staged to shared memory, which is expected.
- AccVGPR drops from `216` to `192`.
- Trace drops from `468.837 us` to `205.846 us`.

## ISA Evidence

HSACO/ISA paths:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_grouped_best_hsaco/`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v29_fused_skeleton_hsaco/`

Static grep:

| kernel | `v_mfma_f32_32x32x8_bf16` | `v_mfma_f32_16x16x16_bf16` | `global_store_dword` |
|:---|---:|---:|---:|
| grouped_v4_best | 8 | 0 | 8 |
| fused_pred_update_skeleton | 16 | 0 | 2 |

The skeleton still generates MFMA32 and avoids MFMA16. The static MFMA count is higher due to compile-time branch structure, but rocprof dynamic MFMA count is the same as grouped-v4.

## Viability

Source-level fusion looks viable enough to continue:

- skeleton is close to the no-VN-store upper bound;
- skeleton reduces VALU/SALU/VMEM sharply;
- it does not require global `vn`;
- it stages corrected values in the layout update wants.

But this is not yet a full path:

- no real state update MFMA is included;
- no decay/gating path is included;
- no final_state correctness is tested;
- no output/chunk_o path is included.

## Stop/Continue Criteria Result

Stop criteria were not met:

- skeleton trace is not near original v29 (`205.846 us` vs grouped `468.837 us`);
- VALU/SALU are not close to original;
- the prototype did not reintroduce global full-VN materialization.

Continue criteria were met:

- skeleton is much closer to `fused_no_vn_store` (`0.238 ms` vs `0.209 ms`) than to original v29 (`0.483 ms`);
- update-friendly shared staging is feasible;
- counters support the materialization bottleneck diagnosis.

## Recommendation

Continue Track B, but the next step should be narrow:

1. Implement `fused_pred_update_partial`: consume staged `u_corr_t_bf16` with one or two update K tiles.
2. Verify state update math against a small torch/reference path.
3. Only after partial correctness, extend to full update.

Do not implement `update_only` reading global `vn`; that would bypass the bottleneck and would not prove full-path viability.

# Qwen MFMA32 L4-L6 Source/ISA Analysis

## Scope

This report maps the existing Qwen-shaped lowering ladder source transitions around `L4 -> L5 -> L6` to the measured counters and static ISA already collected in `qwen_mfma32_lowering_ladder_report.md`.

No new kernel optimization is claimed here.  The new L5/L6 focused repros are separate files and still require docker/rocprof rerun.

## Source Additions

### L4: v_decay Shared Staging

Source: `repro_qwen_mfma32_lowering_ladder.py`, lines 177-192.

```python
v_decay_t = al.make_shared((BV, BT), al.bf16)
for rep_vd in al.range(8):
    linear_vd = tid + rep_vd * WORKGROUP
    tok_vd = linear_vd // BV
    vv_vd = linear_vd - tok_vd * BV
    token_idx_vd = token_base + tok_vd
    offset_vd = token_idx_vd * (8 * 128) + value_head_idx * 128 + value_base + vv_vd
    pred_vd = pred_partial[0, tok_vd, vv_vd] + pred_partial[1, tok_vd, vv_vd]
    u_corr_vd = u_flat[offset_vd] - pred_vd
    decay_v = decay[0, 0, value_head_idx, token_idx_vd]
    v_decay_t[vv_vd, token_idx_vd] = al.convert(u_corr_vd * decay_v, al.bf16)

al.syncthreads()
```

Marks:

- `v_decay_t` shared allocation: line 178.
- No K staging yet.
- No `kall_vec`.
- No update MFMA.

### L5: Runtime Transposed K Staging and `kall_vec`

Source: `repro_qwen_mfma32_lowering_ladder.py`, lines 209-227.

```python
k_all_t = al.make_shared((128, BT), al.bf16)
vdecay_vec = al.view(v_decay_t, al.i32, al.make_layout((BV, 8, 4), (8 * 4, 4, 1)))
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
for rep_k in al.range(64):
    linear_k = tid + rep_k * WORKGROUP
    kk = linear_k // BT
    tok_k = linear_k - kk * BT
    k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
    if level == 5:
        vv_keep = tok_k & 31
        sink[program_id, (linear_k & 511)] = (
            al.convert(k_all_t[kk, tok_k], al.f32)
            + al.convert(v_decay_t[vv_keep, tok_k], al.f32)
        )

al.syncthreads()
```

Marks:

- `k_all_t` shared allocation: line 210.
- `vdecay_vec` packed view: line 211.
- `kall_vec` packed view: line 212.
- K global load: line 217.
- Transposed shared store `[kk, tok_k]`: line 217.
- No update fragment load/MFMA yet.

### L6: First Dependent Update MFMA

Source: `repro_qwen_mfma32_lowering_ladder.py`, lines 229-264.

```python
if level >= 6 or level == 19:
    tile_count = 1
    for tile in al.range(tile_count):
        acc16 = al.full((4,), 0.0, al.f32)
        pack_base = token_tile * 4
        if lane_group == 0:
            a_words16 = vdecay_vec[lane_col, pack_base]
            b_words16 = kall_vec[tile * 16 + lane_col, pack_base]
            a_frag16 = al.view(a_words16, al.Tensor((2, 4, 1), al.bf16))
            b_frag16 = al.view(b_words16, al.Tensor((2, 4, 1), al.bf16))
            acc16 = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag16[0], b_frag16[0], acc16)
        ...
        if level == 6:
            for r6 in al.range(4):
                sink[program_id, tile * 64 + lane_group * 16 + r6 * 4 + (lane_col & 3)] = acc16[r6]
```

Marks:

- Update fragment load from `vdecay_vec`: lines 238/244/250/256.
- Update fragment load from `kall_vec`: lines 239/245/251/257.
- First dependent update MFMA: lines 242/248/254/260.
- The update consumes values produced through `pred_partial -> u_corr -> v_decay_t` and K staged into `k_all_t`.

## Dynamic Counter Deltas

From the existing ladder rocprof table:

| transition | trace_us | VGPR | AccVGPR | VALU | SALU | VMEM | LDS inst | LDS bytes |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| L3 -> L4 | +0.240 | 0 | 0 | +8192 | +192 | +1024 | 0 | 0 |
| L4 -> L5 | +24.196 | +64 | -56 | +82752 | +5120 | +15360 | +8768 | +4096 |
| L5 -> L6 | +1.602 | 0 | +192 | -7872 | -4224 | -7680 | +2496 | +16384 |

Interpretation:

- `L4 -> L5` is the first major runtime/source-size jump.  The dominant increases are scalar/address-generation work (`VALU +82752`, `SALU +5120`), global memory (`VMEM +15360`), and LDS (`LDS +8768`) from runtime K staging and packed view access.
- `L5 -> L6` is the largest AccVGPR jump (`+192`) but not the largest trace jump.  The first dependent update MFMA introduces MFMA16 accumulator state and larger LDS allocation/resource pressure.

## Static ISA Counts

| level | mfma32 | mfma16 | global_load | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_lshl | v_lshl_add | v_or | v_bfe |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4 | 8 | 0 | 80 | 16 | 80 | 4 | 83 | 199 | 327 | 201 | 115 | 72 |
| L5 | 16 | 0 | 140 | 44 | 144 | 9 | 145 | 321 | 507 | 277 | 155 | 112 |
| L6 | 16 | 8 | 256 | 64 | 272 | 11 | 280 | 281 | 801 | 424 | 213 | 113 |

Static growth:

- `L4 -> L5`: global loads +60, ds_read +28, ds_write +64, barriers +5, waitcnt +62, v_add +122, v_lshl +180, v_lshl_add +76, v_or +40, v_bfe +40.
- `L5 -> L6`: mfma16 +8, global loads +116, ds_read +20, ds_write +128, waitcnt +135, v_lshl +294, v_lshl_add +147, v_or +58.

## Source Pattern Attribution

The high-level source pattern most associated with `L4 -> L5` is:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
```

This combines:

- A large shared allocation with transposed logical indexing.
- Runtime K load and shared-store transposition.
- A packed i32 view over the transposed shared tensor.
- Later use as MFMA16 B operand.

The exact dynamic cost cannot be assigned to only `k_all_t` or only `kall_vec` without the new L5 variants.  Existing evidence says the whole K staging/shared-view pattern is expensive.

## Focused Variant Follow-Up

Focused files were added and profiled:

- `repro_qwen_mfma32_l5_kstaging_variants.py`
- `profile_qwen_mfma32_l5_kstaging_variants.py`
- `repro_qwen_mfma32_l6_update_pressure_variants.py`
- `profile_qwen_mfma32_l6_update_pressure_variants.py`

Key follow-up results:

- `L5_direct_global_k_update_probe`: `13.740 us`, diagnostic upper bound.
- `L5_khalf_stage_64`: `23.635 us`, best finite production-relevant staging reduction.
- `L5_subtile_k_stage_16token`: `32.769 us`, modest finite staging reduction.
- `L6_one_update_mfma_only`: AccVGPR `320`, showing one update branch already causes most pressure.
- `L6_update_acc_scope_split_source`: AccVGPR remains `264`, so simple source barriers do not solve update pressure.

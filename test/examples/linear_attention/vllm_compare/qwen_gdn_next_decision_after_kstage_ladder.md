# Qwen GDN Next Decision After K-Stage Ladder

## Summary

The existing Qwen-shaped lowering ladder identified the first major runtime jump at `L4 -> L5`, where the source adds runtime K staging:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]
```

The next transition, `L5 -> L6`, adds the first dependent update MFMA and is the largest AccVGPR jump.

## Current Answers

### Is `L4 -> L5` caused by K staging layout or `kall_vec` view?

Current evidence attributes the cost to the combined K staging/shared-view pattern, not yet to one subcomponent.

Existing measured delta:

- trace `+24.196 us`
- VGPR `+64`
- VALU `+82752`
- VMEM `+15360`
- LDS inst `+8768`

The source pattern includes both runtime transposed shared staging and a packed `kall_vec` view.  The new L5 variants split those factors enough to show that full `k_all_t[128,BT]` staging is the expensive part; simply removing `kall_vec` via token-major staging does not improve trace.

### Did any K staging variant improve trace/counters?

Yes.

| variant | trace_us | speedup vs baseline_L5 | finite | note |
|:---|---:|---:|:---:|:---|
| `baseline_L5` | `36.614` | `1.000x` | yes | current full transposed K staging |
| `L5_khalf_stage_64` | `23.635` | `1.549x` | yes | best production-relevant staging reduction |
| `L5_subtile_k_stage_16token` | `32.769` | `1.117x` | yes | modest but valid staging reduction |
| `L5_direct_global_k_update_probe` | `13.740` | `2.665x` | yes | diagnostic upper bound, not production |
| `L5_packed_i32_contiguous_load` | `40.861` | `0.896x` | no | invalid/non-finite |

This strongly supports that the L4->L5 cost is the full transposed K staging/shared-view pattern, especially full `k_all_t[128,BT]` staging.

### Is `L5 -> L6` AccVGPR growth expected or avoidable?

Existing ladder data:

- `L5 -> L6` AccVGPR `+192`
- trace only `+1.602 us`
- MFMA `+512`
- LDS block `+16384`

The new L6 variants show the jump happens immediately:

| variant | trace_us | AccVGPR | MFMA |
|:---|---:|---:|---:|
| `baseline_L5_no_update` | `36.454` | 144 | 1024 |
| `L6_one_update_mfma_only` | `31.727` | 320 | 1152 |
| `L6_one_ktile_update` | `40.140` | 336 | 1536 |
| `L6_full_update_like_current` | `33.971` | 264 | 5120 |
| `L6_update_acc_scope_split_source` | `34.131` | 264 | 5120 |
| `L6_update_acc_reinit_variant` | `33.930` | 264 | 5120 |

This is not linear tile-count growth.  One dependent update MFMA branch already creates most of the pressure.  Simple source barriers/reinit do not reduce it.

### Is there a small compiler/helper fix?

Not as a compiler patch.

Static compiler/source inspection found:

- `make_shared` lowers to static workgroup alloca plus row-major/default layout machinery.
- `view(memref,dtype,layout)` lowers through general memref view/reinterpret mechanics.
- Packed BF16/i32 support exists, but scalar/vector fragment shape must be explicit.
- No obvious tiny, safe canonicalization was found.
- The measured win points more directly to source/helper staging granularity: K-half/subtile staging, not full `k_all_t[128,BT]`.

### Should we return to full Qwen?

No, not directly from this pass.

The isolated trace gate is technically passed by `L5_khalf_stage_64` and `L5_subtile_k_stage_16token`, but these are still diagnostic L5 staging variants, not full update correctness-equivalent kernels.

So the next step should still be isolated:

1. Build a combined `L6_khalf_update` or `L6_subtile_update` variant.
2. Verify the update sink/correctness trend.
3. Only then copy the source pattern into an experimental full Qwen file.

## Exact Next Action

Build one more isolated ladder experiment:

```text
L6_khalf_update
L6_subtile_update
```

Goal:

- Preserve the real update MFMA path.
- Avoid full `k_all_t[128,BT]`.
- Check whether the L5 K-half/subtile win survives once update MFMA is present.
- Do not touch full Qwen until that passes.

## Remaining Compiler Limitation

The remaining limitation is not yet a proven compiler bug.  The measured issue is that full transposed shared K staging plus packed BF16/i32 view access is much heavier than smaller/direct K staging forms for the fixed Qwen update layout.

Exact unresolved question:

Can we express a correctness-equivalent update path with K-half/subtile staging that keeps the L5 trace benefit after the first dependent update MFMA?

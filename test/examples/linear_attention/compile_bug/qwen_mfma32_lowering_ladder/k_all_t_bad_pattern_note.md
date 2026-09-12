# K-All-T Bad Pattern Note

## Source Pattern

The current broad K staging pattern appears in both the Qwen-shaped L6 repro and the v29 fused full chunk_gdr candidate:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(
    k_all_t,
    al.i32,
    al.make_layout((128, 8, 4), (8 * 4, 4, 1)),
)

for rep_k in al.range(64):
    linear_k = tid + rep_k * WORKGROUP
    kk = linear_k // BT
    tok_k = linear_k - kk * BT
    k_all_t[kk, tok_k] = k[0, tok_k, key_head_idx, kk]

...

b_words = kall_vec[tile * 16 + lane_col, pack_base]
```

The update then consumes this broad packed shared view in the MFMA16 update path.

## Why It Is Expensive

The pattern materializes the full transposed `K[128,BT]` tile even though each update MFMA tile consumes a much narrower token/K subtile.  In the Qwen-shaped L6 repro, this broad staging/view shape keeps more shared-memory data and address-generation state live while pred/v_decay values feed the update MFMA path.

Measured L6 baseline:

| variant | trace_us | VGPR | AccVGPR | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_current_update` | `34.371` | 128 | 264 | 45056 | 5120 | 182144 | 22528 | 28672 |

Validated K-subtile staging:

| variant | trace_us | VGPR | AccVGPR | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_subtile16_stage_full_update_like` | `19.189` | 96 | 168 | 32768 | 5120 | 120128 | 16384 | 21504 |

Delta:

- trace: `-15.182 us`
- AccVGPR: `-96`
- VGPR: `-32`
- VALU: `-62016`
- VMEM: `-6144`
- LDS inst: `-7168`

## Narrow Replacement

Use a K-subtile staging pattern:

```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))

for rep_sub in al.range(16):
    linear_sub = tid + rep_sub * WORKGROUP
    kk_sub = linear_sub // 16
    tok_local = linear_sub - kk_sub * 16
    tok_sub = token_base + tok_local
    k_sub_t[kk_sub, tok_local] = k[0, tok_sub, key_head_idx, kk_sub]

...

b_words = ksub_vec[tile * 16 + lane_col, pack_id]
```

This is a source-level helper/pattern opportunity rather than a proven compiler bug.

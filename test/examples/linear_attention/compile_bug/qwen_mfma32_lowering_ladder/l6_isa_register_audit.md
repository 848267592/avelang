# L6 ISA/Register Audit

## Scope

This audit uses the existing L6 K-stage/update repro.  No new kernels were added.

Artifact directory:

```text
test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/
```

Key aliases:

| audit name | existing variant |
|:---|:---|
| `L6_baseline_no_lifetime` | `L6_baseline_current_update` |
| `L6_subtile_no_lifetime` | `L6_subtile16_stage_full_update_like` |
| `L6_update_mfma_no_pred_dependency` | `L6_update_mfma_no_pred_dependency` |
| `L6_update_mfma_minimal_frag` | `L6_update_mfma_minimal_frag` |

Per-pass IR dumps were not available through the current Python JIT profiling harness.  The audit therefore uses rocprof metadata, HSACO, AMDGPU ISA, and MFMA-local ISA snippets.

## Rocprof Counters

| variant | trace_us | WG | grid | VGPR | AccVGPR | SGPR | scratch | LDS block | MFMA | VALU | SALU | VMEM | LDS inst | occupancy |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 34.331 | 128 | 4096 | 128 | 264 | 112 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.5033 |
| subtile | 19.109 | 128 | 4096 | 96 | 168 | 112 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.4206 |
| no_pred_dependency | 9.774 | 128 | 4096 | 96 | 80 | 112 | 0 | 20480 | 2048 | 69504 | 4928 | 8192 | 7168 | 0.3290 |
| minimal_frag | 3.205 | 128 | 4096 | 12 | 4 | 16 | 0 | 1024 | 256 | 4096 | 1344 | 2432 | 512 | 0.1228 |

## Static ISA Counts

| variant | mfma32 | mfma16 | global_load | global_store | ds_read | ds_write | waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max VGPR idx | max ACC idx | acc write/read |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| baseline | 8 | 32 | 144 | 64 | 88 | 152 | 180 | 192 | 72 | 722 | 401 | 166 | 73 | 255 | 131 | 155 / 155 |
| subtile | 8 | 32 | 96 | 64 | 80 | 104 | 131 | 208 | 73 | 410 | 242 | 116 | 73 | 223 | 15 | 32 / 32 |
| no_pred_dependency | 0 | 32 | 64 | 64 | 32 | 80 | 107 | 80 | 16 | 406 | 194 | 79 | 17 | 93 | 3 | 32 / 32 |
| minimal_frag | 0 | 4 | 2 | 36 | 4 | 4 | 16 | 13 | 2 | 14 | 4 | 3 | 3 | 11 | 3 | 4 / 4 |

## MFMA Region Snippets

Snippet files:

```text
test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/mfma_snippets/
```

Best-effort local maxima inside +/-30 ISA lines around each MFMA:

| variant | region | MFMA count | local max VGPR idx | local max ACC idx |
|:---|:---|---:|---:|---:|
| baseline | `mfma32_pred` | 8 | 251 | 50 |
| baseline | `mfma16_update` | 32 | 249 | 48 |
| subtile | `mfma32_pred` | 8 | 221 | 15 |
| subtile | `mfma16_update` | 32 | 219 | 3 |
| no_pred_dependency | `mfma16_update` | 32 | 93 | 3 |
| minimal_frag | `mfma16_update` | 4 | 11 | 3 |

The global static max ACC index in baseline is higher than the MFMA-local windows.  It comes from a pre-MFMA address/staging region with many `v_accvgpr_write_b32 a100..a131` instructions, not from a single `v_mfma` instruction.

Representative baseline evidence:

```asm
v_lshl_add_u64 v[18:19], s[12:13], 0, v[18:19]
v_sub_u32_e32 v20, v41, v12
v_lshl_add_u64 v[12:13], v[18:19], 0, v[2:3]
v_accvgpr_write_b32 a131, v13
v_accvgpr_write_b32 a130, v12
```

The subtile variant does not show the same high-index AGPR address/staging block; its static max ACC index is only 15 while keeping the same dynamic MFMA count as baseline.

## Main ISA Observation

The high baseline AccVGPR is not explained by the MFMA16 update intrinsic alone:

- `minimal_frag` uses update MFMA and has AccVGPR `4`.
- `no_pred_dependency` keeps a fuller update path and has AccVGPR `80`.
- baseline has AccVGPR `264` and 155 explicit AGPR write/read pairs.

The major structural difference between baseline and subtile is the broad `k_all_t[128,BT]` / `kall_vec` staging/view pattern.  With the same dynamic MFMA count, subtile removes a large amount of address and LDS work:

- trace `34.331 -> 19.109 us`
- AccVGPR `264 -> 168`
- VGPR `128 -> 96`
- LDS block `45056 -> 32768`
- VALU `182144 -> 120128`
- VMEM `22528 -> 16384`
- LDS inst `28672 -> 21504`
- static `v_lshl` `722 -> 410`
- static `v_lshl_add` `401 -> 242`
- explicit AGPR write/read pairs `155/155 -> 32/32`

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_lowering_artifacts.py \
  --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```

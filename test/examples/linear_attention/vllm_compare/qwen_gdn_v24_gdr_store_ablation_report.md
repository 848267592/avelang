# Qwen GDN v24 GDR Store Ablation Report

## Summary

This is a chunk_gdr-only upper-bound experiment.  It does not claim full-pipeline correctness for variants that omit `h` or `vn`, because chunk_o consumes both.

Removing global `h` and `vn` stores helps, but the upper bound is not large enough to close the vLLM gap by itself.

At T=2048:

- baseline copy: `0.359914 ms`
- no_h_no_vn_store: `0.314488 ms`
- speedup: `1.1444x`
- improvement: about `12.6%`

At T=4096:

- baseline copy: `0.664226 ms`
- no_h_no_vn_store: `0.569146 ms`
- speedup: `1.1671x`
- improvement: about `14.3%`

## Files

- `qwen_gdn_chunked_avelang_v24_gdr_store_ablation.py`
- `bench_qwen_gdn_v24_gdr_store_ablation.py`

## Variants

`baseline_copy`:

Same h/vn store behavior as v23 chunk_gdr.

`no_vn_store`:

Skips only global `vn` stores.

`no_h_store`:

Skips only global `h` stores.

`no_h_no_vn_store`:

Skips both global `h` and global `vn` stores.

## Benchmark

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 bench_qwen_gdn_v24_gdr_store_ablation.py \
  --T 512 1024 2048 4096 --warmup 5 --repeat 20
```

| T | baseline | no_vn_store | no_h_store | no_h_no_vn_store | best speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | `0.131876` | `0.124786` | `0.123483` | `0.108300` | `1.2177x` |
| 1024 | `0.213958` | `0.198615` | `0.203462` | `0.188079` | `1.1376x` |
| 2048 | `0.359914` | `0.343710` | `0.349179` | `0.314488` | `1.1444x` |
| 4096 | `0.664226` | `0.607262` | `0.622905` | `0.569146` | `1.1671x` |

## rocprof

Command pattern:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex store_ablation \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_v24_store_ablation_baseline \
  -o v24_store_ablation_baseline_counters \
  -f csv \
  -- python3 test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v24_gdr_store_ablation.py \
       --T 2048 --variant baseline_copy --warmup 2 --repeat 5
```

| metric | baseline_copy | no_h_no_vn_store |
|:---|---:|---:|
| trace median us | `331.012` | `279.495` |
| Workgroup_Size | `256` | `256` |
| Grid_Size | `16384` | `16384` |
| LDS_Block_Size | `25088` | `25088` |
| Scratch_Size | `0` | `0` |
| VGPR_Count | `56` | `16` |
| Accum_VGPR_Count | `136` | `160` |
| SGPR_Count | `112` | `112` |
| SQ_INSTS_MFMA | `327680` | `327680` |
| SQ_INSTS_VALU | `5866240` | `5264128` |
| SQ_INSTS_SALU | `700416` | `699648` |
| SQ_INSTS_VMEM | `921600` | `593920` |
| SQ_INSTS_LDS | `2396160` | `2134016` |
| OccupancyPercent | `2.525705` | `2.532218` |

## Interpretation

VMEM drops substantially, but trace improves by only about `15.6%` in the profiled T=2048 run.  MFMA count is unchanged, and LDS/VALU remain large.  This matches the benchmark result: h/vn stores are real cost, but not the dominant remaining cost.

Using the requested interpretation rule, `no_h_no_vn_store` improves less than 20% at T=2048 and T=4096.  Therefore v24 store removal alone cannot close the long-sequence vLLM gap.

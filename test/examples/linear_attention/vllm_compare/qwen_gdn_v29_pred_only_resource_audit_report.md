
# Qwen GDN v29 Pred-Only MFMA32 Resource Audit Report

## Summary

v29 MFMA32 pred-only has been fully tested in three forms:

1. k-split v29 pred-only
2. token-split no-reduce v29 pred-only
3. token-split stage-all v29 pred-only

All versions pass BF16-level correctness against the torch reference and generate the intended 32x32 BF16 MFMA path. However, all three are slower than the existing v28/v24-style pred/chunk_gdr direction.

Therefore, v29 MFMA32 pred-only is a no-go for now. Do not implement v29_update_only.

## Results

| Variant | T=2048 latency | Main issue |
|---|---:|---|
| k-split v29 pred-only | 0.482717 ms | AccVGPR=204, high VALU/SALU |
| token-split v29 pred-only | 0.524720 ms | VGPR=128, AccVGPR=168 |
| token-split stage-all v29 pred-only | 0.626070 ms | AccVGPR=216, VALU=4.99M, SALU=345K |

## Stage-all rocprof

| Metric | Value |
|---|---:|
| trace avg | 626.386 us |
| Workgroup_Size | 128 |
| Grid_Size | 4096 |
| LDS_Block_Size | 24576 |
| Scratch_Size | 0 |
| VGPR_Count | 48 |
| Accum_VGPR_Count | 216 |
| SGPR_Count | 112 |
| SQ_INSTS_MFMA | 32768 |
| SQ_INSTS_VALU | 4992320 |
| SQ_INSTS_SALU | 345152 |
| SQ_INSTS_VMEM | 198656 |
| SQ_INSTS_LDS | 165888 |

## Diagnosis

The original hypothesis was that k-split v29 was slow mainly because of cross-wave LDS reduction. The token-split version removed cross-wave reduction but became slower. Therefore, cross-wave reduction is not the dominant bottleneck.

The stage-all version shortened ordinary VGPR live range and reduced VGPR count from 128 to 48, but AccVGPR increased to 216 and VALU/SALU exploded. Therefore, the deeper issue is the source-level 32x32 accumulator path itself: accumulator unpack, high-level address generation, and Avelang lowering around the 32x32 fragment are too expensive.

## Decision

Do not implement v29_update_only.

Do not continue full_v29 based on this MFMA32 pred-only path.

Keep v29 as evidence that source-level MFMA32 works, but current high-level schedule/lowering is not performance-viable.

## Next Direction

Return to the v24 production baseline.

Recommended next optimization directions:

1. chunk_o optimization
2. w_u optimization
3. targeted raw_buffer_store_x4/vectorized store integration in existing v24/v28-style kernels
4. small stage-level optimizations rather than another BT64/BV32 rewrite


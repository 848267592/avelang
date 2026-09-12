# Full v29 Late Scalar-Address Generation A/B

## 实验目的

本实验只改变 compact `[128,64]` K producer 的 source-K 全局地址何时具现化。A 在 producer rewrite 中直接创建标量 `memref.load`；B 先保留 `amdgpu_qwen_kfrag_stage_load`，在 GPU outlining 后、紧邻原有 LDS store 时展开为同一标量 BF16 `memref.load`。

四个 update MFMA16 B-fragment consumer、K tile shape、8192 个 source-K global load、8192 个 LDS store、MFMA schedule、barrier、launch `(32,1,1)/(128,1,1)`、数学和输入完全冻结。`AVELANG_QWEN_KFRAG_LATE_BLOAD=0` 固定，故这不是旧 terminal B-load 实验。

## 分叉完整性

- pre-branch MLIR 是否相同：`True`。
- A/B post-rewrite stage-op 计数：`0` / `1`。
- A/B post-late-lowering 残留 stage-op：`0` / `0`。
- B 的 late-address 标记计数：`1`；A 为 `0`。
- post-opt LLVM 是否相同：`True`；规范化 ISA 是否相同：`True`。

这四项共同区分两种情况：若 stage op 根本未穿过 outlining，这是实现失败；若它按预期穿过但 post-opt LLVM/ISA 又收敛，则是后端把 timing 差异消除了，而不是 A/B 漏跑。

## 语义与工作量 gate

- full `h` bit-exact：`True`，max abs `0.0`。
- full final-state bit-exact：`True`，max abs `0.0`。
- persistent rewrite fired：A `True`，B `True`。

| ISA static metric | A early address | B late address |
|:--|--:|--:|
| mfma16 | 128 | 128 |
| mfma32 | 16 | 16 |
| ds_read_b64 | 256 | 256 |
| ds_write | 243 | 243 |
| global_or_buffer_load | 195 | 195 |
| barrier | 17 | 17 |
| accvgpr_write | 274 | 274 |
| high_agpr_write_ge100 | 156 | 156 |
| max_explicit_agpr_write | 255 | 255 |

## Post-greedy MIR 地址压力与 spill

这里的词数是 MIR 虚寄存器从首次定义到最后文本使用的静态 liveness proxy，不是 LLVM LiveIntervals；`av_*` 是 flexible AV class，不能直接等同于物理 AGPR。它仍能机械化比较同一高层图下的地址链是否在 pred 区间保持活跃。

| metric | A early address | B late address | B-A |
|:--|--:|--:|--:|
| pred peak flexible-AV words | 178 | 178 | 0 |
| pred peak address-marked flexible-AV words | 144 | 144 | 0 |
| AV spill words | 190 | 190 | 0 |
| first AV spill | {'mir_line': 247, 'byte_offset': 2952, 'vreg': '%6842', 'register_class': 'vreg_64_align2', 'spill_words': 2, 'phase': 'prelude', 'def_mir_line': 240, 'last_mir_line': 248, 'lexical_span': 8, 'definition': 'undef %6842.sub0:vreg_64_align2 = nsw V_SUB_U32_e32 %907:vgpr_32, %911:vgpr_32, implicit $exec', 'flags': ['address'], 'crosses_pred_mfma32': False, 'crosses_update_mfma16': False} | {'mir_line': 247, 'byte_offset': 2952, 'vreg': '%6842', 'register_class': 'vreg_64_align2', 'spill_words': 2, 'phase': 'prelude', 'def_mir_line': 240, 'last_mir_line': 248, 'lexical_span': 8, 'definition': 'undef %6842.sub0:vreg_64_align2 = nsw V_SUB_U32_e32 %907:vgpr_32, %911:vgpr_32, implicit $exec', 'flags': ['address'], 'crosses_pred_mfma32': False, 'crosses_update_mfma16': False} | - |

## T=2048 rocprof

| metric | A early address | B late address | B-A |
|:--|--:|--:|--:|
| trace_median_us | 1300.473 | 1305.8609999999999 | 5.388 |
| VGPR_Count | 128 | 128 | 0 |
| Accum_VGPR_Count | 384 | 384 | 0 |
| SGPR_Count | 112 | 112 | 0 |
| Scratch_Size | 736 | 736 | 0 |
| LDS_Block_Size | 61440 | 61440 | 0 |
| SQ_INSTS_MFMA | 294912.0 | 294912.0 | 0 |
| SQ_INSTS_VALU | 3180992.0 | 3180992.0 | 0 |
| SQ_INSTS_SALU | 567808.0 | 567808.0 | 0 |
| SQ_INSTS_VMEM | 601984.0 | 601984.0 | 0 |
| SQ_INSTS_LDS | 1242304.0 | 1242304.0 | 0 |
| OccupancyPercent | 0.646141662 | 0.645893202 | -0.00024846 |

## 判定

**No-Go：dedicated stage op 已按约定穿过 GPU outlining 并在 late pass 展开，但后续优化将 A/B 收敛为相同 post-opt LLVM 和 ISA。** 因而没有 retained lowering difference 可以压低 144 个地址 words 或 spill；任何微小 timing 波动都不可解释为编译器收益。

## 复现

```bash
PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_full_chunk_gdr_late_address.py \
  --T 2048 --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```

Raw A/B evidence is under `/workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_full_chunk_gdr_late_address`. The experiment does not resolve the existing original-v29 nonzero-W-vs-reference recurrence issue; it tests bit-exact preservation relative to the same full-v29 implementation only.

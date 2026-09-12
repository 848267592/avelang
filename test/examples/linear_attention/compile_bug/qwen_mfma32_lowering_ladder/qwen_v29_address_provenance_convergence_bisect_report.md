# Qwen v29 Address Provenance And Pipeline Convergence Bisect

## 范围

Experiment 0.5 是纯审计：没有修改 Qwen kernel、tile、workgroup、launch、数学、dtype、barrier 或性能路径。它重跑同一 full-v29 `chunk_gdr` source 的 A/B：A 在 rewrite 中直接创建 source-K 标量 load；B 使用既有 `amdgpu_qwen_kfrag_stage_load`，在 GPU outlining 后展开。唯一目的，是定位 A/B 首次收敛层次，并给 pred 峰值的地址 vreg 建立 machine-level consumer provenance。

## 冻结 gate

- pre-branch SHA 相同：`True`。
- persistent rewrite fired：A `True`，B `True`。
- B stage op 创建：`True`。
- `h` / final-state bit-exact：`True` / `True`。

## 审计 Hook

- `lib/Target/GPU/lower_to_llvm.cc`：仅当 `AVELANG_QWEN_KFRAG_CONVERGENCE_AUDIT=1` 时，保存 outlining 前后、late lowering 前后和 LLVM module-pass 快照。
- `lib/Target/AMDGPU/gpu_to_amdgpu_pipeline.cc`：同一 guard 下在 common/GPU AMDGPU lowering pipeline 各边界打印 MLIR。
- 两处 hook 都只写文件；不改 rewrite 条件、kernel IR、pass 顺序、launch 或 codegen 选项。

## A/B 收敛表

| 层次 | A/B 相同 | stage op A/B | late attr A/B | memref.load A/B | GEP A/B |
|:--|:--|:--|:--|:--|:--|
| pre_kfrag_branch | True | 0/0 | 0/0 | 168/168 | 0/0 |
| post_kfrag_rewrite | False | 0/1 | 0/0 | 163/162 | 0/0 |
| post_gpu_outlining | False | 0/1 | 0/0 | 150/149 | 0/0 |
| post_outline_cleanup | False | 0/1 | 0/0 | 150/149 | 0/0 |
| post_kfrag_load_lowering | False | 0/0 | 0/1 | 150/150 | 0/0 |
| post_late_lowering | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_00_pre_common | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_01_post_one_shot_bufferize | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_02_post_expand_strided_metadata | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_03_post_scf_to_cf | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_04_post_common_cleanup | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_10_pre_gpu_pipeline | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_11_post_rocdl_attach | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_12_post_lower_math | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_13_post_legalize_shuffle | False | 0/0 | 0/1 | 150/150 | 0/0 |
| amdgpu_14_post_gpu_to_rocdl | True | 0/0 | 0/0 | 0/0 | 325/325 |
| amdgpu_15_post_amdgpu_to_rocdl | True | 0/0 | 0/0 | 0/0 | 325/325 |
| amdgpu_16_post_vector_to_llvm | True | 0/0 | 0/0 | 0/0 | 39/39 |
| amdgpu_17_post_gpu_cleanup | True | 0/0 | 0/0 | 0/0 | 38/38 |
| amdgpu_18_post_scalar_to_llvm | True | 0/0 | 0/0 | 0/0 | 38/38 |
| amdgpu_19_post_gpu_pipeline | True | 0/0 | 0/0 | 0/0 | 38/38 |
| post_amdgpu_mlir_pipeline | True | 0/0 | 0/0 | 0/0 | 38/38 |
| final_mlir | True | 0/0 | 0/0 | 0/0 | 38/38 |
| preopt_llvm | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_0_Annotation2MetadataPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_10_GlobalOptPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_11_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 76/76 |
| llvm_pass_12_RequireAnalysisPass_GlobalsAA__Module_ | True | 0/0 | 0/0 | 0/0 | 76/76 |
| llvm_pass_13_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 76/76 |
| llvm_pass_14_RequireAnalysisPass_ProfileSummaryAnalysis__Module_ | True | 0/0 | 0/0 | 0/0 | 76/76 |
| llvm_pass_15_ModuleToPostOrderCGSCCPassAdaptor | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_16_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_17_ModuleInlinerWrapperPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_18_DeadArgumentEliminationPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_19_CoroCleanupPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_1_ForceFunctionAttrsPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_20_GlobalOptPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_21_GlobalDCEPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_22_EliminateAvailableExternallyPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_23_ReversePostOrderFunctionAttrsPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_24_RecomputeGlobalsAAPass | True | 0/0 | 0/0 | 0/0 | 1014/1014 |
| llvm_pass_25_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_26_AMDGPUAttributorPass | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_27_GlobalDCEPass | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_28_ConstantMergePass | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_29_CGProfilePass | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_2_AMDGPUExpandFeaturePredicatesPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_30_RelLookupTableConverterPass | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_31_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 1096/1096 |
| llvm_pass_3_InferFunctionAttrsPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_4_CoroEarlyPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_5_ModuleToFunctionPassAdaptor | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_6_OpenMPOptPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_7_AMDGPUPrintfRuntimeBindingPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_8_IPSCCPPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| llvm_pass_9_CalledValuePropagationPass | True | 0/0 | 0/0 | 0/0 | 32/32 |
| postopt_llvm | True | 0/0 | 0/0 | 0/0 | 1096/1096 |

**第一处 divergence 之后重新相同的快照：`amdgpu_14_post_gpu_to_rocdl`。** 这不是基于猜测：表中每一行保留 SHA、结构计数与首段 unified diff，原始快照在 artifacts 目录。

## LLVM LTO 阶段

| LTO bitcode | bitcode 相同 | llvm-dis 原文相同 | 去 ModuleID 后相同 |
|:--|:--|:--|:--|
| linked.hsaco.0.0.preopt.bc | True | False | True |
| linked.hsaco.0.2.internalize.bc | True | False | True |
| linked.hsaco.0.4.opt.bc | True | False | True |
| linked.hsaco.0.5.precodegen.bc | True | False | True |

`preopt_llvm` 已在前表中出现；LTO 表将 `.preopt.bc`、`.internalize.bc`、`.opt.bc` 与 `.precodegen.bc` 单独比较。原文 llvm-dis 的唯一差异是输出路径写入的 `ModuleID`；bitcode SHA 与仅删除该展示行后的 LLVM IR 均相同。

## 最终 ISA

- raw objdump SHA 相同：`False`（输入 hsaco 路径使原文首行不同属正常）。
- 去除 objdump `file format` 输入路径行后 ISA SHA 相同：`True`。

## MIR A/B Identity

| MIR 阶段 | SHA 相同 |
|:--|:--|
| pre-greedy MIR | True |
| post-greedy MIR | True |

## Pre-greedy MIR 原始地址压力

这里优先使用最终 spill-producing greedy run 之前的 `IR Dump Before Greedy Register Allocator`，避免把 Greedy 新增的 spill save/reload、COPY 与物理寄存器改写当成原始压力来源。pre-greedy 中还没有历史报告里的 `av_*` 144 words；它们仍表现为 `vreg_64`/`vgpr` 地址候选。因此本表统计所有 register class 的 active address provenance candidates。数值仍是 MIR 文本 def/use 的静态代理，不是 LLVM LiveIntervals。

| metric | A early | B late |
|:--|--:|--:|
| pred peak flexible-AV words | 0 | 0 |
| pred peak all address words | 326 | 326 |
| selected address candidates | 326 | 326 |
| peak MIR line | 2576 | 2576 |

### Pre-greedy Consumer Provenance

`pre_greedy_address_objects.csv` 保存每个对象的定义、跨度与 bounded def-use 终止 memory consumer。只有 `k_producer + GLOBAL_LOAD` 才标为 source-K global-load；只有 `k_producer + DS_WRITE` 才标为 source-K LDS-store。若同一地址同时服务 source-K 与其他 consumer，会明确记为 multi-consumer，不会被强行计成单一来源。

| bucket | A words | B words |
|:--|--:|--:|
| multi-consumer (includes source-K) | 42 | 42 |
| pred LDS-read address | 17 | 17 |
| source-K LDS-store address | 1 | 1 |
| source-K global-load address | 128 | 128 |
| state/global-output or non-K memory address | 138 | 138 |

## Post-greedy 历史 144-word 精确归属

历史的 144 words 来自 post-greedy pred-MFMA32 peak：72 个 `av_64` candidate，每个 2 words。它们不是 pre-greedy 的 register class；本节只用于把那个历史数字精确连接到最终 machine consumer。`post_greedy_flexible_av_address_objects.csv` 保存 72 个逐对象记录。

| metric | A early | B late |
|:--|--:|--:|
| pred peak flexible-AV words | 178 | 178 |
| selected flexible-AV address words | 144 | 144 |
| peak MIR line | 3047 | 3047 |

| bucket | A words | B words |
|:--|--:|--:|
| multi-consumer (includes source-K) | 16 | 16 |
| source-K global-load address | 128 | 128 |

## 结论边界

本审计只回答“分叉在哪一层消失”和“pred 峰值对象到哪些 machine memory consumer 可达”。它不把 lexical span 说成硬件物理压力，不会因没有 LTO debug location 而虚构精确 AveLang source 行。

本次 A/B 的差异在 `amdgpu_14_post_gpu_to_rocdl` 首次消失，故普通 stage op 的控制点不足以跨过 `ConvertGpuOpsToROCDLOps`。post-greedy 的历史 144 words 中，`128` words 只终止于 source-K global load，另 `16` words 同时终止于 pred 期 global load 和随后 source-K global load；因此 `144/144` 都可到达 source-K consumer，但并非 `144/144` 都是纯 source-K 地址。这个结果证明 144-word **后 RA 表象**的 consumer 归属，却不证明“延迟一个 source-K scalar load 就能解决整个 full live set”。若继续研究，应控制 pre-greedy 中更宽的 producer/index/address recipe，并把表示保留到这个已定位的转换边界；不能再把普通晚展开 stage op 当成有效控制杆。

## 复现

```bash
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 --target _avelang_bindings -j 16

PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_v29_address_provenance_convergence.py \
  --T 2048 --seed 20260724
```

全部 raw MLIR/LLVM/bitcode/MIR/CSV/JSON 位于 `/workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_v29_address_provenance_convergence_bisect`。

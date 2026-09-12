# Qwen gfx942 C14-CG：Static Physical Encoding Codegen MVP

## 结论

本轮完成 C14-CG Static Physical Encoding -> gfx942 Codegen MVP，不是新的 Qwen
chunk-o 性能版本。C13 的 typed distributed/shared/MFMA/dot/transform attribute
和 ChunkOPhysicalPlan 已经第一次进入 target-specific block-dot lowering：

    typed attrs + plan -> static shared offset
        -> addrspace(3) packed BF16 load
        -> existing gfx942 MFMA32
        -> LLVM -> MIR -> ISA -> HSACO

状态为 C14_CODEGEN_MVP_GO_FOR_FULL_CHUNKO。这里的 GO 只表示允许下一阶段接入
full chunk-o 做数值和性能验证，不表示 full kernel 或 production 晋级。Q/H/K/V 四个 synthetic path 通过真实
Avelang AMDGPU pipeline；V 的 native #linear1 GF(2) basis 通过 4096-slot
bijection；最终形成 v_mfma_f32_32x32x8_bf16。synthetic ISA 没有 ds_bpermute、
除法或取模指令，HSACO 也没有 private segment 或 spill。

这不是性能结论：本轮没有 launch GPU 数值 kernel，没有接入真实 chunk-o，没有跑
PMC、body benchmark 或 Eager public API。

## Preflight

在用户指定的 ljd_qwen_vllm_avelang_rocm722 容器中，Docker full build 通过，
CTest 7/7 通过，C13 static layout/attr/pass-survival 通过，C14 synthetic codegen
通过。本轮没有修改 production selector/dispatch、Qwen full kernel、allocator/RA、
MFMA geometry 或 recurrence。

## Native recipe

selected native chunk-o 工件：
[chunk_fwd_kernel_o.ttgir](./codex_qwen_bt64_stage6z_native_chunko/native_refresh/T16384/triton_cache/2JPYRY6UHK6PQKKKA6ZSRIM3CHXDMZEFNVYACQGONYU2YTYOOHNQ/chunk_fwd_kernel_o.ttgir)

其 literal recipe 是：

    #blocked: sizePerThread=[4,8], threadsPerWarp=[8,8],
             warpsPerCTA=[2,1], order=[1,0]
    #linear1: register=[[1,0],[2,0],[0,1],[0,2],[0,4]]
              lane=[[0,8],[0,16],[0,32],[4,0],[8,0],[16,0]]
              warp=[[32,0]]
    #mma: version=3, warpsPerCTA=[1,2], instrShape=[32,32,8],
          transposed=true
    #shared3: amd_rotating_shared, vec=4, perPhase=1,
              maxPhase=16, order=[0,1]

旧 C13 工厂 V 是 [2,8] x [4,1]；本轮已修正为 native [4,8] x [2,1]。
registerStride=4 将 [r0,r1] 展平为 r0+4*r1，再按 literal GF(2) basis 做 XOR。
新增回归枚举所有 wave/lane/register 坐标，覆盖 [64,64] V tile，无碰撞无遗漏。

current recurrence 的对照工件为：
[chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ttgir](./codex_triton_hsaco_reuse_audit/triton_cache_exact/52F75CUYDCKXZWREEMMJIHXGC37GWPACH6ZGZA7K7A7PJEMLESJQ/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.ttgir)

它实际给出 blocked/blocked1/blocked2/blocked3、mma1 32x32x8 transposed、
rotating vec=4/perPhase=1/maxPhase=16 和 swizzled vec=2/4。selected chunk-o
旋转 V 叫 #shared3；早期文档中的 #shared4 是逻辑别名而非 literal，C14 记录
unknown，不把文档名称冒充 native recipe。shared 公式来自 Triton AMD：

    swizzled: phase=(row/perPhase)%maxPhase
              inner=(vec*phase)%numCols
    rotating: phase=(row/perPhase)%maxPhase
              blockNo=(row/maxPhase/perPhase)%maxPhase
              combinedPhase=phase XOR blockNo
              inner=(vec*combinedPhase)%numCols

当前参数为 2 的幂，C14 使用 shift/mask/xor/add；unsupported recipe 直接
match failure。完整恢复结果见：
[stage6z_c14_exact_native_layout_recipes.json](./stage6z_c14_exact_native_layout_recipes.json)

## C14 lowering

AMDGPUBlockDotMfmaOperandOp 携带五个 typed attr：

| attr | 用途 |
|:--|:--|
| c13.distributed | logical shape/ownership、register/lane/wave recipe |
| c13.shared | swizzled 或 rotating LDS formula |
| c13.mfma | gfx942 v3、32x32x8、transposed |
| c13.dot | op index、K width、MFMA parent |
| c13.transform | fixed permutation、V basis/stride |

ChunkOPhysicalPlan 联合验证 Q/H/K/V、consumer relationship、shared regions 和
lifetimes。C14 只由 c14.static_physical 触发：验证五个 attr，应用 transform
permutation，按 shared recipe 生成 offset，生成 aligned vector<4xbf16>
addrspace(3) load，再调用已有 rocdl_mfma_f32_32x32x8bf16_1k。GPU outlining
保留 c14.mfma_callee 并复用已有 intrinsic。没有 MFMA64、generic fallback 或
隐藏的整个 block-dot schedule。

V basis 在本 MVP 中由 static algebra 做 bijection 验证并作为 typed metadata 保留；
synthetic offset 搬运由 fixed permutation + rotating shared formula 表达，没有伪造
arbitrary lane exchange，也没有用 ds_bpermute 偷渡未实现的 register transpose。

mapping 见：
[stage6z_c14_codegen_mapping.json](./stage6z_c14_codegen_mapping.json)

## Synthetic machine evidence

测试源为 lib/Target/AMDGPU/c14_static_physical_codegen_test.cc，构造
c14_q/c14_h/c14_k/c14_v 四个 detached func.func，每个带两个
memref<64x64xbf16,3> stage 和 vector<16xf32> accumulator；source artifact 记录
one-hot、lane-distinct、wave-distinct、packet-distinct、row/column pattern。

工件目录：
[codex_qwen_gfx942_c14_static_physical_codegen](./codex_qwen_gfx942_c14_static_physical_codegen/)

| 项目 | 结果 |
|:--|:--|
| synthetic functions | 4 |
| MFMA32 call sites | 16 |
| packed BF16x4 LLVM loads | 16 |
| ds_bpermute | 0 |
| s_div / s_rem | 0 |
| named barrier | 0 |
| target | amdgcn-amd-amdhsa--gfx942 |
| wavefront | 64 |
| group segment | 0 B |

这些是 synthetic lexical counts，不是 Qwen dynamic PMC；四个函数各 4 个 MFMA，
stage pointer 已经是 addrspace(3)，所以本 MVP 没有真实 shared allocation/barrier。

HSACO code-object metadata：

| kernel | VGPR | AGPR | SGPR | private | VGPR spill | SGPR spill |
|:--|--:|--:|--:|--:|--:|--:|
| c14_q | 24 | 16 | 12 | 0 B | 0 | 0 |
| c14_h | 28 | 16 | 12 | 0 B | 0 | 0 |
| c14_k | 28 | 16 | 12 | 0 B | 0 | 0 |
| c14_v | 24 | 16 | 12 | 0 B | 0 | 0 |

AGPR 是 code-object agpr_count，不是 rocprof Accum_VGPR_Count；本轮无动态
profiler。MIR 无 spill save/reload，HSACO private segment、VGPR spill、SGPR
spill 全为 0，但 synthetic 没有 full Qwen loop-carried live region，不能外推。

详细 SHA256 和证据：
[stage6z_c14_machine_evidence.json](./stage6z_c14_machine_evidence.json)

## Compiler 结论

可以明确证明 AveLang 能表达并传递 static distributed/shared/MFMA/dot/transform，
并且 gfx942 lowering 能把它们落成 packed LDS BF16 operand 和现有 MFMA32 的
LLVM/MIR/ISA。对已恢复的 power-of-two recipe，不需要 generic runtime div/rem
或 ds_bpermute；非法 recipe 不会静默回退。

所以“AveLang 无法表达 static full-region encoding”已不准确。准确结论是：

> C13/C14 已补齐表示层和第一条 gfx942 codegen 控制点；剩余难题是把它接到真实
> full chunk-o 的 producer ownership、shared allocation、phase/lifetime 和数值
> correctness，而不是 static full-region encoding 从根本上不可表达。

本轮不能证明 synthetic 比 Triton 快，不能证明真实 chunk-o 已减少 VMEM/LDS/barrier，
不能证明 full v29 资源 cliff 已解决，也不能把 synthetic 0 spill 或 16 个静态
MFMA 外推为真实 kernel 动态资源/工作量。

下一步唯一动作是 real physical tile numerical repro：一个 chunk、一个 operand role、
一个 C14 plan，真实分配 shared tile，执行 producer 写入、C14 consumer load、MFMA
输出，再与 reference tile 做 bitwise/误差检查。只有该 repro 证明 shared offset 与
recovered native basis 在真实 GPU lane execution 下匹配，才接 Q/H/K/V producer chain。

机器可读索引：

- [stage6z_c14_exact_native_layout_recipes.json](./stage6z_c14_exact_native_layout_recipes.json)
- [stage6z_c14_codegen_mapping.json](./stage6z_c14_codegen_mapping.json)
- [stage6z_c14_machine_evidence.json](./stage6z_c14_machine_evidence.json)
- [stage6z_c14_regression_results.json](./stage6z_c14_regression_results.json)

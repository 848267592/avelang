# gfx942 FP32 16x16x4 MFMA Intrinsic 功能补齐记录

## 结论

已将 gfx942 已有的 FP32 `16x16x4` MFMA 通过 Avelang 高层 intrinsic 暴露为：

```python
al.amdgpu.mfma_16x16x4_f32_f32(a_fragment, b_fragment, acc)
```

真实 MI300/gfx942 JIT 通过数值检查，并在生成的 HSACO 中确认精确指令：

```text
v_mfma_f32_16x16x4_f32 a[0:3], v1, v2, 0
```

这是一项小范围 compiler/intrinsic 功能补齐：没有修改 AMDGPU RA、没有手写汇编、没有修改 v18，也没有接入任何生产 full path。它只解除 BT64 hierarchical solve S0 的高层 API 门槛；S0 算法本身仍未实现。

## 为什么做这个补齐

Stage 5A/5B 审计已确认 BT64 solve 的局部块矩阵 `A` 与输出 `a_solved` 当前都是 FP32。vLLM/Triton 的 gfx942 code object 已使用 `v_mfma_f32_16x16x4_f32`，说明硬件和 ROCm LLVM 都支持该指令；旧 Avelang 仅缺少 source-level registry 和 ROCDL wrapper。

这不是要求 Q/K/V 输入改成 FP32。公共输入仍可保持 BF16；此处的 FP32 指的是 solve 中的 FP32 中间矩阵乘法及 FP32 accumulator。第一版保持 FP32 是为了严格匹配现有 solve 数学语义和数值稳定性，后续若需要混合精度，应独立作精度与端到端性能实验。

## 改了什么

| 文件 | 修改 | 目的 |
|:--|:--|:--|
| `lib/IR/Intrinsics/amdgpu_mfma_signatures.h` | 注册 `M=16,N=16,K=4,A/B=f32,C=f32` | 让 `al.amdgpu` 能解析 source call，并按每 lane A/B 1 个、C 4 个 FP32 校验 fragment。 |
| `lib/IR/Intrinsics/amdgpu_intrinsics.mlir` | 增加内联 `rocdl.mfma.f32.16x16x4f32` wrapper | 将高层 MFMA call 降到 ROCm 支持的指令。 |
| `lib/IR/mlir_generator_test.cc` | 增加 `GenerateMLIRAMDGPUMFMAFP32` | 防止 registry/前端生成以后回退。 |
| `docs/content/language-reference/hardware-intrinsics.md` | 记录 intrinsic 和 fragment 大小 | 保持公开 API 文档一致。 |
| `compile_bug/.../repro_fp32_mfma16_intrinsic_enablement.py` | 单 wave JIT/HSACO repro | 独立证明数值与实际 ISA。 |

## 从什么问题到什么修正

### 初始尝试

第一版 wrapper 直接将 Avelang 统一 fragment ABI 的 `vector<1xf32>` 传给 ROCDL op：

```mlir
rocdl.mfma.f32.16x16x4f32 %arg0, %arg1, %arg2, ...
  : (vector<1xf32>, vector<1xf32>, vector<4xf32>, ...) -> vector<4xf32>
```

`mlir-opt` 可以解析它，但 JIT 到 LLVM 时失败：`llvm.amdgcn.mfma.f32.16x16x4f32` 的 LLVM declaration 要求 A/B 为标量 `float`，不是 `<1 x float>`。

此外，JIT repro 起初使用 `al.full((1,), x, f32)` 构造 A/B；当前 Avelang frontend 将这个 rank-one full 当作标量表达式，所以 generic MFMA verifier 报 `MFMA operands must be vector types`。

### 最终实现

公开 Avelang ABI 保持一致：A/B 是 `vector<1xf32>`，C/result 是 `vector<4xf32>`。wrapper 在调用 ROCDL 前仅提取第 0 个元素：

```mlir
%a = vector.extract %arg0[0] : f32 from vector<1xf32>
%b = vector.extract %arg1[0] : f32 from vector<1xf32>
%r = rocdl.mfma.f32.16x16x4f32 %a, %b, %arg2, ...
  : (f32, f32, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
```

repro 则把全局 FP32 输入重新解释成 `[64,1]` vector view，使用一次索引得到真正的 one-element vector fragment。这只服务于验证代码，不是 future S0 的固定数据布局要求。

## 验证过程与结果

### 1. ROCDL 语法与前端

| 检查 | 结果 |
|:--|:--|
| `/opt/rocm/llvm/bin/mlir-opt rocdl_mfma16x4_fp32_syntax_probe.mlir` | 通过 |
| `mlir_generator_test --gtest_filter='*AMDGPUMFMAFP32*'` | `1 passed` |
| Avelang bindings 增量重建 | 通过 |
| 更新后的 Stage 5B source availability gate | `available=true`，canonical entry 编译且输出 finite |

构建中出现的 `lower_qwen_kfrag_lds_pass.cc` unused-variable warning 属于工作区已有 kfrag pass，与本 intrinsic 修改无关。

### 2. 单 wave 数值 JIT

repro 用 64 个全 1 FP32 lane fragment，初值 accumulator 为 0。每个 FP32 `16x16x4` 输出 accumulator 元素都应是 `4`，与具体 lane-to-output layout 无关。

| 项目 | 实测 |
|:--|--:|
| 正确性 | `true` |
| max abs | `0.0` |
| mean abs | `0.0` |
| 非 profiler 下 median | `0.017246 ms` |

这个时间仅覆盖一个 64-thread wave，主要用于 smoke，不应解读为 solve 性能。

### 3. HSACO / ISA

`llvm-objdump -d` 的精确命中：

```text
v_mfma_f32_16x16x4_f32 a[0:3], v1, v2, 0
```

这证明 source intrinsic 没有退化为标量 FMA，也没有错用 BF16/FP16 MFMA。

### 4. rocprof 单 kernel 证据

对 7 个 probe dispatch 的 trace duration 排序为 `1.842, 1.963, 1.963, 1.963, 2.043, 2.083, 2.163 us`，中位数为 **`1.963 us`**。该数值是带 profiler 的单波 trace，只作功能与资源 sanity。

| 指标 | 实测 |
|:--|--:|
| workgroup / grid | `64` / `64` work-items |
| LDS / scratch | `0 B` / `0 B` |
| rocprof VGPR / AccVGPR / SGPR | `4 / 4 / 16` |
| code-object VGPR / AGPR / SGPR | `8 / 4 / 10` |
| `SQ_INSTS_MFMA` | `1` |
| `SQ_INSTS_VALU` | `6` |
| `SQ_INSTS_SALU` | `0` |
| `SQ_INSTS_VMEM` | `3` |
| `SQ_INSTS_LDS` | `0` |

rocprof 与 code-object 的 VGPR/SGPR 计数口径不同，因此两行资源数据保留为原样，不将它们错误地当作矛盾。

## 范围与下一步

本次没有：

- 调整 AMDGPU register allocation；
- 引入 AGPR/VGPR 限制；
- 编写或改动手写汇编/HSACO；
- 修改 v18 或其 solve；
- 修改 Stage 4、v23/v24 或生产 dispatch；
- 实现 BT64 hierarchical solve S0。

下一步应单独实现 opt-in 的 S0 hierarchical solve kernel，并按冻结的 `(I+A)X=I` 残差、v18 oracle 和 full-path 数值契约验证。该 kernel 可以使用新 intrinsic，但不能静默回退到 v18；只有 S0 正确性和性能都通过，才讨论后续 DAG 层与 full pipeline 接入。

## 可复现证据

- 契约与命令：[implementation_contract.md](codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/implementation_contract.md)、[commands.sh](codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/commands.sh)
- JIT repro：[repro_fp32_mfma16_intrinsic_enablement.py](repro_fp32_mfma16_intrinsic_enablement.py)
- 数值结果：[live_jit.json](codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/live_jit.json)
- source gate：[feature_gate_after_enablement.json](codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/feature_gate_after_enablement.json)
- HSACO/ISA：[fp32_mfma16x4_intrinsic_probe.s](codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/fp32_mfma16x4_intrinsic_probe.s)
- rocprof trace/counters：`codex_qwen_bt64_fp32_mfma16_intrinsic_enablement/rocprof/`

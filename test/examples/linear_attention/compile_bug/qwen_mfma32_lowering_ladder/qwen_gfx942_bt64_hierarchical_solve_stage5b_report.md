# Qwen gfx942 BT64 Hierarchical Solve Stage 5B

> 历史状态说明：下文记录的是 intrinsic 尚未暴露时的 Stage 5B feature-gate
> 结论。后续的最小 compiler/intrinsic 补齐已完成并通过 gfx942 JIT/ISA 验证；
> 详见
> [`qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md`](qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md)。
> 此后 S0 hierarchical solve 已实现并通过 standalone correctness、W/U consumer、
> performance 和 resource gates。最新结论见
> [`qwen_gfx942_bt64_hierarchical_solve_stage5b_completion_report.md`](qwen_gfx942_bt64_hierarchical_solve_stage5b_completion_report.md)。
> 下文保留为 source feature gate 的历史审计，不再代表当前 Stage 5B 状态。

## 结论

Stage 5B 已完成实现前的契约冻结、源码审计和 **MI300/gfx942 live JIT
feature gate**，但不能进入 SOLVE-S0。原因不是 4x16 数学、LDS 预算、寄存器
压力或性能，而是当前 AveLang 高级代码没有导出 FP32
`16x16x4` MFMA intrinsic。

因此本轮没有修改 v18、Stage 4 KKT/W-U/chunk-o、asm recurrence、compiler、
LLVM RA 或生产 dispatch；也没有用 BF16/FP16 替代 FP32 来制造一个不符合契约
的结果。`full_pipeline_integrated=false`。

## 已冻结的设计

输入/输出保持 contiguous FP32 `[1,T,8,64]`，每个 chunk/head 求
`X=(I+A)^-1`。预定的 4x16 DAG 与 Stage 5A/vLLM 一致：四个 diagonal
inverse，然后 `{X21,X32,X43}`、`{X31,X42}`、`X41` 三层依赖。预定一个
256-thread、4-wave CTA 和一个 solve dispatch，显式 LDS 上限 8192 B。

关键 off-diagonal 公式为：

```text
X21 = -X22 A21 X11            X32 = -X33 A32 X22
X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

这些公式与 v18 的 `M=-A` 递推和 `(I+A)X=I` 严格一致，但其关键 block
product 必须使用 FP32 MFMA16；不能降成 BF16。

## 实际 feature gate

新增的 live probe 是
`test/examples/linear_attention/vllm_compare/repro_qwen_bt64_fp32_mfma16_feature_gate.py`。
它在 MI300 容器中用一个 64-thread JIT kernel，尝试三个合理的源级入口：

| 尝试名 | 实际结果 |
|:--|:--|
| `al.amdgpu.mfma_16x16x4_f32_f32` | `Symbol not found` |
| `al.amdgpu.mfma_f32_16x16x4_f32` | `Symbol not found` |
| `al.amdgpu.mfma_16x16x4_f32` | `Symbol not found` |

编译器还报告 `Unsupported function call target`，因此失败发生在 MLIR/LLVM/
HSACO 之前。原始 stderr 在审计目录的
`standalone/fp32_mfma16_feature_gate.stdout_stderr.txt`，结构化结果在
`standalone/fp32_mfma16_feature_gate.json`。

源码审计与运行时结果相符：
`lib/IR/Intrinsics/amdgpu_mfma_signatures.h` 只注册 F16/BF16
`16x16x16` 和 BF16 `32x32x8`，而
`lib/IR/Intrinsics/amdgpu_intrinsics.mlir` 也没有 FP32 `16x16x4` wrapper。
Stage 5A 保存的 vLLM ISA 确实有 `v_mfma_f32_16x16x4_f32`；这证明 gfx942
硬件和 Triton lowering 可用，不能证明当前 AveLang 高层 API 已暴露它。

## Gate 状态

| gate | 状态 | 原因 |
|:--|:--|:--|
| FP32 MFMA16 source/JIT | 失败 | 所有合理入口均为未注册 symbol |
| SOLVE-S0 correctness / residual | 未运行 | 无可编译的符合契约 kernel |
| W/U 与 recurrence consumer | 未运行 | standalone gate 未通过 |
| ISA / HSACO / rocprof resources | 未运行 | 无 candidate code object |
| T=2048 `<=0.060 ms` | 未运行 | 无 candidate dispatch |
| opt-in full integration | 否 | 任何一项 standalone gate 失败即禁止接入 |

这里的“未运行”不是失败数据的替代说法，`null`/`not_run` 记录在 CSV/JSON 中，
没有把 v18 或 vLLM 的数值填写为新实现结果。

## 基线和下一步

Stage 5A 基线仍为：v18 T=2048 body-only `0.122422 ms`，vLLM
`0.035613 ms`；v18 的主要瓶颈依然是 63-row scalar/LDS recurrence。它们
只是基线，不是 Stage 5B 性能数据。

下一个最小的技术动作是：在 **新的、单独审批的 compiler/intrinsic 工作** 中，
把已经存在于 ROCm 的 FP32 `16x16x4` MFMA 暴露为 AveLang 高级 intrinsic，并做
独立最小 repro 的 ISA 验证。那项工作被当前任务明确禁止，因此本轮没有进行。
完成后应从本报告的 feature gate 重新开始，而不是绕过 gate 修改 v18 或接入 full
pipeline。

新的 opt-in API 已在有效 CUDA FP32 `[1,64,8,64]` 输入上做过 guard smoke；
它抛出了上述 feature-gate `RuntimeError`，没有退回 v18，也没有执行任何不符合
FP32-MFMA 合同的 kernel。

## 产物

- 契约与 DAG：
  `codex_qwen_bt64_hierarchical_solve_stage5b/implementation_contract.{md,json}`、
  `block_dag.md`、`lane_wave_ownership.md`、`storage_plan.md`。
- primitive 审计：`primitive_reuse_audit.md`、`primitive_source_map.json`。
- live 结果：`standalone/fp32_mfma16_feature_gate.{json,stdout_stderr.txt}`。
- 决策：`failure_classification.{md,json}`、`final_decision.json`。
- API guard：
  `vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`。

# gfx942 BT64 Hierarchical Solve Stage 5C 集成报告

## 结论

Stage 5C 已将 Stage 5B 的 FP32 hierarchical BT64 solve 接入当前最高层的
Stage 4 BT64 实验 full path，而不是早期 Stage 2 的通用 pipeline。接入保持为
**显式 opt-in**：`solve_impl="hierarchical_fp32_v1"`；既有默认值仍是
`"v18"`，没有修改 v18 或 production dispatch。

在 gfx942/MI300、T=2048 的三次独立 warmup-10/repeat-50 HIP-event 会话中：

| 实现 | full median ms | 相对 v18 |
|:--|--:|--:|
| Stage 4 + v18 solve | `0.474906` | `1.0000x` |
| Stage 4 + hierarchical FP32 v1 | `0.454215` | `1.0441x` |

新 solve 让完整路径稳定减少约 `20.13 us`。它不是生产替换：当前 production
baseline 仍是 v24；但它已是 Stage 4 BT64 实验路径应采用的 solve 选项。

## 为什么接入 Stage 4

早期 `qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py` 使用通用 KKT、W/U
和 chunk-o，只适合 Stage 2 的 ABI/正确性桥接，不能代表当前最优 BT64 代码。
当前最高层实验入口是：

```text
qwen_gdn_full_bt64_stage4_all_s0
  cumsum v6
  -> Stage 4 native KKT-S0
  -> solve
  -> Stage 4 native W/U-S1
  -> frozen gfx942 asm recurrence
  -> Stage 4 native chunk-o-S0
```

因此本次改动只位于
[`qwen_gdn_bt64_nonrecurrence_mfma_v2.py`](../../vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py)：

```python
def _solve_bt64_stage5c(a, solve_impl):
    if solve_impl == "v18":
        return qwen_gdn_solve_avelang_v18_bt64_layout(a, chunk_size=64)
    if solve_impl == "hierarchical_fp32_v1":
        return qwen_gdn_solve_bt64_hierarchical_fp32_v1(a)
    raise ValueError(...)
```

`qwen_gdn_full_bt64_stage4_all_s0_stages` 与 public wrapper 都增加了该
keyword-only 选择项。非法名字直接 `ValueError`，没有 silent fallback。

## 环境修复

第一次集成运行发现容器实际 import 的
`/opt/avelang/python/_avelang_bindings...so` 是旧版，因此 source call 报：

```text
Symbol not found: al.amdgpu.mfma_16x16x4_f32_f32
```

这不是 S0 或 Stage 4 算法错误。宿主源码已经包含 registry 和 ROCDL wrapper，
但 Docker workspace 与 `/opt` binding 过旧；旧 CMake cache 还引用了已删除的
Ninja/ROCm toolchain 前缀。没有删除旧 build，而是：

1. 同步 `amdgpu_mfma_signatures.h` 和 `amdgpu_intrinsics.mlir`；
2. 在 `/tmp/avelang_stage5c_bindings` 用当前 `/opt/rocm/llvm` 做干净的
   `rocm + WITH_PYTHON + Release` build；
3. 原子替换容器活动的 `_avelang_bindings...so`；
4. 复跑最小 probe，输出精确正确：`max_abs=0`。

没有修改 v18、AMDGPU RA、手写 asm 或任何 production dispatch。

## 正确性

新增测试：
[`test_qwen_gdn_bt64_hierarchical_solve_stage5c.py`](../../vllm_compare/test_qwen_gdn_bt64_hierarchical_solve_stage5c.py)。

### 对 Stage 4 v18 默认路径

| T / 输入 | solve max abs | output max abs | final state max abs |
|:--|--:|--:|--:|
| 64 / random / h0 | `2.9802322e-08` | `0` | `0` |
| 512 / high_dynamic / h0 | `7.4505806e-09` | `0` | `0` |
| 512 / small_values / zero h0 | `9.094947e-12` | `0` | `0` |

### 冻结 vLLM public contract

| T / 输入 | output max abs | final state max abs | 阈值 |
|:--|--:|--:|:--|
| 64 / random / h0 | `4.8828125e-04` | `4.8364401e-03` | `<= 7.8125e-03 / <= 2e-02` |
| 512 / high_dynamic / h0 | `1.953125e-03` | `1.0142088e-02` | `<= 7.8125e-03 / <= 2e-02` |

`6 passed in 22.61s`。HIP-event benchmark 使用另一组 random seed 时，v1 相对
v18 的最大 output 差为 `2.44140625e-04`、final state 差为 `1.3291836e-05`；
这是 FP32 solve 累积顺序在后续 BF16 W/U 写回处的舍入差，仍远小于 frozen
public contract。

此外，默认 `solve_impl="v18"` 的既有 Stage 4 回归套件也在同一容器通过：
`29 passed in 22.71s`。这覆盖 native KKT、W/U、chunk-o、增量 full graph 和
Stage 4 all-S0 full contract，确认 selector 的默认行为没有改变原实验 baseline。

## 性能

### 单次 T=512/2048 A/B

| T | v18 solve ms | v1 solve ms | solve speedup | v18 full ms | v1 full ms | full speedup |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | `0.120539` | `0.026279` | `4.5869x` | `0.316050` | `0.314067` | `1.0063x` |
| 2048 | `0.125987` | `0.028522` | `4.4171x` | `0.474705` | `0.455437` | `1.0423x` |

### T=2048 三会话确认

| session | v18 full ms | v1 full ms | speedup | v18 solve ms | v1 solve ms |
|:--|--:|--:|--:|--:|--:|
| a | `0.474906` | `0.452833` | `1.0487x` | `0.126768` | `0.028642` |
| b | `0.476528` | `0.456398` | `1.0441x` | `0.126828` | `0.029264` |
| c | `0.474105` | `0.454215` | `1.0438x` | `0.126428` | `0.029083` |

单独 solve 的约 `0.0977 ms` 节省没有一比一出现在 full 时间中。不能从独立
solve benchmark 直接相减得到 end-to-end latency；应以完整路径实测的 `20.13 us`
作为可交付收益，不应把 standalone `4.35x` 直接宣称为 full-path 速度提升。下面的
受控诊断进一步说明了这一点。

### 为什么 solve 的大收益没有线性传递到 full

这不是 selector 未接入。使用同一组 `T=2048` 输入、每轮交替执行 v18/v1 的无分段
event 完整图测量，三轮结果为：

| session | v18 full ms | v1 full ms | v18 - v1 |
|:--|--:|--:|--:|
| 0 | `0.474385` | `0.455577` | `18.808 us` |
| 1 | `0.474304` | `0.454395` | `19.910 us` |
| 2 | `0.474405` | `0.456117` | `18.287 us` |

因此，完整路径约 `18.81 us` 的收益是可复现的；但它显著小于 solve-only 的约
`96.6 us`。为定位差异，在**同一条** full graph 中的每个 stage 之间插入 HIP event
进行了诊断。该诊断显示 v1 的 solve 确实少了 `96.583 us`，但紧随其后的 W/U 和 asm
recurrence 在该受扰动的测量中分别增加 `16.865 us` 与 `75.352 us`：

| stage | v18 ms | v1 ms | v18 - v1 |
|:--|--:|--:|--:|
| cumsum | `0.042163` | `0.042383` | `-0.220 us` |
| KKT | `0.061431` | `0.061230` | `0.201 us` |
| solve | `0.110824` | `0.014241` | `96.583 us` |
| W/U | `0.052879` | `0.069743` | `-16.865 us` |
| asm recurrence | `0.147158` | `0.222511` | `-75.352 us` |
| chunk-o | `0.068342` | `0.068261` | `0.081 us` |

这张表只用于诊断，**不可与无分段 full 计时混用**：每个 stage 之间的 event record
本身会改变 dispatch 边界和中间张量的缓存/调度状态；在该受扰动测量中 total 仅从
`0.490008` 降至 `0.485421 ms`。它足以证明 solve kernel 本体的收益没有丢失，也说明
后续 W/U 与 asm 的运行时间不独立于前一个 solve 的 launch/cache 状态；但不能仅凭这张
表把全部抵消严格归因于某一种缓存机制。

目前可信的结论是：v1 solve 已成功，端到端收益受其后连续 kernel 的组合执行状态限制。
下一步应对完整图分别以 v18/v1 运行 rocprof kernel trace，直接比较 W/U 和 asm
dispatch 的 trace/counter，而不是继续优化 solve 或从独立 stage 数字相减。

## T=2048 rocprof

在修复后实际使用的新 binding 下，targeted profile 对 v1 solve 的 7 次 dispatch
trace median 为 `13.701 us`：

| metric | value |
|:--|--:|
| workgroup / grid work-items | `256 / 65536` |
| LDS / scratch | `8192 B / 0 B` |
| VGPR / AccVGPR / SGPR | `44 / 4 / 32` |
| OccupancyPercent | `5.585220` |
| SQ_INSTS_MFMA | `16384` |
| SQ_INSTS_VALU | `637440` |
| SQ_INSTS_SALU | `123904` |
| SQ_INSTS_VMEM | `40960` |
| SQ_INSTS_LDS | `212992` |

此前 Stage 5A 的 v18 historical trace 为约 `110.224 us`，资源比较仍支持 S0
hierarchical block-inverse/MFMA 方向。完整路径的改善较小并不否定 solve kernel
本体收益，而是说明 Stage 4 的剩余成本已更分散，尤其是冻结 asm recurrence。

## 产物与复现

- integration test：
  [`test_qwen_gdn_bt64_hierarchical_solve_stage5c.py`](../../vllm_compare/test_qwen_gdn_bt64_hierarchical_solve_stage5c.py)
- benchmark：
  [`bench_qwen_gdn_bt64_hierarchical_solve_stage5c.py`](../../vllm_compare/bench_qwen_gdn_bt64_hierarchical_solve_stage5c.py)
- 原始 benchmark / pytest / rocprof CSV：
  [`codex_qwen_bt64_hierarchical_solve_stage5c_integration`](codex_qwen_bt64_hierarchical_solve_stage5c_integration)
- 精确命令：
  [`commands_stage5c.sh`](codex_qwen_bt64_hierarchical_solve_stage5c_integration/commands_stage5c.sh)
- 机器可读决策：
  [`final_decision.json`](codex_qwen_bt64_hierarchical_solve_stage5c_integration/final_decision.json)

## 决策

Stage 5C `ready_for_stage4_experimental_default=true`：Stage 4 的实验调用可以明确
选择 `hierarchical_fp32_v1`。保持 API 默认 `v18`，直到后续更大规模 correctness/
benchmark matrix 完成；production v24 不变。

下一步不应再反复优化 solve。新的最大绝对 stage 仍是 frozen asm recurrence；在其
保持 immutable 的约束下，后续优化应先做完整 Stage 4 v1 profile，量化剩余
KKT/W-U/chunk-o/launch 成本，再决定是否值得新开一个非 recurrence stage。

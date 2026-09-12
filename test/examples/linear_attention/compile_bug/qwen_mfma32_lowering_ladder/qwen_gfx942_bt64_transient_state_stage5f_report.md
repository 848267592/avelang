# Qwen gfx942 BT64 Transient-State Stage 5F

## 结论

**No-Go：关闭 Stage 5 transient-state root-cause 支线。** 本轮是严格的
measurement-only 审计，没有修改任何 kernel、solve、W/U、asm、HSACO、编译器、
allocator 或 production dispatch。

Stage 5E 已用 direct-common-out 排除了 output pointer/allocator；求解结果数值、
hidden dispatch 和 downstream 动态工作量也已排除。Stage 5F 的唯一实际 whole-graph
trace 候选 `rocprofv3 --kernel-trace` 未通过预注册的低扰动门槛。因此它的 timestamp、
dispatch gap 和任何从 trace 推出的 cache/clock 解释都不能用于硬件因果结论。

下一阶段唯一建议：停止该根因支线，转向 **BT16/BT64 production-style crossover、
correctness stability 与 dispatch-policy 审计**。

## 冻结图与安全边界

本轮复用 Stage 5E direct-common-out 图：

```text
cumsum -> KKT -> solve_direct_common_out -> W -> U -> asm-v0 -> chunk-o -> cast
```

- GRAPH-A：v18 direct solve，WG=128。
- GRAPH-B：hierarchical v1 direct solve，WG=256。
- 同一 process、stream、输入和预分配；本轮 contract probe 的 common pointer 为
  `0x7f7e26e00000`。
- 忽略 solve symbol/workgroup 后两个 dispatch 图结构相同；solve 与 W/U 间没有
  copy、fill、allocation 或额外 dispatch。

`graph_a.json`、`graph_b.json` 和 `graph_diff.md` 记录了实际 probe。没有对图内
stage 插 HIP event；HIP event 只包住完整 tail 或完整 full 边界。

## 能力盘点

当前 gfx942 Docker 环境实际发现 `/opt/rocm/bin/rocprofv3`、`rocprof`、`rocprofv2`、
`amd-smi`、`rocm-smi`。`rocprofv3 --list-avail`、工具帮助和只读 telemetry 查询的原始
输出保存在 `capability_query_raw.json` 与 `available_counters.txt`。帮助文本表明 kernel
trace、PMC、PC sampling 与 thread trace 入口存在，但这只说明 API 可用，不说明它们
可以在约 64 us 效应上无扰动工作。

## 已执行的标定

每种已执行模式均为 5 sessions、warmup=20、repeat=200，Stage 5E 的 ABBA/order
balanced 计时器保存了 raw samples。

| 模式 | 图 | T | v18 ms | v1 ms | v1-v18 us | session std us | 结论 |
|:--|:--|--:|--:|--:|--:|--:|:--|
| HIP event | tail | 2048 | 0.267218 | 0.330150 | 62.772 | 0.480 | 基线 |
| HIP event | tail | 8192 | 0.988329 | 0.999365 | 11.016 | 0.304 | 基线 |
| HIP event | full | 2048 | 0.449969 | 0.401536 | -47.651 | 0.586 | 基线 |
| HIP event | full | 8192 | 1.370777 | 1.176088 | -194.349 | 0.400 | 基线 |
| rocprofv3 trace-only | tail | 2048 | 0.271163 | 0.358553 | 87.230 | 0.689 | 拒绝 |

对于最关键的 T=2048 tail：

- baseline A/B penalty：`62.772 us`；
- trace-only penalty：`87.230 us`；
- penalty distortion：`24.457 us`，允许上限仅 `6.277 us`；
- 最大 A/B latency distortion：`28.402 us`，允许上限 `16.508 us`。

故 trace-only 同时违反 latency 与 A/B penalty gate。它虽提供 kernel timestamp/trace，
但不再是低扰动观测。timestamp-only、单 PMC、多 PMC/replay、cache、PC sampling、
thread trace、clock/power causal collection 均按 stop rule 标记为 `N/A_gate_failed`，
没有继续尝试。

此前 Stage 5E counter trace 中曾看见 v18/v1 dispatch gap 约 60/90 us，但同一工具也
使本应相同的 cumsum/KKT 漂移，故该现象只是 profiler artifact 线索，不能认定为原生
solve-to-W gap。本轮不对 warm/perturb gap、cache 命中差异、clock/power 或 CU/wave
state 作新的因果声称。

## 因果判定

Measured facts：

- direct common pointer、数值、downstream launch/动态工作量已由 Stage 5E 排除为主因；
- HIP-event baseline 在 T=2048 可稳定看见约 63 us tail penalty；
- kernel trace 将该 penalty 额外扭曲约 24.5 us。

Inference only：原生差异仍可属于 cache residency、runtime/queue pacing、短时间状态或
其组合。没有两种独立、通过低扰动 gate 的观测支持其中任一项，因此
`exact_mechanism_unresolved=true`。

`causal_evidence_matrix.json` 和 `go_no_go_decision.json` 是机器可读 closure。

## 回归

Stage 5E direct-out smoke 在当前 Docker fresh run：`8 passed in 15.26s`。它覆盖 T=2048
direct-common correctness、public output/final-state 阈值和 non-default stream。Stage 5F
新增脚本均通过 `python3 -m py_compile`。

## 产物与复现

全部原始样本、gate、trace CSV、能力查询与命令在：

`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_transient_state_stage5f/`

执行顺序见 `commands.sh`。本轮的正式路径未改动：v18/v1 solve、Stage 4 KKT/W/U/chunk-o、
asm-v0/HSACO/ABI/launch、compiler/RA、allocator、默认 selector 与 production dispatch。

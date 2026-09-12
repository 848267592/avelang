# Qwen gfx942 BT64 Stage 5E Direct Common Output 审计

## 结论

Stage 5E 完成了真正的 caller-provided direct-out A/B。v18 和
`hierarchical_fp32_v1` 直接写同一块预分配 FP32 `[1,T,8,64]` buffer，solve 与
W 之间没有 copy、fill、allocation 或额外 dispatch。T=2048 的下游 tail penalty
仍为 `64.035 us`，几乎等于 Stage 5D 的 `64.255 us`。

因此分类为 **CASE B**：solve 输出 pointer/address 不是主因。仍被数据支持的类别是
“前驱 kernel 诱发的瞬态 downstream execution state”；warm/perturb 能消除差距，
但本轮 cache counters 和粗粒度 telemetry 没有把它唯一定位到 cache、clock 或
runtime queue。Stage 5F 唯一建议是继续更低扰动的 whole-graph cache/dispatch
counter 审计，不修改 W/U、asm、compiler 或 production 路径。

## 实现与写覆盖

新增 audit-only wrapper：

- `qwen_gdn_solve_v18_bt64_direct_out_audit(a, out)`；
- `qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out)`。

文件为 `vllm_compare/qwen_gdn_bt64_solve_direct_out_stage5e_audit.py`。两个 wrapper
直接 launch 原 kernel；不调用原 wrapper 后 copy，不创建 solve output。固定
shape/dtype/device/contiguous/alignment 契约，不合法 alias 和参数会明确报错，无
silent fallback。

源码审计确认两 kernel 均覆盖全部 4096 个 chunk/head 元素。v18 显式写下三角和
对角、上三角写零；v1 kernel 内部先并行清零完整 output，再写下三角/对角。因此
host 无需预初始化，主实验没有额外 fill。两者都不依赖 output 旧值，也没有 atomic
或 read-modify-write。

contract 运行中两次 solve 的 exact pointer 都是 `0x7f78e9600000`，storage
offset=0、大小 4 MiB，对 16/64/128/256 B、4 KiB、64 KiB 均对齐。MODE-2 每图
8 个 dispatch，solve 与 W 直接相邻。

## Correctness

- standalone 68 行：v18 original/direct bitwise 相同；v1 original/direct bitwise
  相同；最大 authority 误差 `1.78814e-07`。
- 87 个 write-coverage 检查：NaN prefill 后无 NaN/Inf；upper triangle 和 repeated
  same-buffer reuse 均通过；最大 residual `2.08616e-07`。
- full T=64/512/2048/8192：direct 对 original 的 output/state 均 bitwise 相同。
- direct v1 对 v18 最坏 BF16 output max_abs `0.000244140625`，FP32 final state
  max_abs `0.0011825562`。本轮重跑的冻结 vLLM public contract 最坏值为
  `0.001953125/0.010142088`，低于阈值 `0.0078125/0.02`。
- 新测试、Stage 5C、Stage 4、asm-v0、external bridge 合并回归：
  `53 passed in 46.45s`。

## Direct-Common Tail

每点 5 个 allocation session，warmup=20、repeat=200，跨 80 个 allocation
session/35 个不同虚拟地址。表中 penalty 为 v1-v18：

| T | v18 tail ms | v1 tail ms | penalty us | bootstrap 95% interval us |
|--:|--:|--:|--:|--:|
| 512 | 0.083284 | 0.173498 | 90.293 | [88.951, 91.076] |
| 2048 | 0.267237 | 0.331152 | 64.035 | [63.254, 64.336] |
| 8192 | 0.988429 | 0.999465 | 10.996 | [10.476, 11.437] |
| 16384 | 1.990618 | 2.002275 | 10.895 | [8.052, 15.984] |

时间列是各实现 session median 的中位数；penalty 是 paired session delta 的
中位数，因此不要求严格等于前两列相减。

T=2048 的 ABAB/BABA/random-ABBA 中位差分别为
`64.186/63.284/64.175 us`，五个 allocation 均复现。相比 Stage 5D，pointer
统一只改变 `-0.220 us`，处于噪声内。

## Solve+Tail 与 Full

| T | solve+tail v18/v1 ms | v1 gain us | full v18/v1 ms | v1 full gain us |
|--:|:--|--:|:--|--:|
| 512 | 0.212195 / 0.204344 | 7.652 | 0.254118 / 0.246025 | 8.152 |
| 1024 | 0.271644 / 0.256361 | 15.282 | 0.312364 / 0.299165 | 12.959 |
| 2048 | 0.398271 / 0.361016 | 37.296 | 0.450670 / 0.403620 | 46.370 |
| 4096 | 0.636947 / 0.556567 | 80.720 | 0.724717 / 0.630537 | 94.200 |
| 8192 | 1.229748 / 1.039104 | 190.884 | 1.373681 / 1.179192 | 194.629 |
| 16384 | 2.356762 / 2.067091 | 290.652 | 2.608817 / 2.335872 | 272.846 |

Stage 5C public full 在 T=2048 的 v1 gain 为 `18.808 us`；本 harness 为
`46.370 us`。收益列使用 paired session delta 的中位数；这说明固定预分配
harness 传递出更多 solve 收益，但绝对值不能跨
harness 当作 production 改善。约 97 us solve 本体收益在 T=2048 被 64 us 级
tail penalty 抵消了相当一部分。

## Cache/Execution-State 控制

| T=2048 控制 | v18 ms | v1 ms | penalty us |
|:--|--:|--:|--:|
| none | 0.267197 | 0.329810 | 62.613 |
| warm | 0.267438 | 0.267478 | 0.021 |
| 512 MiB perturb | 0.274568 | 0.274548 | -0.020 |
| reduction prime | 0.269801 | 0.350041 | 80.280 |

warm 与相同的大工作集 perturb 仍将差距压到噪声内，prime 没有。该现象支持
瞬态状态耦合，但不证明具体 cache 层级。

## rocprof 与 Cache Counters

T=2048 full graph targeted trace：

| stage | v18 us | v1 us | 关键动态指令是否相同 |
|:--|--:|--:|:--|
| cumsum | 11.738 | 14.542 | 是 |
| KKT | 37.736 | 40.981 | 是 |
| solve | 109.163 | 13.260 | 否，算法不同 |
| W | 27.761 | 27.822 | 是 |
| U | 24.256 | 24.816 | 是 |
| asm recurrence | 145.616 | 148.261 | 是 |
| chunk-o | 67.080 | 67.100 | 是 |
| cast | 4.367 | 4.648 | 是 |

下游动态 counts 完全一致，例如 W VALU/VMEM=`5,160,960/458,752`，U=
`4,005,888/393,216`，asm=`2,535,040/91,136`，chunk-o=
`12,918,784/851,968`。TCC/TCP profile 中所有下游
`TCP_TOTAL_CACHE_ACCESSES_sum` 完全相同，TCC 最大相对差为 U hit 的
`-0.232%`。没有发现能解释 64 us 的 cache 流量差。

rocprof 显示 v1 进程中的 dispatch gaps 比 v18 高约 30 us，但 counter
instrumentation 本身把每个 gap 放大到 60--94 us，并且 solve 前的相同 stages
也发生 session 漂移，因此该差异不作为原生 runtime 根因证据。

## Clock/Power Telemetry

`amd-smi` 只读采样覆盖 A-only、B-only、ABBA、warm、perturb。T=8192 none 的
A/B XCP gfx-clock 中位数约 `2008/1988 MHz`、socket power `187/185 W`，范围
高度重叠；warm 也没有稳定分叉。采样粒度远粗于单 kernel，不能区分 cache 与
clock 机制，只能说明没有持续、明显的 A/B 频率/功耗状态差。

## 最终决策

1. direct common output 已完整实现且语义正确。
2. output pointer/address 主因被拒绝；不值得接入 production reusable solve-out。
3. 不修改 W/U、asm、compiler，也不做 fusion 或 dummy warmup。
4. Stage 5F 唯一动作：以更低 profiler 扰动继续 whole-graph cache/dispatch
   counter 审计；可恢复收益暂记 `N/A`。

本轮没有修改 v18/v1 kernel 数学、Stage 4 KKT/W/U/chunk-o、asm-v0 HSACO/ABI、
compiler、production dispatch 或默认 solve selector。

## 证据路径

所有 raw samples、profiles、telemetry、pytest 和机器可读决策位于
`codex_qwen_bt64_direct_common_out_stage5e/`。核心文件包括
`final_decision.json`、`downstream_counter_comparison.csv`、
`cache_counter_comparison.csv`、`clock_power_observation.csv` 和
`tests/pytest_results.txt`。

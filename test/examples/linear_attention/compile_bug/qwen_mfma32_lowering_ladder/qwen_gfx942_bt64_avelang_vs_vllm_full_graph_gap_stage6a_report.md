# Qwen gfx942 BT64 Stage 6A: Avelang-vLLM Full-Graph Gap Audit

## 结论

Stage 6A 关闭了 Stage 5 的 transient-state 根因支线，改用同口径的
full-graph 审计。当前 BT64 Avelang Stage 4 图在短序列已经接近 vLLM，
但每个 64-token chunk 仍额外花费 **4.406 us**。在 `T=16384`，这累积成
`1.123 ms` 的差距。

唯一的 Stage 6B 建议是：**只重做 native BT64 `chunk_o` ownership**。
它是可修改 stage 中最大的 T=2048 body 差距（`46.169 us`）、最大的长文本
差距（T=16384 为 `309.600 us`）和最大的可修改 slope（`1.196 us/chunk`）。
本轮不修改 kernel；也不把独立 cast 或 W/U 合并顺手塞进 6B。

`recurrence` 的 slope 更大，但它是冻结的 asm-v0，明确不属于本次可修改
候选。`solve` 已不是瓶颈：Avelang standalone 反而快于 vLLM。

## 1. 严格计时口径

固定目标为 gfx942、`B=1,Hk=4,Hv=8,K=V=128,BT=64`，外部输入均为同一组
contiguous tensor：BF16 `q/k/v`，FP32 `g/beta/initial_state`，布局 `[B,T,H,D]`。

- 两边在同一 Python process、同一 current stream 上运行。
- 先完成 JIT/autotune/module load，再在 CUDA Graph capture 中完成所有输出和
  中间 buffer 的分配；计时区间只 replay graph。
- vLLM 的公开 API 不提供 caller-owned output 参数。因此 graph capture 是两边都
  能使用的严格预分配机制；capture 之后 replay 不做 allocation。
- 每个 `T` 使用 20 次 warmup、100 个 ABBA 样本、5 sessions。HIP event 只包住
  完整 graph 或单一 standalone body，没有在 full graph 内插 stage event。
- `T=512,1024,2048,4096,8192,16384` 都通过 public output 和 final-state 门槛。
  最大 output abs 为 `0.0009765625`（门槛 `0.0078125`），最大 state abs 为
  `0.00646675`（门槛 `0.02`）。

ROCprof trace 仅用于 dispatch/resource 结构，**不是**下表的 latency 来源。
这避免了 Stage 5F 已证实的 profiler 扰动问题。

## 2. 同口径 Full 延迟

| T | chunks | Avelang ms | vLLM ms | gap us | Avelang/vLLM |
|--:|--:|--:|--:|--:|--:|
| 512 | 8 | 0.122823 | 0.100930 | 21.893 | 1.217x |
| 1024 | 16 | 0.191845 | 0.128952 | 62.893 | 1.488x |
| 2048 | 32 | 0.335539 | 0.188900 | 146.639 | 1.776x |
| 4096 | 64 | 0.607022 | 0.324142 | 282.880 | 1.873x |
| 8192 | 128 | 1.156198 | 0.602416 | 553.783 | 1.919x |
| 16384 | 256 | 2.302501 | 1.179092 | 1123.409 | 1.953x |

对 session median 按 `latency = intercept + slope * chunks` 最小二乘拟合：

| series | intercept ms | slope us/chunk |
|:--|--:|--:|
| Avelang full | 0.049170 | 8.771643 |
| vLLM full | 0.054013 | 4.365784 |
| Avelang-vLLM gap | -0.004843 | **4.405859** |

因此差距的主形态是 chunk-linear，而不是一次性 launch 常数。

## 3. 实际 Dispatch Graph

T=2048 的 graph replay 尾部由 rocprof kernel trace 捕获。Avelang 是 8 个
dispatch；vLLM 是 7 个，而不是假定的“完全融合”。

| 顺序 | Avelang 实际 kernel | vLLM 实际 kernel / 逻辑归属 |
|--:|:--|:--|
| 1 | `chunk_cumsum` | `chunk_local_cumsum_scalar_kernel` |
| 2 | native BT64 KKT | `chunk_scaled_dot_kkt_fwd_kernel` |
| 3 | hierarchical FP32 solve | BF16 fill + `merge_16x16_to_64x64_inverse_kernel`（solve） |
| 4 | W kernel | `recompute_w_u_fwd_kernel`（单一 W/U kernel） |
| 5 | U kernel | 无独立 U dispatch |
| 6 | frozen `qwen_gdn_bt64_gfx942_asm_v0` | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` |
| 7 | native BT64 chunk-o | `chunk_fwd_kernel_o` |
| 8 | FP32-to-BF16 cast | 无独立 cast；`chunk_fwd_kernel_o` 已写 BF16 output |

所以 Avelang 多一个 W/U dispatch 和一个 cast dispatch；vLLM 则在 solve 内拥有一
个 fill dispatch。这个结构差异是测得的，不是事前假设。

## 4. 已物化的 Global Intermediates

下表为 T=2048 capture 结果的 tensor contract。`h_bf16` 和 final state 两边相同；
不同的 dtype 是可见的全局物化契约，不代表单独已经证明某一个 dtype 改动会带来收益。

| intermediate | Avelang | vLLM |
|:--|:--|:--|
| `g_cumsum` | FP32, 64 KiB | FP32, 64 KiB |
| `a` | FP32, 4 MiB | FP32, 4 MiB |
| `a_solved` | FP32, 4 MiB | BF16, 2 MiB |
| `w` | FP32, 8 MiB | BF16, 4 MiB |
| `u` | FP32, 8 MiB | BF16, 4 MiB |
| `h_bf16` | BF16, 8 MiB | BF16, 8 MiB |
| `v_new` | FP32, 8 MiB | BF16, 4 MiB |
| output staging | FP32 `output_fp32`, 8 MiB | 无独立 FP32 output |
| public output | BF16, 4 MiB | BF16, 4 MiB |

只计算这些公开 stage 边界，Avelang 在 T=2048 已至少额外物化约 **22 MiB**：
`a_solved` 2 MiB、`w/u/v_new` 各 4 MiB、`output_fp32` 8 MiB。该数字只是 storage
下界；它不把后续 consumer reread 或 kernel 内部临时值重复计入。

## 5. Standalone Body 延迟与 Slope

body 使用同一固定输入和 graph replay/ABBA 计时，但将每一个逻辑 stage 独立 capture。
它用于定位，不要求各 body gap 在每个小 T 精确相加为 full gap。尤其在短文本，
launch 与 event resolution 会使非加性更明显。

| stage | T=2048 A-vLLM us | gap slope us/chunk | T=16384 A-vLLM us |
|:--|--:|--:|--:|
| cumsum | +6.850 | +0.001 | +5.529 |
| KKT | +18.828 | +0.878 | +219.206 |
| solve | **-14.902** | -0.100 | **-41.601** |
| W/U | +36.214 | +0.990 | +255.339 |
| recurrence (frozen) | +39.980 | +1.261 | +322.559 |
| chunk-o | **+46.169** | **+1.196** | **+309.600** |
| cast | +15.483 | +0.069 | +32.688 |

`chunk_o` 是最大的可修改项。W/U、KKT 是第二、第三候选；它们应保留为 6B 后的
排序，而不是与 chunk-o 同时改动。

## 6. T=2048 Full-Graph Resource 对比

下表来自实际 full graph 的 rocprof tail。trace us 会被 profiling 扰动，故只作同工具
下的资源结构参考。CTA 由 `grid_work_items / workgroup` 计算。W/U 是 Avelang W+U
的合计；vLLM 为一个 `recompute_w_u_fwd_kernel`。vLLM solve 包含 fill 和 merge。

| logic | A trace us / CTA / MFMA / VMEM / LDS | vLLM trace us / CTA / MFMA / VMEM / LDS |
|:--|:--|:--|
| cumsum | 12.939 / 256 / 0 / 32,768 / 0 | 2.243 / 256 / 0 / 512 / 1,536 |
| KKT | 35.532 / 4,096 / 20,480 / 210,944 / 184,320 | 7.451 / 256 / 16,384 / 26,624 / 49,152 |
| solve | 12.699 / 256 / 16,384 / 40,960 / 212,992 | fill+merge: 27.200 / 256 / 32,768 / 69,632 / 324,608 |
| W/U | 51.516 / 4,096 / 524,288 / 851,968 / 1,638,400 | 15.463 / 256 / 32,768 / 59,392 / 38,912 |
| recurrence | 145.176 / 32 / 196,608 / 91,136 / 588,928 | 106.318 / 32 / 65,536 / 58,368 / 305,472 |
| chunk-o | **66.699 / 2,048 / 458,752 / 851,968 / 1,343,488** | **14.662 / 512 / 81,920 / 71,680 / 245,760** |
| cast | 4.287 / 512 / 0 / 16,384 / 0 | not materialized |

尤其是实际 full `chunk_o`：Avelang 使用 4x CTA、5.6x MFMA、约 11.9x VMEM 和
约 5.5x LDS 指令。这与它在 timing 表中最大的可修改 slope 一致，足以支持把它排在
Stage 6B 第一位。

完整 full replay 的资源 metadata 如下。Scratch 在两边所有实际图 kernel 均为零；
occupancy 是 rocprof 原始报告值，不把它和 HIP-event timing 混用。

| logic | Avelang VGPR / AccVGPR / occupancy | vLLM VGPR / AccVGPR / occupancy |
|:--|:--|:--|
| cumsum | 4 / 4 / 1.350% | 8 / 0 / 0.294% |
| KKT | 20 / 4 / 8.919% | 72 / 16 / 3.640% |
| solve | 44 / 4 / 5.149% | fill 12 / 4 / 1.189%; merge 68 / 20 / 3.782% |
| W/U | W 52 / 4 / 42.052%; U 48 / 8 / 39.026% | 60 / 164 / 2.775% |
| recurrence | 128 / 192 / 1.201% | 104 / 160 / 0.596% |
| chunk-o | 112 / 64 / 17.666% | 100 / 36 / 9.473% |
| cast | 28 / 4 / 3.934% | not materialized |

独立 body rocprof 也已执行并保存在 `rocprof_bodies_v2/`。其 vLLM solve/WU/chunk-o
部分在新进程中出现了不同 autotune launch config，因此报告不把它们和 full graph
counter 混合；full graph 表才是实际 end-to-end launch 的权威资源表。

## 7. 解释与 6B 选择

1. **不是 solve。** Avelang standalone solve 在所有长文本点均快于 vLLM；继续优化
   solve 没有 recoverable-gap 依据。
2. **不是单独 cast 的优先级。** 它是明确的额外 launch/global read-write，但仅为
   T=2048 `15.5 us`、slope `0.069 us/chunk`。应记录，不能抢在 chunk-o 前面。
3. **不是立即改 W/U。** Avelang 的两个 W/U kernel 与 vLLM 单 kernel 的资源差距很大，
   是有价值的第二候选；但它的 body slope/长文本 gap 仍小于 chunk-o。
4. **Stage 6B：只做 native BT64 chunk-o ownership。** 目标是先降低 Avelang 的
   chunk-o CTA 数、重复 MFMA、VMEM 和 LDS 流量，接口维持当前 FP32 output staging 和
   独立 cast，不在同一 patch 中改输出 dtype 或融合 cast。这样因果归属和正确性风险都
   可控。
5. asm recurrence 保持冻结；任何 future chunk-o work 不得改变 asm/HSACO、recurrence
   ABI 或 production selector。

## 8. 复现与产物

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_full_graph_gap_stage6a/stage6a_full_graph_audit.py \
  --mode all --T 512 1024 2048 4096 8192 16384 \
  --warmup 20 --repeat 100 --body-repeat 100 --sessions 5

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_full_graph_gap_stage6a/stage6a_profile.py \
  --scope full --T 2048 --replay 3

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_full_graph_gap_stage6a/stage6a_profile.py \
  --scope bodies --T 2048 --replay 3 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/\
codex_qwen_bt64_full_graph_gap_stage6a/rocprof_bodies_v2
```

Raw results and scripts:

- `codex_qwen_bt64_full_graph_gap_stage6a/full_raw.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/full_summary.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/full_slope.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/body_raw.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/body_summary.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/body_slopes.csv`
- `codex_qwen_bt64_full_graph_gap_stage6a/frozen_measurement_contract.json`
- `codex_qwen_bt64_full_graph_gap_stage6a/rocprof/`
- `codex_qwen_bt64_full_graph_gap_stage6a/rocprof_bodies_v2/`

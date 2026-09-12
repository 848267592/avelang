# Direct-K64 全递归 P0-P2：非零 W 映射与反馈正确性审计（历史 pre-fix）

> 本文件保留 BF16 V-new boundary 修复前的诊断、停止条件与原始数值。该问题已被
> 后续 U0 审计定位并修复；最新通过结果见
> [BF16 V-new boundary fix report](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_direct_k64_update_bf16_vnew_boundary_fix_report.md)。

## 结论

本轮**没有做性能优化，也没有运行任何 benchmark**。先前登记的
`split_w0` 对 `fused_w0` 全序列性能实验已经正式撤销：递归关系要求
`pred_i` 读取 `H_i`，并由 `update_i` 产生 `H_(i+1)`；将整段序列拆开不再
保持原数学。`W=0` 也不能作为非零 W 正确性的主门槛，因为它会消去最需要验证的
pred 路径。

P0 和 P1 均通过：MFMA32 pred 的非零 W lane/fragment 映射已修正，单个 BT64
chunk 的 pred -> V-new -> V-decay -> Direct-K64 update 组合符合参考。P2 的
多 chunk feedback 阶梯在 `T=128`、`T=512` 通过；`T=2048` 在第 23 个
chunk（从 0 开始）第一次越过严格的 P0 `pred_f32=5e-5` 门槛，故按预注册
hard stop 停止，未运行余下 chunk、phase-lifetime A/B 或任何性能测试。

这个失败不能表述为“P0 pred lane mapping 又错了”。P0 的单 chunk 映射证据很强，
而 P2 首个越界发生在已经带有微小状态偏差的 feedback 后：第 23 chunk 的
`pred_f32` 最大误差为 `5.206061e-05`，但 state 最大误差仍仅
`2.279105e-04`，远低于冻结的 FP32 state `0.02` 门槛。下一步应只做
**feedback 来源分解**，而不是进行性能路线或修改寄存器分配。

## 范围与冻结条件

本轮保持以下条件不变：

- current-vLLM BF16 recurrence ABI：K/W/U/H/V-new 为 BF16；g、initial state、
  final state 为 FP32；
- `BT=64`、`BV=32`、`WG=128`、32 CTA、两个 wave 协作 ownership；
- pred 使用 `v_mfma_f32_32x32x8_bf16`；update 使用已通过 C0 的
  `block_dot_bf16_f32_staged_vdecay` / `persistent_typed_block`；
- K32 累加顺序、state 逻辑布局、global IO、输出数学均冻结；
- 未修改 production selector、external HSACO、allocator/RA、MFMA geometry、
  old broad-K/compact-K、D0-P LDS layout、ping-pong 或 double buffering。

新增实验代码：

- [P0 pred mapping](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0.py)
- [P1 单 chunk 组合](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1.py)
- [P2 feedback 阶梯](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2.py)

P2 是 host 端按 chunk 顺序调用 P1 的**正确性显微镜**：实际 `state_after`
和独立参考 state 分别前传，以记录第一处误差放大。它不代表建议的 runtime
split，不可用于延迟比较。

## P0：single-chunk 非零 W pred 映射

### 为什么先做 P0

旧 v29 系列的 nonzero-W 问题不能由 W=0 对照排除。P0 直接导出 raw FP32
MFMA accumulator、两 K-half `pred_partial`、unpacked FP32 pred、BF16 pred 与
BF16 `V-new`，因此第一处错误不会被最终 state 掩盖。

测试输入包含：

1. one-hot initial state + one-hot W + zero U；
2. one-hot initial state + one-hot W + one-hot U；
3. 单 head / 单 value / 128 个 K feature 的扫描；
4. sparse diagonal；
5. permutation；
6. low-amplitude random nonzero-W。

### 发现并修复的映射问题

首次运行的第一个错误已发生在 raw accumulator：一个 one-hot 应落在
`token=11,value=5,K=73` 的结果，实际落到相邻错误 value row，且幅度为 2。
这说明问题发生在 pred MFMA operand ownership / accumulator unpack，而不是
update 或 feedback。

错误的思路把一个 64-lane wave 当作四个 16-lane group。修正后采用 MFMA32
wave-local ownership：

```python
lane_row = lane & 31
mfma_lane_group = lane >> 5
k_vec = kpack * 2 + mfma_lane_group

out_row = lane_row
out_col = ((acc_i >> 2) * 8) + mfma_lane_group * 4 + (acc_i & 3)
```

这里 workgroup 的 `wave_id` 仍只选择 K64 half；不能替代 wave 内的
`mfma_lane_group`。该规则与项目的 MFMA intrinsic 教程一致，并在 P0 中通过
机器可判定输入验证。

### P0 结果

| 用例 | raw acc / pred partial / pred / V-new | 结果 |
|:--|:--|:--|
| one-hot state + W，U=0 | 全部 `0` | 通过 |
| one-hot state + W + U | 全部 `0` | 通过 |
| 128-feature 扫描 | 全部 `0` | 通过 |
| sparse diagonal | 全部 `0` | 通过 |
| permutation | 全部 `0` | 通过 |
| random nonzero-W | raw `9.313e-10`，pred BF16 `4.768e-07`，V-new `0` | 通过 |

此外，P0 使用唯一编码的 W/state 输入生成了 `4096` 条 empirical mapping 记录：
2 个 K-half、32 个 shift、每 shift 64 个逻辑 row，`unresolved=0`。这些不是
仅靠最终张量近似比较得到的结论。

证据：

- [P0 摘要](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p0/p0_pred_mapping_summary.json)
- [公式 lane 映射](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p0/lane_to_logical_pred_mapping.csv)
- [实测 lane 映射](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p0/empirical_lane_to_logical_pred_mapping.csv)

## P1：single-chunk fused composition

P1 只在 P0 通过后执行。每个 CTA 完成：

```text
runtime BF16 W/U + FP32 input state
  -> MFMA32 pred
  -> pred_partial unpack
  -> BF16 V-new
  -> BF16 V-decay
  -> C0 persistent typed Direct-K64 update
  -> BF16 H snapshot + FP32 state_after/final_state
```

P1 逐层比较 raw accumulator、pred partial、FP32/BF16 pred、V-new、V-decay、
delta、H、state_after 与 final_state。六个 P0 machine case 全部 finite 且通过。

| P1 用例 | 最大重要误差 | 结果 |
|:--|:--|:--|
| 前五个结构化 one-hot / scan / sparse / permutation 用例 | 所有导出中间量 `0` | 通过 |
| random nonzero-W | pred FP32 `1.863e-09`；pred BF16 `4.768e-07`；V-new `0`；V-decay `4.883e-04`；state `4.377e-05` | 通过 |

`V-decay` 与 state 的非零差异来自 BF16 V-decay 边界和 Direct-K64 更新参考的
冻结精度合同；仍低于对应 BF16 / FP32 门槛。它不是 lane permutation。

证据：[P1 摘要](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1/p1_single_chunk_composition_summary.json)。

## P1 机器资源审计

这不是性能 profiling，只用于确认正确性组合没有因隐式 spill 或 scratch 改变语义。
捕获的 HSACO metadata 为：

| 字段 | 值 |
|:--|--:|
| VGPR | `346` |
| AGPR | `90` |
| SGPR | `58` |
| LDS group segment | `36,864 B` |
| private segment / scratch | `0 B` |
| VGPR spill count | `0` |
| SGPR spill count | `0` |
| workgroup | `128` |

exact LTO replay 的 greedy 前/后、virtregrewriter 后与 prologepilog 后 MIR 均为
`SI_SPILL_AV32_SAVE=0`、`SI_SPILL_AV64_SAVE=0`。静态 ISA 还确认：

| 静态 ISA 项 | 数量 |
|:--|--:|
| `v_mfma_f32_32x32x8_bf16` | `40` |
| `v_mfma_f32_16x16x16_bf16` | `0` |
| `s_barrier` | `9` |
| LDS read / write | `61 / 140` |
| global load / store | `96 / 228` |
| scratch/spill 指令 | `0` |

这些是**静态** ISA 数，不等于动态 PMC；尤其 `VGPR=346` 也不应被误写成
rocprof 的 `Accum_VGPR_Count`。本轮没有用它们作性能结论。

证据：

- [P1 HSACO](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1_mir/hsaco/_qwen_gdn_direct_k64_bv32_full_p1_kernel.hsaco)
- [LTO MIR 摘要](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1_mir/replay/summary.json)
- [post-greedy MIR](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p1_mir/replay/kernel_section_07.mir)

## P2：feedback ladder

### 方法

对 `T=128/512/2048` 的 random low-amplitude nonzero-W 输入，P2 每个 BT64 chunk
同时推进两条链：

- actual chain：上一 chunk 的 actual FP32 `state_after`；
- reference chain：上一 chunk 的独立 FP32 reference `state_after`。

每一 chunk 记录 pred、V-new、V-decay、delta、state_after 的 max/mean 误差以及
第一处超限。P2 还记录 `H` 的 BF16 snapshot 是否等于实际输入 state 的 BF16
roundtrip。该检查在所有已执行 chunk 都为 `0`，因此当前 contract 没有把 H snapshot
误当作 BF16 feedback carrier；反馈 state 仍为 FP32。

### 结果

| 长度 | 执行 chunk | 结果 | 关键结果 |
|--:|--:|:--|:--|
| 128 | 2 / 2 | 通过 | 最后 state max `6.621e-05`；H snapshot readback `0` |
| 512 | 8 / 8 | 通过 | 最后 pred FP32 `2.804e-05`；state max `1.169e-04` |
| 2048 | 24 / 32 | 严格门槛失败，停止 | chunk 23 的 pred FP32 `5.206e-05` 首次超过 `5e-05` |

T=2048 首个超限坐标为 `[0,46,5,2]`：expected
`4.184877e-04`，actual `3.664271e-04`。在同一 chunk：

- raw MFMA accumulator max：`3.961e-05`；
- pred partial max：`3.961e-05`；
- pred BF16 max：`6.104e-05`；
- V-new max：`4.883e-04`；
- delta max：`5.601e-05`；
- FP32 final state max：`2.279e-04`；
- H snapshot vs actual input-state BF16：`0`；
- 无 NaN/Inf。

误差从 chunk 0 的 pred FP32 `9.313e-10` 开始，随 actual/reference state 的微小差异
在后续 pred 中逐渐体现；chunk 22 为 `4.996e-05`，chunk 23 首次达到
`5.206e-05`。因此诊断顺序是“**state/update 细微偏差 -> 下一 chunk pred 读取不同
BF16 state -> pred_f32 门槛越界**”，不是 P0 的一开始 lane 写错。

P2 记录：[T=128 摘要](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2_t128/p2_recurrence_feedback_summary.json)、[T=512/2048 摘要](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_recurrence_p2/p2_recurrence_feedback_summary.json)。

## 停止决策与下一步

本轮严格状态为：

```text
P0 非零 W pred mapping       PASS
P1 单 chunk pred+update      PASS
P2 T=128 / T=512             PASS
P2 T=2048 strict pred gate   FAIL at chunk 23
performance / phase lifetime NOT RUN
```

不得把 P1/P2 的 static resource 信息当作性能晋级证据，也不得把 P2 的 host
sequential microscope 接入生产。下一步只允许一个 correctness-origin 对照：固定同一
T=2048 输入，分别让下一 chunk 的 pred 读取 actual state、reference state、以及
actual state 的 BF16 snapshot；逐项比较 raw accumulator / pred / V-new。它将区分：

1. 首 chunk update 的 BF16 V-decay / delta 舍入是否是种子；
2. state feedback 的 FP32-to-BF16 operand conversion 是否放大该种子；
3. 是否仍存在仅在 composed feedback 中才出现的 pred lowering 问题。

在该对照通过前，不实施 phase-lifetime A/B、不运行性能、不改 RA 或 LDS layout，也
不把此 Direct-K64 full recurrence 推进为生产候选。

## 复现命令

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p0.py --json
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1.py --json
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p2.py --T 128 512 2048

# 仅 MIR / HSACO 审计，不是 benchmark。
export AVELANG_AMDGPU_LINK_DEBUG_DIR=.../p1_mir/link
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_recurrence_p1.py \
  --dump-hsaco-dir .../p1_mir/hsaco
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py \
  --argv-file .../p1_mir/link/amdgpu-link-0.argv.txt \
  --out-dir .../p1_mir/replay \
  --kernel _qwen_gdn_direct_k64_bv32_full_p1_kernel
```

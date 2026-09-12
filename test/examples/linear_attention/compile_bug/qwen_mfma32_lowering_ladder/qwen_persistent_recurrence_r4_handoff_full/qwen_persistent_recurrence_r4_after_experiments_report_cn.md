# Qwen GDN R4 之后的 recurrence / chunk_gdr 优化复盘

## 0. 这份报告回答什么

这是一份给后续指导者使用的技术交接报告，范围是 Qwen GDN 的 recurrence
也就是 `chunk_gdr` 中负责生成 `H` 和 `V-new` 的核心更新，以及它之后的
R4 相关实验。它回答四个问题：

1. R4 的纯 Avelang kernel 和 current-vLLM Triton kernel 分别在哪里；
2. R4 之后我们按什么顺序尝试了哪些方案；
3. 每个方案到底改善了什么，为什么没有继续成为主线；
4. 最后证据把问题收敛成了什么，而不是继续笼统地说“编译器可能不够好”。

本文只讨论 recurrence body。它不把 X2+Z5B 的整体算子结果误写成纯
Avelang：X2+Z5B 的 recurrence 是复用 current-vLLM Triton 的 Stage 6R
HSACO，Z5B 是 Avelang 的 chunk-o 路径。R4 则是我们自己用 Avelang 写的
native recurrence experimental baseline。

## 1. 交接包位置

当前目录：

`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff/`

其中：

| 内容 | 文件 |
|:--|:--|
| R4 主 kernel | `avelang/repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` |
| R4 correctness wrapper | `avelang/repro_qwen_gdn_persistent_recurrence_r4.py` |
| R4 统一 body benchmark | `avelang/bench_qwen_gdn_persistent_recurrence_r4.py` |
| R4 rocprof launcher | `avelang/run_qwen_gdn_persistent_recurrence_r4_body.py` |
| 可读 Triton kernel | `triton/chunk_delta_h.py` |
| Triton exact TTIR/TTGIR/LLVM/ISA/HSACO | `triton/kernel.ttir`、`kernel.ttgir`、`kernel.llir`、`kernel.amdgcn`、`kernel.hsaco` |
| 运行和文件说明 | `README.md` |

依赖的语义 helper 也一起复制在 `avelang/` 中，因此别人可以从 R4 wrapper
顺着 import 阅读 P0/P1/P2 和 B0 reference。原始仓库中的详细机器工件没有
全部复制进交接包，避免把数 GB 的中间产物混进代码交付；报告保留了它们在
仓库中的原始路径和 SHA/数值索引。

## 2. 两个需要先分开的实现

### 2.1 R4：我们自己写的 Avelang recurrence

文件：

`avelang/repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py`

主入口：

`_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel`

主 body 从源文件约第 40 行开始。它采用：

- gfx942；
- BT64；
- BV32；
- WG128；
- 32 CTA；
- two-wave cooperative ownership；
- K/W/U/H/V-new 为 BF16；
- g、persistent state、final state 为 FP32；
- pred 和 update 都使用 MFMA32；
- Direct-K64 update；
- `v_new = BF16(corrected)` 后再以 `FP32(v_new)` 进入 update；
- state 保持 FP32 loop-carried feedback；
- H 是 pre-update BF16 snapshot；
- R2/R3 已验证的 one-chunk lookahead 和 same-bank tail commit；
- R4 特有的 LDS-mediated K retile。

源文件里的主数据流可以简化为：

```text
current FP32 state
  -> W/U pred MFMA
  -> corrected = U - pred
  -> BF16 V-new round-trip
  -> BF16 V-decay
  -> K0/K1 Direct-K64 MFMA32 update
  -> FP32 state feedback
  -> BF16 H/V-new output and FP32 final state
```

R4 的 K 路径不是 R3 的寄存器 cross-lane packet：

```text
BF16x8 global K packet
  -> token-major LDS producer store
  -> LDS address-mediated retile
  -> consumer-side local LDS gather
  -> Direct-K64 MFMA32 operand
```

这里的 `Direct-K64` 也不要和旧 v29 的 broad-K/compact-K 标签混用。R4 的
`k_stage` 是两个 `[BT=64, K-half=64]` 区域，逻辑上对应：

```text
K[0:64, 64 tokens] + K[64:128, 64 tokens]
```

它不建立旧 broad-K 的 transposed `k_all_t/kall_vec` shared view；从“两个 K64
half、直接交给 MFMA32 consumer”的形状看更接近 compact/direct-K 思路，但
R4 不是旧 v29 compact-K kernel。R4 的固定组合是 MFMA32、BV32、WG128、
LDS-mediated retile；旧 v29 compact-K 的 MFMA16 结果不能直接当作 R4 结果。

这样做的直接动机，是删除 R3 生成的约 512 条 `ds_bpermute_b32`，同时不
改 MFMA 数学、K32 累加顺序、ownership 和 ABI。

### 2.2 current-vLLM Triton recurrence

可读源码：

`triton/chunk_delta_h.py`

kernel：

`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

它是从当前 vLLM/Flash-Linear-Attention capture 保存的 Triton Python 源码，
不是我们重新手写的 ISA。主要结构是一个 device-side `scf.for` recurrence：

```text
load initial H/state
for chunk:
    load W and run pred dot
    load U/V and form corrected V-new
    store BF16 V-new
    load g and form decay
    load K blocks
    update persistent H/state
store H snapshot and final state
```

精确 Stage 6R 编译工件在原始仓库：

`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/`

交接包中对应复制为 `triton/kernel.*`。其关键 metadata 是：

- arch `gfx942`；
- warp64；
- `num_warps=2`；
- `num_stages=2`；
- dynamic shared `40960 B`；
- symbol `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`；
- grid `(4,8,1)`，WG128，合计 32 CTA；
- HSACO SHA256：
  `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`。

要严格区分两类证据：可读 Python 源码和 exact Stage 6R HSACO 是分开保存
的历史工件，但 symbol、ABI 和 source location 对得上。不能把它们描述成
“同一份 Python 文件经过一次可复现 hash 链直接得到的 HSACO”。

## 3. R4 本身的基线结果

R4 详细报告：

`../qwen_persistent_recurrence_r4_lds_mediated_retile_report.md`

R4 的 correctness 已覆盖 T=64/128/512/2048，与 R1/R2/R3/B0/P2 对比通过，
并且没有 scratch、MIR spill。T=2048 机器资源如下；这里的动态数来自
rocprof，静态资源来自 code object，不能混用：

| 项目 | R4 |
|:--|--:|
| dynamic MFMA | 65,536 |
| dynamic VMEM | 202,240 |
| dynamic VALU | 2,294,144 |
| dynamic SALU | 163,072 |
| dynamic LDS instructions | 381,952 |
| LDS block | 53,248 B |
| VGPR / AccVGPR / SGPR | 128 / 192 / 112 |
| scratch | 0 |
| spill | 0 |
| R4 HSACO SHA256 | `1f80bc215d7373c763d7761db083217af9d9cd9d5830d91f5d83d4bac21b70a4` |

body benchmark 的统一表如下。它不是 Eager public API 最终排名，而是把
recurrence body 放进同一个预分配、HIP-event、fresh-process harness 的
诊断比较：

| T | B0 ms | R1 ms | R2 ms | R3 ms | R4 ms | direct Triton ms | external bridge ms |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 0.156633 | 0.139187 | 0.137905 | 0.157273 | 0.133698 | 0.073830 | 0.040741 |
| 1024 | 0.357292 | 0.283502 | 0.278634 | 0.294538 | 0.269541 | 0.103394 | 0.068402 |
| 2048 | 0.677707 | 0.530889 | 0.521675 | 0.547474 | 0.501926 | 0.156192 | 0.118616 |
| 8192 | 2.778611 | 2.125920 | 2.090668 | 2.072541 | 2.007665 | 0.472783 | 0.421246 |

对应 per-chunk slope：

| arm | slope us/chunk |
|:--|--:|
| B0 | 21.771093 |
| R1 | 16.526212 |
| R2 | 16.249126 |
| R3 | 15.918738 |
| R4 | 15.590970 |
| direct Triton | 3.311115 |
| external bridge | 3.160771 |

所以 R4 相对 direct Triton 的 body 比例约为：

- T=512：`1.81x`；
- T=1024：`2.61x`；
- T=2048：`3.21x`；
- T=8192：`4.25x`；
- slope：约 `4.71x`。

这个结果说明 R4 已经修掉了 R3 的明显 cross-lane 代价，但距离 Triton 的
长期 per-chunk 数据流仍然很远。

## 4. R4 之后实验时间线

下面按“问题假设 -> 实验 -> 结果 -> 保留结论”的顺序记录。R4-tail 是
从 R4 主线派生的 tail/production attribution 分支，不能把它和 R4 主
baseline 的源文件混成同一个版本。

### 4.1 R4-tail：先拆 gap，而不是盲目改 kernel

相关原始报告和脚本在：

- `vllm_compare/qwen_persistent_recurrence_r4_tail_pred_mn_axis_swap_report.md`
- `vllm_compare/qwen_persistent_recurrence_r4_tail_iopacket_wide_access_report.md`
- `vllm_compare/bench_qwen_gdn_r4_tail_vs_current_vllm_eager.py`

第一步做了 production attribution 和 full-attention bucket audit，没有把
它当成优化结果。T=2048、以物理 V32/chunk 归一化后，R4-tail 对当前 vLLM
的差距约为：

| 指令类别 | R4-tail | current vLLM | 差值 |
|:--|--:|--:|--:|
| MFMA | 64 | 64 | 0 |
| VMEM | 197.5 | 57 | +140.5 |
| VALU | 2187.375 | 1362.875 | +824.5 |
| SALU | 159.25 | 63.313 | +95.938 |
| LDS | 373 | 298.313 | +74.688 |

更细的归因是：

- VMEM 差值的约 99.7% 可以由 U/g 和 H/V-new 相关 I/O bucket 解释；
- R4-tail 的 H/V-new/terminal store bucket 约 98 条，U/g bucket 约 66 条，
  末端 W/K 只约 32 条；
- VALU 最大的新增 bucket 是 update operand preparation，约 +490；
- tail/loop W-K 约 +373.5；
- SALU 至少约 +131.063 来自 16 组重复 mask/reconvergence/control。

这一步把问题从“是不是 K MFMA 慢”改成了“operand ownership、I/O packet
和 fragment feeding 是否造成重复工作”。

### 4.2 State-KV pred M/N swap 和 U/V-new packet I/O

相关报告：

- `vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_report.md`
- `vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_v4_io_report.md`
- `vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_v4_io_v3_performance_report.md`

尝试的理由：Triton 的 pred operand ownership 与 Avelang 的 state/KV 组织不同，
而 U/V-new 的 BF16 I/O 也明显存在窄 load/store 机会。因此分别尝试：

1. 调整 pred M/N 轴和 state-KV B fragment 归属，并修正 MFMA physical operand
   顺序；
2. 用 T1xV4 BF16 b64 packet 处理 U/V-new I/O。

correctness 通过。v3 的 T=2048 动态数据是：

| 项目 | v3 每 dispatch | 归一化每 V32/chunk |
|:--|--:|--:|
| VMEM | 123,904 | 121 |
| LDS | 451,584 | 441 |
| MFMA | 65,536 | 64 |
| VALU | 1,927,712 | 1882.531 |
| SALU | 208,096 | 203.219 |

U/V-new 的 I/O 子路径从 32 条 VMEM 降为 8 条，减少 75%，但没有消除
整体 LDS/operand feeding 代价。public Eager fresh-process 样本：

| T | R4-tail ms | v3 ms | current vLLM ms |
|--:|--:|--:|--:|
| 1024 | 0.224043 | 0.186978 | 0.104976 |
| 2048 | 0.382338 | 0.327226 | 0.155040 |
| 8192 | 1.412989 | 1.190058 | 0.457039 |

v3 slope 约 `8.956 us/chunk`，current vLLM 约 `3.143 us/chunk`，约
`2.85x`。结论是：宽 I/O 有真实收益，但只消除了一个明确子路径，不能
解决完整 operand feeding gap。

### 4.3 U/V-new LDS bridge：静态宽 store 没有变成有效 packet pipeline

报告：

`vllm_compare/qwen_persistent_recurrence_r4_tail_u_vnew_lds_bridge_report.md`

假设是：把 U/V-new 从 scalar 路径先写入一个已有死 pred plane 的 LDS bridge，
也许可以让后面的 consumer 重用。结果 correctness 正确，但 lowering 生成了
exec-mask packet loop/backedge，导致动态机器工作反而增加：

| T=2048 | R4 | bridge |
|:--|--:|--:|
| VMEM | 202,240 | 632,320 |
| SALU | 163,072 | 1,228,480 |
| VALU | 2,239,872 | 2,774,720 |
| LDS | 381,952 | 422,912 |
| MFMA | unchanged | unchanged |

结论：看到 ISA 中有 `buffer_load/store_dwordx4` 并不等于运行时就是一次
高效的 packet producer。真正要匹配的是 ownership、mask、控制流和
lowering；“把某一条 load 改宽”不是充分条件。

### 4.4 State-KV dual-dot / typed fragment 尝试

相关报告：

- `vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_dual_dot_report.md`
- `vllm_compare/qwen_persistent_recurrence_r4_tail_iopacket_wide_access_report.md`

dual-dot 变体在 source/correctness 层面有候选，但最终 ISA gate 没有形成
预期的 producer/consumer 结构，因此没有进入 PMC/Eager 正式结论。

另一个 same-source typed BF16x4 fragment 尝试把 vector.extract/from_elements
换成 `extract_strided_slice`。它 correctness 通过，但 final HSACO 与 ISA
完全没有变化：

- HSACO SHA256：
  `7d7eb721de8109aee0264393854ef814f732ce12a7ead1b8f6c8a5c6453a47ea5`；
- ISA SHA256：
  `c19050518e51fbcd6e993e955825896ba488f5704baef76af1508d45709ed255`。

这是一条重要的 negative result：高级代码里换一个看起来更“向量化”的
表达式，如果在 block-dot / LLVM lowering 处被重新规范化，机器图可以完全
不变。下一步必须提供 first-class typed dot-fragment 或 address/lifetime
表示，而不是继续改表面 AST。

### 4.5 BV consume tile sweep

这组实验只改变 logical BV consume 组织，物理 MFMA 仍是 BV32，目的在于
确认是不是简单把 BV 改成 16 或 64 就能接近 Triton。T=2048 动态数据：

| logical arm | MFMA | VALU | VMEM | LDS |
|:--|--:|--:|--:|--:|
| BV16 | 131,072 | 4,771,584 | 407,552 | 935,936 |
| BV32 | 65,536 | 2,239,872 | 202,240 | 381,952 |
| BV64 | 98,304 | 3,227,968 | 187,392 | 475,648 |

T=8192 body：

- R4/BV32：`1.401803 ms`；
- BV16：`1.617833 ms`，约 `1.147x`；
- BV64：`3.819003 ms`，约 `2.708x`。

BV64 的 serial subtiles/four-wave 组织反而更差，所以没有创建 joint_v6。
这排除了“只要把 BV 调大就会自然接近 Triton”的想法。

### 4.6 Microtile、last-use 和 software-pipeline sweep

这些实验想回答：如果保持数学和 MFMA 不变，只改变 W/K issue、LDS last-use、
tail commit 或 chunk 间距，能否把 Triton 的隐藏 latency 能力搬过来。

涉及的脚本和工件包括：

- `bench_qwen_gdn_persistent_recurrence_microtile_d1_w_granularity.py`；
- `bench_qwen_gdn_persistent_recurrence_microtile_top3.py`；
- `bench_qwen_gdn_persistent_recurrence_core_lastuse.py`；
- `bench_qwen_gdn_persistent_recurrence_software_pipeline.py`；
- `bv_consume_artifacts_t2048/`；
- `microtile_search_artifacts/`；
- `microtile_d1_w_granularity_artifacts/`。

结果很一致：

- d0 的多个 issue/tail plan 机器计数基本和 R4-tail 相同，差异只有噪声；
- `lastuse+d1` 会把 HSA VGPR 推到约 236--256，而 R4-tail 约 228，资源
  压力变差；
- d1 不是免费的 overlap：T=8192 R4-tail 约 `1.410786 ms`，w1/w2/w4
  分别约 `1.425918/1.440290/1.443805 ms`；
- 对应 slope 约增加 `0.92%/2.50%/2.91%`；
- 没有 spill，但也没有真实的 dynamic global-to-LDS async copy 或有效
  overlap。

结论是：当前 opaque stage token 的 payload 仍然以 SSA/register value
形式存在。把 issue 在源码里提前，不会自动产生 Triton 那种非寄存器驻留的
异步 copy；它往往只会拉长 live range，最后增加 wait 或寄存器压力。

### 4.7 R5 full superblock lowering

报告：

`../qwen_persistent_recurrence_r5_superblock_lowering_report.md`

R5 试图在一个 late superblock lowering 中统一安排：

```text
W pred -> corrected -> BF16 V-new -> V-decay -> K update -> FP32 state -> next chunk
```

这次不是只做 isolated K 或 pred，而是完整 recurrence candidate。它确实
改变了 MLIR、LLVM、MIR、ISA/HSACO 的静态图，但没有改变实际动态工作：

| T=2048 | R4 | R5 | 差值 |
|:--|--:|--:|--:|
| body ms | 0.500084 | 0.509197 | +9.113 us |
| dynamic MFMA | 65,536 | 65,536 | 0 |
| dynamic VMEM | 202,240 | 202,240 | 0 |
| dynamic VALU | 2,294,144 | 2,294,144 | 0 |
| dynamic SALU | 163,072 | 163,072 | 0 |
| dynamic LDS | 381,952 | 381,952 | 0 |

T=8192：R4 约 `2.007603 ms`，R5 约 `2.037427 ms`，回退约 `29.824 us`。
R4 slope 约 `15.604582 us/chunk`，R5 约 `15.825837 us/chunk`，增加约
`1.42%`。

静态图虽然出现了 global load/store、ds read/write、wait/barrier 数量的
变化，但动态 PMC 没有下降。根因仍是 stage token 没有被 lower 成真正的
非寄存器异步传输和 completion/wait primitive；它只是改变了 SSA issue
位置和静态控制流。R5 因此 No-Go，但它证明“planner 能改变机器图”并不等于
“planner 已经改变了有效运行时数据流”。

## 5. 最终把问题抽象成什么

### 5.1 已经基本排除的解释

**不是 MFMA 数学工作太多。** R4、R4-tail、R5 与 current Triton 的
MFMA 数在对齐实验中一致或可按同一物理 tile 归一化；差距主要出现在 VMEM、
VALU、SALU、LDS 和 feeding。

**不是简单的 BV 选择。** BV16 增加了工作，BV64 造成 serial subtiles 和
多 wave 组织退化；BV32 是当前合理的 native source ownership。

**不是单一 scalar load 宽度。** U/V-new b64 packet 有明确局部收益，但
LDS bridge 的 wide load 反而因为 mask/backedge 变慢；静态 dwordx4 不是
动态 packet pipeline 的证明。

**不是 register allocator 单独造成的 spill。** R4 的 scratch/spill 为零，
R5 也没有资源灾难却仍然比 R4 慢。继续改 RA 不能解释主要 gap。

**不是把 issue 提前就自动有 overlap。** R5 和 software-pipeline sweep
说明如果没有真正的 async copy/token completion，提前 issue 只是改变 SSA
live range、waitcnt 和寄存器压力。

### 5.2 目前最有证据的根因

问题应该抽象为：

> Avelang 当前可以表达 recurrence 的数学和 MFMA，但还不能自然地把
> Triton 使用的“typed distributed producer -> swizzled/rotating shared
> layout -> typed dot operand consumer -> persistent update”完整保留到
> gfx942 machine lowering；结果是 operand ownership、global I/O packet、
> LDS materialization、fragment reconstruction 和跨 phase lifetime 被拆成
> 多个普通 SSA/vector/memref 步骤，产生重复的机器工作。

更具体的证据是 R4-tail 每个物理 V32/chunk 相对 current-vLLM：

| 类别 | R4-tail | Triton | 相对倍数 |
|:--|--:|--:|--:|
| VMEM | 197.5 | 57 | 3.46x |
| VALU | 2187.375 | 1362.875 | 1.61x |
| SALU | 159.25 | 63.313 | 2.52x |
| LDS | 373 | 298.313 | 1.25x |
| MFMA | 64 | 64 | 1.00x |

R4-tail attribution 进一步显示，VMEM gap 主要由 U/g 和 H/V-new/global
output 相关 I/O 造成；VALU gap 的最大 bucket 是 update operand preparation，
并伴随 W/K loop/address/control 计算。也就是说，最大问题是**数据流和
表示在 lowering 中丢失**，不是核心矩阵乘法本身。

### 5.3 “这是编译器问题”应该怎样严谨表述

可以说：有强证据指向 Avelang 的 lowering/IR 表示能力，而不是说“任何
性能差都必然是 compiler bug”。证据链是：

1. 同一数学工作和 MFMA geometry 下，Avelang 与 Triton 的主要差异出现在
   operand materialization、地址/布局和 I/O 数量；
2. 高级源码换成看似等价的 vector/fragment 表达后，某些候选在最终 ISA
   完全收敛，HSACO/ISA SHA 一致，说明表面高级代码没有控制住 machine graph；
3. U/V-new 的宽 packet 实验在真正的 producer ownership/straight-line
   lowering 形成时有收益，但把单条 load 改宽并不能普遍解决问题；
4. R4/R5 无 spill，且 R5 只改变静态计划、不改变动态 PMC，说明需要控制的
   是 typed operand、shared layout、async completion 和 lifetime 表示；
5. Triton TTGIR 明确保留了 `amd_mfma`、swizzled shared、rotating shared
   等布局信息，而当前 Avelang 某些路径在 lower 前后会退化成 generic
   vector/memref/i32/view/fragment 重建。

因此更准确的结论是：**当前 Avelang compiler 的中间表示和目标 lowering
还没有把这类完整 producer-consumer contract 保留下来；这是 compiler
capability/lowering gap 的证据。** 它不是证明高级算法永远不能优化，也
不是证明只改一条 LLVM 指令就能得到 Triton 性能。

## 6. 为什么不能直接“抄 Triton 汇编”解决

Triton 的 ISA/HSACO 可以作为：

- lane/ownership/layout 的观察 oracle；
- 指令类别、LDS/shared encoding、MFMA operand feeding 的机器证据；
- external HSACO bridge 的诊断 control。

但直接把 ISA 抄成 Avelang kernel 不是正确的 native 解决方案，原因是：

1. ISA 中的寄存器分配、waitcnt、barrier、LDS offset 和 packet order 是
   特定 ABI/shape/编译器版本下的结果；
2. 它绕过了 Avelang 的 source-to-MLIR-to-LLVM contract，不能证明 Avelang
   能表达这个算子；
3. 它会变成 external-kernel integration，而不是 Avelang source kernel；
4. 一旦 shape、T、ABI、ROCm 或 symbol 改变，手写/复用的 code object 可能
   失效；
5. 我们已经有成功的 Stage 6R external bridge，但它的含义是“复用 Triton
   code object 作为性能 control”，不是“已经用 Avelang 实现了 Triton”。

真正值得交给 compiler 团队的问题应是：能否提供通用的 first-class typed
dot operand、distributed/shared encoding、async copy/completion token 和
明确的 phase lifetime，使同一份高层 block-dot/recurrence source 在
generic/specialized lowering A/B 中只改变 lowering 而不改变数学和 ownership。

## 6.1 R4 相关的 compiler 文件地图

下面是继续优化时应优先交给指导者看的源码边界。它们不是“每个文件都只
为 R4 改过”的声明，而是 R4 所依赖的 compiler surface；具体 commit 中还
包含了前后阶段的实验能力。

### Recurrence op、planner 和 phase lowering

- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.{cc,h}`：
  把高层 persistent recurrence 组织成 compiler 可识别的 recurrence region。
- `lib/Dialect/AveLang/Transforms/qwen_recurrence_schedule_plan.h`：
  `QwenRecurrenceSchedulePlan`，统一描述 pred/update、current/next stage、
  shared bank 和 lifetime。
- `lib/Dialect/AveLang/Transforms/lower_qwen_gdn_recurrence_step_pass.{cc,h}`：
  recurrence step 的语义和 BF16 V-new/state feedback lowering。
- `lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.{cc,h}`：
  K64/K32 pipeline stage、producer issue 和 stage commit。
- `lib/Dialect/AveLang/Transforms/lower_qwen_kfrag_lds_pass.{cc,h}`：
  R4 使用的 K fragment LDS-mediated retile 路径。
- `lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.{cc,h}`：
  K producer 到 MFMA consumer 的 producer/consumer 关系重写。

### Block-dot、layout 和 IR 表达

- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.{cc,h}`：
  `block_dot_bf16_f32` 以及 logical/typed operand 的 late lowering；后续
  first-class dot operand A/B 应从这里进入，而不是在 Qwen Python 中复制
  一整套固定 ISA。
- `lib/Dialect/AveLang/IR/AveLangOps.{td,h,cc}`：AveLang operation/intrinsic
  定义和 verifier。
- `lib/Dialect/AveLang/IR/AveLangAttrs.{td,h,cc}`：如果要把 distributed、
  swizzled、rotating shared encoding 保留到后端，应在 attribute/type 层表达。
- `lib/Dialect/AveLang/IR/static_physical_layout.{h,cc}`：静态 physical layout
  和 affine mapping 的通用表示，不应写成 Qwen-specific 地址表。
- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc` 与
  `lower_gpuop_to_intrinsics_pass.cc`：观察 typed/layout 信息在哪一步退化
  为 generic memref/vector/i32 view。

### 后端证据与调试

- `lib/Target/AMDGPU/gpu_to_amdgpu_pipeline.cc`、
  `lib/Target/GPU/lower_to_llvm.cc`：检查 layout/operand 表示到 LLVM 的丢失
  位置。
- `lib/Target/AMDGPU/amdgpu_backend.cc`：记录 pre-link bitcode、full-LTO
  replay argv 和 code object；它帮助做 exact MIR/ISA 审计，但不是 R4 数学
  或性能路径的修复。

这张地图对应的核心追问是：在 `qwen_persistent_recurrence_pass` 形成
plan 之后，到 `lower_qwen_block_dot_pass`、GPU-to-LLVM、AMDGPU LTO 的哪一
层，Triton 的 typed shared/dot operand 语义第一次变成了普通 scalar/vector
地址和 fragment 重建。找到这个收敛点，比继续添加 `w_next`、`k_next` 或
单条 `ds_read` 变体更重要。

## 7. 当前最值得继续指导的方向

不要再从 R4 继续做没有证据的 BV、单条 ds_read、store width 或 RA sweep。
应以这两个可复现 arm 为起点：

### Arm A：R4 native baseline

使用交接包中的 R4 source，保留现有数学、ABI、ownership 和 K retile，继续
定位完整 I/O packet 和 typed consumer feeding。

### Arm B：current Triton machine oracle

使用 `triton/chunk_delta_h.py` 和 `triton/kernel.ttgir`/`kernel.amdgcn`
对照实际 mapping。重点回答：

- 一个 global packet 到底服务多少个 consumer；
- 哪些数据在 shared 中跨 dot consumer 复用；
- 哪些 LDS read 是 typed dot operand 而不是 generic gather；
- 哪些 load/store 是数学必需，哪些是 Avelang materialization 造成；
- 哪个 Avelang IR pass 第一次丢失了 encoding/ownership 信息。

推荐的下一项 compiler 实验不是“重写整个 kernel”，而是对同一 high-level
block-dot source 做 strict same-source A/B：

```text
same pre-lowering MLIR
  -> generic vector/memref lowering
  -> specialized gfx942 typed shared/dot-operand lowering
```

要求最终保留到 LLVM/MIR/ISA，并比较 dynamic VMEM/LDS/VALU/SALU、live range、
scratch/spill 和 latency。如果 specialized 只在 MLIR 里存在，到了 LLVM/ISA
又收敛，就继续追 representation loss；如果机器图真正不同，再判断是哪一类
数据流收益。

## 8. 给指导者的一句话版本

R4 已经证明 Avelang 能正确实现完整 BF16 recurrence 和 MFMA32，但它与
Triton 的差距不在 MFMA 数，而在 Triton 保留的 typed distributed/shared
operand 数据流在 Avelang lowering 中被拆成了较多 global I/O、LDS
materialization、地址/fragment 重建和控制工作；R4 之后的 packet、BV、
microtile、pipeline 和 R5 实验分别排除了单点宽 load、BV、提前 issue 和
普通 superblock 能解决问题，下一步应围绕 first-class typed dot/shared
encoding 与真正的 async copy/lifetime contract 做 same-source compiler A/B。

## 9. 原始详细资料索引

核心 R4/R5：

- `../qwen_persistent_recurrence_r4_lds_mediated_retile_report.md`
- `../qwen_persistent_recurrence_r5_superblock_lowering_report.md`
- `../qwen_persistent_recurrence_r0_architecture_and_legacy_lowering_report.md`
- `../qwen_persistent_recurrence_r1_joint_planner_report.md`
- `../qwen_persistent_recurrence_r2_joint_planner_report.md`
- `../qwen_persistent_recurrence_r3_full_typed_operand_pipeline_report.md`

R4-tail 与 I/O：

- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_pred_mn_axis_swap_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_iopacket_wide_access_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_v4_io_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_pred_mn_swap_v4_io_v3_performance_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_u_vnew_lds_bridge_report.md`
- `../../../vllm_compare/qwen_persistent_recurrence_r4_tail_state_kv_dual_dot_report.md`

BV、microtile、pipeline 工件：

- `../../../vllm_compare/bench_qwen_gdn_persistent_recurrence_bv_consume.py`
- `../../../vllm_compare/bench_qwen_gdn_persistent_recurrence_microtile_d1_w_granularity.py`
- `../../../vllm_compare/bench_qwen_gdn_persistent_recurrence_core_lastuse.py`
- `../../../vllm_compare/bench_qwen_gdn_persistent_recurrence_software_pipeline.py`
- `../../../vllm_compare/bv_consume_artifacts_t2048/`
- `../../../vllm_compare/microtile_search_artifacts/`
- `../../../vllm_compare/microtile_d1_w_granularity_artifacts/`

Triton exact artifact：

- `../codex_qwen_bt64_recurrence_reconciliation_stage6r/current_kernels/vllm/`

## 10. 交付边界

本交接包没有把任何 R4 变体提升为 production，也没有把 external Triton
HSACO 伪装成 Avelang 编译结果。benchmark 是 body diagnostic；最终如果要
提交性能结论，还需要在同一设备、同一 public API、同一输入和同一 Eager
口径下重新确认。报告中的历史数字用于学习和复盘，不应在硬件、ROCm 或
selector 改变后直接当作新鲜 benchmark。

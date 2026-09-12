# Avelang / Qwen GDN 在 AMD gfx942 上的算子优化完整学习复盘

> 面向第一次进行 GPU 算子优化的学习版完整报告  
> 覆盖：正确性修复、v14–v29 kernel 演进、编译器 lowering 与寄存器压力调查、Stage 4–6W 系统优化、失败路径、测量合同与下一步推导

## 0. 文档定位与合并说明

这份文档由以下三份材料整理而成：

1. `Avelang_AMD_MFMA_编译器问题与性能优化交接_更新版.md`：作为前篇，保留 FullOp/CSE
   correctness 修复、v14–v29 演进、MFMA32 与 compiler lowering 调查；
2. `qwen_gfx942_bt64_stage4_to_stage6s_complete_cn(1).md`：作为历史快照使用；
3. `report_final.md`：作为 Stage 4–6W 的当前权威记录。

第二份文件与第三份文件的大部分内容重复，而且它在 Stage 6T–6W 完成之前仍把后续内容写成
“未来计划”。因此本报告没有把它再次原样拼接，而是以 `report_final.md` 的已完成结果替代其
重复段落。这样既保留 Stage 4–6S 的全部实验链，又避免同一个阶段出现两套相互冲突的当前状态。

### 0.1 本报告保留什么

- 每个主要版本或 Stage 的问题背景；
- 为什么提出该优化假设；
- 修改了什么代码结构或数据边界；
- correctness、延迟、资源、ISA/profiler 结果；
- 失败方案与失败原因；
- 如何由本轮证据推导下一轮实验；
- 哪些实现进入 production、experimental baseline 或仅作为证据保留。

### 0.2 本报告不做什么

- 不把未实现的设想写成已完成结果；
- 不用模型知识补造源报告中不存在的性能数字；
- 不把 isolated kernel、Graph replay 或 profiler trace 当作正式排行榜；
- 不删除负结果，只删除逐字重复的段落。

### 0.3 最重要的测量口径

文中早期版本保留了当时的 body、Graph、rocprof 或独立 kernel 测量，因为这些数据对解释
优化方向有历史价值。但从 Stage 6T 开始，正式 correctness 与性能排名的权威合同是：

```text
完整 Eager public API
包含真实 dispatch、cast、allocation/边界等公开路径成本
timing_contract = eager_public_api
cuda_graph_used = false
```

以下数据只能用于诊断，不能单独支撑“整体更快”的结论：

```text
CUDA/HIP Graph replay
private kernel body
isolated HSACO
rocprof trace latency
单独某条 ISA 指令数量
```

## 目录（按学习主线）

- 第 0–3 章：合并原则、测量合同、算法数据流与术语
- Part I：FullOp/CSE correctness、v14–v24 稳定优化、v25/v29 大 tile 探索、compiler lowering 与 RA 调查
- Part II-A：Stage 4 非 recurrence MFMA 化
- Part II-B：Stage 5 solve 与 transient-state 审计
- Part II-C：Stage 6A–6S full graph、current-vLLM specialization 与 BF16 recurrence bridge
- Part II-D：Stage 6T–6W fusion、BF16 boundary、predicate collapse 与 paired Eager confirmation
- Part III：成功/失败索引、实验模板与最终方法总结

> 本报告很长。第一次阅读时可先看每个 Stage 的“为什么做”“结果”“决策”，再回看代码与 ISA 细节。

## 1. 建议的学习顺序

第一次学习算子优化时，不建议直接从最后的最快版本向前看。推荐顺序：

```text
第一遍：读第 2–4 章，理解硬件、数据流和测量方法
第二遍：读 Part I，理解如何从 correctness bug 和早期瓶颈一步步推进到 v24/v29
第三遍：读 Part II，理解如何从 Stage 4 的局部 kernel 优化推进到 6W 的 full public API 优化
第四遍：对照“实验账本”和失败索引，练习自己预测下一步
```

每个实验都可以用五个问题复盘：

1. 当前最可信的瓶颈证据是什么？
2. 这次只改变了哪个主要变量？
3. correctness 如何证明？
4. standalone、full graph 和 public API 是否一致？
5. 结果支持继续、停止还是转向哪条路线？

## 2. 核心算法与整体数据流

Qwen GDN / Qwen3Next linear attention 不是一个单独 kernel，而是一条多阶段 pipeline：

```text
g cumsum
  -> KKT
  -> solve
  -> W/U
  -> chunk_gdr recurrence
  -> chunk_o
  -> BF16 public output
```

各阶段可以先这样理解：

| 阶段 | 主要作用 | 为什么可能成为瓶颈 |
|---|---|---|
| `g cumsum` | 对 gate/decay 相关量做前缀累积 | 有多个下游 consumer，不容易简单融合 |
| KKT | 计算 chunk 内 token-token 的 K 相关矩阵，并施加 causal/gate/decay | 本质是矩阵乘，但通用 scalar 写法会浪费 MFMA 能力 |
| solve | 求解 chunk 内 lower-triangular 依赖 | row 之间有依赖，错误 schedule 会串行化 |
| W/U | 根据 solved 系数与 K/V 计算 recurrence 输入 | 涉及 FP32 系数、BF16 MFMA 和 residual correction |
| `chunk_gdr` recurrence | 根据当前 state 计算 pred、`v_new` 和 state update | chunk 间串行，是最难融合、最容易产生长 live range 的部分 |
| `chunk_o` | 组合 state contribution 与 chunk 内 causal contribution | Q/K/V/H 复用、score 重算和 output boundary 都会影响性能 |
| public output | 返回 BF16 输出与可选 final state | 额外 FP32 staging/cast 可能成为可删除的边界成本 |

固定 shape 中常见符号：

| 符号 | 含义 | 本项目典型值 |
|---|---|---:|
| B | batch size | 1 |
| T | sequence length | 512、1024、2048、8192 等 |
| Hk | key head 数（TP4 per-rank） | 4 |
| Hv | value head 数（TP4 per-rank） | 8 |
| K | key feature dimension | 128 |
| V | value feature dimension | 128 |
| BT | token/chunk tile size | 16、32、64 |
| BV | value tile size | 常见 16、32 |

recurrence 的核心数据流近似为：

```text
pred_i = W_i @ H_i^T
v_new_i = U_i - pred_i
v_decay_i = v_new_i * decay_i

delta_i = v_decay_i^T @ K_i
H_{i+1} = H_i * exp(g_last_i) + delta_i
```

其中 `H_{i+1}` 会成为下一 chunk 的输入，因此 chunk 之间存在真实串行依赖。这一点直接限制了
kernel fusion、kernel split 和跨 chunk 并行方式：很多看似能减少寄存器的拆分方案，会引入
每 chunk launch 与 global handoff，甚至破坏计算顺序。

## 3. 专业术语与硬件概念解释

本章是学习辅助说明，用来解释源报告中的术语；它不新增实验结论。

### 3.1 GPU 执行层级

| 术语 | 含义 | 在本项目中的作用 |
|---|---|---|
| kernel | 一次在 GPU 上执行的函数 | KKT、solve、W/U、recurrence、chunk-o 都可能是独立 kernel |
| grid | 一次 launch 的全部工作单元集合 | 决定总 CTA/workgroup 数量 |
| CTA / workgroup | 一组可共享 LDS、可使用 barrier 同步的线程 | AMD 文档常称 workgroup；报告中有时沿用 CTA |
| wave / wavefront | AMD GPU 的基本 SIMD 执行组，gfx942 通常为 64 lanes | divergent branch 会让同一 wave 分多次执行不同路径 |
| lane | wave 中的单个线程位置 | MFMA fragment 的元素归属通常与 lane 编号有关 |
| ownership | 哪个 CTA/wave/lane 负责哪块数据或输出 | ownership 不合理会造成重复 load、重复 compute 或串行瓶颈 |
| tile / subtile | 把大矩阵切成的小块 | 例如 BT64 拆成 4×4 个 16×16 token tile |

### 3.2 数据类型与矩阵指令

| 术语 | 含义 | 学习重点 |
|---|---|---|
| BF16 | 16 位浮点，指数范围接近 FP32，但尾数较短 | 省带宽和存储，但量化误差可能被 recurrence 放大 |
| FP32 accumulation | 输入可为 BF16，但累加器用 FP32 | 提高矩阵乘累加稳定性 |
| MFMA | AMD Matrix Fused Multiply-Add 指令 | 类似矩阵乘加硬件原语，是本项目主要计算加速手段 |
| MFMA16 / MFMA32 | 输出 tile/指令形状不同的 MFMA 变体 | 更大形状不必然更快，可能增加 accumulator footprint |
| accumulator | 矩阵乘加的累积结果寄存器 | pred 与 update 必须使用独立 accumulator 初值 |
| residual correction | BF16 主乘法后，用残差补偿 FP32→BF16 量化损失 | 删除它可能更快，但可能破坏 full correctness |

### 3.3 GPU 存储和寄存器

| 术语 | 含义 | 学习重点 |
|---|---|---|
| global memory / VMEM | GPU 显存及其向量访存指令 | 容量大但延迟高，中间量 materialization 会产生写回与重读 |
| LDS | workgroup 内共享的低延迟片上内存 | 可复用 tile，但容量、barrier 和地址计算都有成本 |
| VGPR | 每 lane 的向量通用寄存器 | 数量过高会降低 occupancy |
| SGPR | wave 共享的标量寄存器 | 常保存地址、循环和 uniform 控制信息 |
| AGPR / AccVGPR | MFMA accumulator 使用的寄存器区域 | 高压力下普通 temporary 也可能被 RA 放入 AGPR 区域 |
| scratch | 寄存器放不下后使用的私有显存空间 | 常意味着 spill，并带来额外 VMEM traffic |
| spill | virtual register 被写到 scratch、以后再读回 | 可能造成明显性能 cliff |
| occupancy | 一个计算单元可同时驻留的 wave/workgroup 数 | VGPR、AccVGPR、LDS 增长都可能降低 occupancy |
| resource cliff | 资源超过某个阈值后，occupancy 或 spill 突然恶化 | 解释了“减少工作量却反而更慢”的多次实验 |

### 3.4 编译器层级

| 术语 | 含义 | 在本项目中的作用 |
|---|---|---|
| AveLang / MLIR dialect | 高层算子与 GPU 表达所在的 IR 层 | 负责 tile、shared memory、MFMA 等语义 |
| lowering | 从高层 IR 逐步变成更低层 IR/机器指令 | 过早 lowering 会丢失专用 fragment 语义 |
| TableGen | LLVM/MLIR 中声明 operation、trait 等的机制 | FullOp 的 `Pure` trait 就在此处修复 |
| SSA | 每个值只定义一次的 IR 表示 | 源级变量少不代表机器级 live range 一定短 |
| CSE | Common Subexpression Elimination，公共子表达式消除 | 错误地合并两个独立 FullOp 导致 accumulator 污染 |
| Pure trait | 表示 operation 无副作用、相同输入可视为相同结果 | 对 materialize mutable storage 的 FullOp 不成立 |
| LLVM IR | 更低层、接近目标机器的中间表示 | 用于确认 update MFMA 是否错误继承 pred accumulator |
| MIR | LLVM 后端寄存器分配前后的机器中间表示 | 用于审计 virtual register、spill 和 live interval |
| ISA | 最终 GPU 汇编指令 | 用于核对 MFMA、load、barrier 和寄存器资源 |
| RA | Register Allocation，寄存器分配 | 决定 virtual register 放 VGPR/AGPR 还是 spill 到 scratch |
| live range | 一个值从产生到最后一次使用之间必须保持存活的范围 | full composition 的主要压力来源之一 |

### 3.5 软件接口与测量

| 术语 | 含义 | 学习重点 |
|---|---|---|
| ABI | kernel 参数、tensor dtype/layout、symbol 等二进制接口合同 | specialization 不同不能只替换一段 asm 就认为兼容 |
| HSACO | AMD GPU code object | Stage 6R/6S 用于桥接 current-vLLM recurrence |
| specialization | 针对固定 shape、dtype、layout、配置生成的实现 | old asm-v0 与 current-vLLM specialization 不是同一份机器代码 |
| dispatch / launch | CPU 向 GPU 提交一次 kernel 执行 | 少 dispatch 可能降低 overhead，但 fusion 可能提高资源压力 |
| Eager public API | 每次通过真实公开接口执行，不使用 Graph replay | 本报告最终性能排名的权威口径 |
| body latency | 单个 kernel 本体或私有 harness 的时间 | 适合定位瓶颈，不等于用户看到的 full latency |
| full latency | 整条 pipeline 的时间 | 必须包含真实边界、cast、launch 等成本 |
| paired comparison | 在共享环境中交替/成对测量两个候选 | 降低 GPU 状态漂移对微秒级差异的污染 |
| HIP 95% CI | 成对差值置信区间 | 用来判断改进是否稳定，而不是只看一次 median |

### 3.6 实验方法

| 术语 | 含义 | 示例 |
|---|---|---|
| ablation | 删除或替换一个功能，观察成本变化 | `no_vn_write`、`pred_only` |
| control experiment | 控制其他变量，只验证某个原因 | 相同 output pointer、canonical data |
| standalone | 只测一个 kernel/stage | solve 单独快，不保证接回 full 仍同样收益 |
| full integration | 把新实现接回完整 pipeline | 检查 downstream 与边界成本 |
| materialization | 把临时结果真正写入 global/shared storage | `v_new_fp32`、`a`、`h_bf16` 等中间量 |
| producer-consumer handoff | 让生产者结果更直接交给唯一消费者 | 当前登记的 KKT `a` → solve 候选 |
| bit-exact | 比较结果逐 bit 完全相同 | 比仅在 tolerance 内更强 |
| N/A | 当前没有安全实现或可信测量 | 不是 pass，也不是 fail |

---

## Part I：早期正确性修复、v14–v29 演进与编译器 lowering 调查

### Part I 阅读目标

这一部分回答三个问题：

1. 一个算子第一次优化时，如何从最慢 stage 开始，而不是直接重写整条图；
2. 当 full kernel 错误时，如何用最小 repro 找到编译器语义 bug；
3. 为什么 microbenchmark 的 compiler 优化迁移到 full recurrence 后可能完全失败。

本文档汇总 Avelang 在 AMD MI300 / gfx942 上实现 Qwen GDN / Qwen3Next linear attention `chunk_gdr` 路径时，已经完成的 correctness 修复、kernel 优化、编译器 lowering 实验、正负结果、当前进度与未解决卡点。

本文只记录已经验证过的事实与结论。dead-code 争议已经解决，不是当前问题。

---

### 1. 项目背景

目标平台：

```text
AMD MI300 / gfx942
ROCm 7.2.x
Avelang
BF16 MFMA
FP32 accumulation
```

真实应用：

```text
Qwen GDN / Qwen3Next linear attention
主要瓶颈路径：chunk_gdr
```

固定 benchmark shape：

```text
B=1
Hk=4
Hv=8
K=128
V=128
T=512 / 1024 / 2048，部分实验包含 T=4096

q/k/v: BF16
g/beta/intermediate/state/output: FP32
layout: [B,T,H,D]
state: [B,Hv,V,K]
TP4 per-rank target
```

> **历史合同说明：** Part I 记录的是早期 v14–v29 路线，当时中间量和 output 主要按 FP32
> 路径描述；Part II 的 Stage 6S–6W 随 current-vLLM specialization 与边界优化逐步引入 BF16
> recurrence/public output。两者属于不同阶段，不能把早期 dtype 描述直接套到最终 W1 图上。

主要目录：

```text
/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/
/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/
```

代表性文件与报告包括：

```text
prototype_qwen_gdn_mfma_delta_staged.py
repro_mfma_fullop_pure_cse.py
avelang_mfma_final_compiler_bug_report.md

qwen_gdn_chunked_avelang_v23_gdr_distributed_layout_fixed.py
qwen_gdn_chunked_avelang_v24_kkt_mfma_layout_fixed.py
qwen_gdn_chunked_avelang_v25_gdr_bt64_layout_fixed.py

qwen_mfma32_lowering_ladder_l4_l6_analysis.md
qwen_mfma32_l5_kstaging_variants_report.md
qwen_mfma32_l6_update_pressure_variants_report.md
qwen_mfma32_l6_kstage_update_variants_report.md
qwen_kfrag_helper_lowering_report.md
qwen_kfrag_producer_consumer_rewrite_report.md
qwen_full_kfrag_rewrite_regression_root_cause_report.md
qwen_late_bfrag_lowering_report.md
qwen_phase_boundary_real_pred_report.md
```

---

### 2. 真实算法数据流与 recurrence 约束

真实 chunk recurrence 大致为：

```text
pred_i = W_i @ H_i^T
v_new_i = U_i - pred_i
v_decay_i = v_new_i * decay_i

delta_i = v_decay_i^T @ K_i

H_{i+1}
    = H_i * exp(g_last_i)
    + delta_i
```

kernel 内存在两组不同的 MFMA accumulator：

```python
pred_acc = zero
pred_acc = mfma(W, H, pred_acc)

v_new = U - pred_acc
v_decay = v_new * decay

update_acc = zero
update_acc = mfma(v_decay, K, update_acc)

state = state * scale + update_acc
```

关键语义：

```text
pred_acc 和 update_acc 是两个独立 accumulator。

pred_acc 通过 v_new / v_decay 影响 update 的输入矩阵，
但 pred_acc 不能成为 update MFMA 的 accumulator 初值。

update MFMA 必须从独立的 zero accumulator 开始。
```

chunk 之间存在真实串行依赖：

```text
pred_i 依赖 H_i
H_{i+1} 依赖 update_i
pred_{i+1} 又依赖 H_{i+1}
```

因此不能把所有 chunk 的 pred/v_decay 全部提前计算，再统一执行 update。

---

### 2.1 初学者导读：为什么必须先解决 correctness，再谈性能

算子优化的第一原则不是“先让它快”，而是先确认每一个数学中间量都正确。原因是 GPU
kernel 往往包含并行执行、共享内存、向量 fragment、寄存器复用和编译器变换。一旦结果
错误，表面上看起来可能只是“某个 MFMA mapping 不对”，真实原因却可能来自更高层的 CSE、
内存别名或 accumulator 初始化。

本项目采用的排查顺序是：

```text
先做 update-only / pred-only 等最小正确性实验
-> 再逐步加入 prior MFMA、LDS、barrier 和真实数据流
-> 比较 AveLang IR、LLVM IR 与最终结果
-> 找到第一个发生语义变化的编译阶段
```

这种方法的价值在于：每次只增加一个触发因素，最终才能把错误从“整个 full kernel 不对”
缩小到“两个独立 FullOp 被 CSE 合并”。这也是后续所有性能优化都沿用的单变量方法。

### 3. 已解决的 FullOp / CSE correctness 编译器问题

#### 3.1 现象

`update-only` kernel 正确。

旧编译器在同一个 kernel 中先执行 pred MFMA、再执行 update MFMA 时，可能让 update MFMA 的 accumulator 错误继承 pred MFMA result。

错误 LLVM IR 曾表现为：

```llvm
%pred = call <4 x float> @llvm.amdgcn.mfma...(
    ...,
    zeroinitializer,
    ...
)

%update = call <4 x float> @llvm.amdgcn.mfma...(
    ...,
    %pred,
    ...
)
```

正确形式应为：

```llvm
%pred = call <4 x float> @llvm.amdgcn.mfma...(
    ...,
    zeroinitializer,
    ...
)

%update = call <4 x float> @llvm.amdgcn.mfma...(
    ...,
    zeroinitializer,
    ...
)
```

该问题会导致 silent numerical corruption，包括：

```text
delta_h 错误
state 错误
final_state 错误
最终 output 错误
```

#### 3.2 Root cause 与修复

旧 TableGen 定义：

```td
def FullOp : AveLang_Op<"full", [Pure]>
```

修复后：

```td
def FullOp : AveLang_Op<"full", []>
```

原因：

```text
AveLang FullOp 在当前实现中会 materialize fresh mutable storage。

两个语法相同的 al.full((4,), 0.0, al.f32)
可以代表两个独立 accumulator storage。

FullOp 被错误标记为 Pure 后，CSE 可以合并两个 logically independent FullOp，
导致 pred accumulator initializer 与 update accumulator initializer 失去独立性。
```

对应修复：

```text
[fix][IR] Make full non-Pure in TableGen
```

当前状态：

```text
FullOp/CSE sequential-MFMA correctness bug 已修复，regression 已通过。
```

---

### 4. FullOp correctness 问题的隔离过程

已经完成以下隔离实验：

| 实验 | 结果 | 结论 |
|---|---|---|
| update-only | pass | update MFMA fragment mapping、BF16 staging、FP32 accumulation 与写回正确 |
| 只增加 pred-like LDS footprint | pass | 不是 LDS 静态大小本身导致 |
| pred MFMA + update MFMA | 旧编译器 fail | 与 sequential MFMA regions 有关 |
| no-op pred region，无 pred MFMA | pass | staging、control flow、barrier 不是触发条件 |
| 一个 prior MFMA | fail | 一个 prior MFMA 即可触发污染 |
| 两个 prior MFMA | fail | 进一步确认 accumulator lowering 问题 |
| extra barrier/wait/dummy LDS | 无效 | 不是简单同步问题 |
| 独立 padded LDS + canary | 无 LDS overlap 证据 | 不是 shared layout overlap |
| LLVM IR audit | 定位成功 | update accumulator 错误引用 pred result |

移除 `FullOp` 的 `Pure` trait 后：

```text
每个 accumulator initializer 独立保留；
update MFMA 重新从 zeroinitializer 开始；
相关 correctness regression 全部通过。
```

---

### 5. 当前稳定生产基线

当前稳定 full-path baseline 为：

```text
v24
```

T=2048 的典型结果随具体 benchmark 路径和环境略有波动：

```text
full        约 0.585–0.625 ms
chunk_gdr   约 0.349 ms
chunk_o     约 0.108 ms
w_u         约 0.083 ms
solve       约 0.048 ms
KKT         约 0.029 ms
```

历史 vLLM/Triton T=2048 full 约为：

```text
0.36–0.39 ms
```

当前稳定基线与 Triton 的主要差距仍集中在 `chunk_gdr`。

---

### 5.1 从 v14 到 v24 的决策链：为什么下一步总是由上一步结果决定

下面先用一张因果表解释版本推进逻辑。后续各小节保留原始数字和实验结论。

| 阶段 | 当时观察到的问题 | 因此提出的假设 | 最小改动 | 结果如何决定下一步 |
|---|---|---|---|---|
| v14 | `w_u` 仍是大头，scalar 计算无法利用矩阵硬件 | W/U 可改成 MFMA tile 计算 | 只替换 W/U | W/U 大幅下降，瓶颈转移到 `chunk_gdr` |
| v16 | recurrence 由少数线程承担过多工作 | 多 wave cooperative ownership 可分摊工作 | 2-wave `chunk_gdr` | 明显成功，因此继续增加并行度 |
| v17 | wave0 仍承担 `v_new/v_decay` 等串行工作 | 4-wave 与 predecay 可进一步分散 | 4-wave + predecay | 成功但边际递减，说明要看其他 stage |
| v18 | 增大 chunk 后 solve 成本爆炸 | row 内 column/update 可并行 | parallel solve | 解除了 BT32/BT64 solve blocker，允许测试大 chunk full path |
| v19 | 更大 BT 理论上可减少 chunk 数和 recurrence 次数 | BT32 full 可能整体更快 | 整条 BT32 full path | recurrence 略快但其他 stage 变贵，证明局部收益不等于 full 收益 |
| v20 | v19 的 KKT/WU 变贵 | 为 BT32 写 native MFMA 可补回损失 | 分别改 KKT/WU | KKT 成功、WU 失败，说明不同算子不能机械套用相同 tile |
| v21 | 怀疑 load 与 compute 缺乏重叠 | double buffering 可能隐藏内存延迟 | 同步双缓冲 | 无 async copy，资源增加且更慢，因此停止 |
| v22 | `chunk_gdr` 内部到底哪部分最贵仍不清楚 | 用 ablation 删除单个工作可定位成本 | no-store、pred-only、update-only 等 | 定位 `v_new` materialization 与 wave0-only 路径 |
| v23 | wave0-only `v_new/v_decay` 是明确瓶颈 | 把它分发到多 wave | distributed `v_new/v_decay` | 成功，full 继续下降 |
| v24 | KKT 成为可见剩余成本 | BT16 native KKT MFMA 可消除 scalar/通用开销 | 只改 KKT | 成功并成为 production baseline |

这张表体现一个重要原则：**下一版不是因为“某种技术听起来高级”，而是因为上一版留下了
一个被测量证据支持的具体瓶颈。**

### 6. 已完成的主要 kernel 性能优化

#### 6.1 v14：MFMA w_u

```text
w_u 约 0.96 ms -> 0.082 ms
full 约 2.55 ms -> 1.65 ms
```

结果：成功。

#### 6.2 v16：2-wave cooperative chunk_gdr

```text
chunk_gdr 约 1.36 ms -> 0.56 ms
full 约 1.65 ms -> 0.85 ms
```

结果：成功。

#### 6.3 v17：4-wave cooperative + predecay

```text
chunk_gdr 约 0.56 ms -> 0.41 ms
full 约 0.85 ms -> 0.69 ms
```

结果：成功，但开始出现边际递减。

#### 6.4 v18：parallel solve

```text
BT32 solve 约 2.38 ms -> 0.034 ms
BT64 solve 约 21.1 ms -> 0.127 ms
```

结果：成功，larger-chunk solve blocker 被解除。

#### 6.5 v19：BT32 full path

```text
BT16 v17 full 约 0.698 ms
BT32 v19 full 约 0.877 ms
```

虽然 `chunk_gdr` 略降：

```text
0.410 ms -> 0.389 ms
```

但 KKT、w_u、chunk_o 变贵，full path 变慢。

结果：失败。

#### 6.6 v20：BT32 native KKT / w_u MFMA

KKT：

```text
约 0.193 ms -> 0.030 ms
```

w_u：

```text
约 0.193 ms -> 0.268 ms
```

结果：KKT 成功，w_u 失败，full path 仍输 BT16。

#### 6.7 v21：同步 double buffering

尝试：

```text
W double buffer
W + K double buffer
```

结果：负面。

原因：

```text
没有真正 async copy；
VMEM 没下降；
LDS、VGPR、AccVGPR 增加；
trace 变慢。
```

#### 6.8 v22：chunk_gdr ablation

T=2048：

```text
full_baseline           0.429 ms
no_h_write              0.409 ms
no_vn_write             0.343 ms
no_h_no_vn_write        0.314 ms
pred_only               0.049 ms
update_only_dummy       0.169 ms
pred_vn_only_no_update  0.274 ms
update_no_state_decay   0.424 ms
distributed_vn_vdecay   0.350 ms
```

结论：

```text
h write 不是主要问题；
vn materialization 有真实成本；
wave0-only vn/v_decay 是明确瓶颈；
pred MFMA 指令本身不是当时最大的单独瓶颈；
state decay 不值得单独优化。
```

#### 6.9 v23：distributed vn/v_decay

```text
chunk_gdr 约 0.410 ms -> 0.349 ms
full 约 0.694 ms -> 0.637 ms
```

结果：成功。

#### 6.10 v24：native BT16 KKT MFMA

```text
KKT 约 0.115 ms -> 0.029 ms
full 约 0.636 ms -> 0.585 ms
```

结果：成功，成为当前 production baseline。

#### 6.11 v25：BT64/BV32 chunk_gdr-only

T=2048：

```text
v23 BT16 chunk_gdr       约 0.350 ms
v25 BT64/BV32 chunk_gdr  约 0.833 ms
```

结果：明显失败。

结论：

```text
BT64/BV32 参数本身不等于 Triton-like kernel。
Avelang 当前依赖显式 16x16 MFMA subtile、LDS、barrier 和手写 coordination 拼接大 tile，
导致 per-chunk 成本和资源压力过高。
```

---

### 7. v24 store ablation

v24/v23 chunk_gdr store ablation 结果：

```text
T=2048 baseline chunk_gdr: 0.359914 ms
T=2048 no_h_no_vn:         0.314488 ms
speedup:                   1.1444x

T=4096 baseline:           0.664226 ms
T=4096 no_h_no_vn:         0.569146 ms
speedup:                   1.1671x
```

rocprof T=2048：

```text
baseline trace: 331.012 us
no_h_no_vn:    279.495 us
VMEM:          921600 -> 593920
```

结论：

```text
store/materialization 有真实成本，但收益上限约在 15% 左右，
不足以单独解释或消除与 Triton 的全部差距。
```

---

### 7.1 为什么在 v24 已经稳定后仍要探索 v29

v24 已经正确且稳定，但与 vLLM/Triton 的主要差距仍集中在 recurrence。此前 v25 说明只把
参数改成 BT64/BV32 并不会自动得到 Triton-like 性能，因此 v29 的探索目标不是简单扩大
维度，而是同时改变 pred 的矩阵指令形状、tile ownership 与 `v_new` materialization。

推理链是：

```text
BT64 减少 chunk 数
+ MFMA32 可能更贴合较大 pred tile
+ 融合后避免全局 v_new 中间量
=> 有机会减少 recurrence 的每序列固定成本
```

但这个假设同时提高了三个风险：

1. 更大的 accumulator footprint；
2. pred、state、v_decay、update 的 live range 重叠；
3. recurrence feedback 会放大单 chunk 的 BF16 误差。

因此 v29 后续必须同时检查 correctness、资源和 full latency，而不能只看 pred-only。

### 8. v29 BT64/BV32/MFMA32 路线

v29 的目标是：

```text
保留 BT64 带来的较少 chunk 数，
使用 MFMA32 pred 与 MFMA16 update，
尝试接近 Triton 的大 tile 路线。
```

#### 8.1 v29 pred-only 与 skeleton

关键结果：

```text
pred-only original, T=2048: 约 0.482 ms
grouped_v4 best:            约 0.461 ms
fused skeleton:             约 0.238 ms
```

说明：

```text
MFMA32 primitive 能生成并运行；
pred-only correctness 可达到 BF16 级；
移除 global vn materialization 的 skeleton 有明显收益。
```

#### 8.2 v29 fused full

T=2048：

```text
v29 fused full: 约 0.8364 ms
v24 full:       约 0.6251 ms
```

rocprof T=2048：

```text
trace:          约 833.9 us
VGPR:           128
AccVGPR:        264
LDS block:      61440 B
VALU:           约 4.98M
LDS inst:       约 1.24M
```

correctness：

```text
T=512 final_state max_abs 约 5.84e4
T=1024 max_abs 约 2.07e8
T=2048 max_abs 约 2.10e15
```

`w=0` update isolation 正确。

结论：

```text
v29 source-level full Qwen 同时存在性能失败与 nonzero-w recurrence correctness 失败。
MFMA32 primitive 本身可用，但 BT64/BV32 full schedule 对当前 Avelang 后端过于激进。
```

#### 8.3 v29 correctness/debug 结论

关键结果：

```text
state_readback_vs_written_bf16 max_abs = 0
chunk0 pred max_abs 约 2.78e-02
chunk0 state_after max_abs 约 2.62e-01
chunk1 normal pred max_abs 约 1.06e+01
```

补充实验：

```text
decay_off 仍失败；
feedback_disabled 后误差保持 chunk0 级；
w=0 update sanity 继续正确。
```

结论：

```text
state 写回布局与读取布局一致；
不是简单 decay/g_last indexing 错误；
updated state feedback 会放大误差；
原始 v29 nonzero-w recurrence correctness 问题尚未解决。
```

---

### 8.4 v29 失败后的问题树：如何从“full 不行”缩小根因

v29 的 pred-only 和 skeleton 有正结果，但 full kernel 同时出现数值失真和性能倒退。此时不能
直接断言“MFMA32 不适合”，因为 primitive 本身已经证明可运行。后续调查按以下问题树推进：

```text
A. 是否是所有 sequential MFMA 都会造成 accumulator 污染或资源相加？
B. 是否是 K 的 broad shared view / address lowering 制造了压力？
C. 是否只要缩短 source-level lifetime 就能改善 RA？
D. 是否需要让专用 fragment 语义存活到更晚 lowering？
E. isolated 修复迁移到 exact full composition 后是否仍成立？
```

每个后续实验都只回答其中一个问题。这样即使最终没有得到 production 修复，也能明确排除
错误方向，并把卡点收敛到 full live-set composition 与 backend scheduling。

### 9. Generic sequential MFMA lifetime bug 已排除

独立 repro `mfma_region_lifetime_v3`：

```text
update16_only:                    VGPR=92, AccVGPR=84
pred32_only_sink:                 VGPR=80, AccVGPR=184
pred32_then_update16:             VGPR=92, AccVGPR=172
pred32_two_regions_then_update16: AccVGPR=172
```

结论：

```text
一个 kernel 中 pred32 MFMA 后接 update16 MFMA，
不会普遍导致 AccVGPR 相加式爆炸。

问题是 Qwen-shaped source/dataflow/lowering 的组合，
不是 generic sequential MFMA contamination。
```

---

### 10. Qwen-shaped L5/L6 lowering ladder

#### 10.1 L5 K-staging variants

```text
L5 baseline trace:                   36.614 us
L5 direct-global-K update probe:     13.740 us
L5 khalf stage64:                    23.635 us
L5 subtile16 stage:                  32.769 us
L5 packed-i32 contiguous load:       non-finite
```

结论：

```text
K staging/view/address lowering 是明确压力来源；
direct-global probe 很快，但不是 production-safe 方案；
khalf/subtile 可以局部减压。
```

#### 10.2 L6 update pressure variants

第一组：

```text
L5 no update:                   AccVGPR=144
L6 one update MFMA only:        AccVGPR=320
L6 one K-tile update:           AccVGPR=336
L6 full update-like current:    AccVGPR=264
scope split/reinit:             不降低 AccVGPR
```

第二组 isolated K-stage/update：

```text
L6 baseline current update:
  trace 34.371 us
  VGPR 128
  AccVGPR 264
  LDS block 45056

L6 khalf stage64:
  trace 27.200 us
  VGPR 76
  AccVGPR 188
  LDS 36864

L6 subtile16 stage:
  trace 19.189 us
  VGPR 96
  AccVGPR 168
  LDS 32768

L6 no-shared-K direct-global probe:
  trace 14.381 us
  VGPR 112
  AccVGPR 152

L6 update MFMA no pred dependency:
  trace 10.175 us
  AccVGPR 80

L6 minimal fragment:
  trace 3.164 us
  AccVGPR 4
```

结论：

```text
update MFMA intrinsic 本身不是主因；
broad K staging / kall_vec shared-view address lowering 是明确嫌疑；
pred/v_decay dependency 是重要 pressure source；
isolated 结果不能直接外推到 full Qwen。
```

---

### 11. Source-level K-subtile / helper 路线

#### 11.1 Isolated L6 成功

```text
baseline trace: 34.331 us, AccVGPR 264
subtile trace:  19.308 us, AccVGPR 168
helper trace:   19.029 us, AccVGPR 168
```

说明局部 K-subtile 能消除 isolated L6 的 broad K view/address pressure。

#### 11.2 Full Qwen 迁移失败

T=2048：

```text
original full v29: 约 0.834 ms
K-subtile full:    约 2.281 ms
```

rocprof：

```text
original:
  trace 817.775 us
  AccVGPR 264
  Scratch 0
  VMEM 399360
  LDS 1242304

K-subtile:
  trace 2239.567 us
  AccVGPR 384
  Scratch 84 B
  VMEM 1347008
  LDS 2159808
```

correctness：

```text
K-subtile experiment 与 original v29 输出完全一致；
original v29 对 nonzero-w 仍不正确；
w=0 update sanity 正确。
```

结论：

```text
单独 source-level K-subtile 不是 full 根因修复。
局部 isolated 收益无法自动迁移到 full recurrence kernel。
```

---

### 12. end_lifetime / lifetime-boundary 路线

#### 12.1 实现过的两种形式

##### Marker-only

```text
al.end_lifetime / al.discard 在 AveLang IR 中可见；
AveLang-to-memref lowering 时被 erase；
没有生成 LLVM lifetime.end；
AMDGPU backend/RA 看不到有效信息。
```

##### Late-survive marker

```text
ave.end_lifetime 存活到更晚 pipeline；
在进入 AMDGPU backend 前被 erase；
仍未生成能影响 RA/shared reuse 的有效 lifetime 约束。
```

#### 12.2 L6 counter

```text
L6 baseline no lifetime:
  trace 34.371 us, VGPR 128, AccVGPR 264, Scratch 0

L6 with end_lifetime:
  trace 34.291 us, VGPR 128, AccVGPR 264, Scratch 0

L6 subtile no lifetime:
  trace 19.349 us, VGPR 96, AccVGPR 168, Scratch 0

L6 subtile with lifetime:
  trace 19.189 us, VGPR 96, AccVGPR 168, Scratch 0
```

结论：

```text
end_lifetime 没有改变 backend 可见 graph，
对 AccVGPR、Scratch 与 trace 均无实际效果。
```

operand audit 还表明：

```text
pred_acc 是 MFMA SSA/vector accumulator，不是稳定 memref base；
pred_partial/state_bf16/w_bf16 多为 workgroup memref；
v_decay_t_bf16 在 update 中仍需使用；
K staging 是 active producer-consumer graph，不能通过 marker 删除。
```

---

### 13. MIR / ISA regalloc audit

L6 MIR/ISA audit 发现：

```text
baseline ISA 存在 v_accvgpr_write_b32 a100..a131；
subtile 中没有这些高 AGPR writes；
两者 MFMA 数量相同；
MIR hasSpilledVGPRs=false，Scratch=0。
```

高 AGPR 对应：

```text
global K load
LDS staging
shared-view / addrspace(3) GEP
vector.load
相关地址与数据 temporary
```

它们不是 MFMA accumulator 本体，而是 register allocator 在高压下把普通 temporary 停到 AGPR。

关键静态结果：

```text
baseline:
  max AGPR index 131
  v_accvgpr_write/read 155/155
  high AGPR writes 51
  MFMA32 8
  MFMA16 32

subtile:
  max AGPR index 15
  v_accvgpr_write/read 32/32
  high AGPR writes 0
```

结论：

```text
Qwen broad k_all_t[128,BT] + kall_vec lowering 会制造过重的 producer-consumer/address graph；
主要问题应在 Avelang/MLIR 层处理，而不是全局修改 LLVM/AMDGPU RA。
```

---

### 14. Read-side K-fragment helper

实现过：

```python
al.amdgpu.qwen_update_kfrag_load_bf16x4(shared_k, k_col, token_base)
```

结果：

```text
baseline/helper max_abs=0, mean_abs=0
MFMA count unchanged
Scratch=0

baseline: trace 34.251 us, VGPR 128, AccVGPR 264, high AGPR a131
helper:   trace 34.572 us, VGPR 128, AccVGPR 264, high AGPR a131
subtile:  trace 19.188 us, VGPR 96,  AccVGPR 168
```

原因：

```text
helper 过早 lowered 成 AveLangMemRefLoadVecOp / vector.load；
optimized LLVM 又生成与 baseline 同构的 addrspace(3) GEP 与 broad K staging；
RA 看到的 graph 基本不变。
```

结论：

```text
只替换 consumer 表达式不够，必须同时重写 producer-consumer graph。
```

---

### 15. Persistent K-fragment producer-consumer rewrite

#### 15.1 编译器设计

新增 opt-in persistent op：

```text
ave.gpu.amdgpu_qwen_update_kfrag_load
```

主要改动文件：

```text
lib/Dialect/AveLang/IR/AveLangOps.td
lib/Dialect/AveLang/IR/AveLangOps.h
lib/Dialect/AveLang/IR/AveLangOps.cc
lib/IR/Intrinsics/amdgpu_module.cc
lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.h
lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.cc
lib/Dialect/AveLang/Transforms/CMakeLists.txt
lib/Target/GPU/lower_to_llvm.cc
```

设计目标：

```text
保留 Qwen K-fragment load 语义；
在 AveLang-to-memref 后、LLVM lowering 前运行专门 pass；
匹配 broad [128,64] K producer + 四个 MFMA16 B-fragment consumers；
将其重写为 compact physical K tile + direct vector<4xbf16> consumer；
在 LLVM RA 前删除旧 broad producer/shared chain。
```

#### 15.2 Isolated L6 正结果

```text
baseline -> rewrite
max_abs / mean_abs: 0 / 0
trace median:       34.412 us -> 18.628 us
AccVGPR:            264 -> 180
VGPR:               128 -> 84
LDS block:          45056 -> 32768
MFMA:               5120 -> 5120
high AGPR >= a100:  31 -> 0
Scratch:            0 -> 0
speedup:            1.847x
```

静态 ISA/MIR：

```text
MFMA32:                 8 -> 8
MFMA16:                32 -> 32
global_load:          144 -> 96
ds_read:               88 -> 80
ds_write:             152 -> 104
total ISA instructions:3144 -> 2082
v_accvgpr_write_b32:   155 -> 32
high AGPR >=100:        31 -> 0
max AGPR write index:  131 -> 3
```

结论：

```text
这是一次真实有效的局部 compiler lowering 修复。
它证明 broad K producer-consumer graph 是 isolated L6 的真实问题。
```

---

### 16. Persistent rewrite 迁移到 full v29

#### 16.1 Correctness

T=512/1024/2048：

```text
rewrite 相对 original v29 的 h 与 final_state：
max_abs=0, mean_abs=0
```

说明 rewrite 保持了 original v29 的语义。

原始 v29 的 nonzero-w recurrence correctness 问题仍然存在，rewrite 没有引入该问题，也没有修复它。

#### 16.2 性能负结果

T=2048：

```text
original v29: 0.837143 ms
rewrite full: 1.338669 ms
```

rocprof：

```text
trace:      830.974 us -> 1302.635 us
AccVGPR:    264 -> 384
Scratch:    0 -> 736 B
VGPR:       128 -> 128
LDS block:  61440 -> 61440
MFMA:       unchanged, 294912
VALU:       4977280 -> 3180992
SALU:       810496 -> 567808
VMEM:       399360 -> 601984
LDS inst:   1242304 -> 1242304
```

结论：

```text
isolated compiler rewrite 的局部收益没有在 full recurrence kernel 中保持；
full v29 出现更高 AccVGPR、scratch 与 VMEM spill traffic；
该 rewrite 不能迁移到 production full path。
```

---

### 17. Full rewrite regression root-cause audit

root-cause audit 的主要分类为：

```text
Category E：full live-set composition pressure
```

关键证据：

```text
original 与 rewrite static global-load count 都是 198；
动态 MFMA 与 LDS instruction count 不变；
旧 broad producer 已被匹配并 erase；
没有证据表明 old/new K staging 同时存在；
VMEM 增长与 scratch traffic 一致，而不是重复 global K load。
```

Reduced R0-R4 ladder：

```text
R0 isolated rewrite:              AccVGPR 84,  Scratch 0
R1 + 32-window loop:              AccVGPR 132, Scratch 0
R2 + pred/v-decay-like live:      AccVGPR 136, Scratch 0
R3 + state update/writeback:      AccVGPR 192, Scratch 0
R4 full-loop skeleton:            AccVGPR 192, Scratch 0
```

只有 exact full v29 同时加入真实 MFMA32 pred accumulator、pred_partial/state/v_decay 与 update path 后，才出现：

```text
AccVGPR 384
Scratch 736 B
```

唯一局部修复尝试：hoist replacement compact alloca。

```text
normal:   1.338669 ms -> 1.335064 ms
trace:    1302.635 us -> 1301.974 us
AccVGPR:  384 -> 384
Scratch:  736 -> 736 B
VMEM:     unchanged
```

结果：无效，已撤回。

当前定性：

```text
full failure 不是 repeated K load、old broad staging 或 scalar reload cloning 的单一问题；
真实 MFMA32 pred/live region 与 update 路径共同跨过 register-allocation 阈值。
```

---

### 18. Late persistent B-fragment lowering

目标：

```text
让专用 B-fragment op 存活到 GPU outlining 后；
避开 generic vector.load rewrite；
直接生成 workgroup address-space 的单个对齐 8-byte LLVM vector load，目标选择为 ds_read_b64。
```

实现结果：

```text
专用 op 成功存活到 late lowering；
没有临时 alloca；
没有走原来的 generic vector.load / memref-view chain；
reduced repro 可编译、可运行且 finite。
```

真实 MFMA32 pred reduced repro：

```text
generic B load:
  latency 0.5292 ms
  VGPR 128
  AccVGPR 144
  Scratch 0

late direct LDS load:
  latency 0.5634 ms
  VGPR 124
  AccVGPR 148
  Scratch 0
```

结论：

```text
局部 B-load 替换只减少 4 个普通 VGPR，AccVGPR 反而增加 4；
没有降低整体 pressure，也没有提供消除 full v29 scratch 的证据；
full v29 未恢复测试。
```

---

### 19. Real pred/update hard phase-boundary experiment

目标：

```text
在同一个 kernel 内建立真实数据流边界：
pred accumulator 只写 pred_partial shared；
barrier 后 downstream 全部 reload shared；
update 不再直接使用 pred_acc 或 pre-boundary pred_live SSA value。
```

A/B/C 结果：

| Variant | Trace | VGPR | AccVGPR | Scratch |
|---|---:|---:|---:|---:|
| A current fused | 513.423 us | 128 | 144 | 0 B |
| B hard shared boundary | 539.181 us | 128 | 144 | 0 B |
| C no real pred accumulator | 314.126 us | 124 | 132 | 0 B |

A 与 B 的以下指标完全相同：

```text
VGPR
AccVGPR
SGPR
Scratch
MFMA
VALU
SALU
VMEM
LDS instructions
LDS block
```

B 只增加 barrier，同步成本使 trace 增加 5.02%。

C 移除真实 MFMA32 pred accumulator 后：

```text
AccVGPR 144 -> 132
trace 513.423 us -> 314.126 us
```

结论：

```text
真实 MFMA32 pred 区域是明确 pressure amplifier；
但 pred/update 之间的 shared materialization + barrier 并不能降低 backend 看到的资源峰值；
问题不是一根直接 SSA edge 没切断，而是 pred/full composition 本身的峰值成本更深。
```

---

### 20. 两 kernel 拆分方向的实验依据

曾考虑：

```text
kernel 1: pred / u_corr / v_decay
kernel 2: update state
```

但 recurrence 决定：

```text
pred_i 依赖 H_i
H_{i+1} 依赖 update_i
```

因此不能对所有 chunk 先跑 kernel 1，再统一跑 kernel 2。

若每个 chunk 都交替 launch：

```text
会产生大量 kernel launch；
需要 global materialization v_decay；
增加 global write/read handoff；
破坏 fused 数据局部性。
```

v22 近似 ablation：

```text
pred/v_decay-like: 约 0.274 ms
update-like:       约 0.169 ms
粗略合计:         约 0.443 ms
```

v23 fused chunk_gdr：

```text
约 0.349 ms
```

结论：

```text
两个独立 kernel 不是当前有竞争力的性能路线，也不能简单保持全序列语义。
```

---

### 21. 已排除或已闭环的实验方向

以下方向已经有明确实验结果：

```text
FullOp/Pure CSE correctness bug：已修复。
Generic sequential MFMA contamination：已排除。
更多 barrier / wait / dummy LDS：无效。
LDS overlap / padding：无 root-cause 证据。
同步 double buffering：负结果。
盲目 BT32/BT64/BV sweep：多次负结果。
Source-level K-subtile：isolated 正、full 负。
Read-side helper：lowering 同构，无效。
end_lifetime / discard marker：无 backend 效果。
Persistent producer-consumer rewrite：isolated 正、full 负。
Hoist compact alloca：无效，已撤回。
Late B-fragment direct LDS load：无 pressure 收益。
Hard shared pred/update boundary：只增加 barrier 开销。
Pred/update 两 kernel 拆分：语义与性能均不具备主线价值。
```

---

### 21.1 如何理解这些“失败”的价值

这一阶段最重要的成果并不只是 isolated L6 的 `1.847x`，而是建立了以下边界：

- 能在 isolated repro 中下降的 AccVGPR，不保证在 full recurrence 中下降；
- source-level 变量看起来生命周期结束，不代表 backend RA 能看见这个事实；
- 少一个 `vector.load` 表达式，不代表 LLVM/MIR producer-consumer graph 真正改变；
- 同一 kernel 内增加 shared materialization 和 barrier，可能只增加同步成本而不降低峰值；
- scratch=0 也不代表寄存器压力低，普通 temporary 仍可能被放进高 AGPR 区域；
- full kernel 的性能由“同时活跃的所有值”决定，而不是由单个 microkernel 的最好结果决定。

因此这些负结果应当保留。它们是后续选择停止 v29 compiler line、转向更稳健 Stage 4 路线的
证据，而不是可以删除的“无用尝试”。

### 22. v29 compiler line 在该阶段的进度

> **时间范围说明：** 以下“当前”只表示 Part I 结束时 v29/compiler 分支的阶段性状态。整份
> 项目的最新状态请看 Part II 的 Stage 6W 和第 41 节。

目前已经获得三类关键结论。

#### 22.1 Correctness 层

```text
旧 FullOp/CSE sequential-MFMA correctness bug 已解决。

v29 full nonzero-w recurrence 仍存在独立 correctness 问题：
chunk0 误差尚小，但 state feedback 在后续 chunk 中放大；
该问题不是 state read/write layout mismatch，也不是简单 decay indexing。
```

#### 22.2 Isolated compiler lowering 层

```text
broad K producer-consumer/shared-view lowering 是真实 compiler lowering 问题；
通过 persistent K-fragment producer-consumer rewrite，
isolated L6 达到 1.847x speedup，AccVGPR 264 -> 180，high AGPR temporary 消失。
```

#### 22.3 Full composition 层

```text
上述 isolated 修复迁移到 full v29 后失败：
AccVGPR 264 -> 384，Scratch 0 -> 736 B，VMEM 与 latency 明显增加。

重复 K load、old broad staging、scalar reload、final B load、单一 SSA crossing、shared phase boundary
均已被实验排除为主要原因。

真实 MFMA32 pred 区域是最明确的 pressure amplifier。
```

---

### 23. v29 compiler line 当时的卡点

当前卡点不再是“如何优化单独 K-fragment load”，而是：

```text
为什么 exact full v29 中，MFMA32 pred、pred_partial、state、v_decay、update 与长 recurrence loop
组合后会从 reduced repro 的 AccVGPR 144–192 / Scratch 0，
跃迁到 AccVGPR 384 / Scratch 736 B。
```

目前缺失的最关键后端证据是：

```text
exact full v29 的 pre-RA / post-RA MIR；
具体 spill virtual registers；
这些 spill 对应的源 IR op；
MFMA32 pred accumulator、unpack/store temporary、state/v_decay temporary 的真实 live intervals；
为什么 full rewrite 改变了寄存器分配阈值，而 reduced repro 没有完全复现。
```

现阶段最可能的更深层问题包括：

```text
MFMA32 pred 区域自身的 peak accumulator footprint；
pred accumulator epilogue/unpack/store temporary 的内部重叠；
full recurrence 中 loop-carried state/v_decay/address values 与 pred accumulator 的峰值叠加；
Avelang 当前缺少 block-tensor/MFMA fragment-aware scheduling，无法保持 BT64 大 tile 的复用优势同时控制寄存器峰值。
```

当前不能再把问题简化为：

```text
某一个 K view；
某一个 vector.load；
某一个 barrier；
某一根 SSA use chain；
某一个 lifetime marker。
```

它已经收敛为：

```text
Qwen full BT64/MFMA32 composition 的 backend register-pressure 与 scheduling 问题。
```

---

### 24. Part I 阶段性路线状态总结

```text
v24：稳定、正确、当前 production baseline。

v29：研究性路线；MFMA32 primitive 与 isolated compiler rewrite 均有正结果，
但 full nonzero-w correctness 未解决，full performance 与资源分配失败。

K-fragment compiler rewrite：作为 isolated compiler lowering 正结果保留，
不能泛化为 full v29 修复。

Full v29 compiler line：当前停止在 root-cause evidence 阶段，
尚未得到能消除 AccVGPR=384 / Scratch=736 B 的局部修复。
```

---

## Part II：Stage 4–6W 的系统优化、完整实验链与当前结论

### Part II 与 Part I 的时间关系

Part I 的 v24 是稳定 production baseline；v29 是研究性的大 tile/MFMA32 路线，并在 full
correctness、寄存器压力和性能上遇到卡点。之后项目没有继续无边界扩大 v29，而是重新建立
Stage 4 路线：冻结 recurrence，先把非 recurrence 的 KKT、W/U、chunk-o 与 solve 分别做好，
再逐步处理真实 vLLM specialization、dtype ABI、fusion 和 storage boundary。

本部分主体来自当前 `report_final.md`。其中 Stage 4–6S 已覆盖旧版 Stage 4–6S 报告，Stage
6T–6W 使用已完成结果替换旧版未来计划。

> 路径说明：正文保留了原工程中的相对链接，下载本单文件后这些链接可能无法直接打开；它们
> 主要用于记录原始证据位置。

Part II 的固定 Stage 4 目标合同为：

```text
B=1, Hk=4, Hv=8
K=128, V=128
q/k/v: BF16
g/beta/intermediate/final_state: FP32
chunk_size = BT = 64
layout = [B,T,H,D]
T % 64 == 0
target = gfx942 / MI300
```

这是一份完整的中文实验档案，不是结果摘要。它保留 Stage 4--6S 的详细历史，
并用真实已经完成的 Stage 6T、6T-Golden、6U、6V、6W 和 6W 成对确认结果替换了
旧版“未来计划”。读者可以从中看到：问题如何缩小、每段源码为什么改、失败如何被
记录、哪些数字可以比较，以及什么时候必须停止猜测。

**必须先读的口径更正：**

```text
正式 correctness 与性能排名：完整 Eager public API
timing_contract = eager_public_api
cuda_graph_used = false

Graph replay / private body / isolated HSACO / rocprof trace：仅诊断，不进正式排行榜
```

当前状态：v24 BT16 production/default 不变；U1 是上一 BT64 experimental baseline；
W1/Stage 6W 是当前 **Avelang experimental baseline**，但只在
`paired_shared_environment` 的严格成对 Eager 比较范围内成立，未改 production selector。
T=2048 W1 比 U1 快 `8.994 us`（HIP 95% CI `[7.246,10.545] us`），同批中为 vLLM 的
`0.8607x`；T=8192 W1 比 U1 快 `27.352 us`（`[25.398,29.163] us`），但仍是 vLLM 的
`1.2259x`。因此不能声称全面超过 vLLM。

本文把 Stage 6R 所捕获的 ABI/config/code-object 统称为 **current-vLLM specialization**；
它与历史 asm-v0 不是同一个 recurrence specialization。本文的唯一后续登记项写作
**KKT FP32 `a` 到 solve** handoff：它尚未实现，不能被误读成已经融合。

原始旧报告已原样备份为
[report_final_before_stage6w_update.md](report_final_before_stage6w_update.md)。具体旧结论如何修正，见
[修订日志](report_final_revision_log.md)和[事实冲突记录](report_final_fact_corrections.md)。源码、测试、数据和报告的路径见[证据索引](report_final_evidence_index.json)。

---

### 2. 为什么要做 Stage 4

#### 2.1 Stage 2 的情况：功能正确，但上游和下游是 scalar fallback

Stage 2 的 BT64 full pipeline 已经能跑通完整接口，也能连接冻结的 asm
recurrence，但是 KKT、W/U、chunk-o 仍然使用通用 scalar 风格实现。

T=2048 时，Stage 2 大致是：

| 阶段 | Stage 2 延迟 |
|:--|--:|
| full | `16.0413 ms` |
| W/U | `约 4.0051 ms` |
| chunk-o | `约 12.1517 ms` |

rocprof 显示 Stage 2 的典型问题：

- workgroup 只有 1 个 thread；
- MFMA count 为 0；
- VALU/VMEM 数量巨大；
- chunk-o 还有 private scratch；
- 大量循环完全在单线程中展开或串行执行。

所以第一步很明确：先把 scalar fallback 消掉。

#### 2.2 Stage 3 的情况：已经 MFMA 化，但 BT64 ownership 还不够好

Stage 3 复用了 v14/v24 的 MFMA16 microkernel，把 W/U 和 chunk-o 改成了
可用的 MFMA 版本：

| 阶段 | Stage 3 T=2048 |
|:--|--:|
| full | `1.0922 ms` |
| KKT | `约 0.3502 ms` |
| solve | `约 0.109~0.127 ms` |
| W/U | `约 0.2949 ms` |
| asm recurrence | `约 0.205 ms`，未插桩 HIP event |
| chunk-o | `约 0.2562 ms` |

Stage 3 的进步很大，但仍然明显慢于：

```text
v24 BT16: 约 0.5882 ms
vLLM:     约 0.3613 ms
```

Stage 3 的关键问题不是“有没有 MFMA”，而是：

- KKT 仍然是 BT64 scalar 单线程版本；
- W/U 的 ownership 仍然接近机械执行多个 token16 小块；
- chunk-o 的跨 token16 复用不够充分；
- scalar correction、重复 load/store 和 kernel ownership 仍有优化空间。

因此 Stage 4 的假设是：

```text
Stage 3 的主要剩余损失来自高级代码的 tile/ownership/data reuse，
而不是先去改 compiler、LLVM RA 或 asm recurrence。
```

这个假设必须通过 kernel-level benchmark、rocprof 和 ISA 证据验证。

---

### 3. 实验总策略

本次没有一开始直接写一个“大而全”的新 full kernel，而是采用增量接入：

```text
Stage 3 full
  -> 只替换 KKT
  -> KKT + W/U
  -> KKT + W/U + chunk-o
```

对应的 incremental full T=2048 结果：

| 版本 | full latency | 相对前一阶段 |
|:--|--:|--:|
| Stage 3 | `1.089079 ms` | baseline |
| Stage 4 KKT-S0 | `0.795162 ms` | `1.3696x` |
| Stage 4 KKT + WU-S1 | `0.610087 ms` | `1.3034x` |
| Stage 4 KKT + WU-S1 + chunk-o-S0 | `0.475867 ms` | `1.2821x` |

这样做的好处是每一步都能回答：

1. 单 kernel 是否正确；
2. 单 kernel 是否变快；
3. 接回 recurrence 后收益是否传递到 full；
4. 如果 full 失败，究竟是哪一个新增阶段造成问题。

---

### 4. KKT：从单线程 scalar 到 BT64 native MFMA

#### 4.1 原来的代码和问题

原来的 BT64 KKT 走 v6 风格的通用 kernel：

```text
grid  = B * T * Hv
workgroup = 1 thread
MFMA = 0
```

数学上，KKT 是 chunk 内的严格下三角 token-token 矩阵：

```text
a[t,h,s] = beta[t,h]
           * dot(k[t, key_head], k[chunk_start+s, key_head])
           * exp(g[t,h] - g[chunk_start+s,h])
```

当 `s >= local_token_position` 时输出 0。

也就是说，KKT 不是 solve，也不是一个任意的大矩阵；它本质上是：

```text
一个 BT64 chunk 内部的 K[64,128] @ K[64,128].T
然后施加 causal mask / beta / decay
```

这正适合复用 v24 已验证的 `mfma_16x16x16_bf16_f32`。

#### 4.2 Stage 4 KKT-S0 的代码变化

新增 kernel：

```text
_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0
```

新的 ownership：

```text
每个 CTA = 一个 (chunk, value_head, row_tile, col_tile)
row_tile = 0..3
col_tile = 0..3
每个 tile = 16x16
workgroup = 64 threads
```

BT64 不再由一个线程处理，而是拆成 4x4 个 token16 tile：

```text
BT64 matrix
  -> 4 rows of 16 tokens
  -> 4 cols of 16 source tokens
  -> 16 independent CTA tiles
```

每个 lower/diagonal tile：

1. 把两侧的 `[16,128]` K tile staging 到 LDS；
2. 用 8 次 MFMA16 覆盖 K=128；
3. 在 writeback 时应用 strict-lower mask；
4. 同时处理 FP32 beta 和 decay；
5. 写回与 solve 完全兼容的 `[1,T,8,64]` layout。

#### 4.3 为什么先选这个方案

原因不是单纯“MFMA 越多越好”，而是：

- v24 BT16 KKT 已经证明数学和 layout 是可靠的；
- v20 BT32 KKT 已经证明 token matrix 可以 MFMA 化；
- 16x16 tile 的 fragment mapping 在仓库中已有稳定范例；
- 不需要引入新的 32x32 fragment layout 风险；
- causal mask 可以在 MFMA 后安全处理。

#### 4.4 结果

| T | 旧 Stage 3 KKT | 新 KKT-S0 | speedup |
|--:|--:|--:|--:|
| 512 | `0.145897 ms` | `0.034731 ms` | `4.2007x` |
| 1024 | `0.213637 ms` | `0.035513 ms` | `6.0157x` |
| 2048 | `0.351423 ms` | `0.046589 ms` | `7.5430x` |

T=2048 rocprof：

| 指标 | KKT-S0 |
|:--|--:|
| workgroup | 64 |
| MFMA | 20,480 |
| VGPR | 20 |
| AccVGPR | 4 |
| LDS | 8,192 B |
| scratch | 0 B |
| VGPR/SGPR spill | 0 / 0 |
| trace | 40.941 us |

KKT correctness：

- T=64/128/512 对比旧 KKT，max abs `4.47e-8`；
- KKT 后接 solve，max abs `3.73e-8`；
- causal mask 和跨 token16 tile 均通过。

结论：KKT-S0 是明确成功的方案，保留并接入 full。

---

### 5. W/U：从重复 token16 MFMA 到 BT64 四 wave ownership

#### 5.1 W/U 的数学

W 和 U 的共同形式是一个 chunk-local 小矩阵乘：

```text
A_w[t,s] = a_solved[t,s] * beta[s] * exp(g[s])
A_u[t,s] = a_solved[t,s] * beta[s]

W = A_w @ K
U = A_u @ V
```

输出为 FP32，供 asm recurrence 使用。

#### 5.2 Stage 3 之前的主要问题

历史 v14/v24 的 MFMA16 microkernel 是为 BT16 设计的。直接把它扩展到
BT64，容易形成：

```text
4 个独立 token16 ownership
每个 tile 都重复读取 A / K / V / g / beta
每个 tile 都单独做 correction 和 store
```

这种方案虽然出现了 MFMA，却没有充分利用 BT64 内部的共享数据。

#### 5.3 W/U-S0：先保守地改 ownership

新增 kernel：

```text
_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0
_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0
```

新的 launch：

```text
grid = num_chunks * 8 value_heads * 8 column_tiles
workgroup = 256 threads
wave0..wave3 = 四个 token16 row tile
```

和 Stage 3 相比，Stage 4 让一个 CTA 处理完整 BT64 row extent，而不是让
多个 CTA 分别处理各个 token16 row。这样可以：

- 共享 A staging；
- 共享 K/V source tile；
- 减少 kernel program 数；
- 减少重复的 address generation；
- 保留 MFMA16 的稳定 fragment mapping。

#### 5.4 W/U-S0 的第一版：保留 scalar residual correction

为了保证 FP32 correctness，主 BF16 MFMA 后保留了原来的 residual：

```python
main = bf16(A) @ B
correction = (A_fp32 - bf16(A)) @ B
result = main + correction
```

但第一版 correction 仍然是 scalar loop：

```text
for source_offset in range(64):
    correction += (coeff_fp32 - coeff_bf16) * B[source_offset]
```

结果：正确，但收益不够强。

T=2048：

```text
Stage 3 W/U       0.248470 ms
W/U-S0            0.229160 ms
```

只有约 `1.084x`，说明 ownership 改善存在，但 scalar residual 仍然是明显
负担。

#### 5.5 失败尝试：删除 correction

下一步尝试是直接删除 correction，只保留 BF16 main MFMA：

```python
result = bf16(A) @ B
```

这个版本非常快，所以它是一个有价值的性能上界诊断。但 correctness
完全不能接受，尤其是在：

- nonzero gate；
- high dynamic range；
- cancellation；
- 多 chunk recurrence。

失败原因是：FP32 A 先转 BF16 后，量化误差会被 W/U 计算和 recurrence 放大。

这个函数被保留为：

```text
qwen_gdn_w_u_bt64_mfma_v2_no_correction_failed
```

但 final path 永远不调用它。

#### 5.6 W/U-S1：把 residual correction 也 MFMA 化

最终采用的方案是：

```text
第一次 MFMA：bf16(A) @ B
第二次 MFMA：bf16(A_fp32 - bf16(A)) @ B
最后相加
```

代码层面新增了 constexpr 控制：

```python
use_residual_correction = False
use_mfma_residual = True
```

residual A 的生成：

```python
coeff_staged = float(bf16(coeff_residual))
residual = coeff_residual - coeff_staged
a_bf16[...] = bf16(residual)
```

然后使用和主路径相同的 MFMA fragment mapping。

这一步的逻辑是：

```text
不牺牲 correction 的数学作用，
但把 correction 的逐元素串行计算改成矩阵 tile 计算。
```

#### 5.7 W/U-S1 结果

| T | Stage 3 | W/U-S0 | W/U-S1 residual MFMA |
|--:|--:|--:|--:|
| 512 | `0.123203` | `0.095522` | `0.060971` |
| 1024 | `0.149723` | `0.146318` | `0.064196` |
| 2048 | `0.248470` | `0.229160` | `0.084165` |

T=2048 的 S1 相对 Stage 3 为 `2.9522x`。

S1 和 exact S0 的误差：

```text
W/U max abs <= 1.8775e-6
```

T=2048 rocprof：

| 指标 | W | U |
|:--|--:|--:|
| workgroup | 256 | 256 |
| MFMA | 262,144 | 262,144 |
| VGPR | 52 | 48 |
| AccVGPR | 4 | 8 |
| LDS | 2,560 B | 2,560 B |
| scratch | 0 B | 0 B |
| trace | 27.722 us | 24.797 us |

相对于 Stage 3，MFMA count 因为加入 residual MFMA 而增加，但 VALU、VMEM、
SALU 和 trace 显著下降。这说明本次收益来自 ownership、数据复用和消除
scalar correction，而不是单纯减少 MFMA 数量。

结论：保留 W/U-S1，不保留 no-correction 版本。

---

### 6. chunk-o：从多个小 tile 到一个 BT64 cooperative CTA

#### 6.1 chunk-o 的计算结构

chunk-o 需要计算两类贡献：

```text
inter-state contribution:
    q @ H

intra-chunk causal contribution:
    score(q,k,g) @ V_new
```

然后把两者相加，得到输出。

#### 6.2 Stage 2/Stage 3 的问题

Stage 2 的 generic chunk-o 是单线程 scalar fallback：

- workgroup=1；
- MFMA=0；
- 需要大量 q/k/vn/h global load；
- private scratch；
- T=2048 trace 超过 11 ms。

Stage 3 已经使用 MFMA16，但仍然保留比较分散的 token16 tile ownership，
跨 tile 的 Q/K/H/V-new 复用不充分。

#### 6.3 chunk-o-S0 的代码变化

新增 kernel：

```text
_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0
```

launch：

```text
workgroup = 256 threads
四个 wave = 四个输出 token16 tile
一个 CTA = 一个 chunk/value-head/V16 block
```

shared staging：

```text
q_scaled_bf16
h_bf16
k_bf16
score_decay_bf16
vn_t_bf16
```

目标是让一个 CTA 之内的四个 token16 tile 共享：

- Q scaled；
- H state；
- K source tile；
- V_new；
- intra score/decay 中间结果。

#### 6.4 失败尝试：合并 inter/intra accumulator

从减少寄存器和最终合并指令的角度，曾经尝试把：

```text
inter_acc
intra_acc
```

合并成同一个 accumulator。

结果是数值误差约 `2e-2 ~ 3.4e-2`，超过 frozen contract。原因是 MFMA
累积顺序改变后，FP32 累加的舍入路径发生变化，尤其会影响 chunk 内 causal
贡献和 recurrence 后的最终 state/output。

这个方案没有继续“调容差”，而是直接放弃，保留两个 accumulator：

```text
inter_acc: q @ H
intra_acc: score_decay @ V_new
final = inter_acc + intra_acc
```

它们仍然在 register/accumulator 中保持到最终写回，只是在最后一步相加。

#### 6.5 chunk-o-S0 结果

| T | Stage 3 | chunk-o-S0 | speedup |
|--:|--:|--:|--:|
| 512 | `0.087570` | `0.044406` | `1.9720x` |
| 1024 | `0.141130` | `0.062573` | `2.2554x` |
| 2048 | `0.228620` | `0.095342` | `2.3979x` |

T=2048 rocprof：

| 指标 | chunk-o-S0 |
|:--|--:|
| workgroup | 256 |
| MFMA | 458,752 |
| VGPR | 112 |
| AccVGPR | 64 |
| LDS | 27,136 B |
| scratch | 0 B |
| trace | 68.622 us |

chunk-o correctness：

- T=64/128/512 对比 Stage 3，max abs 为 0；
- inter-only、intra-only、source0/1/2 均通过；
- cross-token16 causal tile 通过。

结论：保留 separate inter/intra accumulator 版本。

---

### 7. solve：为什么审计了但没有重写

#### 7.1 当前 solve 是什么

v18 solve 的数学是 lower-triangular recurrence：

```text
M[row,col] = M0[row,col]                         if col >= row
M[row,col] = M0[row,col]
                + sum_i<row M0[row,i] * M[i,col]  if col < row
a_solved[row,col] = M[row,col] + diagonal_identity
```

v18 使用 128-thread workgroup，把一个 row 内的 column/update 并行化，
但 row dependency 仍然必须顺序推进。

#### 7.2 审计结果

| T | v18 FP32 | vLLM FP32 | v18/vLLM |
|--:|--:|--:|--:|
| 512 | `0.121140 ms` | `0.053359 ms` | `2.270x` |
| 2048 | `0.126888 ms` | `0.053779 ms` | `2.359x` |

这个差距足以允许进行一个 solve v2 实验，但没有足够理由在本轮直接改：

- vLLM 使用 hierarchical block inverse/dot schedule；
- 它不是对 v18 row recurrence 的简单局部修改；
- 当前 solve 已经 correctness 近似 exact；
- KKT/W/U/chunk-o 已经通过完整 37-case gate；
- 一个不成熟 solve 改动可能把所有下游误差放大。

因此本轮决定：

```text
solve 只做 audit，不修改实现。
下一阶段单独设计 hierarchical FP32 BT64 solve。
```

---

### 8. full pipeline 的最终结果

#### 8.1 完整延迟

三组独立 session，每组 warmup=10、repeat=50，取 session median 后再取三组
中位数：

| T | Stage 4 | Stage 3 | v24 | vLLM |
|--:|--:|--:|--:|--:|
| 512 | `0.313706` | `0.490168` | `0.300726` | `0.304873` |
| 2048 | `0.475867` | `1.089079` | `0.587173` | `0.365043` |
| 8192 | `1.399619` | `3.655211` | `2.122734` | `0.725157` |
| 16384 | `2.632552` | `7.213319` | `4.252098` | `1.268545` |

T=512 时 Stage 4 稍慢于 v24/vLLM，说明 kernel launch 和固定开销仍然重要。
从 T=2048 开始，BT64 的收益足以超过 v24。

#### 8.2 frozen correctness

37-case full matrix 覆盖：

- T=64/128/512/2048 随机输入；
- zero/nonzero initial state；
- neutral gate；
- high dynamic range；
- cancellation；
- small-value；
- T=8192 smoke。

最终结果：

| 检查项 | 最大误差 | 门槛 | 结果 |
|:--|--:|--:|:--|
| public BF16 output | `0.001953125` | `0.0078125` | 通过 |
| FP32 final state | `0.013475478` | `0.020000000` | 通过 |

另外：

```text
Stage 4 pytest: 29 passed
immutable regressions: 80 passed, 1 skipped
```

#### 8.3 资源和 ISA

四个新 kernel 都满足：

- Scratch = 0；
- VGPR spill = 0；
- SGPR spill = 0；
- 没有 v29 那种 high-AGPR/resource cliff。

静态 ISA 中的 MFMA16 数量：

| kernel | MFMA16 指令数 | MFMA32 |
|:--|--:|--:|
| KKT | 8 | 0 |
| W | 8 | 0 |
| U | 8 | 0 |
| chunk-o | 56 | 0 |

---

### 9. 哪些方案最终保留，哪些方案放弃

#### 保留

1. KKT：BT64 拆成 4x4 token16 tile，64-thread MFMA16。
2. W/U：256-thread BT64 ownership，主 MFMA + residual MFMA。
3. chunk-o：256-thread cooperative CTA，共享 Q/K/H/V-new，inter/intra accumulator 分离。
4. v18 solve：不改，作为当前正确 baseline。
5. asm recurrence：不改，直接复用冻结版本。

#### 放弃

1. W/U 只保留 BF16 main MFMA：速度快但数值错误。
2. W/U 保留 scalar residual：正确，但 T=2048 只有约 `1.084x`，不够好。
3. chunk-o 合并 inter/intra accumulator：误差约 `2e-2~3.4e-2`。
4. 继续改 v29 accumulator/lifetime/compiler lowering：不属于本轮 BT64 高级代码目标，且已有 resource cliff 证据。
5. 修改 asm recurrence：当前 asm 是稳定且高效的冻结组件，没有理由在本轮冒险。
6. 直接用 vLLM full wrapper：违反实验边界，不能作为 Avelang 候选实现。

---

### 10. 最终瓶颈和下一步

需要区分两个概念：

#### 整体最大 stage

T=2048 时，冻结 asm recurrence 约 `0.2015 ms`，是单个 stage 中最大的。
但它是本轮明确禁止修改的 asm kernel，而且已经是稳定实现。

#### 当前可继续优化的最大高级 stage

solve 约 `0.1257 ms`，并且相对 vLLM FP32 solve 约慢 `2.36x`。所以当前
真正适合作为下一步的方向是：

```text
设计一个独立的 hierarchical FP32 BT64 solve v2
```

下一步建议顺序：

1. 先构造 solve block inverse/dot 的 isolated repro；
2. 对比 v18 row recurrence 的 correctness；
3. 检查 workgroup、LDS、VGPR、scratch；
4. 只有 solve standalone 通过后才接回 full pipeline。

目前没有证据表明需要继续修改 compiler 或 AMDGPU RA，也没有必要马上写
新的 assembly。Stage 4 的结果已经证明：通过正确的高级 ownership 和 data
reuse，BT64 非 recurrence 路径可以从 `1.089 ms` 降到 `0.476 ms`。

---

### 11. 如何重现实验

在 `ljd_qwen_vllm_avelang_rocm722` 容器中执行，必须使用历史 Avelang Python
环境，避免误导入 `/opt/avelang`：

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_full_pipeline_stage2
```

Stage 4 correctness：

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  -s --tb=short
```

Full frozen matrix：

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/stage4_runner.py \
  --random-cases 30
```

Full benchmark：

```bash
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/bench_stage4.py \
  --T 512 2048 8192 16384 --warmup 10 --repeat 50 --session a
```

Microbenchmark：

```bash
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_nonrecurrence_mfma_v2.py \
  --T 512 1024 2048 --warmup 10 --repeat 50
```

rocprof、HSACO、ISA 和汇总命令已经集中在：

[commands.sh](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_nonrecurrence_stage4/commands.sh)

---

### 12. 一句话总结

这次真正有效的不是“把 kernel 改得更复杂”，而是逐级找到正确的 ownership：

```text
KKT：把 BT64 变成 4x4 个 MFMA16 tile
W/U：让一个 256-thread CTA 拥有完整 BT64 row，并把 residual 也 MFMA 化
chunk-o：让一个 256-thread CTA 复用 Q/K/H/V-new，保持数值安全的双 accumulator
solve：先审计，暂不冒险重写
recurrence：保持冻结不动
```

最终得到：

```text
Stage 3 BT64: 1.089079 ms
Stage 4 BT64: 0.475867 ms
```

并且所有 correctness、MFMA、scratch/spill、full-sequence 门槛均通过。

---

### 阶段分界：Stage 5 到 Stage 6S 的完整实验链、失败原因与推理方法

> 本续篇从上文 Stage 4 结束的位置继续记录。  
> 它不仅汇总“做成了什么”，也保留所有重要的负结果、停止条件和测量口径变化。
>
> 阅读时必须区分三种结论：
>
> 1. **kernel standalone 结论**：只说明某个 kernel 本体；
> 2. **full graph 结论**：说明它接入上下游以后是否仍有收益；
> 3. **production/public API 结论**：说明真实公开接口下的端到端表现。
>
> 三者不能互相替代。后续曾经因为混用不同 harness、CUDA Graph 与 Eager
> public API 数字，造成“差距怎么反而变大”的观感。本文会把这些口径明确拆开。

---

### 13. Stage 4 之后，我们为什么先优化 solve

Stage 4 完成后，非 recurrence 的灾难性 scalar fallback 已经消失：

- KKT 已使用 MFMA；
- W/U 已使用 BT64 ownership 和 residual MFMA；
- chunk-o 已使用 cooperative CTA；
- scratch 和 spill 均为 0；
- full 从 Stage 3 的约 `1.089 ms` 降至约 `0.476 ms`。

此时不能再用“哪个 kernel 看起来大就直接改哪个”的方式推进。我们重新审计了
各阶段，发现当前仍可修改的阶段里，`solve` 是最明显的候选：

```text
v18 solve T=2048：约 0.126 ms
vLLM FP32 solve：  约 0.054 ms
差距：约 2.36x
```

但这并不意味着应立即重写 solve。首先要回答：

1. v18 慢在数学工作量，还是慢在 lowering/寄存器？
2. vLLM 使用的是同一种逐行 recurrence，还是完全不同的算法 DAG？
3. 是否能在不修改 compiler 和 assembly 的前提下复现其结构？

因此 Stage 5A 被定义成 **root-cause audit-only**，而不是直接开始写新 kernel。

---

### 14. Stage 5A：solve 根因审计

#### 14.1 为什么做这个实验

v18 solve 正确、无 scratch、无 spill，但 body 明显慢。这里有两种完全不同的可能：

```text
可能 A：算法结构本身串行，应该换算法；
可能 B：算法合理，只是编译器 lowering 或寄存器分配不好。
```

如果是 A，修改 compiler 没有意义；如果是 B，贸然换数学结构又会增加正确性风险。

所以 Stage 5A 的目标不是“让 solve 变快”，而是把这两个可能分开。

#### 14.2 审计发现

v18 的核心结构是：

```text
对 row = 1..63 顺序推进
每一行内部并行更新 columns
每一行依赖前面所有行
```

它具有以下特征：

- 一个 chunk/head 一个 CTA；
- workgroup 128，2 waves；
- 63 个 row 依次推进；
- recurrence 区域约 189 个 barrier，加初始 barrier 共约 190；
- MFMA count 为 0；
- 主要依赖 LDS 和标量/向量更新；
- 输入即使是对角矩阵，固定 barrier 结构也基本不变。

当前 vLLM 使用的不是同一种逐行算法，而是 **4×4 个 16×16 block 的层次化下三角逆**：

```text
先求四个 16x16 对角块的局部逆
再按 block DAG 计算 X21、X32、X43
然后计算 X31、X42
最后计算 X41
```

典型公式是：

```text
X21 = -X22 A21 X11
X32 = -X33 A32 X22
X43 = -X44 A43 X33

X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)

X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

最终机器代码使用 FP32 MFMA，而不是 63 行顺序 recurrence。

#### 14.3 关键测量

同输入 body 测量中，v18 相比 vLLM 的差距稳定存在：

| T | v18 body | vLLM body | v18/vLLM |
|--:|--:|--:|--:|
| 512 | `0.116694 ms` | `0.035372 ms` | `3.30x` |
| 2048 | `0.122422 ms` | `0.035613 ms` | `3.44x` |
| 8192 | `0.229501 ms` | `0.043484 ms` | `5.28x` |

T=2048 的 profiler 结构也明显不同：

| 指标 | v18 | vLLM |
|:--|--:|--:|
| WG | 128 | 256 |
| MFMA | 0 | 65,536 |
| LDS 指令 | 约 1.178M | 约 0.397M |
| Scratch | 0 | 0 |

#### 14.4 结论

Stage 5A 的结论是：

> solve 的主因是算法调度结构，不是 AMDGPU RA 或普通 lowering 失误。

因此唯一合理的下一步是：

```text
高层 AveLang 实现
FP32 BT64 4×16 hierarchical block inverse
一个 256-thread / 4-wave CTA
使用现有 FP32 MFMA primitive
```

#### 14.5 这一步教会我们的东西

遇到慢 kernel 时，不要先问“怎样减少几条指令”，而要先问：

```text
当前依赖图是否天然串行？
参考实现是否使用了不同的数学分解？
```

Stage 5A 避免了一次错误的 compiler/assembly 优化支线。

---

### 15. Stage 5B：hierarchical FP32 solve v1

#### 15.1 为什么做

Stage 5A 已经证明 v18 的问题是 63-row 串行依赖。因此 Stage 5B 不再微调 v18，
而是实现一个独立的层次化 FP32 solve：

- BT64 拆成 4×4 个 16×16 block；
- 一个 chunk/head 对应一个 CTA；
- workgroup 256，4 waves；
- block product 使用 `mfma_16x16x4_f32_f32`；
- accumulator 保持 FP32；
- 明确限制 LDS、scratch 和 spill。

#### 15.2 实现原则

最重要的不是“使用 MFMA”本身，而是缩短依赖链：

```text
v18：
row0 -> row1 -> row2 -> ... -> row63

hierarchical：
4个对角块并行/局部处理
-> 三层有限的block DAG
```

这把 63 层依赖压缩成少量 block-level 阶段。

#### 15.3 结果

新的 hierarchical solve：

- 正确性通过；
- 资源正常；
- scratch = 0；
- spill = 0；
- LDS 为 8 KiB；
- T=2048 集成 wrapper 中约 `0.029 ms`；
- rocprof kernel trace median 约 `13.701 us`；
- v18 同口径 wrapper 约 `0.126 ms`。

因此 solve 本体大约获得 `4.3x` 级别加速。

#### 15.4 为什么这次算成功

它同时满足三项：

1. 数学结构改变有明确依据；
2. 正确性没有依赖放宽阈值；
3. 性能和资源都朝正确方向变化。

它没有修改 compiler 或 assembly，说明高级代码足以表达这个 block DAG。

---

### 16. Stage 5C：把新 solve 接回 Stage 4 full graph

#### 16.1 为什么 standalone 成功后还必须做 full integration

一个 kernel standalone 快，不代表 full 一定按相同幅度变快。上下游会影响：

- 数据布局；
- 中间张量；
- kernel launch 顺序；
- cache/设备状态；
- dtype；
- 分配与 wrapper 开销。

所以 Stage 5C 只做一件事：

```text
在 Stage 4 full path 中
把 v18 solve 替换为 hierarchical_fp32_v1
其它 stage 完全不变
```

并且保持：

- 默认 selector 仍是 `v18`；
- 新 solve 只通过显式 `solve_impl="hierarchical_fp32_v1"` 选择；
- 没有 silent fallback；
- production 不变。

#### 16.2 正确性

- 新集成测试：`6 passed`；
- 原 Stage 4 默认路径：`29 passed`；
- public output 和 final state 均在冻结阈值内；
- v1 solve 资源为 `VGPR=44`、`AccVGPR=4`、`Scratch=0`。

#### 16.3 性能结果

T=2048 三个 session：

```text
Stage 4 + v18：约 0.474906 ms
Stage 4 + v1： 约 0.454215 ms
full gain：    约 20.13 us
```

但 solve 本体大约节省：

```text
0.126 ms -> 0.029 ms
约 97 us
```

于是出现了一个新问题：

> 为什么 solve 本体节省约 97 us，full 只节省约 20 us？

#### 16.4 第一轮分段 event 诊断

在每个 stage 之间插 HIP event 后，观察到：

```text
solve：约省 96.6 us
W/U：  约多 16.9 us
recurrence：约多 75.4 us
```

但这张表不能直接相加，因为 stage event 会改变：

- dispatch 边界；
- 同步；
- cache 驻留；
- queue pacing；
- 中间张量状态。

因此 Stage 5C 没有武断地写成“L2 cache 导致 75 us”，而只得出：

> 新 solve 已正确接入；其后连续 kernel 的组合执行状态发生了变化。

这就是 Stage 5D 的来源。

---

### 17. Stage 5D：downstream state-coupling 审计

#### 17.1 为什么做

Stage 5C 的核心矛盾是：

```text
solve-only gain：约 97 us
public full gain：约 19~20 us
```

若直接继续优化 solve，会忽略已经发生的 full-path 抵消；若直接改 recurrence，
又没有证据证明 recurrence 代码本身有问题。

Stage 5D 因此是 audit-only，目标是排除以下变量：

- solved tensor 数值；
- consumer pointer；
- alignment；
- allocator；
- 隐藏 dispatch；
- stage-event instrumentation；
- cache/执行状态。

#### 17.2 做过的控制实验

##### A. 冻结两条执行图

两图都是 8 个 dispatch，只有 solve 不同：

```text
cumsum
-> KKT
-> solve
-> W
-> U
-> recurrence
-> chunk-o
-> cast
```

W/U、recurrence、chunk-o 的代码和 launch 不变。

##### B. 连续 tail 单 event

只用一个 event 包住：

```text
W -> U -> recurrence -> chunk-o -> cast
```

solve 在 event 前执行，tail 内不插 event。

T=2048：

```text
tail after v18：约 0.2679 ms
tail after v1： 约 0.3322 ms
v1 penalty：   约 64.3 us
```

这证明 downstream 抵消并不是分段 event 完全伪造的。

##### C. Canonical-data 控制

两种 solve 都执行，但丢弃输出；downstream 始终消费同一份 bitwise-identical
canonical tensor。

T=2048：

```text
after v18：约 0.2675 ms
after v1： 约 0.3321 ms
差距：     约 64.6 us
```

因此 solved 数值差异不是主因。

##### D. 同 consumer pointer 控制

两种 solve 的结果 copy 到同一个 canonical consumer buffer，copy 放在计时区间外。

差距仍约 `68 us`，说明 downstream 可见 pointer 和 alignment 不是主因。

但这一步仍未关闭“两个 solve 自身写入不同物理输出地址”这一变量，因此后来还需要
Stage 5E。

##### E. Warm 与大工作集扰动

T=2048：

```text
正常连续执行：差距约 68.6 us
先运行一次相同 tail：差距约 0.12 us
512 MiB workload 扰动：差距约 -0.08 us
reduction prime：差距仍约 75 us
```

这说明差距属于一种可被较长 GPU workload 重置的瞬态状态。

#### 17.3 得到什么结论

高置信度结论：

> v18 solve 会留下一个对后续 tail 有利的瞬态执行状态；v1、no-solve 和短 dummy
> predecessor 不会。

但没有足够证据把它唯一归因于：

- L2/TCC/TCP cache；
- 短时频率/功耗爬升；
- runtime queue pacing；
- CU/wave 状态。

#### 17.4 为什么这个“没有精确根因”的实验仍有价值

它排除了很多看似合理但错误的修复：

- 不应因为输出误差小就怪数值；
- 不应随意修改 W/U；
- 不应修改 recurrence 汇编；
- 不应增加 dummy warmup；
- 不应根据插 event 的分段数字做加法。

负结果的价值在于收缩问题空间。

---

### 18. Stage 5E：direct common-output pointer

#### 18.1 为什么做

Stage 5D 虽然让 downstream 读取同一 pointer，但两个 solve 自身仍先写入不同 allocation。
这留下一个可能：

```text
solve store 的物理地址/cache-set 不同
-> 影响后续 GPU 状态
```

所以 Stage 5E 要求：

```text
v18 solve ------------------\
                             -> exact same solved_common.data_ptr()
hierarchical solve ---------/
```

不允许中间 copy、fill 或额外 dispatch。

#### 18.2 实现和正确性

新增 direct-out wrapper：

```text
qwen_gdn_solve_v18_bt64_direct_out_audit(a, out)
qwen_gdn_solve_hierarchical_fp32_v1_direct_out_audit(a, out)
```

结果：

- 两种 solve 直接写 exact same `data_ptr`；
- original wrapper 与 direct-out 均 bitwise 相同；
- NaN prefill、upper triangle、重复复用、residual 全部通过；
- 共测试 80 个 allocation session、35 个不同地址；
- 合并回归 `53 passed`。

#### 18.3 关键结果

T=2048：

```text
Stage 5D 原始 tail penalty：64.255 us
Stage 5E direct-common：    64.035 us
变化：                     -0.220 us
```

95% bootstrap 区间约：

```text
[63.254, 64.336] us
```

不同 ABAB/BABA/ABBA 顺序、不同 allocation 都稳定复现。

#### 18.4 资源和 profiler 结果

- downstream 动态指令数相同；
- TCP 总访问数相同；
- TCC 最大相对差约 `0.232%`；
- 没有发现可解释 64 us 的 cache 流量差；
- 粗粒度 clock/power telemetry 没有稳定分叉；
- profiler 本身会放大 dispatch gap。

#### 18.5 结论

> output pointer/address 被排除为主因。

此时只剩“前驱诱发的瞬态 GPU 状态”这一操作层结论，但精确硬件机制仍未识别。

这一步失败在“没有找到可修复的 pointer 问题”，但成功关闭了一个重要变量。

---

### 19. Stage 5F：低扰动可观测性 Go/No-Go

#### 19.1 为什么做

Stage 5D/5E 已经证明现象真实，但没有分离：

- cache residency；
- runtime pacing；
- clock/power；
- CU/wave 状态。

继续 profiler 之前必须先确认：

> profiler 本身是否足够低扰动，能观察一个约 63 us 的效应？

如果 profiler 会改变该效应，任何 timestamp 和 counter 因果分析都不可信。

#### 19.2 预注册 gate

只有同时满足以下条件才允许继续：

- 不修改 graph；
- graph 内不插 event；
- latency 扰动足够小；
- A/B penalty 扭曲足够小；
- 能提供 cache、dispatch 或短时状态信息。

#### 19.3 实测结果

T=2048 tail：

```text
HIP eager/event baseline penalty：62.772 us
rocprofv3 trace-only penalty：    87.230 us
penalty distortion：              24.457 us
```

trace 还带来约 `28.402 us` 的 latency distortion。

因此 `rocprofv3 --kernel-trace` 未通过 gate。

没有任何 profiler/counter/telemetry 模式能同时满足：

```text
可观察
+
足够低扰动
```

#### 19.4 决策

Stage 5F 正式宣布：

```text
Stage 5 transient-state root-cause branch closed
```

没有继续 PMC replay、PC sampling 或更多 profiler 花样，因为这会违反预注册 stop rule。

#### 19.5 这一步最重要的学习点

优化研究中必须允许：

> “现有工具无法可靠归因，因此停止。”

这不是放弃，而是防止无限追逐不可观测机制。一个严格的 No-Go
通常比一个听起来合理但无证据的“L2 根因”更有科研价值。

---

### 20. 为什么后来感觉“离 vLLM 越来越远”

这里必须解释测量口径变化。

#### 20.1 Avelang 实际一直在变快

按当时各阶段自己的测量环境：

```text
Stage 2：约 16.04 ms
Stage 3：约 1.09 ms
Stage 4：约 0.476 ms
Stage 5C：约 0.454 ms
Stage 6A CUDA Graph：约 0.336 ms
```

Avelang 并没有因为优化而变慢。

#### 20.2 为什么 gap 看起来扩大

旧测量中，大量固定开销仍在：

- Python wrapper；
- allocation；
- launch；
- module/dispatch 固定成本。

Stage 6A 使用 CUDA/HIP Graph replay 后，两边固定成本都下降，但 vLLM 下降更多：

```text
旧口径附近：
Avelang 约 0.455 ms
vLLM    约 0.365 ms
gap     约 90 us

Stage 6A graph replay：
Avelang 0.335539 ms
vLLM    0.188900 ms
gap     146.639 us
```

这不代表 Avelang 退化，而是 graph replay 暴露了 vLLM 更低的每-chunk执行成本。

#### 20.3 Stage 6A 的关键 slope

```text
Avelang：8.771643 us/chunk
vLLM：   4.365784 us/chunk
gap：    4.405859 us/chunk
```

所以真正问题是：

> Avelang 每处理一个 BT64 chunk 仍做了更多结构性工作。

#### 20.4 后续统一规则：Eager public API

从后续 Stage 6T 开始，权威测量规则改为：

```text
Avelang 与 vLLM 都调用各自 public API
每个 timed sample 都是完整 Eager public API 调用
public API 内部 allocation/cast/dispatch 均计时
CUDA/HIP Graph 不再作为权威性能结论
private body 只用于诊断
rocprof trace latency 只用于结构，不用于性能排名
```

因此：

- Stage 6A/6S 的 CUDA Graph 数字仍有结构诊断价值；
- 但未来不能拿它们和 Eager public API 数字直接相减；
- 最终“SOTA/超过 vLLM”必须基于同口径 Eager public API。

---

### 21. Stage 6A：严格 full-graph gap audit

#### 21.1 为什么做

Stage 5 关闭后，我们不再追不可观测的 transient state，而是回到主目标：

```text
Avelang full forward 超过 vLLM
```

Stage 6A 的目标是把 full gap 分解成：

- 哪些逻辑 stage；
- 哪些 dispatch；
- 哪些中间 tensor；
- 哪些 dtype；
- 哪些随 chunk 线性增长的成本。

#### 21.2 同口径 full 结果

当时采用 CUDA Graph replay 作为结构化对比口径：

| T | Avelang | vLLM | gap |
|--:|--:|--:|--:|
| 512 | `0.122823 ms` | `0.100930 ms` | `21.893 us` |
| 1024 | `0.191845 ms` | `0.128952 ms` | `62.893 us` |
| 2048 | `0.335539 ms` | `0.188900 ms` | `146.639 us` |
| 4096 | `0.607022 ms` | `0.324142 ms` | `282.880 us` |
| 8192 | `1.156198 ms` | `0.602416 ms` | `553.783 us` |
| 16384 | `2.302501 ms` | `1.179092 ms` | `1123.409 us` |

#### 21.3 实际 dispatch 图

Avelang 8 个 dispatch：

```text
cumsum
KKT
solve
W
U
recurrence
chunk-o
final cast
```

vLLM 7 个 dispatch：

```text
cumsum
KKT
solve fill
solve merge
fused W/U
recurrence
chunk-o直接写BF16
```

注意：vLLM solve 虽然逻辑上是一个 stage，但机器图中包含 fill 和 merge；
而 W/U 是一个 fused dispatch，chunk-o 直接写 public BF16 output。

#### 21.4 中间 dtype 差异

T=2048 可见 contract：

| intermediate | Avelang | vLLM |
|:--|:--|:--|
| solved | FP32 4 MiB | BF16 2 MiB |
| W | FP32 8 MiB | BF16 4 MiB |
| U | FP32 8 MiB | BF16 4 MiB |
| v_new | FP32 8 MiB | BF16 4 MiB |
| output staging | FP32 8 MiB | 无 |
| public output | BF16 4 MiB | BF16 4 MiB |

Avelang 至少额外物化约 22 MiB 中间 storage，尚未计算 consumer reread。

#### 21.5 各 stage body gap

| stage | T=2048 gap | gap slope |
|:--|--:|--:|
| cumsum | `+6.850 us` | `+0.001 us/chunk` |
| KKT | `+18.828 us` | `+0.878 us/chunk` |
| solve | `-14.902 us` | `-0.100 us/chunk` |
| W/U | `+36.214 us` | `+0.990 us/chunk` |
| recurrence | `+39.980 us` | `+1.261 us/chunk` |
| chunk-o | `+46.169 us` | `+1.196 us/chunk` |
| cast | `+15.483 us` | `+0.069 us/chunk` |

#### 21.6 为什么当时先选 chunk-o

`chunk-o` 是最大的可修改 body gap，并且资源差异巨大：

```text
CTA：Avelang 2048，vLLM 512
MFMA：约 5.6x
VMEM：约 11.9x
LDS inst：约 5.5x
```

所以 Stage 6B 被设计为只改 ownership，不同时修改 BF16 output 或 cast。

这是一种正确的实验隔离：

```text
先验证“重复工作”假设
再验证“dtype/store”假设
```

---

### 22. Stage 6B：chunk-o ownership O0/O1

#### 22.1 当前 ownership

Stage 4 current：

```text
一个 CTA = token16 × V16
一个 chunk-head 共 8 个 V16 CTA
T=2048 总 CTA = 2048
```

当前 vLLM：

```text
一个 CTA = token64 × V64
一个 chunk-head 共 2 个 CTA
T=2048 总 CTA = 512
```

Avelang 会在八个 V16 CTA 中重复读取/变换 Q/K 和 score；
vLLM 在 V64 CTA 内复用这些数据。

#### 22.2 O0：token64 × V64

##### 为什么做

目标是把 CTA 降到与 vLLM 一致，同时让同一 CTA 复用 lower score。

O0：

- T=2048 CTA `2048 -> 512`；
- 4 waves 管 4 个 token16 row；
- CTA 内缓存 4×4 的 lower score tiles；
- output FP32 staging 和 final cast 不变。

##### 正确性

O0 对 current Stage 4：

- random、zero-H、zero-V_new；
- inter-only、intra-only；
- high/small/cancellation；
- cross-token16；
- 所有 V16 boundary；
- T=8192；

均 bit-exact。

##### 工作量改善

T=2048：

| metric | current | O0 |
|:--|--:|--:|
| CTA | 2048 | 512 |
| MFMA | 458,752 | 188,416 |
| VMEM | 851,968 | 335,872 |
| LDS inst | 1,343,488 | 520,192 |

这证明 ownership 假设成立：O0 真实减少了重复工作。

##### 为什么仍然失败

为了跨 V16 复用 score，O0 引入了更长的 score LDS lifetime：

| resource | current | O0 |
|:--|--:|--:|
| AccVGPR | 64 | 188 |
| LDS block | 27,136 B | 33,280 B |
| occupancy | 17.90% | 8.86% |
| static barriers | 13 | 19 |

T=2048 body：

```text
current：0.076073 ms
O0：     0.062012 ms
speedup：1.227x
要求：   >=1.5x
```

它只回收约 `14.061 us`，未达到晋级门槛。

##### 这个失败告诉了我们什么

> 减少理论工作量不等于减少 latency。

必须同时检查：

- fragment lifetime；
- AccVGPR；
- LDS 容量；
- barrier；
- occupancy。

O0 不是“方向完全错误”，而是“工作复用成功，但资源组织不足以把收益完全兑现”。

#### 22.3 O1：token64 × V32

##### 为什么做

O1 是唯一允许的第二个变量，只把 V64 改为 V32，试图：

- 减小单 CTA accumulator；
- 缩短部分 lifetime；
- 提高 occupancy。

##### 结果

O1 反而产生：

- 1024 CTA；
- 更多 MFMA；
- 更多 LDS；
- 更多 barrier；
- T=2048 慢于 current；
- 长序列 slope 更差。

因此 O1 是明确负结果。

#### 22.4 最终决策

- O0/O1 都不接 full；
- Stage 4 current 保持不变；
- 不继续 O2/O3 tile sweep；
- 不修改 compiler 或 assembly；
- 暂不进入 direct-BF16 chunk-o。

#### 22.5 学习总结

vLLM 和 O0 表面上都是：

```text
token64 × V64
WG256 / 4 waves
512 CTA
```

但资源差异仍然非常大。说明：

> 几何参数相同，不代表 fragment lowering、寄存器生命周期和 block-dot
> lowering 相同。

这是 GPU kernel 优化中非常常见的陷阱。

---

### 23. Stage 6R：recurrence baseline reconciliation

#### 23.1 为什么突然回头核对 recurrence

历史上 asm-v0 曾与当时提取的 Triton recurrence 等速：

```text
约 0.195~0.197 ms
```

但 Stage 6A 显示：

```text
Avelang recurrence 比当前 vLLM 慢约 40 us
动态 MFMA：196608 vs 65536
```

这两条结论互相矛盾。继续优化 chunk-o 之前，必须先判断：

1. profiler 聚合是否算错；
2. 当前 vLLM 是否换了 specialization；
3. dtype/ABI 是否不同；
4. 旧 asm-v0 是否真的退化。

#### 23.2 Counter aggregation 审计

结果确认 `196608 / 65536` 是同口径、每个 recurrence dispatch 的动态 MFMA：

```text
按 chunk：     6144 / 2048
按 chunk-head：768 / 256
```

因此 3x 差异真实存在，不是 replay 聚合错误。

#### 23.3 当前两份机器代码

旧 asm-v0：

```text
HSACO: eede...c226
W/U/v_new：FP32
WG256
dynamic LDS 57344 B
VGPR/AccVGPR 128/192
```

当前 vLLM actual：

```text
HSACO: 6320...077e
W/U/v_new：BF16
BV=32
num_warps=2
num_stages=2
WG128
dynamic LDS 40960 B
VGPR/AccVGPR 104/160
```

两者 kernarg segment 都是 88 B，但只是物理 slot 布局类似，语义 ABI 不兼容。

#### 23.4 同口径 body

| T | asm-v0 FP32 | current vLLM BF16 | gap |
|--:|--:|--:|--:|
| 512 | `0.049554 ms` | `0.039199 ms` | `10.356 us` |
| 2048 | `0.154670 ms` | `0.114570 ms` | `40.100 us` |
| 8192 | `0.574333 ms` | `0.413274 ms` | `161.059 us` |
| 16384 | `1.172622 ms` | `0.841410 ms` | `331.212 us` |

#### 23.5 Isolated actual-vLLM bridge

Codex 提取当前 vLLM HSACO，建立 isolated external bridge：

- T=512/2048/8192/16384；
- h/v_new/final_state 对 native vLLM bit-exact；
- T=2048 性能差约 `-0.052%`；
- hash、grid、WG、LDS 和 ABI 都有 guard。

#### 23.6 结论

CASE B：

> asm-v0 没有退化，它仍忠实对应历史 FP32/WG256 specialization；当前 vLLM
> 选择了新的 BF16/WG128 specialization。

因此正确下一步不是重写 asm，而是测试：

```text
当前更快的 BF16 recurrence bridge
能否接入 Avelang FP32 上下游
```

---

### 24. Stage 6S：BF16 recurrence full-contract integration

#### 24.1 为什么不能直接替换 recurrence

当前 Avelang 周边 contract 是：

```text
W FP32
U FP32
old recurrence consumes FP32
v_new FP32
```

新的 vLLM recurrence 要求：

```text
W BF16
U BF16
recurrence outputs v_new BF16
```

所以 Stage 6S 使用显式、可见、可测的 boundary conversion：

```text
W FP32 -> BF16
U FP32 -> BF16
-> current-vLLM recurrence bridge
v_new BF16 -> FP32
-> current chunk-o
```

没有 reinterpret，没有 silent fallback。

#### 24.2 三条图

Graph A：旧 Avelang，8 dispatch。

Graph B：Stage 6S，11 dispatch：

```text
cumsum
KKT
solve
W FP32
U FP32
W cast
U cast
new BF16 recurrence
v_new cast FP32
chunk-o
final cast
```

Graph C：native vLLM，7 dispatch。

#### 24.3 Correctness

- recurrence bridge 对 native recurrence：
  `h/v_new/final_state` bit-exact；
- public output max abs：`0.001953125`；
- final state max abs：`0.0172200203`；
- 均低于冻结阈值；
- non-default stream、graph replay、invalid guard 均通过。

注意 final state 已接近 `0.02` 门槛，所以后续 BF16 propagation 必须扩大 seed 稳定性测试，
不能随意放宽阈值。

#### 24.4 Body 成本

T=2048：

```text
old recurrence：154.669 us
new recurrence：114.570 us
body gain：      40.099 us

W/U/v_new 三个边界 cast 合计：20.750 us
```

这些 isolated body 数不能简单相减当作 full gain，但能解释为什么 full 只能保留部分收益。

#### 24.5 Full 结果

同一 CUDA Graph harness 下：

| T | old A | Stage6S B | vLLM C | B gain |
|--:|--:|--:|--:|--:|
| 512 | `0.121721` | `0.120579` | `0.100450` | `1.370 us` |
| 1024 | `0.191685` | `0.181350` | `0.129493` | `10.319 us` |
| 2048 | `0.335679` | `0.309019` | `0.188800` | `26.647 us` |
| 4096 | `0.608985` | `0.550138` | `0.324282` | `58.960 us` |
| 8192 | `1.159743` | `1.033295` | `0.604138` | `126.288 us` |
| 16384 | `2.302501` | `2.024268` | `1.182217` | `278.926 us` |

T=2048 gain 的 95% CI：

```text
[26.595, 26.700] us
```

gap slope：

```text
old A - vLLM：4.396726 us/chunk
Stage6S - vLLM：3.284716 us/chunk
改善：1.112009 us/chunk
```

#### 24.6 决策

Stage 6S 是成功候选：

- 保留为 opt-in；
- 不改默认路径；
- 不改 recurrence HSACO；
- 不改 compiler/assembly；
- 下一步传播 BF16 storage boundary，消除三次显式 cast。

#### 24.7 学习总结

真正的性能边界不只是 kernel 本体，还包括 ABI：

```text
更快 kernel
+
不匹配的 dtype boundary
=
收益被 cast/materialization 吃掉
```

Stage 6S 说明：先用显式 cast 验证算法和性能，再逐步把 producer/consumer contract
改成原生 BF16，是一种低风险推进方式。

---

### 25. 从 Stage 6S 到 Stage 6W：为什么接下来不是再改旧 asm

Stage 6S 已经确认一个事实：current-vLLM 的 BF16 recurrence 本体比历史 asm-v0
快，但外围的 W/U 与 chunk-o 仍因 dtype boundary 产生额外 dispatch 和 global
materialization。于是路线从“优化 recurrence 数学”转向：

```text
先检查 producer/consumer 合同
-> 再消除不必要的 dtype boundary
-> 再看真正剩下的 global intermediate
```

这是和 Stage 4 很不同的目标。Stage 4 主要处理 scalar fallback、tile ownership 与
MFMA；Stage 6 的核心问题是图的边界：谁写什么 dtype、谁读什么 dtype、是否多发
cast kernel、是否重复把一个 single-use tensor 写回 global memory。

#### 25.1 本阶段统一的公开 API 合同

每个 Stage 6T--6W 的正式例子满足：

```text
start HIP event
-> 一个完整 Avelang 或 vLLM public API call
-> end HIP event
```

预热允许包含 import、JIT、Triton autotune、module load 和首次 runtime 初始化；
计时中则必须包含：

- API 内 allocation；
- numeric cast；
- kernel dispatch；
- wrapper glue；
- output object construction；
- 当前 stream 上真正执行的所有工作。

这也是为什么 Stage 6W 可以出现“private chunk-o body 略慢，而 full public API
仍更快”的正确结论。body 不计入被删掉的 casts，Eager full 会计入。

#### 25.2 为什么不能继续用 CUDA Graph 当正式排名

Graph replay 能把固定 buffer、allocation 与 Python wrapper 影响移开，适合回答：

```text
这个 device graph 的 steady-state dispatch 结构是什么？
```

但它不能回答：

```text
真实使用者调用 public API 时谁更快？
```

所以旧章节中出现的 Graph 数字仅保留为结构诊断。任何新旧表若没有同一 harness、
相同 stream、相同 allocation contract、相同 session/order，禁止跨表相减。

---

### 26. Stage 6T-Eager：融合 W/U 的 F0 与 F1

#### 26.1 问题

Stage 6S 的 Avelang 图在 W/U 周围存在：

```text
W kernel -> FP32 W -> cast -> BF16 W
U kernel -> FP32 U -> cast -> BF16 U
```

而 captured vLLM 使用一个 `recompute_w_u_fwd_kernel`。因此要区分两个可能收益：

1. 把 W/U 合成一个 CTA schedule，少一次 dispatch，复用 solved/K/V/beta/decay；
2. 最终直接写 BF16，删除两次 FP32-to-BF16 cast 与 FP32 W/U staging。

不能同时把两者混在一个未经控制的改动里，否则性能变化无法归因。于是设计两个
相同 ownership 的 variant。

#### 26.2 F0：只测试 fusion，仍写 FP32

F0 的图是：

```text
fused W/U FP32
-> W FP32-to-BF16 cast
-> U FP32-to-BF16 cast
-> unchanged BF16 recurrence
```

F0 不改变 solved A dtype、MFMA16 geometry、CTA/WG、recurrence、chunk-o 或 output
contract。它回答的只是：**融合是否本身值得？**

实现要求不是把两个 kernel 文本拼起来。正确 lifetime 顺序是：

```text
shared input staging
-> W accumulator/full W writeback
-> W accumulator 已不再使用
-> U accumulator/full U writeback
```

否则 W/U 两套 accumulator 同时跨长区域活跃，会制造高 AccVGPR、spill 或 occupancy
cliff。F0 以一 CTA 覆盖一个 `(chunk,value_head)` 的方式把 CTA 数从历史 separate
W+U 的 4096 降到 256，但必须拿资源检查证明没有用并行度换来寄存器灾难。

#### 26.3 F1：同 schedule，只把最终存储改为 BF16

F1 唯一相对 F0 的语义边界是 output dtype：

```text
F0: FP32 W/U global store -> numeric cast -> BF16 recurrence input
F1: BF16 W/U global store -> BF16 recurrence input
```

因此 F0/F1 的差异才可以归因于 cast/materialization，而不能归因于不同 tile。
F1 不是把 accumulator 改成 BF16；MFMA 的 accumulator 和中间计算仍按原计划保持 FP32。

#### 26.4 Correctness：先分开证明，再接 full

F0/F1 分别与原 W/U 参考比较 W、U，再比较完整 public output/final state。测试与
driver 为：

- [fused W/U source](../vllm_compare/qwen_gdn_bt64_fused_wu_eager_stage6t.py)
- [correctness test](../vllm_compare/test_qwen_gdn_bt64_fused_wu_eager_stage6t.py)
- [Eager benchmark](../vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py)

不通过 numeric contract 时，禁止用 full output “似乎还行”作为替代证据。

#### 26.5 Eager public 结果

这一表来自当时同批 Eager public measurement，含 warmup、repeat、five sessions 和
wall-clock confirmation；它不是 Graph 或 profiler trace：

| T | Stage 6S | F0 | F1 | vLLM |
|---:|---:|---:|---:|---:|
| 512 | `0.265433 ms` | `0.255137 ms` | `0.249769 ms` | `0.372131 ms` |
| 1024 | `0.292393 ms` | `0.298161 ms` | `0.292753 ms` | `0.372130 ms` |
| 2048 | `0.380383 ms` | `0.394043 ms` | `0.390177 ms` | `0.426070 ms` |
| 4096 | `0.597484 ms` | `0.598586 ms` | `0.588331 ms` | `0.547390 ms` |
| 8192 | `1.084264 ms` | `1.054961 ms` | `1.031687 ms` | `0.793073 ms` |
| 16384 | `2.079838 ms` | `1.999439 ms` | `1.951007 ms` | `1.336598 ms` |

这组绝对数是本阶段的同批观测，不应和 Stage 6U/6W 其他 session 的绝对数混合。它说明：

- F0/T=2048 反而慢，fusion 不自动赢；
- F1 在短文本没有稳定 promotion，长文本 slope 有改善；
- F1 slope `6.886 us/chunk`，Stage6S `7.411`，但仍高于 vLLM `3.956`。

#### 26.6 为什么少 3840 CTA 仍未直接获胜

资源证据：

| W/U path | CTA | WG | VGPR | AccVGPR | LDS | scratch | MFMA | VALU | VMEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical separate W+U | 4096 | 256 | W52/U48 | W4/U8 | N/A | 0 | 524288 | N/A | 851968 |
| F0 | 256 | 256 | 64 | 8 | 3072 B | 0 | 524288 | 4518912 | 491520 |
| F1 | 256 | 256 | 64 | 8 | 3072 B | 0 | 524288 | 4846592 | 491520 |

CTA 与 VMEM 流量下降，但 BF16 epilogue 引入额外 VALU，fusion 还改变了 work per CTA
与 launch scheduling。故不能只用 “CTA 4096 -> 256” 预测 T=2048 一定快。

#### 26.7 Stage 6T 决策

F0/F1 是有效的 source 与 Eager 实验，但不是新 baseline。最需要知道的不是“是否再调
F1”，而是 native vLLM fused W/U 在数值 ABI、MFMA 预算和 ownership 上究竟做了什么。
因此下一步是 Golden audit，不是立即 tile sweep。

详细证据见 [Stage 6T report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_fused_wu_eager_stage6t_report.md)。

---

### 27. Stage 6T-Golden：真实 vLLM fused W/U 的数学、ABI 与 MFMA 预算

#### 27.1 为什么要做 Golden Audit

看到 F1 的 MFMA 或 CTA 仍远高于 vLLM 时，不能凭函数名猜 native kernel 的行为。需要
实际捕获 Triton source、TTIR/TTGIR、HSACO、autotune specialization 与 kernel trace，
核对：

- CTA 对应哪个 logical unit；
- solved A 的 dtype；
- W/U 是否同一 dispatch；
- 是否有 residual；
- tile/warp/stage；
- static/dynamic MFMA、LDS/barrier、资源与输出 layout。

#### 27.2 Golden Audit 的关键发现

native specialization 中：

```text
one CTA = one (BT64 chunk, value head)
native solved A = BF16
W/U = one fused dispatch
no residual coefficient or residual MFMA
```

F1 虽也 fused 且 BF16 output，但它接收的 solved A 是 FP32，并构造 main+residual：

```text
F1: W main 512 + W residual 512 + U main 512 + U residual 512
    = 2048 MFMA/CTA

native vLLM: W 64 + U 64
    = 128 MFMA/CTA
```

这 `16x` 不是“小的 cndmask 或 loop unroll”差异，而是上游 producer-consumer 数值
合同差异。输出 layout 相同不意味着 FP32 solved A 与 BF16 solved A 可以直接替换：
累计顺序、rounding 点和 residual 的数学意义不同。

#### 27.3 为什么不能直接桥接 native W/U

若只把 native vLLM W/U HSACO 接在 FP32 solved A 后，输入 ABI 不对；若无证明地把 FP32
pointer reinterpret 成 BF16，是 silent data corruption。正确路线是先验证：

```text
FP32 solve internal arithmetic
-> final BF16 solved storage
```

是否等价于：

```text
FP32 solve
-> explicit numeric BF16 cast
```

只有这个 producer gate 通过，才能合法消除 residual consumer path。这正是 Stage 6U。

Golden 原始审计见 [Stage 6T-Golden report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_vllm_fused_wu_golden_audit_stage6tg_report.md)。

---

### 28. Stage 6U：BF16 solved boundary，P0 producer 与 C0 consumer

#### 28.1 P0 的最小改变：FP32 compute，不再 FP32 storage

P0 的 solver 仍然：

- 读取 FP32 `a`；
- 用 FP32 LDS `x/work`；
- 使用 FP32 row recurrence；
- 用 `mfma_16x16x4_f32_f32` 算 block products；
- 保持原 4x16 block DAG。

唯一改变是 output pointer/layout 的 element dtype 和 final global writeback：

```python
@avelang.jit
def _qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u(
    a_ptr: al.Pointer(al.f32),
    out_ptr: al.Pointer(al.bf16),
    num_tokens: al.constexpr,
    num_chunks: al.constexpr,
):
    ...
    out[0, chunk_start + diag_block * BLOCK + row, head_idx,
        diag_block * BLOCK + col] = al.convert(x[diag_block, row, col], al.bf16)
```

lower-block writeback 也同样只在 store 前 `al.convert(..., al.bf16)`。因此 P0 不是
FP16/BF16 solve；它是 **FP32 solve + native BF16 terminal storage**。

#### 28.2 P0 correctness gate

P-REF 是原 FP32 solve 后执行 numeric BF16 cast。P0 覆盖 T=64/128/512/1024/2048/8192，
zero、identity-like、high-dynamic、small、cancellation、sparse-lower、NaN prefill、output reuse
与 non-default stream，共 42 producer cases。结果是 BF16 bit-exact：mismatch `0`、max abs `0`。

P1 packed x4 global store 没有实现。当前 lane ownership 不能安全地给每 lane 连续四个元素；
ISA 是 scattered `global_store_short`/`global_store_short_d16_hi`。它是 N/A，不是失败后悄悄
fallback。

#### 28.3 C0：改变 consumer 数值 ABI，删除 residual

C0 接收 BF16 solved A/K/V，输出 BF16 W/U。它不再生成 residual coefficient，不再执行
residual MFMA。P0 的 bit-exact producer gate 使这个改变有数学依据，而不是“看起来 BF16
更快”就删修正项。

| implementation | W main | W residual | U main | U residual | MFMA/CTA | T2048 MFMA |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| C0 | 512 | 0 | 512 | 0 | 1024 | 262144 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

C0 isolated T=64/512/2048 W/U max abs 相对 BF16-coefficient reference 为 `0.0009765625`。

#### 28.4 U0/U1 full graph 与 correctness

U0 是 P0 + numeric boundary path，U1 是 P0 + C0 native BF16 solved path。U1 不 materialize：

```text
FP32 solved A
FP32 W
FP32 U
solved cast
W/U casts
```

U1 full matrix T=64/128/512/1024/2048/8192 均接受；T2048 额外 30 seeds、T8192 10 seeds、
T16384 3 seeds 也接受。U0/U1 output/state bit-exact；public BF16 output max `0.001953125`，
FP32 final state max `0.0152155161`，non-default stream 通过。

#### 28.5 U1 Eager gain、资源和限制

U1 相对 Stage6S 的 paired gain：

| T | U1 gain vs 6S | paired 95% CI |
|---:|---:|---:|
| 1024 | `29.277 us` | `[26.564,31.980] us` |
| 2048 | `122.551 us` | `[108.741,136.236] us` |
| 8192 | `181.001 us` | `[158.452,203.506] us` |
| 16384 | `232.026 us` | `[163.207,278.647] us` |

C0 profile 资源：VGPR68、AccVGPR12、SGPR32、LDS3072 B、scratch0、MFMA262144；F1 的
MFMA524288、VALU4846592、VMEM491520，C0 分别降至 MFMA262144、VALU2087936、VMEM311296。
native vLLM 仍只 32768 MFMA，故 U1 虽成为当时最佳 experimental candidate，长文本仍不能
宣称追平 vLLM。

#### 28.6 Stage 6U 的推理结论

这一步是整个路线的重要转折：

```text
不是先压低 accumulator precision，
而是证明 precision 只需在 global storage boundary 变化；
然后把该事实传播到 consumer，移除重复的 correction schedule。
```

详见 [Stage 6U report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_solved_boundary_stage6u_report.md)。

---

### 29. Stage 6V：四个 predicated MFMA region 到一次 uniform MFMA

#### 29.1 假设

C0 每个 source tile 有四段同构的 `lane_group` region：

```python
if lane_group == 0:
    acc = mfma(fragment0, acc)
if lane_group == 1:
    acc = mfma(fragment1, acc)
if lane_group == 2:
    acc = mfma(fragment2, acc)
if lane_group == 3:
    acc = mfma(fragment3, acc)
```

每 lane 只走一个分支，却不代表 wave 只发一条 MFMA。wave 会以不同 EXEC mask 动态执行
四段区域。假设是：选择每 lane 对应 packed fragment 后，整个 wave 做一次 uniform MFMA，
dynamic MFMA 应从 1024/CTA 接近 256/CTA。

#### 29.2 V0 的写法与首次失败

第一次尝试在 statement-level `if` 中给临时 fragment 赋值。它失败并让所有 lane 使用
default group-0 operand，原因是 Avelang `scf.if` 的分支 scope 不会产生 if 后可用的 SSA result。
这不是硬件错误，也不是 A/B fragment order 错。

正确 V0 用 conditional expressions，使其 lower 为 `arith.select`：

```python
a_operand = a_frag0[0] if lane_group == 0 else (
    a_frag0[1] if lane_group == 1 else (
        a_frag1[0] if lane_group == 2 else a_frag1[1]))
b0_operand = b0_frag0[0] if lane_group == 0 else (...)
b1_operand = b1_frag0[0] if lane_group == 0 else (...)
w_acc0 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b0_operand, w_acc0)
w_acc1 = al.amdgpu.mfma_16x16x16_bf16_f32(a_operand, b1_operand, w_acc1)
```

这个例子对理解 GPU 很重要：**每 lane 的逻辑选择可以先变为 vector select，再避免 wave
级的控制流分裂**；但 select 本身也有成本和 live-range 后果。

#### 29.3 V0 correctness 与 ISA/resource gate

V0 对 C0 不 bit-exact，因为 EXEC/multiplication timing 改变导致少量 BF16 LSB 差异；冻结
gate 是 delta `<=1e-3`，而不是假装 bit-exact。T=64/512/1024/2048 最大 W/U delta 分别不超过
`1.4901e-08`、`4.8828e-04`、`9.7656e-04`、`9.7656e-04`，通过。

| metric | C0 predicated | V0 collapse |
|---|---:|---:|
| static MFMA16 | 64 | 16 |
| static `v_cndmask_b32` | 56 | 200 |
| static branch | 38 | 6 |
| profiler VGPR | 68 | 100 |
| AccVGPR | 12 | 12 |
| dynamic MFMA | 262144 | 65536 |
| dynamic MFMA/CTA | 1024 | 256 |
| scratch/spill | 0/0 | 0/0 |
| isolated T2048 body | 0.073129 ms | 0.071827 ms |

静态/动态 MFMA 都精确达到目标，但 cndmask 和 VGPR 增长抵消大部分 gain；trace 也有 run-to-run
扰动，所以只用它确认 instruction gate，不做性能晋级。

#### 29.4 V1 full integration

V1 图严格为：

```text
P0 BF16 solved -> V0 predicate-collapse fused W/U
-> unchanged BF16 recurrence -> unchanged chunk-o
```

没有增加 dispatch 或 fallback。full correctness 覆盖 T=64/512/2048/8192/16384；output
`<=1/128`、state `<=0.02`。Eager 结果：

| T | U1 | V1 | V1-U1 | V1/vLLM |
|---:|---:|---:|---:|---:|
| 512 | 0.249551 | 0.248410 | -1.141 us | 0.6931x |
| 2048 | 0.364962 | 0.361397 | -3.565 us | 0.8678x |
| 8192 | 0.969402 | 0.968219 | -1.182 us | 1.2467x |
| 16384 | 1.834768 | 1.829000 | -5.768 us | 1.3811x |

九 session T2048 high-repeat 的 paired bootstrap CI 为 `[-2.003,43.375] us`，跨零；
slope 只从 `6.460055` 到 `6.448760 us/chunk`。结论：V0 通过 lowering probe，V1 不通过
promotion。不要为了微秒级中心趋势继续 source tuning。

完整记录见 [Stage 6V report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_predicate_collapse_stage6v_report.md)。

---

### 30. Stage 6W：chunk-o 直接消费 BF16 V-new，直接写 BF16 public output

#### 30.1 问题：两次明显却未被消除的 storage boundary

Stage 6S/U1 recurrence 已产生 BF16 `v_new`，但旧 chunk-o 图仍将它扩到 FP32；chunk-o
内部又为 BF16 MFMA operand 构造 BF16 LDS tile。输出同样先写 FP32 staging，再单独 cast
为 BF16 public output：

```text
BF16 recurrence V-new
-> BF16-to-FP32 cast dispatch
-> FP32 V-new global staging
-> chunk-o FP32 global load -> BF16 LDS/MFMA operand
-> FP32 output staging
-> FP32-to-BF16 cast dispatch
```

这两个 cast 并不大，但它们是确定的 dispatch、allocation、global write/read 和 dtype
rounding boundary。W1 只改这个变量，不改 recurrence HSACO、KKT、solve、W/U、chunk-o
MFMA geometry/LDS tile、compiler 或 default selector。

#### 30.2 真实 kernel 改动

旧 chunk-o 入口为 FP32 `vn/out`。W1 的 tensor 合同改为：

```python
vn = al.make_tensor(vn_ptr, al.bf16, ...)
out = al.make_tensor(out_ptr, al.bf16, ...)
```

在 intra loop：

```python
# 原路径会读取 FP32，再马上 convert 到 BF16 LDS tile
# W1 直接读 recurrence 的 BF16 V-new
vn_t_bf16[value_offset, source_offset] = vn[
    0, chunk_start + source_base + source_offset,
    value_head_idx, value_base + value_offset
]
```

在 public writeback：

```python
result = inter_acc[r_out] * al.exp(g[0, token_idx_out, value_head_idx]) + intra_acc[r_out]
out[0, token_idx_out, value_head_idx, value_base + lane_col] = al.convert(result, al.bf16)
```

因此 accumulator 仍 FP32；改变的是 global input/output storage。wrapper 严查 `v_new/h/out`
为 contiguous BF16、BT64、同设备；任何不匹配均 `ValueError`，无 fallback。

#### 30.3 正确性：为什么能 bit-exact

W1 的 direct BF16 load 与旧路径：

```text
BF16 -> FP32 -> BF16 LDS
```

在数值上使用同一 BF16 value。writeback 的 `FP32 result -> BF16` 也与旧 output staging 后
cast 具有同一 terminal rounding。测试结果：

| 层次 | 覆盖 | 结果 |
|---|---|---|
| isolated chunk-o | T=64/512/2048 | int16 mismatch=0，max/mean abs=0 |
| full vs U1 | output/state | bit-exact |
| full vs native vLLM | 64..8192、state/zero/high dynamic/cancellation | output <=1/128，state <=0.02 |
| guards | FP32 v_new、不连续、非法 T | 正确抛错，无 fallback |

#### 30.4 ISA 与资源：证明“身体”没有被偷换

| 位置 | U1 chunk-o | W1 chunk-o |
|---|---|---|
| V-new stride | `s_lshl_b64 ..., 2` | `s_lshl_b64 ..., 1` |
| V-new load | `global_load_dword` | `global_load_ushort` |
| output store | `global_store_dword` | `global_store_short_d16_hi` |
| MFMA mnemonic | `v_mfma_f32_16x16x16_bf16` | 相同 |

T2048 rocprof 的 WG256、grid work-items 524288、VGPR112、AccVGPR64、SGPR112、
LDS27136 B、scratch0、MFMA458752、VALU12918784、VMEM851968、LDS1343488 都相同；
SALU 仅少 45056。VMEM 指令数相同不等于传输字节相同，BF16 load/store 仍减少了每元素
字节宽度。

PMC rocprof 却给 W1 报 occupancy `0.81%`、trace约1498 us，而 current 为 `17.75%`、
66.86 us；这与 zero scratch、相同资源/动态 count、无 profiler body 约0.09 ms矛盾。
它被记录为 collector perturbation 反例，绝不拿来否定或解释 W1 性能。

#### 30.5 private body 变慢为何不否定 full 图收益

预分配 body 只包 chunk-o kernel，不包括删掉的 casts：

| T | U1 current body | W1 body | W1 变化 |
|---:|---:|---:|---:|
| 512 | 0.036154 | 0.041822 | 慢 |
| 2048 | 0.086869 | 0.091136 | 慢 4.267 us |
| 8192 | 0.250192 | 0.251434 | 慢 |
| 16384 | 0.468576 | 0.473244 | 慢 |

这精确回答了一个常见误区：W1 的收益不是“BF16 MFMA 更快”，而是删除整个 public graph
里的两个 boundary dispatch/staging。用 body 反驳 full 改善是口径错误。

#### 30.6 初始 Eager sweep：正向候选，不足以晋级

5 session Eager sweep 的中心趋势：

| T | U1 | W1 | W1 gain | W1/vLLM |
|---:|---:|---:|---:|---:|
| 512 | 0.250191 | 0.223572 | 26.619 us | 0.6190x |
| 1024 | 0.275829 | 0.267397 | 8.432 us | 0.7308x |
| 2048 | 0.361818 | 0.351862 | 9.956 us | 0.8581x |
| 4096 | 0.539100 | 0.523217 | 15.884 us | 0.9801x |
| 8192 | 0.968759 | 0.936471 | 32.289 us | 1.2119x |
| 16384 | 1.829537 | 1.768947 | 60.590 us | 1.3401x |

W1 slope `6.252842 us/chunk`，相对 U1 `6.439639` 回收 `0.186797 us/chunk`，仍高于 vLLM
`3.926166`。9-session confirmation 的中心趋势一致，但 call-level bootstrap 仍跨零；
正确状态只能是 opt-in candidate，不能改 baseline。这使后续 clustered confirmation 必要。

完整实现和报告：[Stage 6W boundary report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_chunko_boundary_stage6w_report.md)。

---

### 31. Stage 6W clustered confirmation：从历史污染样本到当前成对共享环境基线

#### 31.1 为什么小样本 Eager 不够

W1 的预期收益只有数到数十 us，短 kernel 很容易被 queue、clock、外部 GPU context 或
order 影响。必须记录每个 call、block、session，而不是只报一个 global median。历史严格
设计使用 12 process-isolated session、每 session warmup Williams blocks 后 6 个计时
block；一个 block 覆盖六种三实现 Williams order：

```text
U1 W1 vLLM   W1 vLLM U1   vLLM U1 W1
U1 vLLM W1   vLLM W1 U1   W1 U1 vLLM
```

这保证每个实现以相同次数出现在每个位置、相邻于每个其他实现。它保存 event、wall、
session/block/order/position/timestamp、GPU clock/temperature/utilization 与 process snapshot。

#### 31.2 历史 12-session run 为什么不能判定 W1

历史 T2048/T8192 三种 CI（block、session mean、nested cluster）均跨零；但原始 event
出现异常长尾：T2048 U1/W1/vLLM 的 min 约 `0.335/0.333/0.312 ms`，max 却到
`445.222/390.990/1276.906 ms`。GPU context 审计有外部 context，`evicted_time`
约 `90,332 ms`。

因此正确解释是：

```text
该样本被环境长尾污染，不能确认微秒收益；
它也不能否定 W1 的正向候选。
```

把“CI 跨零”写成“W1 一定无收益”是统计错误；把它写成“W1 已稳定收益”同样错误。

#### 31.3 新鲜 paired shared-environment retest

随后重新设计为显式 `--allow-shared-gpu` 的严格成对相对排名：每 T 8 个 process-isolated
session、每 session 8 个完整 Williams block，共 64 paired block/1152 public calls。
primary statistic 是 **每 session 的 paired median HIP-event gain**，block 与 nested
cluster 是 sensitivity；每次同一 API call 周围还记录 wall-clock，要求 direction 一致。

该范围的理由是：外部负载对 U1/W1/vLLM 都是共同条件，随机交错与 paired difference
能可靠回答“在这个共享环境中 W1 是否相对 U1 更快”；它不能回答“独占 GPU 上的绝对 latency
是多少”。

| T | U1 session median mean | W1 session median mean | W1-U1 gain | HIP primary CI | wall CI | nested CI |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 0.358282 ms | 0.348283 ms | 8.994 us | [7.246,10.545] | [7.138,10.517] | [5.671,10.819] |
| 8192 | 0.966426 ms | 0.939173 ms | 27.352 us | [25.398,29.163] | [26.100,29.373] | [26.015,30.168] |

两个 T 下 HIP/wall 都有 8/8 positive session paired medians。机器可读 decision 也写明：

```json
{
  "timing_contract": "eager_public_api",
  "cuda_graph_used": false,
  "promotion_scope": "paired_shared_environment",
  "promotion_gate": true
}
```

#### 31.4 W1 与 vLLM：按倍数表述，不能只看绝对微秒

同一批：

| T | W1 | vLLM | 关系 |
|---:|---:|---:|---:|
| 2048 | 0.348283 ms | 0.404651 ms | W1 是 `0.8607x` vLLM，快 `56.685 us` |
| 8192 | 0.939173 ms | 0.766289 ms | W1 是 `1.2259x` vLLM，慢 `172.933 us` |

这就是准确的当前结论：短/中 T 在该同批共享环境可领先，长文本仍有明显 slope gap；
不是“vLLM 产生两个真性能”，而是不同 harness/session/specialization/GPU 环境测到的
不同样本。只有同一公开 API contract 的 paired comparison 才能做该表中的倍数结论。

#### 31.5 当前 W1 status

```text
W1 = current Avelang experimental baseline
scope = paired_shared_environment
production/default = unchanged
exclusive GPU absolute confirmation = 尚未获得
```

完整统计、历史污染说明和图 accounting 见
[Stage 6W clustered confirmation report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md)。

---

### 32. Stage 6W 后的完整 intermediate accounting：下一步为什么是 KKT -> solve

W1 已删除两条尾部 FP32 intermediate：

```text
v_new_fp32 staging
output_fp32 staging
```

更新图的所有剩余 global materialization：

| tensor | dtype | T2048 / T8192 | producer -> consumer | single use | write+read traffic |
|---|---|---:|---|---|---:|
| g_cumsum | FP32 | 0.0625 / 0.25 MiB | cumsum -> KKT,W/U,recurrence,chunk-o | no, four consumers | 0.3125 / 1.25 MiB |
| a | FP32 | 4 / 16 MiB | KKT -> solve | yes | 8 / 32 MiB |
| a_solved_bf16 | BF16 | 2 / 8 MiB | solve -> fused W/U | yes | 4 / 16 MiB |
| w_bf16 | BF16 | 4 / 16 MiB | fused W/U -> recurrence | yes | 8 / 32 MiB |
| u_bf16 | BF16 | 4 / 16 MiB | fused W/U -> recurrence | yes | 8 / 32 MiB |
| h_bf16 | BF16 | 8 / 32 MiB | recurrence -> chunk-o | yes | 16 / 64 MiB |
| v_new_bf16 | BF16 | 4 / 16 MiB | recurrence -> chunk-o | yes | 8 / 32 MiB |
| final_state | FP32 | 0.5 / 0.5 MiB | recurrence -> public optional output | public | n/a |
| output_bf16 | BF16 | 4 / 16 MiB | chunk-o -> public output | public | n/a |

不能只选最大流量：`h/w/u/v_new` 跨 immutable recurrence ABI，`g_cumsum` 有四个
consumer，均不适合立即改。`a` 是 KKT/solve 两端都是 Avelang source、single-use，且
T2048 8 MiB/T8192 32 MiB roundtrip。因此唯一登记候选是：

```text
KKT FP32 a -> solve 的 producer-consumer handoff
```

它不是“直接把整个 a 放 private memory”，更不是加一个 BF16 cast。后续必须先审计：

- KKT 的 16x16 CTA tiles 与 solve 的 4x16 block DAG 如何对应；
- 是否能在合理 LDS/CTA 范围内交接小 tile；
- 是否会扩大 KKT/solve live region、减少 occupancy 或增加 spill；
- 是否能保持既有 FP32 solve math 和 full correctness。

在这之前它只能称为下一候选，绝不称已经实现。

---

### 33. 完整实验账本：从 Stage 4 到 W1

| Stage | 问题 | 单变量改变 | correctness | Eager/正式结论 | 资源诊断 | 决策 |
|---|---|---|---|---|---|---|
| 4 | scalar fallback | KKT/WU/chunk-o MFMA ownership | 37 full cases | historical T2048 0.475867 ms | scratch/spill0 | 保留历史 BT64 图 |
| 5A | solve gap | audit only | 54 authority cases | body v18/vLLM 3.44x | 63-row vs block DAG | 做 5B |
| 5B | sequential solve | 4x16 FP32 block DAG | 7 tests | solve T2048 4.15x | 8KiB LDS, scratch0 | 可接 full |
| 5C | full propagation | only replace solve | pass | public gain约18.808us | stage event 有扰动 | 做 5D |
| 5D | downstream penalty | canonical/pointer/warm controls | pass | 非 leaderboard | tail penalty约64us | direct common out |
| 5E | pointer root cause | exact same output pointer | 53 tests | 非 leaderboard | pointer 不是主因 | 5F low perturb |
| 5F | transient mechanism | measurement only | N/A | profiler gate fail | trace扰动 | stop branch |
| 6A | structural graph gap | audit only | source/graph captured | graph diagnostic | 8 vs 7 dispatch | 6B |
| 6B | chunk-o ownership | O0/O1 | pass | no promotion | AccVGPR/LDS increase | close branch |
| 6R | recurrence identity | HSACO/ABI audit | bridge check pass | current BF16 body faster | WG/ABI changed | 6S |
| 6S | current recurrence full contract | 3 explicit casts + bridge | 12 full cases | opt-in Eager candidate | 11 vs 7 dispatch | propagate BF16 |
| 6T | W/U fusion | F0 FP32, F1 BF16 | pass | slope improves; short gain unstable | BF16 VALU rises | Golden audit |
| 6T-Golden | native W/U facts | audit only | ABI audit pass | N/A | 2048 vs128 MFMA/CTA | 6U |
| 6U | solved boundary | P0 BF16 store+C0 no residual | full matrix/bit-exact | U1 gains vs6S | 1024 MFMA/CTA | U1 baseline |
| 6V | predicate collapse | select+one MFMA | frozen error pass | V1 CI crosses zero | MFMA down, VGPR up | do not promote |
| 6W | tail boundaries | BF16 V-new+BF16 output | bit-exact vs U1 | initial candidate | body slower; graph fewer casts | strict confirmation |
| 6W confirm | W1 promotion | no kernel change, paired protocol | existing W1 contract | W1>U1 at 2048/8192 in scope | shared env caveat | W1 experimental baseline |

机器可读版本见 [timeline CSV](report_final_stage_timeline.csv)。

---

### 34. 失败实验、N/A 与环境限制如何记录

#### 34.1 数学/正确性失败

- Stage4 no-correction：错误删除 residual；不可以通过放松 tolerance 解决。
- Stage5B initial diagonal identity order：单位阵太早加入会重复第一条 sub-diagonal；修正计算顺序。
- Stage5B X42 offset：只含 A43 的 identity-diagonal case 定位为 output offset bug，而非 MFMA mapping。
- Stage6V statement `if`：SSA/scope 失败；要改为 conditional expression/select。

#### 34.2 性能/资源失败

- accumulator merge：缩短源级变量数不保证降 live range；
- O0/O1：少 CTA/少 score 重复，却增加 AccVGPR/LDS，occupancy 下降；
- F0/F1：少 dispatch/CTA，却可能引入 fusion lifetime 与 BF16 epilogue VALU；
- V1：MFMA 4x 少，却被 cndmask/VGPR 与全图其他成本抵消；
- W1 body：BF16 boundary body 不是更快的 MFMA body，但 full 仍获益。

#### 34.3 环境/工具 N/A

- `P1 packed solve writeback`：没有安全 x4 lane packing，N/A；
- U2：没有实施，不应被写成 predicate-collapse 之后的版本；
- Stage5F cache/clock：profiler 低扰动 gate 失败，N/A；
- historical v29 exact artifact：报告存在但当前 worktree 无原始 artifact directory，证据索引注明 availability note；
- shared GPU：绝对延迟确认受外部 context 限制，但 paired shared ranking 的结论范围明确保留。

**N/A 既不是 pass 也不是 fail。** 它指出当前问题尚没有可信测量或安全实现，不允许拿旧数据补空格。

---

### 35. 当前状态、严谨表述和下一步

#### 35.1 可以说什么

1. Stage 4 消除了 BT64 上游/下游 scalar fallback；
2. Stage 5 将 solve 根因定位为依赖图，并完成 FP32 MFMA16x4 hierarchical solve；
3. current-vLLM recurrence 与旧 asm-v0 是不同 specialization；
4. BF16 solved boundary 使 U1 大幅减少 residual 工作和 boundary dispatch；
5. predicate collapse 精确降了 MFMA，但没有 full promotion；
6. W1 删除尾部两个 materialized boundary，在严格 paired shared environment 下稳定快于 U1，因而是当前 Avelang experimental baseline；
7. W1 在同批 T2048 快于 vLLM，T8192 仍慢于 vLLM。

#### 35.2 不能说什么

1. 不能说 W1 已经是 production/default；
2. 不能说 W1 全序列超过 vLLM；
3. 不能用 Graph replay/body/trace 代替 Eager leaderboard；
4. 不能把 old asm-v0 叫 current vLLM kernel；
5. 不能把 `a` handoff fusion 写成已经完成；
6. 不能把 historical polluted CI 说成 W1 无收益；
7. 不能从没有 scratch 推出没有 AGPR/VGPR pressure。

#### 35.3 唯一下一候选

在不动 recurrence HSACO、v24、default selector、compiler/RA 或多个 kernel 的前提下：

```text
先做 KKT FP32 a -> solve producer-consumer handoff 的可行性审计，
再决定是否实现一个单变量实验。
```

标准仍不变：correctness 先行；只改一类边界；Eager public API paired result 决定是否晋级；
ISA/rocprof 只解释，不替代性能；发现资源 cliff 或测量不可信时停止。

---

### 36. 文档、证据与复现入口

- [原报告备份](report_final_before_stage6w_update.md)
- [修订日志](report_final_revision_log.md)
- [事实冲突与更正](report_final_fact_corrections.md)
- [阶段时间线 CSV](report_final_stage_timeline.csv)
- [证据索引 JSON](report_final_evidence_index.json)
- [源码修改索引](report_final_source_change_index.md)
- [链接检查记录](report_final_link_validation.md)
- [Stage 4 原始报告](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_nonrecurrence_stage4_report.md)
- [Stage 5A solve audit](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_solve_rootcause_stage5a_report.md)
- [Stage 5F closure](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_transient_state_stage5f_report.md)
- [Stage 6S report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_recurrence_full_contract_stage6s_report.md)
- [Stage 6T report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_fused_wu_eager_stage6t_report.md)
- [Stage 6U report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_solved_boundary_stage6u_report.md)
- [Stage 6V report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_predicate_collapse_stage6v_report.md)
- [Stage 6W report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_chunko_boundary_stage6w_report.md)
- [Stage 6W confirmation](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md)

这份报告的中心方法可以压缩为：

```text
问题
-> 证据
-> 可证伪假设
-> 单变量最小实验
-> correctness
-> paired Eager full
-> ISA/资源诊断
-> 保留、停止或唯一下一步
```

这比反复根据单个 profiler 数字“猜一个更快 kernel”慢一点，但会让每次失败都变成
下一次选择的可靠输入。

---

## Part III：跨阶段实验索引与第一次写算子的复盘模板

### 37. 主要成功路径索引

| 实验 | 为什么做 | 成功证据 | 后续影响 |
|---|---|---|---|
| FullOp 去除 `Pure` | sequential MFMA 出现 silent corruption | LLVM IR 恢复独立 `zeroinitializer`，regression 通过 | 先保证 accumulator 语义正确 |
| v14 MFMA W/U | W/U scalar 成为大头 | 约 `0.96 -> 0.082 ms` | 瓶颈转移到 recurrence |
| v16 2-wave | `chunk_gdr` 串行 ownership 过重 | 约 `1.36 -> 0.56 ms` | 证明 cooperative wave 有效 |
| v17 4-wave + predecay | wave0 仍是热点 | 约 `0.56 -> 0.41 ms` | 边际递减，开始优化其他 stage |
| v18 parallel solve | BT32/64 solve 爆炸 | BT64 `21.1 -> 0.127 ms` | 允许 larger-chunk full 实验 |
| v23 distributed v_new/v_decay | v22 ablation 定位 wave0-only 瓶颈 | full 约 `0.694 -> 0.637 ms` | 为 v24 production 奠基 |
| v24 native BT16 KKT | KKT 成为剩余可见成本 | full 约 `0.636 -> 0.585 ms` | production/default baseline |
| persistent K-frag rewrite（isolated） | broad K graph 制造高 AGPR | isolated `1.847x`，AccVGPR `264 -> 180` | 证明 lowering 根因在 isolated 环境成立 |
| Stage 4 KKT/WU/chunk-o | BT64 非 recurrence 仍有 scalar/低复用 | T2048 full `1.089079 -> 0.475867 ms` | 建立稳定 BT64 experimental 图 |
| Stage 5B hierarchical solve | row-sequential DAG 是 solve 根因 | solve T2048 约 `4.15x` | 可接回 full |
| Stage 6R/6S bridge | old asm-v0 不是 current vLLM recurrence | ABI/HSACO 桥接 correctness 通过 | 开始沿真实 BF16 specialization 优化 |
| Stage 6U BF16 solved boundary | FP32 solved storage/residual 是边界成本 | U1 correctness 与 Eager gain | U1 成为上一 experimental baseline |
| Stage 6W tail BF16 boundaries | `v_new_fp32` 与 `output_fp32` 为多余 materialization | paired Eager 中 W1 稳定快于 U1 | W1 成为当前 experimental baseline |

### 38. 主要失败或关闭路径索引

| 路线 | 初始理由 | 失败/关闭证据 | 学到什么 |
|---|---|---|---|
| v19 BT32 full | 减少 chunk 数 | recurrence 小降但 KKT/WU/chunk-o 增长，full 更慢 | 必须看 full，不可只看主 kernel |
| v20 BT32 W/U MFMA | 补回 BT32 非 recurrence 成本 | KKT 成功但 W/U 更慢 | 相同 MFMA 思路不能机械复制 |
| v21 同步 double buffer | 隐藏 load latency | 无真正 async，VMEM 不降，资源与 trace 变差 | buffering 需要真实 overlap |
| v25 BT64/BV32 | 模仿大 tile | per-chunk 与资源压力过高 | 参数变大不等于 Triton-like schedule |
| v29 fused full | MFMA32 + 少 chunk + 去 v_new materialization | 性能慢、nonzero-w recurrence 巨大误差 | pred-only 成功不等于 full 成功 |
| source K-subtile full | isolated L6 明显降压 | full `0.834 -> 2.281 ms`，AccVGPR 384、scratch | isolated 不能直接外推 |
| `end_lifetime` marker | 告诉 RA 值已结束 | marker 在 backend 前被 erase，资源不变 | 优化提示必须后端可见 |
| read-side helper | 替换 K consumer 表达式 | lowering 后与 baseline 同构 | 必须重写完整 producer-consumer graph |
| persistent rewrite full | isolated 编译器修复有效 | full AccVGPR `264 -> 384`、scratch 736 B | full live-set 跨过 RA 阈值 |
| late B-fragment load | 直接生成 LDS 8-byte load | VGPR略降但 AccVGPR升、latency变慢 | 单个 load 不是峰值根因 |
| hard shared phase boundary | 切断 pred/update SSA | 资源完全不变，仅 barrier 变慢 | 峰值不是单根 SSA edge |
| pred/update 两 kernel | 降低单 kernel live set | recurrence 顺序、launch 与 global handoff 不利 | 拆分必须尊重算法依赖 |
| Stage 4 no-correction | 删除 residual 提速 | full correctness 失败 | 数值补偿不能随意删 |
| chunk-o accumulator merge | 减少 accumulator 数量 | 误差超过 contract | 改变累加顺序会改变舍入路径 |
| Stage 5F profiler 深挖 | 解释 transient-state penalty | profiler 低扰动 gate 失败 | 不可信的测量应标 N/A 并停止 |
| Stage 6B O0/O1 | 少 CTA、少 score 重复 | AccVGPR/LDS 上升，occupancy 下降 | 工作量少不等于 latency 少 |
| Stage 6V predicate collapse | 四段 MFMA 合成一次 uniform MFMA | MFMA 降但 full CI 跨零 | 指令数下降可能被 select/VGPR 抵消 |

### 39. 第一次进行算子优化时可直接使用的实验记录模板

```markdown
## 实验名称

### 1. 当前现象
- full public API 延迟：
- 最慢 stage：
- correctness 状态：
- 资源状态：VGPR / AccVGPR / LDS / scratch

### 2. 证据
- profiler / event / source / ISA 显示什么：
- 哪些数据仅为诊断，哪些属于正式口径：

### 3. 可证伪假设
- 我认为瓶颈来自：
- 如果假设正确，应观察到：
- 如果假设错误，应观察到：

### 4. 单变量改动
- 只修改：
- 明确保留不变：数学、layout、MFMA 数量、ABI 或其他：

### 5. Correctness gate
- 单 kernel 对比：
- full pipeline 对比：
- tolerance / bit-exact：
- 特殊用例：nonzero state、high dynamic range、cancellation、长序列：

### 6. 性能结果
- Eager public API paired result：
- 置信区间：
- standalone/body 结果（仅诊断）：

### 7. 资源与 ISA 解释
- VGPR / AccVGPR / LDS / scratch：
- MFMA / VALU / VMEM / barrier：
- occupancy 或 resource cliff：

### 8. 决策
- 晋级 / 保留为证据 / 放弃 / N/A：
- 原因：

### 9. 下一步
- 下一候选由哪条证据推出：
- 下一实验只改变哪个变量：
- 停止条件：
```

### 40. 这条优化路线最值得学习的十条原则

1. correctness 是性能实验的前置条件，而不是最后补的测试；
2. 优化目标必须来自测量，不来自“某种技术应该更快”的直觉；
3. 每轮尽量只改变一个主要变量；
4. isolated、full graph 与 public API 是三个不同层级；
5. 少 CTA、少 MFMA、少 load 都不自动等于更低 latency；
6. 资源峰值由同时存活的所有值决定，而不是单个局部代码块决定；
7. BF16 storage 可以省流量，但必须验证 recurrence 误差传播；
8. 编译器优化必须确认最终 MIR/ISA graph 真正变化；
9. 失败实验要记录“它排除了什么”，不能只写“变慢了”；
10. 下一步应是唯一被当前证据支持、且能用单变量实验验证的候选。

## 41. 当前最终状态

```text
production/default：v24 BT16
当前 Avelang experimental baseline：W1 / Stage 6W
权威排名范围：paired shared-environment 的完整 Eager public API
不能声称：所有长度、所有环境全面超过 vLLM
唯一登记的下一候选：KKT FP32 a -> solve producer-consumer handoff 可行性审计
```

整条路线可以概括为：

```text
正确性
-> 测量瓶颈
-> 可证伪假设
-> 单变量实现
-> 单 kernel correctness
-> full correctness
-> paired Eager public API
-> ISA/资源解释
-> 晋级、停止或推导唯一下一步
```

## 42. 2026-08-18 性能提交记忆：X2+Z5B

本节覆盖第 41 节之后的 X2 和 Stage 6Z 进展，并取代其中“W1 是当前
experimental baseline”的历史状态描述。

### 42.1 当前版本结论

截至 **2026-08-18**，固定 shape `gfx942, B=1, Hk=4, Hv=8, K=V=128,
BT=64, T%64=0` 下，当前最快的已验证完整 Avelang Qwen GDN 性能候选为：

```text
X2+Z5B
= cumsum
  -> X2 CTA-local KKT+solve
  -> Stage 6U fused BF16 W/U
  -> hash-guarded current-vLLM BF16 recurrence HSACO bridge
  -> Stage 6Z Z5B direct-Q-cache BF16 chunk-o
```

入口是：

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py`

这不是把 isolated 数字拼在一起：Z5B 已在完整 Eager public API 中只替换 X2 的
最后一个 chunk-o dispatch，并完成 47 项 full correctness gate。它相对 X2 的
`final_state` 保持 FP32 bit-exact；BF16 public output 因两种合法 chunk-o 的
MFMA/LDS 累加路径不同而非 bit-exact，但最大差异 `0.0009765625`，低于
`1/128` public-output contract，且直接 vLLM contract 均通过。

### 42.2 性能提交选择

没有一个不分长度的版本在所有 T 都最优：

| 目标 | 应提交/选择的完整版本 | 原因 |
|:--|:--|:--|
| 单一性能提交，主测 T>=1024 | **X2+Z5B** | 在已测 `1024/2048/4096/8192/16384` 都稳定快于 X2。 |
| 只测 T=512 | X2 | Z5B 的 32 KiB Q-cache 有固定成本，T=512 慢约 8.9 us。 |
| 允许长度 selector | T=512 走 X2；T>=1024 走 X2+Z5B | 这是现有实测点的最佳选择；本阶段未修改 production selector。 |
| production/default | v24 BT16 | X2+Z5B 仍是 experimental：依赖 hash-guarded current-vLLM recurrence HSACO，且未完成 production dispatch policy。 |

T=512--16384 的本轮 Eager public-API 数值以及 raw session/block evidence 在
`qwen_gfx942_bt64_x2_z5b_chunko_full_integration_report.md` 和
`codex_qwen_bt64_x2_z5b_chunko_full_eager/`。X2+Z5B 的拟合 slope 为
`4.839 us/chunk`，原 X2 为 `5.620 us/chunk`，本轮 native vLLM 为
`3.630 us/chunk`。

### 42.3 Fork 提交边界

个人 fork 应以
`Avelang_Qwen_GDN_gfx942_X2_Z5B_2026-08-18_提交清单.md` 中的 source、external
artifact 和四文件 compiler patch 为准。不要把工作树中后续 persistent-recurrence、
block-dot V2、`lower_qwen_*`、`qwen_*pipeline*` 等研究性 pass 当成 X2+Z5B 的
依赖一起提交；它们没有参与当前完整图的执行或性能结论。

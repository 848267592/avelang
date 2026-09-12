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
# Qwen v29 Full K-Fragment Rewrite: Final Root Cause and Fix Audit

## Executive Summary

The full-v29 regression is a **mixed register-allocation cliff**: the opt-in
producer-consumer rewrite replaces broad K staging with a compact tile, but
its generic dynamic vector-load lowering creates nearly equal amounts of
long-lived address arithmetic and packed fragment copy/extract pressure while
the real MFMA32 pred, state, v-decay, and MFMA16 update regions coexist.

The exact post-greedy MIR classifies all `190` spilled VGPR words:

| machine-level family | spill words | share |
|:--|--:|--:|
| scalar/vector address formation | 96 | 50.5% |
| fragment `REG_SEQUENCE`/`COPY` packing | 94 | 49.5% |
| unclassified | 0 | 0.0% |

There is no defensible single-family fix. The existing narrow compiler
implementation, `qwen-kfrag-producer-consumer-rewrite`, is semantically
correct and helps the isolated L6 case, but is a negative full-v29 result.
The one evidence-guided pred lane-major serialization experiment reduces the
full rewrite from `190` to `151` spill words and scratch from `736 B` to
`600 B`, while remaining bit-exact relative to the rewrite. It does **not**
change `Accum_VGPR_Count=384` or the high-AGPR range, and is still slower than
original v29. Therefore it is kept only as an experiment; v24 remains the
production baseline.

## Scope and Safety

- No v23/v24/v26/v27/v28 production implementation was modified.
- The compiler change is opt-in and guarded by the existing fixed-layout
  Qwen K-fragment match. Default compilation is unchanged.
- The known original-v29 nonzero-W recurrence-vs-reference failure remains
  unresolved. All equivalence claims below are only against original v29 or
  the full K-fragment rewrite as stated.

## A/B/C Full-Kernel Matrix

All three variants use the same fixed BT64/BV32, 128-thread geometry.

| variant | role | T=2048 normal median ms | relation |
|:--|:--|--:|:--|
| A: original v29 | unrewritten source baseline | `0.836063` | baseline |
| B: full K-fragment rewrite | existing opt-in compiler rewrite | `1.338250` | regression |
| C: B + lane-major pred serialization | experimental live-range repair | `1.271430` | `1.0526x` faster than B, still slower than A |

The latest A/B/C run used warmup `5`, repeat `20`. Shorter sequence timing was
noisy on the shared device, but T=2048 is stable enough for the decision:

| T | A ms | B ms | C ms | C vs B |
|--:|--:|--:|--:|--:|
| 512 | `0.634663` | `0.639670` | `0.642756` | `0.9952x` |
| 1024 | `0.647583` | `1.312992` | `1.320884` | `0.9940x` |
| 2048 | `0.836063` | `1.338250` | `1.271430` | `1.0526x` |

`test_qwen_gdn_v29_kfrag_pred_streaming_exp.py` passed at `T=64,512,1024,2048`:
C's `h` and final state are bit-exact to B (`max_abs=0`). B was previously
shown bit-exact to original v29 at the same interface.

## Exact Spill Attribution

Artifacts generated by
[`analyze_qwen_v29_spill_vregs.py`](analyze_qwen_v29_spill_vregs.py):

- `codex_final_audit/rewrite_spills/spill_vregs.json`
- `codex_final_audit/rewrite_spills/spill_vregs.csv`
- `codex_final_audit/rewrite_spills/spill_summary.json`
- matching `rewrite_streaming_spills/` files for C.

The script reads each `SI_SPILL_AV32_SAVE` and `SI_SPILL_AV64_SAVE`, preserves
the defining instruction, all retained uses, block/line locations, register
class, spill width, stack object, and machine-only classification. ROCm full
LTO stripped source debug locations, so the script deliberately leaves LLVM
and Avelang producer fields unavailable rather than inventing a mapping.

### B: Full Rewrite

| item | result |
|:--|--:|
| AV32 saves/restores | `70 / 70` |
| AV64 saves/restores | `60 / 60` |
| spill words | `70 + 2*60 = 190` |
| frame objects referenced by saves | `130` |
| sum of object extents | `760 B` |
| final private frame | `736 B` |
| final code-object VGPR spills | `190` |

The `760 B` object-extent sum is not expected to equal the final `736 B`
private frame. The post-greedy dump has no final colored offsets; later frame
coloring/reuse produces the code-object footprint. Likewise, `190 * 4` is not
the private frame calculation. It is the profiler/code-object VGPR spill-word
count.

The defining opcode families identify a mixed cluster:

- `V_LSHL_ADD_*`, shifts, add/or/and arithmetic: address tuple and dynamic
  indexing pressure from generic vector/memref lowering;
- `REG_SEQUENCE`, subregister extraction, and `COPY`: packed BF16 fragment
  assembly/copy pressure around update consumption.

This explains why L6 transfers positively while full v29 does not. L6 has the
same compact K producer/consumer idea but not the complete 32-window loop
with live MFMA32 pred accumulators, pred correction, state/v-decay staging,
and update recurrence. The full composition crosses greedy RA's resource
threshold; it is not an MFMA-count problem.

### C: Rewrite + Pred Serialization

| item | B | C | change |
|:--|--:|--:|--:|
| AV32 saves | 70 | 63 | -7 |
| AV64 saves | 60 | 44 | -16 |
| spill words | 190 | 151 | -39 (20.5%) |
| address spill words | 96 | 79 | -17 |
| fragment-copy spill words | 94 | 72 | -22 |
| private frame | 736 B | 600 B | -136 B |
| final `vgpr_spill_count` | 190 | 151 | -39 |
| `Accum_VGPR_Count` | 384 | 384 | 0 |

Pred serialization does not cross the cliff. It lowers both machine-level
families, somewhat more on fragment packing, but the post-RA physical MIR
still contains `166` `COPY` lines touching `$agpr >= 100`, with maximum
`$agpr254`, exactly as B does. It also leaves code-object `.agpr_count: 256`.

## Existing Compiler Repair and Why It Does Not Transfer

The only compiler repair retained in this line is the fixed-layout, default-off
producer-consumer rewrite in
`lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.cc`.
It matches exactly four MFMA16 B-fragment consumers of a BF16 `[128,64]`
workgroup K tile, removes the broad producer, stages the compact persistent
tile, and replaces each consumer. Its L6 result is positive; its full B
result is not.

The alternative late direct-LDS B-load lowering was already tested and did
not improve the full combined live region. It is intentionally not revived.
The present attribution also rejects choosing only address or only fragment
materialization: the split is `96/94`, and C lowers both but leaves the same
384-AccVGPR allocation class.

The minimal valid conclusion is therefore a negative one: **do not promote
this rewrite to full v29 and do not add another local B-load lowering pass.**
The next architectural change would need a compiler-owned scheduling/block
operation that represents the full MFMA32-pred-to-MFMA16-update composition,
not another generic producer or consumer rewrite. That is outside this narrow
pass and should be separately designed and benchmarked.

## Exact LLVM, MIR, ISA, and Rocprof Evidence

The final LTO replay artifacts are under:

- A: `rocprof_outputs/qwen_v29_full_mir_regalloc/original_exact_lto/`
- B: `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/`
- C: `rocprof_outputs/qwen_v29_full_mir_regalloc/kfrag_pred_streaming_lto/`

The fresh A replay has no `SI_SPILL_AV32_SAVE` or `SI_SPILL_AV64_SAVE` in
its post-greedy kernel section. That directly distinguishes the original
register allocation from B/C; it is not inferred only from rocprof scratch.

For B, post-LTO LLVM contains `224` `<2 x i64>` and `688` `<2 x i32>`
occurrences, alongside `416` `extractelement`s and `1326` GEPs. C reduces
some accumulator-epilogue work (`384` `extractelement`s and `1283` GEPs),
but retains the same `<2 x i64>` dynamic-address family (`224`) and still has
`704` `<2 x i32>` occurrences. This is consistent with C lowering spill
pressure without replacing the generic compact-K address path.

Both ISA listings retain the intended MFMA geometry:

| ISA mnemonic count | B | C |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 16 | 16 |
| `v_mfma_f32_16x16x16_bf16` | 128 | 128 |
| `scratch_store` | 126 | 106 |
| `scratch_load` | 126 | 106 |

The targeted T=2048 rocprof comparison is:

| metric | B | C | change |
|:--|--:|--:|--:|
| trace median | `1302.317 us` | `1234.936 us` | -5.17% |
| VGPR / AccVGPR / SGPR | `128 / 384 / 112` | `128 / 384 / 112` | unchanged |
| scratch | `736 B` | `600 B` | -18.5% |
| LDS block | `61440 B` | `61440 B` | unchanged |
| MFMA | `294912` | `294912` | unchanged |
| VALU | `3180992` | `3173632` | -0.55% |
| SALU | `567808` | `568000` | +0.03% |
| VMEM | `601984` | `555776` | -7.68% |
| LDS instructions | `1242304` | `1164480` | -6.26% |
| occupancy | `0.6424%` | `0.6431%` | effectively unchanged |

`Accum_VGPR_Count` is a profiler resource metric, not a one-to-one virtual
register identifier. The direct vreg evidence is the 190/151 spill-word and
MIR save/restore accounting above.

## Reproducibility

```bash
# Host-side classification from exact post-greedy MIR.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/analyze_qwen_v29_spill_vregs.py \
  --mir test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_07.mir \
  --llvm-ir test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/post_lto_precodegen.ll \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_final_audit/rewrite_spills

# Existing Docker environment.
docker exec -w /workspace/project/avelang \
  -e PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
  -e PYTHONDONTWRITEBYTECODE=1 ljd_qwen_vllm_avelang_rocm722 \
  bash -lc 'python3 -m pytest -q test/examples/linear_attention/vllm_compare/test_qwen_gdn_v29_kfrag_pred_streaming_exp.py -s'

# Exact LTO replay for C after compiling with AVELANG_AMDGPU_LINK_DEBUG_DIR.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py \
  --argv-file test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/kfrag_pred_streaming_link/amdgpu-link-0.argv.txt \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/kfrag_pred_streaming_lto \
  --kernel _qwen_gdn_fused_chunk_gdr_full_kfrag_pred_streaming_exp_bf16_kernel_v29_mfma32
```

## Files Added or Updated in This Pass

- `test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_fused_chunk_gdr_full_kfrag_pred_streaming_exp.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_v29_kfrag_pred_streaming_exp.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_full_kfrag_matrix.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/analyze_qwen_v29_spill_vregs.py`
- this report and `codex_final_audit/{benchmark_matrix.csv,rocprof_summary.json,`
  `rewrite_spills,rewrite_streaming_spills}/`

`git diff --check` passes. No commit was created.

At final verification, the tracked workspace diff stat was `13 files changed,
492 insertions(+), 6 deletions(-)`. That is a workspace-wide figure containing
pre-existing user/compiler work, not a claim that every line belongs to this
audit. The experiment, analyzer, and artifacts are under the paths listed
above and were left uncommitted.

## Final Decision

The full rewrite+streaming result is a useful, correct experimental
mitigation, not a production fix. Stop the local v29 K-fragment compiler
rewrite line here; preserve the artifacts for any future full-composition
block/scheduling design, and keep v24 as the production baseline.
# Qwen gfx942 ASM v0 Integration Report

## Summary

The experimental raw BT64 recurrence route is implemented and passes its
target stage gate. The contract is **CASE C**: the frozen Triton operator is
correct for the raw FLA/vLLM recurrence, but its pred cast/order and public
`h` dtype differ from the historical Avelang BT64/v24 pipeline.

Final symbol:

```text
qwen_gdn_bt64_gfx942_asm_v0
```

The core assembly body was not algorithmically changed. The compiler-stage
Triton AMDGCN source was copied and mechanically renamed to the new symbol;
after normalizing the symbol name, the source is byte-identical to the frozen
golden source.

The v0 path is opt-in and opaque. It loads the dedicated HSACO through the
existing HIP external-kernel bridge and does not lower the recurrence into
generic AveLang memref/vector/MFMA operations. v24 and all production
defaults remain unchanged.

## Contract Difference

| Item | asm v0 / Triton | historical Avelang BT64 or v24 |
|:--|:--|:--|
| pred | FP32 resident `w/state`, `v_mfma_f32_32x32x4_xf32` | BF16-staged pred path |
| `h` | BF16 `[1,T/64,8,128,128]` | public Avelang containers are FP32; v24 is BT16 |
| gate input | raw `g`, exponentiation in kernel | v29 commonly receives precomputed decay/scale |
| chunk | BT64, BV32 | v24 production chain is BT16 |
| initial state | strict non-null FP32 in v0 | Avelang wrappers commonly allow `None` |

The common recurrence pieces are `key_head = value_head // 2`, `v_new =
u - pred`, BF16 corrected-decay/key update, and FP32 final state. Because the
differences are observable, the adapter does not convert v0 output into a
false v24-equivalent full-forward result. Detailed records are in
`codex_qwen_asm_v0_integration/contract_diff.md` and `.json`.

## Runtime Integration

The experimental high-level entry point is:

```python
qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g, initial_state)
```

The explicit container adapter is
`qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0(...)`. Runtime checks cover
gfx942, fixed shapes/dtypes/layouts, contiguity, `T % 64`, HSACO SHA256,
current HIP stream, and the exact 88-byte kernarg ABI. Guard failures use an
explicit caller-provided fallback; launch failures are raised.

| Field | Value |
|:--|:--|
| grid | `[4,8,1]` |
| workgroup | `[256,1,1]` |
| dynamic LDS argument | `57344` bytes |
| kernarg segment | 88 bytes |
| stream | current PyTorch HIP stream |

## Correctness

The GPU correctness runner completed **46 cases**: 40 random nonzero-W cases
plus six special modes: `zero_w`, `zero_state`, `unit_decay`, `high_dynamic`,
`cancellation`, and `small_scale`. It covered T=64, 128, 512, and 2048.

Against the real vLLM golden kernel, original HSACO, rebuilt HSACO, and asm v0
all had zero max absolute error, zero mean absolute error, and no first
mismatch for `h`, `v_new`, and `final_state` in every case.

| Test | Result |
|:--|:--|
| asm v0 pytest | `5 passed` |
| external HSACO bridge tests | `8 passed` |
| v31 P16 existing tests | `55 passed` |
| 46-case raw correctness runner | exit 0; all three artifacts exact vs vLLM |
| nonzero-W | passed against vLLM authority |
| T=64/128/512/2048 | passed against vLLM authority |

The project/v29 diagnostic reference is intentionally separate. It shows the
expected CASE-C numerical difference from BF16 staging and recurrence order;
it is not used as the v0 authority. In ordinary random cases, the largest
observed project-reference errors were approximately `0.001953` for `h`,
`0.000608` for `v_new`, and `0.000931` for final state. `high_dynamic`
amplifies this cast/order difference. This prevents claiming v0 is a drop-in
v24/v29 numerical replacement, but does not weaken the bit-exact vLLM gate.

## Same-Harness Benchmark

The strict comparison used the same C++ HIP harness, same inputs, same dynamic
LDS launch, module loading before timing, warmup 10, repeat 50, and three
independent sessions. T=512 is shown as session medians because its
sub-0.1-ms workload had a visible cold/cache outlier.

| T | golden original ms | golden rebuilt ms | asm v0 ms | old v29 context ms | v31 context ms |
|---:|---:|---:|---:|---:|---:|
| 512 | `1.586597/0.081921/0.086609` | `1.615040/0.082843/0.089292` | `1.615480/0.082963/0.154350` | `0.433924` | `0.647702` |
| 2048 | `0.195331` | `0.197573` | `0.196371` | `0.833899` | `1.582211` |
| 8192 | `0.567604` | `0.567364` | `0.567084` | `3.227254` | `6.230597` |
| 16384 | `1.164608` | `1.164450` | `1.164971` | `6.649940` | `12.539170` |

At T=2048, asm v0 is `+0.53%` relative to original golden and `-0.61%`
relative to rebuilt golden. At T=8192 and 16384 it remains within about
`0.1%` of golden. The T=512 row is not treated as a stable optimization
claim. Old-v29 and v31 are context only: their visible outputs and contracts
are different and they are not semantic speedup baselines.

## rocprof Resources

All rows below were collected with the same harness and rocprof command at
T=2048. `LDS_Block_Size=0` is the profiler's static LDS field; the launch
still supplies the required 57,344-byte dynamic LDS argument.

| implementation | trace us | grid work-items | WG | LDS field | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| golden original | 145.755 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.201728 |
| golden rebuilt | 145.776 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.210285 |
| asm v0 | 145.656 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.204427 |
| old v29 context | 798.866 | 4096 | 128 | 61440 | 0 | 128 | 264 | 112 | 294912 | 4977280 | 810496 | 399360 | 1242304 | 0.644988 |
| v31 context | 1550.784 | 4096 | 128 | 27648 | 0 | 36 | 228 | 112 | 327680 | 4671424 | 1447616 | 661504 | 1690816 | 0.644323 |

Static v0 metadata reports private segment 0, VGPR spill count 0, and SGPR
spill count 0. Its disassembly has no scratch loads/stores and no direct
`v_accvgpr_read/write_b32` reference at or above `a100`; the maximum direct
accumulator index is `a63`. The old failed generic full-v29 rewrite remains
documented separately as `AccVGPR=384`, `736 B` scratch, and 190 spilled VGPR
words. The v0 path avoids that compiler region; it does not repair generic
Avelang lowering.

## ISA Evidence

The v0 disassembly contains:

- `v_mfma_f32_32x32x4_xf32`: 96 static instructions;
- `v_mfma_f32_32x32x8_bf16`: 48 static instructions;
- `s_barrier`: 44;
- LDS read/write instructions: 483;
- scratch load/store instructions: 0;
- direct high-AGPR references at `a100` or above: 0.

This confirms the Triton compiler-stage MFMA schedule was preserved,
including the XF32 pred path and BF16 update path. No padded MFMA32 schedule
was introduced.

## Full-Forward Boundary

The asm v0 recurrence was not wired into v24 full forward. This is a
correctness boundary:

1. v24 is a BT16 cumsum/KKT/solve/w_u/chunk_gdr/chunk_o pipeline;
2. asm v0 is BT64 and emits BF16 chunk-start `h`;
3. the pred cast/order differs;
4. v24 chunk_o expects BT16 chunk layout.

Widening BF16 `h` to FP32 alone does not repair chunk boundaries or numerical
semantics. `full_forward_benchmark.csv` therefore records the comparison as
not applicable, and no v24/asm-v0 full-forward latency is claimed.

The next full-forward bottleneck is a separately validated BT64 upstream and
downstream contract: KKT/solve/w_u/chunk_o plus the XF32/BF16 cast policy.

## Reproduction and Decision

The consolidated runner is:

```bash
cd /workspace/project/avelang
bash test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/commands.sh
```

The raw recurrence gate passes: exact vLLM correctness, golden-matched
resources, and same-stage trace within 0.1% at long-text sizes. The old
generic-lowering resource cliff is avoided for this opaque route.

It is not production-ready as a v24 full-forward replacement because the
contract is CASE C. Keep it opt-in. `ready_for_step_2_profile` is false until
a correct BT64 upstream/downstream full-forward boundary is defined and
validated; no full-forward performance claim is made here.
# Qwen gfx942 BT64 Full-Pipeline Stage 2

## Result

Stage 2 is complete as an opt-in experimental pipeline. The new entry point is
`qwen_gdn_full_bt64_gfx942_asm_v0(...)` in
`vllm_compare/qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py`. Its formal
execution path contains no `vllm` import and does not call the vLLM full
wrapper. vLLM is used only by the golden and benchmark harnesses.

The observed next bottleneck is the **generic-v6 BT64 `chunk_o` fallback**.
This is not the v24 production MFMA output path. Stage 2 did not implement a
native BT64 output kernel, fuse stages, or alter the immutable assembly
recurrence.

## Real Operator Boundary

The authoritative entry is
`vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule`.
Its source-level graph is:

```text
chunk_local_cumsum
  -> chunk_scaled_dot_kkt_fwd
  -> solve_tril
  -> recompute_w_u_fwd
  -> chunk_gated_delta_rule_fwd_h
  -> chunk_fwd_o
  -> BF16 public output + optional FP32 final state
```

The candidate graph and exact source mapping are recorded in
`codex_qwen_bt64_full_pipeline_stage2/full_operator_execution_graph.md` and
`stage_source_map.md`.

## Frozen Contract

- Fixed target: gfx942, `B=1,Hk=4,Hv=8,K=V=128`, `[B,T,H,D]`, contiguous.
- `BT=64`; reject `T % 64 != 0`; no tail support is claimed.
- Inputs: BF16 `q/k/v`; FP32 `g/beta`; optional FP32 initial state.
- `initial_state=None` becomes a zero FP32 state outside asm.
- Reused asm is unchanged CASE C: FP32 `w/u/g/h0`, XF32 prediction, BF16
  `h`, FP32 `v_new` and final state.
- Candidate cumsum/KKT/solve/W-U reuse v6/v18 BT64-capable stages. Candidate
  W/U is FP32 for the asm ABI; the vLLM captured W/U is BF16.
- Experimental chunk-o uses the generic v6 primitive with `chunk_size=64` and
  a FP32 view of BF16 `h`; it is not the v24 BT16 MFMA chunk-o wrapper.
  Likewise, the experimental W/U is generic v6 rather than v24's v14 MFMA
  W/U implementation.
- Public output is BF16 and final state is FP32. Acceptance thresholds were
  frozen before the matrix: output abs <= `1/128`, final-state abs <= `0.02`.

Details are in `bt64_numerical_contract.md`, `bt64_numerical_contract.json`,
`cast_policy.md`, and `shape_layout_contract.md`.

## Correctness

`stage2_runner.py --random-cases 30 --capture` executed 37 cases: 30 random
nonzero cases covering T=64/128/512/2048, plus neutral-gate, high-dynamic,
cancellation, small-value, and two T=8192 smoke cases. Zero and nonzero
initial states are both included. Every full result was accepted.

| quantity | maximum abs error | threshold |
|:--|--:|--:|
| public BF16 output | `0.001953125` | `0.0078125` |
| FP32 final state | `0.014624655` | `0.020000000` |
| KKT | `5.78135e-04` | diagnostic |
| solve | `1.28478e-03` | diagnostic |
| W | `1.83117e-03` | diagnostic |
| U | `6.01139e-02` | diagnostic |
| recurrence H | `0.03125` | diagnostic |
| recurrence V_new | `0.0869646` | diagnostic |

The first non-bitwise difference is the FP32 cumsum (`9.53674e-06` max).
Downstream stage values differ further because the candidate's FP32
intermediates intentionally bridge to the immutable asm ABI while vLLM's
captured solve/W/U are BF16. This is documented rather than hidden; the public
outputs remain under the predeclared thresholds. Per-case data and first
coordinates are in `stage_correctness_results.csv` and
`first_divergence_report.md`.

## Regression Tests

- asm-v0 recurrence: `5 passed`.
- external HSACO/full bridge, gfx942 smoke, and P16: `63 passed, 1 skipped`.
- new full BT64 pipeline suite: `4 passed`.

The assembly body, symbol, grid/workgroup ABI, LDS, MFMA order, and dispatch
were not modified. The recurrence's resource tuple remains `VGPR=128`,
`AccVGPR=192`, `SGPR=80`, and zero scratch/spills.

## End-to-End Timing

Each row is the median of three independent HIP-event session medians, with
warmup=10 and repeat=50. Candidate/v24 and vLLM run in separate Python
processes. Candidate full timing is explicitly marked cached-allocator because
the reused generic wrappers retain internal allocation behavior.

| T | candidate BT64 asm-v0 ms | v24 BT16 ms | vLLM ms |
|--:|--:|--:|--:|
| 512 | `4.925` | `0.303` | `0.308` |
| 2048 | `16.041` | `0.590` | `0.364` |
| 8192 | `80.907` | `2.584` | `0.735` |
| 16384 | `201.451` | `5.206` | `1.273` |

At T=2048 the experimental candidate is `27.19x` v24 and `44.12x` vLLM.
These comparisons are diagnostic: v24 uses a different BT16 numerical path,
while the candidate has the correct BT64 full API but deliberately generic
upstream/downstream stages. The raw three-session samples and p10/p90 context
are retained in `full_pipeline_benchmark.csv`.

## T=2048 Targeted Profile

The table uses the median of the last five matching rocprof dispatches. These
are device trace measurements, not the HIP-event full latency above.

| stage | trace ms | workgroup | grid work-items | scratch | MFMA | VALU | SALU | VMEM |
|:--|--:|:--|:--|--:|--:|--:|--:|--:|
| cumsum | `0.012` | `1` | `256` | 0 | 0 | 17,152 | 22,272 | 32,768 |
| KKT | `0.325` | `1` | `16,384` | 0 | 0 | 207,568,896 | 12,812,288 | 11,116,544 |
| solve | `0.109` | `128` | `32,768` | 0 | 0 | 2,695,424 | 2,535,680 | 32,768 |
| W/U | `3.612` | `1` | `16,384` | 0 | 0 | 800,227,328 | 17,006,592 | 34,603,008 |
| asm recurrence | `0.472` | `256` | `1024 x 8` | 0 | 196,608 | 2,535,040 | 214,016 | 91,136 |
| chunk-o | `11.761` | `1` | `16,384` | 260 B | 0 | 1,079,058,432 | 317,063,168 | 215,351,296 |

For asm, rocprof reports global work-items `1024 x 8`; this normalizes to the
frozen launch grid `(4,8,1)` with workgroup 256. Its uninstrumented preallocated
HIP-event time at T=2048 was `0.205 ms`, or about `1.28%` of the candidate's
full median. Profiling perturbation explains why its trace value is higher.

This generic-v6 `chunk_o` fallback dominates the experimental profile: it has
no MFMA or LDS work, one-thread workgroups, 260 B scratch, and 1.079B VALU /
317M SALU / 215M VMEM instructions. Its `11.761 ms` trace is about 3.26x the
generic-v6 W/U trace and roughly 72% of the profiled stage sum. This identifies
the first missing native BT64 downstream stage; it is not evidence that the
existing v24 MFMA `chunk_o` is slow.

## Decision

`ready_for_stage3=true` because the candidate is opt-in, does not call the
vLLM full wrapper, passes the ordinary/multi-chunk/T8192 correctness gates,
and has a complete targeted profile. Stage 3 should port the **v24-style MFMA
output design to a native BT64 chunk-o** against the frozen BF16-H/V-new
interface. It cannot directly call v24's kernel: both v24 `w_u` and `chunk_o`
hard-reject `chunk_size != 16`, and splitting a BT64 recurrence snapshot into
four BT16 output calls changes the causal intra-chunk math. The work should not
touch asm v0, alter v24, or modify compiler/register allocation. Native BT64
W/U is the next missing companion stage, but BT64 chunk-o is the first target
because its generic fallback has the largest measured trace.

## Evidence

All generated artifacts and exact commands live under
`codex_qwen_bt64_full_pipeline_stage2/`; the structured decision is
`final_decision.json`. No Stage 3 kernel was implemented in this pass.
# Qwen gfx942 BT64 Native W/U + chunk-o Stage 3

## Summary

Stage 3 replaces the Stage 2 **generic scalar BT64** W/U and chunk-o fallbacks
with opt-in 64-thread MFMA implementations derived from v24/v14's validated
BT16 microkernel.  It does not modify v23/v24/v26/v27/v28, the frozen asm-v0
recurrence, compiler lowering, LLVM, AMDGPU RA, or default dispatch.

The native stages pass standalone numerical gates, produce real
`v_mfma_f32_16x16x16_bf16` instructions, have zero scratch, and pass the
frozen 37-case BT64 full contract.  At T=2048 the full candidate is
`1.0922 ms`, a `14.69x` reduction from Stage 2's `16.0413 ms`.

It is **not** a production promotion: the valid BT64 candidate is still slower
than v24 BT16 (`0.5882 ms`) and the two-session vLLM reference (`0.3613 ms`).
The next measured bottleneck is generic BT64 KKT.

## Scope and Source Audit

The detailed source audit is in
`codex_qwen_bt64_wu_chunko_stage3/source_audit.md`; the machine-readable map
is `source_map.json`.

- v24 imports its W/U and chunk-o fast kernels from v17, not from a scalar
  fallback.  They use 64 threads and `mfma_16x16x16_bf16_f32`.
- Stage 2 uses v6 W/U and chunk-o only to establish the BT64 numerical bridge
  to asm-v0.  Their T=2048 rocprof workgroup is one thread, with no MFMA.
- Stage 3 creates
  `qwen_gdn_bt64_native_wu_chunko_mfma_v1.py` and the explicit opt-in public
  entry `qwen_gdn_full_bt64_gfx942_asm_v0_native_wu_o_v1`.
- BT64 is four 16-token source subtiles.  W/U accumulate four subtiles; the
  chunk-o source loop handles the full 4x4 lower-triangular tile relation.

The contracts are frozen in `wu_contract.md/json` and `chunko_contract.md/json`.
W/U output FP32 for the asm ABI; native chunk-o consumes BF16 asm H directly
and yields FP32 before public BF16 conversion.

## Standalone Correctness

GPU pytest results:

| gate | cases | maximum abs error | result |
|:--|:--|--:|:--|
| native W vs v6 BT64 | T=64,128 | `2.9802322e-08` | pass |
| native U vs v6 BT64 | T=64,128 | `2.9802322e-08` | pass |
| native chunk-o vs v6 BF16-H consumer | T=64,128 | `8.353265e-06` | pass |
| early-source to later-output causal tile check | T=64 | `2.5258523e-06` | pass |
| native full graph vs frozen vLLM contract | T=64,128,512 | output `<=0.001953125`, state `<=0.017289162` | 3 passed |

The standalone suite was `5 passed in 13.63s`; the fast full smoke suite was
`3 passed in 28.07s` on the MI300 environment.

## Full BT64 Contract Matrix

The local Stage 3 runner pins the entire 37-case plan so it does not depend on
the version of Stage 2 helper code copied into a Docker workspace.  It covers
30 random cases, neutral-gate, high-dynamic, cancellation, small-value, and
two T=8192 cases with and without initial state.

| quantity | observed maximum abs | frozen threshold | result |
|:--|--:|--:|:--|
| public BF16 output | `0.001953125` | `0.0078125` | pass |
| FP32 final state | `0.013978422` | `0.020000000` | pass |

`full_correctness.json` and `full_correctness.csv` record all 37 accepted
cases.  The largest public-output/final-state pair occurred in the T=8192
neutral-gate case; both remain under the predeclared contract.

## Microbenchmark

HIP-event median, warmup 10/repeat 50, actual MI300 runs:

| T | stage | generic Stage 2 ms | native Stage 3 ms | speedup |
|--:|:--|--:|--:|--:|
| 512 | W/U | `1.447091` | `0.194790` | `7.43x` |
| 512 | chunk-o | `3.694571` | `0.067280` | `54.91x` |
| 2048 | W/U | `4.005133` | `0.294919` | `13.58x` |
| 2048 | chunk-o | `12.151661` | `0.256241` | `47.42x` |

The requested T=2048 W/U and chunk-o gates therefore pass comfortably.  This
table compares the same Stage 2 generic Avelang fallbacks to native Stage 3;
it is not a direct vLLM per-stage measurement.

## Full Pipeline Timing

Each Stage 3/v24 row is the median of three independent Python sessions,
warmup 10/repeat 50.  vLLM completed two independent sessions before the
environment rejected the third Docker execution on quota grounds; its `n=2`
median is labelled accordingly and is not represented as a three-session
result.

| T | Stage 3 BT64 native ms | v24 BT16 ms | vLLM ms (`n=2`) |
|--:|--:|--:|--:|
| 512 | `0.489788` | `0.304152` | `0.307978` |
| 2048 | `1.092222` | `0.588173` | `0.361317` |
| 8192 | `3.659696` | `2.124136` | `0.747861` |
| 16384 | `7.223932` | `4.256944` | `1.263817` |

At T=2048 Stage 3 is `14.69x` faster than Stage 2 (`16.041299 ms`), but
`1.86x` slower than v24 and about `3.02x` slower than the two-session vLLM
reference.  The raw sessions are in `full_pipeline_benchmark_stage3_[abc].csv`
and `full_pipeline_benchmark_vllm_[ab].csv`.

## Stage Breakdown

The following T=2048 HIP-event stage medians are from independent Stage 3
session `stage3_a` (warmup 10/repeat 50):

| stage | ms |
|:--|--:|
| cumsum | `0.031327` |
| KKT generic BT64 | `0.350221` |
| solve | `0.125367` |
| native W/U | `0.235329` |
| frozen asm recurrence, preallocated | `0.201620` |
| native chunk-o | `0.227899` |
| full native graph, cached allocator | `1.089960` |

KKT is the largest individual stage.  Native W/U and chunk-o are no longer
the scalar disasters from Stage 2, but their remaining composition with KKT,
solve, and the frozen recurrence is still above v24/vLLM.

## T=2048 rocprof Resources

Tail median of five matching dispatches.  Stage 2 figures are the prior
targeted profile, while native figures are measured from this Stage 3 source.

| kernel/group | trace us | WG | grid work-items | scratch | VGPR | AccVGPR | LDS B | MFMA | VALU | VMEM |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Stage 2 generic W/U | `3611.807` | 1 | 16384 | 0 | 128 | 144 | 0 | 0 | 800227328 | 34603008 |
| native W | `131.034` | 64 | 524288 | 0 | 44 | 4 | 1024 | 131072 | 55279616 | 6127616 |
| native U | `88.572` | 64 | 524288 | 0 | 44 | 4 | 1024 | 131072 | 28295168 | 4521984 |
| Stage 2 generic chunk-o | `11761.158` | 1 | 16384 | 260 | 128 | 136 | 0 | 0 | 1079058432 | 215351296 |
| native chunk-o | `203.422` | 64 | 524288 | 0 | 12 | 140 | 13312 | 458752 | 18096128 | 1933312 |

The native W+U trace is `219.606 us`, roughly `16.45x` below the Stage 2 W/U
trace.  Native chunk-o is `57.82x` below the Stage 2 generic chunk-o trace.
The detailed counters, including SALU, LDS instructions, SGPR, and occupancy,
are in `resource_profile.json/csv`.

## ISA Evidence

All three native HSACOs were captured from the MI300 container and disassembled
with ROCm `llvm-objdump`:

- native W: 16 static `v_mfma_f32_16x16x16_bf16` instructions;
- native U: 16 static `v_mfma_f32_16x16x16_bf16` instructions;
- native chunk-o: 56 static `v_mfma_f32_16x16x16_bf16` instructions;
- all three: zero `v_mfma_f32_32x32x8_bf16` instructions.

The files are under `native_wu/isa/` and `native_chunko/isa/`.  This is the
intended v24 microkernel port, not an accidental scalar fallback.  The profile
also proves zero scratch for all three native kernels.

## Tests and Reproduction

New source/test/benchmark files:

- `vllm_compare/qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `vllm_compare/test_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `vllm_compare/test_qwen_gdn_full_bt64_native_wu_o_v1.py`
- `vllm_compare/bench_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py`
- `codex_qwen_bt64_wu_chunko_stage3/stage3_runner.py`
- `codex_qwen_bt64_wu_chunko_stage3/bench_stage3.py`
- `codex_qwen_bt64_wu_chunko_stage3/profile_native_stages.py`

Exact commands are in `codex_qwen_bt64_wu_chunko_stage3/commands.sh`.
The immutable asm-v0/P16 regression suites were not re-run in this pass after
the environment exhausted Docker execution quota; they were untouched, and
the new full matrix repeatedly dispatches the unchanged asm recurrence.

## Decision

The Stage 2 W/U and chunk-o fallbacks should be replaced **only through the
new opt-in Stage 3 API** for further BT64 experimentation.  Do not change any
production/default dispatch.  The next isolated optimization should be a
native BT64 KKT: at T=2048 it is the largest measured stage (`0.350221 ms`) and
remains the Stage 2 generic scalar implementation.  Re-evaluate full BT64
after that gate; do not alter the validated asm recurrence merely because it is
part of the remaining cost.

# Qwen gfx942 BT64 Non-Recurrence Stage 4

## Summary

Stage 4 is complete as an opt-in, high-level Avelang experiment. It replaces
the Stage 3 BT64 KKT, W/U, and chunk-o schedules while reusing cumsum, v18
solve, and the frozen gfx942 asm-v0 recurrence unchanged.

At T=2048, median of three independent warmup-10/repeat-50 sessions:

| implementation | full ms | relative to Stage 4 |
|:--|--:|--:|
| Stage 3 BT64 | `1.089079` | `2.2887x` slower |
| v24 BT16 | `0.587173` | `1.2339x` slower |
| **Stage 4 BT64** | **`0.475867`** | `1.0000x` |
| vLLM BT64 | `0.365043` | Stage 4 is `1.3036x` slower |

The main `<0.75 ms`, v24-beating, and `<=1.5x vLLM` targets all pass. This
does not change production dispatch. The experimental public entry is
`qwen_gdn_full_bt64_stage4_all_s0`; the historical name contains KKT-S0,
residual-MFMA WU-S1, and chunk-o-S0.

## Historical Audit

The audit covered v14 W/U, v18 parallel solve, v20 BT32 MFMA, v24 KKT/full,
Stage 2 scalar BT64, Stage 3 native W/U/chunk-o, and the frozen asm-v0 path.
Detailed successful/failed method mapping is in
`codex_qwen_bt64_nonrecurrence_stage4/historical_method_matrix.md/json`.

- KKT reused v24's verified MFMA16 token-dot primitive.
- W/U reused v14's main-product/residual idea but made both terms MFMA.
- chunk-o reused Stage 3/v24 MFMA16 math with BT64 CTA ownership.
- solve retained v18's correctness-proven parallel recurrence after audit.
- recurrence reused asm-v0 byte-for-byte.

Stage 3 was reproduced with the same full-path methodology. Three new vLLM
full sessions were obtained. A complete isolated vLLM KKT/W/U/chunk-o split
was not captured; only full latency and a direct solve comparison are claimed.

## Native KKT

Kernel: `_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`.

Each 64-thread CTA computes one of the 4x4 token16 tiles of a BT64
chunk/value-head matrix. Lower/diagonal tiles stage two `[16,128]` K tiles,
run eight MFMA16 reductions, and apply causal mask, beta, and FP32 decay at
writeback. The output remains layout-compatible with v18 solve.

| T | Stage 3 ms | Stage 4 ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.145897` | `0.034731` | `4.2007x` |
| 1024 | `0.213637` | `0.035513` | `6.0157x` |
| 2048 | `0.351423` | `0.046589` | `7.5430x` |

KKT maximum absolute error was `4.47e-8`; solve-after-KKT was `3.73e-8`.

## Native W/U

Kernels: `_qwen_gdn_w_bf16_kernel_bt64_mfma_v2_s0` and
`_qwen_gdn_u_bf16_kernel_bt64_mfma_v2_s0`; final wrapper:
`qwen_gdn_w_u_bt64_mfma_v2_s1`.

One 256-thread CTA owns a full BT64 row extent and a 16-column output tile;
four waves own four token16 rows and reuse shared A/K-or-V tiles. This cuts
CTA count fourfold. The main BF16 product is followed by a second BF16 MFMA
of `A_fp32 - bf16(A)`, replacing the exact S0 scalar 64-term correction.

| T | Stage 3 ms | exact S0 ms | residual-MFMA S1 ms | S1 speedup |
|--:|--:|--:|--:|--:|
| 512 | `0.123203` | `0.095522` | `0.060971` | `2.0207x` |
| 1024 | `0.149723` | `0.146318` | `0.064196` | `2.3323x` |
| 2048 | `0.248470` | `0.229160` | `0.084165` | `2.9522x` |

S1 differs from exact S0 by at most `1.88e-6`. The faster no-correction
candidate failed the frozen full contract and remains named as a failed
diagnostic; it is not called by the final path.

## Native chunk-o

Kernel: `_qwen_gdn_chunk_o_bf16_kernel_bt64_mfma_v2_s0`.

One 256-thread CTA owns `(chunk,value-head,V16)` and its four waves own all
four output token16 tiles. It stages/reuses Q, H, K, and V-new across the
BT64 tile. Inter-state and causal intra-chunk accumulators remain separate in
FP32 and are summed once at writeback. A single merged accumulator was tested
but rejected after numerical error reached roughly `2e-2` to `3.4e-2`.

| T | Stage 3 ms | Stage 4 ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.087570` | `0.044406` | `1.9720x` |
| 1024 | `0.141130` | `0.062573` | `2.2554x` |
| 2048 | `0.228620` | `0.095342` | `2.3979x` |

Random T=64/128/512 and dedicated inter/intra/source0/source1/source2
cross-token16 checks are bit-exact against Stage 3.

## Solve Audit

| T | v18 FP32 ms | vLLM FP32 ms | ratio | FP32 max abs |
|--:|--:|--:|--:|--:|
| 512 | `0.121140` | `0.053359` | `2.270x` | `2.98e-8` |
| 2048 | `0.126888` | `0.053779` | `2.359x` | `2.98e-8` |

The evidence permits a future solve experiment, but vLLM uses a distinct
hierarchical block inverse/dot schedule. Stage 4 does not add a rushed solve
variant after already meeting every full target. A separately gated FP32
hierarchical solve is the next high-level action.

## Incremental Integration

T=2048, three-session median:

| graph | full ms | gain from previous |
|:--|--:|--:|
| Stage 3 | `1.089079` | baseline |
| + KKT-S0 | `0.795162` | `1.3696x` |
| + WU-S1 | `0.610087` | `1.3034x` |
| + chunk-o-S0 | `0.475867` | `1.2821x` |

## Stage Breakdown

| T | cumsum | KKT | solve | W/U | asm recurrence | chunk-o |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 0.0311 | 0.0331 | 0.1193 | 0.0574 | 0.0959 | 0.0429 |
| 2048 | 0.0310 | 0.0453 | 0.1257 | 0.0812 | 0.2015 | 0.0928 |
| 8192 | 0.0313 | 0.1473 | 0.2326 | 0.2038 | 0.6238 | 0.2574 |
| 16384 | 0.0312 | 0.2699 | 0.3394 | 0.3725 | 1.2233 | 0.4793 |

Stage timings are independent dispatch measurements and do not sum exactly
to cached full latency.

## Full Scaling

| T | Stage 4 | Stage 3 | v24 | vLLM |
|--:|--:|--:|--:|--:|
| 512 | `0.313706` | `0.490168` | `0.300726` | `0.304873` |
| 2048 | `0.475867` | `1.089079` | `0.587173` | `0.365043` |
| 8192 | `1.399619` | `3.655211` | `2.122734` | `0.725157` |
| 16384 | `2.632552` | `7.213319` | `4.252098` | `1.268545` |

Stage 4 is slightly slower than v24/vLLM at T=512, but clearly beats v24
from T=2048 onward. Its long-sequence gap to vLLM remains about 1.9x to 2.1x.

## Correctness

The frozen 37-case matrix covers random T=64/128/512/2048, zero/nonzero
initial state, neutral gate, high dynamic range, cancellation, small values,
and T=8192 smoke cases.

| public quantity | maximum abs | threshold | result |
|:--|--:|--:|:--|
| BF16 output | `0.001953125` | `0.0078125` | pass |
| FP32 final state | `0.013475478` | `0.020000000` | pass |

All 37 cases passed. Standalone Stage 4 pytest was `29 passed in 23.06s`.
The unchanged asm-v0, external bridge, P16, gfx942 smoke, Stage 2, and Stage 3
regressions were `80 passed, 1 skipped in 79.38s`.

## rocprof and ISA

T=2048 counter-instrumented tail medians:

| kernel | trace us | WG | LDS B | VGPR | AccVGPR | scratch | MFMA | VALU | VMEM |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| KKT | 40.941 | 64 | 8192 | 20 | 4 | 0 | 20480 | 1565696 | 210944 |
| W | 27.722 | 256 | 2560 | 52 | 4 | 0 | 262144 | 5160960 | 458752 |
| U | 24.797 | 256 | 2560 | 48 | 8 | 0 | 262144 | 4005888 | 393216 |
| chunk-o | 68.622 | 256 | 27136 | 112 | 64 | 0 | 458752 | 12918784 | 851968 |

Every code object reports private segment `0`, VGPR spill count `0`, and SGPR
spill count `0`. Static ISA contains MFMA16 counts KKT/W/U/chunk-o =
`8/8/8/56`; all contain zero MFMA32. No v29-style scratch/high-AGPR cliff
appeared.

## Decision

Stage 4 validates the hypothesis that the remaining Stage 3 gap was primarily
high-level ownership, reuse, and correction schedule. Only experimental
high-level Python/Avelang code and evidence files were added; asm-v0,
compiler/RA, and production v23/v24/v26/v27/v28 were not modified.

The largest absolute T=2048 stage is the frozen recurrence (`0.2015 ms`). The
largest mutable non-recurrence stage is solve (`0.1257 ms`), and its measured
2.36x gap to vLLM makes a hierarchical FP32 BT64 solve the next single action.
No compiler or new assembly work is justified by Stage 4 data.

All evidence, exact commands, raw sessions, tests, HSACOs, ISA, and rocprof
CSVs are under `codex_qwen_bt64_nonrecurrence_stage4/`; the structured result
is `final_decision.json`.
# Qwen gfx942 BT64 FP32 Solve Stage 5A Root-Cause Audit

## 结论

Stage 5A 完成，且严格保持 audit-only：没有新增 solve、没有改 v18、没有改
Stage 4、asm-v0、compiler、LLVM 或 AMDGCN assembly。

当前 v18 solve 与 vLLM solve 计算的是同一个 FP32 契约：对每个 strict-lower
64x64 `A` 输出 `X=(I+A)^-1`，布局均为连续的 `[1,T,8,64]`。54 个同输入
case 均通过 authority 检查。造成差距的主因不是 input、kernel 数、scratch、
spill，也不是单纯的 global store；是 **v18 的 63 行串行 LDS/标量 recurrence
结构**，相对 vLLM 的 **4x16 hierarchical block inverse + FP32 MFMA block dot**
拥有更长的依赖关键路径、更高 SALU/LDS/VALU，以及更差的 wave/resource 形态。

唯一建议的 Stage 5B 是：先独立实现一个 FP32 BT64 4x16 hierarchical block
inverse，使用 256-thread CTA 和 MFMA16 block product；不接入 full pipeline
直到 standalone correctness/resource/performance gate 全部通过。

## 1. 真实调用路径

### Avelang v18

Stage 4 的
`qwen_gdn_full_bt64_stage4_all_s0_stages` 在
[qwen_gdn_bt64_nonrecurrence_mfma_v2.py](../../vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py:773)
调用 `qwen_gdn_solve_avelang_v18_bt64_layout`，该 wrapper 再发射
`_qwen_gdn_solve_kernel_v18_parallel`。源码和 launch 在
[qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py](../../vllm_compare/qwen_gdn_chunked_avelang_v18_bt64_layout_fixed.py:36)：

- 输入/输出：FP32、contiguous `[1,T,8,64]`；
- 一 CTA 对应一个 `(chunk, value_head)`；
- grid：`T/64 * 8`；workgroup：128 threads，即 2 waves；
- wrapper 的 `torch.empty_like` 已在 body-only benchmark 中排除；
- solve 后直接由 Stage 4 的 native W/U 读取 `a_solved`。

### vLLM/FLA

安装版的 `solve_tril` 位于审计副本
[`solve_tril_vllm_installed.py`](codex_qwen_bt64_solve_rootcause_stage5a/ir/vllm/solve_tril_vllm_installed.py:506)。
BT64 选择 `merge_16x16_to_64x64_inverse_kernel`（源码第 238 行），其 wrapper
分配 `torch.zeros_like(A)`，再以 grid `(T/64, B*H)` 发射一个 Triton kernel。

正常调用的实际 autotune 选择已捕获，而非猜测：`num_warps=4`、
`num_stages=5`、`num_ctas=1`、`precision=ieee`、`USE_TMA=false`。body-only
harness 用这个 exact config 直接发射；必要的输出清零在每个 event 之前完成，
不混入 body 时间。

完整 source map 见
[`source_call_graph.md`](codex_qwen_bt64_solve_rootcause_stage5a/source_call_graph.md)、
[`avelang_solve_source_map.json`](codex_qwen_bt64_solve_rootcause_stage5a/avelang_solve_source_map.json)、
[`vllm_solve_source_map.json`](codex_qwen_bt64_solve_rootcause_stage5a/vllm_solve_source_map.json)。

## 2. 数学、数值与依赖

v18 先把 `-A` stage 到 `mat[64,64]`，再按 `r=1..63` 做前向递推；每一行按列
并行，但下一行依赖上一行。每个 row 有三次 workgroup barrier：复制 row、完成
group partial、写回 final result。即 63 个串行 row stage，189 个 recurrence
barrier（加初始 staging barrier 共 190）。

vLLM 将 64x64 分成 4x4 个 16x16 block。四个 diagonal block 先做 local inverse，
然后计算：

```text
X21 = -X22 A21 X11      X32 = -X33 A32 X22      X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

因此它不是不同数学，而是等价的 block 重排。它有四条局部的 14-step 16x16
inverse 链，以及 `{21,32,43}` -> `{31,42}` -> `{41}` 的三层 block-DAG；其
off-diagonal 工作变成 `tl.dot`。最终 HSACO 实际包含
`v_mfma_f32_16x16x4_f32`，不是只从 Triton `tl.dot` 文本推断。

两条路径的 FP32 结合顺序不同，所以不要求 bitwise identical。54 cases 都使用
同一个 `A` 对象，authority 是 `torch.linalg.solve_triangular(I+A,I)`：

| 全部 case 最大值 | v18 | vLLM |
|:--|--:|--:|
| max abs vs authority | `4.5776367e-05` | `3.0517578e-05` |
| max residual inf | `4.8995018e-05` | `1.9311905e-05` |
| 两实现 max abs | `6.1035156e-05` | - |

最大误差来自刻意构造的 high-dynamic case；所有 case 均 `passed=true`。详见
[`correctness.csv`](codex_qwen_bt64_solve_rootcause_stage5a/correctness.csv) 与
[`numerical_order_comparison.md`](codex_qwen_bt64_solve_rootcause_stage5a/numerical_order_comparison.md)。

## 3. 同口径性能

Body-only 表排除了 allocation、compile 和 vLLM `zeros_like` 准备；每点为 3 个
独立 session 的中位数，每 session warmup=20/repeat=100：

| T | chunks | v18 body ms | vLLM body ms | v18/vLLM |
|--:|--:|--:|--:|--:|
| 512 | 8 | 0.116694 | 0.035372 | 3.30x |
| 2048 | 32 | 0.122422 | 0.035613 | 3.44x |
| 8192 | 128 | 0.229501 | 0.043484 | 5.28x |

为和历史 Stage 4 对齐，public wrapper 的 T=2048 是 `0.125386 ms` vs
`0.053179 ms`，比例 `2.36x`，复现了先前约 2.36x 的观测。body-only 更严格，
因此揭示了被 vLLM 输出初始化掩盖的 compute 差距。

chunk sweep 拟合：

| implementation | intercept | slope/chunk | R2 |
|:--|--:|--:|--:|
| v18 | 103.069 us | 0.895079 us | 0.9586 |
| vLLM | 33.866 us | 0.087651 us | 0.9728 |

所以不只是 fixed overhead：v18 的 per-chunk slope 为 vLLM 的 10.21x。T=512
和 2048 看起来接近，是因为两者在这些 grid 下都能并行调度大量 CTA；到 T=8192
后 gap 明显扩大。原始 sample 与方法在
[`benchmark_methodology.md`](codex_qwen_bt64_solve_rootcause_stage5a/benchmark_methodology.md)、
[`standalone_benchmark.csv`](codex_qwen_bt64_solve_rootcause_stage5a/standalone_benchmark.csv)、
[`latency_fit.json`](codex_qwen_bt64_solve_rootcause_stage5a/latency_fit.json)。

## 4. rocprof 和 ISA

T=2048 的 selected dispatch 比较：

| metric | v18 | vLLM |
|:--|--:|--:|
| trace median | 110.224 us | 23.875 us |
| workgroup / waves | 128 / 2 | 256 / 4 |
| grid global workitems | 32768 x 1 | 8192 x 8 |
| CTA 数 | 256 | 256 |
| VGPR / AccVGPR / SGPR | 96 / 128 / 112 | 56 / 16 / 48 |
| LDS block / scratch | 17920 B / 0 | 0 B / 0 |
| OccupancyPercent | 4.733 | 6.991 |
| MFMA | 0 | 65536 |
| VALU | 2695424 | 1419264 |
| SALU | 2535680 | 570368 |
| VMEM | 32768 | 135168 |
| LDS instructions | 1177856 | 397312 |

两边都只有一个 solve CTA dispatch，均为零 scratch；HSACO metadata 也显示零
private segment、零 VGPR spill。没有证据表明需要先修 compiler/RA 或写 assembly。
vLLM 的 VMEM 更高却更快，也反证 global traffic 不是主解释；v18 的 SALU 是
4.45x，LDS instruction 是 2.96x，且没有 MFMA block update。

barrier 的精确动态 counter 不可得，故未伪造。v18 的 189 次 recurrence barrier
来自真实源码；两边 static ISA 的 `s_barrier` text count 受 unrolling/loop 影响，
不用于直接时延归因。完整资料在
[`resource_comparison.csv`](codex_qwen_bt64_solve_rootcause_stage5a/resource_comparison.csv)、
[`dynamic_counter_comparison.csv`](codex_qwen_bt64_solve_rootcause_stage5a/dynamic_counter_comparison.csv)、
[`static_isa_comparison.md`](codex_qwen_bt64_solve_rootcause_stage5a/static_isa_comparison.md)。

## 5. 受控消融

| T/input | real v18 | launch floor | load/store only | no-store checksum |
|:--|--:|--:|--:|--:|
| 64 random | 0.116053 | 0.018027 | 0.018908 | 0.105997 |
| 2048 random | 0.122302 | 0.017866 | 0.024397 | 0.106959 |
| 2048 diagonal-boundary | 0.122061 | 0.017266 | 0.022473 | 0.106419 |

`load/store-only` 仅保留原始地址映射和 full output 写回；它远小于 real solve。
`no-store` 保留 LDS staging/recurrence，并以最后一行 checksum 防止整个 kernel 被
DCE；它不是数学等价 variant，故只用于界定 final output store 不是主因。对角边界
输入几乎不改变 v18 时间，也支持 fixed control/barrier schedule 才是主要负担。

## 6. Stage 5B 决策

**继续，但只做一个方向：独立的 FP32 BT64 4x16 hierarchical block inverse。**

- 每 CTA 一个 chunk/head，256 threads/4 waves；
- 先四个 16x16 diagonal local inverse，再按三层 block-DAG 完成 six lower blocks；
- block product 目标使用 FP32 MFMA16；不改 public layout；
- 首先只测 standalone，先跑当前 54-case matrix 与 W/U consumer；
- gate：scratch=0、显式 LDS <=8 KiB、T=2048 body <=0.060 ms，强目标 <=0.050 ms；
- 若实现失败 gate，保留 v18，不开始 compiler 或 assembly 支线。

vLLM body `0.035613 ms` 是硬件/算法参考下界，不承诺 Avelang 可直接相同。若达到
0.06-0.05 ms，按 Stage 4 historical `solve=0.125687 ms` 和 full `0.475867 ms`
作简单替换估算，T=2048 full 可到约 `0.410-0.400 ms`；完整 pipeline 必须重测，
不得将该加减当结果。

实现蓝图与 gate 见
[`stage5b_implementation_blueprint.md`](codex_qwen_bt64_solve_rootcause_stage5a/stage5b_implementation_blueprint.md)。

## 7. 回归与产物

- standalone solve: 54/54 authority cases passed；
- Stage 4 KKT/W-U/chunk-o/solve regression: `29 passed in 23.42s`；
- asm-v0 full contract: `4 passed in 24.48s`；
- 新跑 Stage 4 T=2048 smoke: `stage4_all=0.482517 ms`，与历史
  `0.475867 ms` 同量级；此次 smoke 使用 warmup=2/repeat=5，因此只验证路径未被
  audit 改动，不替代正式 benchmark。

所有命令、原始 CSV、HSACO/ISA、rocprof trace 和最终 JSON 均在
[`codex_qwen_bt64_solve_rootcause_stage5a`](codex_qwen_bt64_solve_rootcause_stage5a) 下。
最终可机器读取结论是
[`final_decision.json`](codex_qwen_bt64_solve_rootcause_stage5a/final_decision.json)。
# Qwen gfx942 BT64 Hierarchical Solve Stage 5B 完成报告

> 后续接入已完成：详见
> [Stage 5C 集成报告](qwen_gfx942_bt64_hierarchical_solve_stage5c_integration_report.md)。
> S0 在当前最高层 Stage 4 BT64 full path 的 T=2048 三会话中位数为
> `0.474906 ms -> 0.454215 ms`；v18 和 production dispatch 保持不变。

## 结论

Stage 5B 已完成到可接入的 standalone S0 solve：新增的 opt-in BT64
hierarchical FP32 solve 在 gfx942/MI300 上通过 source feature、数学正确性、
W/U consumer、性能、LDS、scratch 和 ISA 门。

- T=2048：v1 `0.030646 ms`，同一 benchmark harness 的 v18 `0.127149 ms`，
  speedup `4.15x`。
- v1 对 v6 的最大 solve 误差为 `3.73e-08`，最大 residual inf 为 `3.73e-08`。
- 256-thread CTA、8 KiB LDS、scratch `0 B`、VGPR `44`、AccVGPR `4`。
- ISA 实际含有 `v_mfma_f32_16x16x4_f32`，没有 BF16/F16 16x16x16 fallback。

v18 和任何 production dispatch 都没有改动。当前 v1 仍是 opt-in standalone
API，尚未改写 BT64 full pipeline；因此“可接入”不等于“已成为 production”。

## 1. 前置门与改动范围

历史 Stage 5B 被高层 API 门挡住：硬件和 Triton 已能用
`v_mfma_f32_16x16x4_f32`，但 AveLang source 没有对应 intrinsic。前一小步已
在 `amdgpu_mfma_signatures.h`、ROCDL wrapper、语言参考和 C++ test 中补齐
`al.amdgpu.mfma_16x16x4_f32_f32`。其独立 ISA/JIT 证据见
[`qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md`](qwen_gfx942_fp32_mfma16_intrinsic_enablement_report.md)。

本阶段新建而非替换基线的文件：

- `vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/test_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/bench_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `vllm_compare/profile_qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`
- `repro_fp32_mfma16x4_gemm_mapping.py`

没有改动：v18、v24、full production dispatch、AMDGPU RA、asm recurrence。

## 2. 数学与实现

输入/输出严格保持现有 solve ABI：contiguous FP32 `[1,T,8,64]`，`T > 0` 且
`T % 64 == 0`。每个 CTA 负责一个 `(chunk, value_head)`，求：

```text
X = (I + A)^-1
```

将 64x64 strict-lower matrix 切为 4x4 个 16x16 blocks。四个 diagonal block
保留 v6/v18 的 FP32 行递推；六个 lower block 用下列 DAG：

```text
X21 = -X22 A21 X11            X32 = -X33 A32 X22
X43 = -X44 A43 X33
X31 = -X33 (A31 X11 + A32 X21)
X42 = -X44 (A42 X22 + A43 X32)
X41 = -X44 (A41 X11 + A42 X21 + A43 X31)
```

每个 16x16 product 由四次 FP32 16x16x4 MFMA 累积。MFMA mapping 先由随机
16x16 GEMM repro 验证：

```text
A lane: A[lane & 15, 4*piece + (lane >> 4)]
B lane: B[4*piece + (lane >> 4), lane & 15]
C row:  4*(lane >> 4) + acc_i
C col:  lane & 15
```

该 repro 对 `torch.matmul` 的 max/mean abs 均为 `0`。

### LDS 计划

`x[7,16,16]` 保存四个 diagonal 与三个一阶 lower blocks，大小为 `7 KiB`；
`work[16,16]` 在 diagonal phase 复用为 row snapshot、在 block-DAG phase
复用为一个 FP32 matrix product workspace，大小为 `1 KiB`。总计正好 `8 KiB`。

X33、X43、X32 在其最后一个 consumer 完成后分别被 X31、X42、X41 复用，但
它们先写回全局输出。因此不会丢失最终矩阵元素，也不会扩大 LDS。

## 3. 实施中发现并修复的两个问题

1. 初版在 diagonal recurrence 前先写入单位对角。v6/v18 的递推对象是
   `M=-A`，单位阵必须在 strict-lower rows 全部完成后再加；提前加入会把第一条
   sub-diagonal 的贡献加两次。修正后 diagonal block 与 v18 对齐。
2. 初版将 `X42` 写到 0-based column block 2，即 `X43` 的全局位置，覆盖了
   正确的 X43，同时遗漏真正的 column block 1。隔离的“identity diagonal +
   only A43 nonzero”案例把它定位为 output offset bug，而不是 MFMA lane mapping
   或 wave control bug。改为 `BLOCK + lane_col` 后，完整 block DAG 通过。

这两项都记录在新测试/实现中，没有以放宽 tolerance 或 fallback 掩盖。

## 4. 正确性

完整原始输出在
[`standalone/s0_v1_pytest_results.txt`](codex_qwen_bt64_hierarchical_solve_stage5b/standalone/s0_v1_pytest_results.txt)。

| 输入 / T | max abs vs v6 | residual inf | 额外检查 |
|:--|--:|--:|:--|
| random strict-lower / 64 | `2.24e-08` | `2.24e-08` | multi-wave base case |
| random strict-lower / 128 | `1.86e-08` | `2.24e-08` | two chunks |
| random strict-lower / 512 | `3.73e-08` | `2.98e-08` | eight chunks |
| real KKT / 64 | `2.24e-08` | `2.24e-08` | also checks v18 |
| real KKT / 512 | `2.98e-08` | `2.98e-08` | also checks v18 |

下游 W/U consumer gate 也通过：T=512 时 `W` max abs `7.45e-09`、`U` max abs
`2.38e-07`。总 pytest 结果为 `7 passed in 13.53s`。

## 5. Solve-only 性能

每个数字是 HIP-event median，warmup=5、repeat=20，同一输入和 wrapper 计时
方式下比较 v1、v18、v6。原始 JSON 在
[`standalone/s0_v1_benchmark.json`](codex_qwen_bt64_hierarchical_solve_stage5b/standalone/s0_v1_benchmark.json)。

| T | v1 ms | v18 ms | v6 ms | v1 vs v18 |
|--:|--:|--:|--:|--:|
| 512 | `0.030185` | `0.122422` | `12.259758` | `4.06x` |
| 1024 | `0.028663` | `0.122141` | `18.318080` | `4.26x` |
| 2048 | `0.030646` | `0.127149` | `20.855812` | `4.15x` |

Stage 5A 的 standalone target 是 T=2048 `< 0.060 ms`；v1 以 `0.030646 ms`
通过。历史 vLLM body-only 数据为 `0.035613 ms`，但它不与本次完全同一 session，
所以这里只把它当参考，不宣称正式 end-to-end 胜过 vLLM。

## 6. T=2048 rocprof 与 ISA

Targeted rocprof 使用 7 个 dispatch，trace median 为 `13.700 us`。原始 CSV/HSACO
在 [`rocprof_s0_v1`](codex_qwen_bt64_hierarchical_solve_stage5b/rocprof_s0_v1)。

| metric | v1 S0 | Stage 5A v18 historical |
|:--|--:|--:|
| trace median | `13.700 us` | `110.224 us` |
| workgroup / waves | `256 / 4` | `128 / 2` |
| CTA count | `256` | `256` |
| LDS / scratch | `8192 B / 0 B` | `17920 B / 0 B` |
| VGPR / AccVGPR / SGPR | `44 / 4 / 32` | `96 / 128 / 112` |
| OccupancyPercent | `5.138` | `4.733` |
| SQ_INSTS_MFMA | `16384` | `0` |
| SQ_INSTS_VALU | `637440` | `2695424` |
| SQ_INSTS_SALU | `123904` | `2535680` |
| SQ_INSTS_VMEM | `40960` | `32768` |
| SQ_INSTS_LDS | `212992` | `1177856` |

v1 每 CTA 动态执行 64 条 MFMA，`64 * 256 = 16384`，与 counter 一致。静态
assembly 有 56 处 mnemonic，因为 wave0/wave1 的部分 Level-1 代码共享同一段
动态分支；这不改变动态执行数。

HSACO 反汇编明确包含：

```text
v_mfma_f32_16x16x4_f32 a[0:3], ...
```

`v_mfma_f32_16x16x16_*` 匹配数为 `0`。这证明 S0 使用的是新增的 FP32
intrinsic，而不是 BF16/F16 fallback。

## 7. 决策与下一步

S0 满足 Stage 5B standalone gate：source API 可用、正确、consumer-preserving、
scratch=0、LDS=8 KiB、T=2048 小于 0.060 ms。因此下一步可以将这一 **opt-in**
solve 接入 `qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py`，随后重跑该 full
pipeline 的冻结 correctness matrix 和 stage timing。

不要修改 v18 或 production dispatch；BT64 full pipeline 仍有已知 generic KKT、W/U
和 chunk-o 瓶颈，solve 的单独成功并不等价于 full-path 性能成功。

## 8. 复现

完整命令见
[`commands_s0_v1.sh`](codex_qwen_bt64_hierarchical_solve_stage5b/commands_s0_v1.sh)。
机器可读决策见
[`s0_v1_final_decision.json`](codex_qwen_bt64_hierarchical_solve_stage5b/s0_v1_final_decision.json)。
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
# Qwen GDN gfx942 BT64 Downstream State-Coupling Stage 5D Report

## 1. 结论摘要

Stage 5D 是 audit-only 实验。本轮没有修改 solve、KKT、W/U、chunk-o、
asm-v0、compiler 或 production dispatch。

现有数据可以高置信度确认一个操作层面的根因：

> v18 solve 会留下一个有利于紧随其后的 W/U -> asm recurrence -> chunk-o 的
> 瞬态执行状态；hierarchical_fp32_v1、no-solve 和短 dummy predecessor 不会。

这个状态与 solved tensor 的数值内容无关，也不是下游可见 pointer alignment 的差异。
在 T=2048、下游读取同一份 bitwise canonical data 和同一个 consumer pointer 时：

- v18 predecessor 后的连续 tail：`0.267518 ms`
- v1 predecessor 后的连续 tail：`0.332073 ms`
- v1 tail penalty：`64.555 us`

将同一个 tail 预热一次，差距缩小到 `0.120 us`；在 tail 前执行相同的 512 MiB
cache/execution-state 扰动，差距为 `-0.080 us`，也就是测量分辨率内相同。这个结果
强烈支持“前驱造成的 cache residency、频率/功耗爬升或二者组合”这一类机制，但本轮
没有拿到 cache counter 和 clock telemetry，因此不能把最终根因写成确定的 L2 cache，
也不能写成确定的 clock ramp。

Stage 5C 中 solve 单阶段约节省 `96.583 us`，无分段 event 的 public full 只节省
`18.808 us`，相差约 `77.775 us`。本轮连续 tail 单 event 实际复现了 `64.255 us`
的下游抵消，说明抵消并非主要由分段 event 伪造；剩余部分来自运行条件、测量扰动和
尚未关闭的 direct common solve-output pointer/硬件状态变量，不能强行精确分摊。

唯一 Stage 5E 建议是：

> 让 v18 和 hierarchical_fp32_v1 直接写入同一个固定预分配 output buffer，再运行
> 完整 fixed graph。

这是当前最小、低风险、可证伪的实验。它不改变数学、W/U、asm、compiler 或生产路径，
并关闭当前 copy-to-canonical 控制仍未关闭的 solve-store 物理地址/cache-set 变量。

## 2. 审计范围与工作区安全

审计目录：

`codex_qwen_bt64_downstream_state_coupling_stage5d/`

新增内容只有独立 harness、测试、profile/telemetry 驱动、CSV 和报告。未修改：

- v18 solve；
- hierarchical_fp32_v1 solve；
- Stage 4 cumsum/KKT/W/U/chunk-o；
- asm-v0 HSACO、symbol、ABI、grid/WG；
- Avelang compiler、generic lowering、LLVM/AMDGPU RA；
- production 默认 dispatch；
- v23/v24/v26/v27/v28/v29 production baseline。

开始前的工作区状态保存在 `git_before.txt` 和 `git_before.diff`。本轮没有执行
`git reset`、`git clean` 或创建 commit。

## 3. 冻结执行图

两条审计图均为八个 dispatch：

```text
chunk cumsum
  -> KKT
  -> solve
  -> W
  -> U
  -> gfx942 asm-v0 recurrence
  -> chunk-o
  -> BF16 output cast
```

GRAPH-A 使用 `_qwen_gdn_solve_kernel_v18_parallel`，workgroup 128。

GRAPH-B 使用 `_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1`，workgroup 256。

除 solve symbol 和 solve workgroup 外，两图使用相同的输入、stream、预分配输出、
dispatch 数量和顺序。W/U/chunk-o 在同一个 Python process 内调用相同 JIT function、
相同 constexpr specialization，因此使用同一 JIT cache entry。asm-v0 使用同一外部
HSACO，SHA256 为：

```text
eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226
```

T=2048 的关键 launch：

| stage | grid | WG | dynamic LDS |
|:--|--:|--:|--:|
| cumsum | 256 | 1 | 0 |
| KKT | 4096 | 64 | 0 |
| solve A | 256 | 128 | 0 |
| solve B | 256 | 256 | 0 |
| W | 2048 | 256 | 0 |
| U | 2048 | 256 | 0 |
| asm recurrence | `(4,8,1)` | 256 | 57344 B |
| chunk-o | 2048 | 256 | 0 |
| BF16 cast | identical | identical | identical |

逐 dispatch 的 tensor shape、dtype、stride、pointer、storage offset、依赖关系记录在
`graph_a_dispatch_map.json` 和 `graph_b_dispatch_map.json`。

限制：本轮没有额外导出 W/U/chunk-o JIT binary 的字节 hash；“same code object”的
证据是同 process、同 JIT callable、同 constexpr specialization 和同 cache key。
asm-v0 的 HSACO hash 是直接记录的。

## 4. Correctness 与 solve 输出审计

已实际执行的 smoke：

| T | A/B output max abs | A/B final-state max abs | canonical-vs-A output/state |
|--:|--:|--:|:--|
| 64 | `0` | `0` | `0 / 0` |
| 512 | `2.44140625e-4` | `2.7179718e-5` | `0 / 0` |
| 8192 | `2.44140625e-4` | `1.5418977e-5` | `0 / 0` |

T=2048 solved tensor A/B 统计：

| metric | value |
|:--|--:|
| max abs | `2.980232e-8` |
| mean abs | `3.80825e-10` |
| bitwise mismatch | `317998 / 1048576` |
| NaN | 0 |
| Inf | 0 |
| subnormal | 0 |

两种 solve 输出均为 contiguous FP32，shape、stride、storage offset 完全相同。微小数值
差异确实存在，所以不能仅凭误差小就宣布数据无关；数据主因是由 canonical-data
控制实验排除的。

Stage 5C 六项 integration、Stage 4 29 项回归、asm-v0/external bridge 回归没有在本轮
重新执行。原因是完成 benchmark/control 后，平台拒绝新的 Docker execution；这些项在
`pytest_results.txt` 中明确标为 N/A，不沿用旧结果冒充本轮结果。

## 5. 无分段 Event 的完整 A/B

固定预分配审计 harness 在完整 graph 边界放一个 HIP event，graph 内不插 event。
每个 T 使用 warmup=20、repeat=200、5 sessions，并覆盖 ABAB、BABA 和随机 ABBA。

| T | v18 full ms | v1 full ms | v1 gain us |
|--:|--:|--:|--:|
| 512 | `0.277933` | `0.242881` | `35.052` |
| 1024 | `0.357131` | `0.316590` | `40.541` |
| 2048 | `0.548115` | `0.470680` | `77.435` |
| 4096 | `0.724257` | `0.630296` | `93.961` |
| 8192 | `1.373361` | `1.178431` | `194.930` |
| 16384 | `2.611822` | `2.336312` | `275.509` |

T=2048 固定-buffer 数据存在明显 order/session 异常：四个 session 的 v1 gain 为
`75--96 us`，随机 ABBA session 却反转为 v18 `0.480975 ms`、v1 `0.616737 ms`。
因此不能把上表 T=2048 的 `77.435 us` 当作生产结论。

更可信的 Stage 5C public/no-segment A/B 是：

| graph | T=2048 ms |
|:--|--:|
| v18 | `0.474385` |
| v1 | `0.455577` |
| v1 gain | `18.808 us` |

完整 raw samples 在 `full_ab_raw_samples.csv`，order 分析在
`order_effect_analysis.md`。

## 6. 连续 Downstream Tail

tail 的一个 HIP event 连续包围：

```text
W -> U -> asm recurrence -> chunk-o -> BF16 cast
```

solve 在 event 之前执行，tail 内没有分段 event。

### Solve + Tail Boundary

| T | v18 ms | v1 ms | v1 gain us |
|--:|--:|--:|--:|
| 512 | `0.228479` | `0.200939` | `27.540` |
| 2048 | `0.393144` | `0.351943` | `41.201` |
| 8192 | `1.226943` | `1.032995` | `193.948` |
| 16384 | `2.354199` | `2.067292` | `286.907` |

### Tail Only

| T | after v18 ms | after v1 ms | v1 tail penalty us |
|--:|--:|--:|--:|
| 512 | `0.100089` | `0.176903` | `76.814` |
| 2048 | `0.267938` | `0.332193` | `64.255` |
| 8192 | `0.991834` | `1.004613` | `12.779` |
| 16384 | `2.220320` | `2.252848` | `32.528` |

T=2048 的 ABAB、BABA、random order 都稳定复现约 `64 us` penalty。这证明真实下游
抵消在无内部 event 的连续 tail 中存在，不能归因于 Stage 5C 分段 event 的简单相加。

## 7. Canonical Data 控制

下游始终读取同一个预分配、bitwise 固定的 `solved_canonical`：

- A：先执行 v18，丢弃输出，再消费 canonical；
- B：先执行 v1，丢弃输出，再消费 canonical；
- C：不执行 solve，直接消费 canonical；
- D：执行短 dummy predecessor，再消费 canonical。

| T | after v18 ms | after v1 ms | v1 penalty us |
|--:|--:|--:|--:|
| 2048 | `0.267518` | `0.332073` | `64.555` |
| 8192 | `0.991954` | `1.005834` | `13.880` |

T=2048 no-solve 是 `0.334737 ms`，dummy predecessor 是 `0.343390 ms`，均更接近
v1，而不是 v18。这说明有利状态是 v18 predecessor 特有的，不是 canonical 数据内容
本身，也不是“任意 predecessor 都能预热”的简单现象。

## 8. Same Pointer 与 Alignment 控制

当前 same-pointer 控制的准确语义是：

1. 两种 solve 仍分别写自己的 output buffer；
2. solve 后把真实输出 copy 到同一个 canonical consumer buffer；
3. copy 在 tail event 外；
4. downstream 只读取同一个 data pointer。

T=2048：

| predecessor | tail ms |
|:--|--:|
| v18 + copy-to-canonical | `0.267838` |
| v1 + copy-to-canonical | `0.336020` |
| v1 penalty | `68.181 us` |

copy 成本约 `9.8--10 us`，不计入 tail。这个结果排除了 downstream consumer pointer
和 visible alignment 作为主因，但没有关闭 solve 自身写入哪个物理 output pointer 这一
变量。因此不能把它误写成“两种 solve 已直接写同一 pointer”。

T=2048/8192 的 solved A、solved B、canonical 均为：

- contiguous FP32；
- storage offset 0；
- address mod 16/64/128/256 = 0；
- address mod 4 KiB/64 KiB = 0。

可见低位 alignment 不是主因；更高物理地址/cache-set 交互仍未由 direct common-out
实验关闭。timed region 内无 allocation，allocator state 没有观察到 A/B graph 差异。

## 9. Cache/Execution-State 控制

所有控制都让 downstream 消费相同 canonical pointer 和相同数据：

- none：solve 后直接 tail；
- warm：tail 前先运行一次同样的 tail；
- controlled perturbation：tail 前执行 512 MiB tensor add；
- prime：用 reduction 访问相关输入。

512 MiB 操作只称 controlled cache/execution-state perturbation，不宣称精确清空某一级
cache。

| T | control | after v18 ms | after v1 ms | v1 penalty us |
|--:|:--|--:|--:|--:|
| 2048 | none | `0.267898` | `0.336520` | `68.622` |
| 2048 | warm | `0.267638` | `0.267758` | `0.120` |
| 2048 | 512 MiB perturb | `0.274528` | `0.274448` | `-0.080` |
| 2048 | reduction prime | `0.280777` | `0.356189` | `75.412` |
| 8192 | none | `0.993256` | `1.009660` | `16.404` |
| 8192 | warm | `0.993717` | `0.993937` | `0.220` |
| 8192 | 512 MiB perturb | `1.011784` | `1.012064` | `0.280` |
| 8192 | reduction prime | `1.001168` | `1.020457` | `19.289` |

结论边界：

- warm 和大 buffer perturb 都把 A/B 差距压到 sub-us；
- reduction prime 没有压平差距；
- 这强烈支持可被 GPU 工作负载重置的瞬态执行状态；
- 没有 cache counter，不能确定是 L2/TCC/TCP 中哪一级；
- v18 本身比 v1、dummy/no-solve 更长，warm/大 buffer 也是长 workload，因此 GPU
  clock/power ramp 仍是同样合理的解释；
- 最严谨表述是 cache residency 与 clock/power ramp 尚未分离。

## 10. Profiling 与时钟项的状态

完成 benchmark/control 后，平台拒绝了新的 Docker execution，并返回 usage-limit
blocker。因此以下项目未执行，全部标为 N/A：

- 当前 gfx942 可用 cache counter 查询；
- GRAPH-A/B whole-graph rocprof timeline；
- W/U、asm、chunk-o 的 A/B trace/counter；
- dispatch gap、CU/wave 分布；
- cache/TCC/TCP counter；
- amd-smi/rocm-smi clock、memory clock、power、temperature 采样；
- rocprof instrumentation overhead 的本轮重测。

对应 CSV 已保留 N/A schema，`rocprof/README.md` 记录 blocker。未使用不存在的 counter
名称，也没有根据旧 profile 编造新 A/B counter。

由于 graph contract 已证明下游 kernel symbol/specialization 相同，静态 code object
资源应相同；但动态 instruction count、trace duration、cache behavior 和 dispatch gap
仍需要实际 rocprof 才能回答。本报告把这些字段保持为 `null`。

## 11. Measurement Perturbation

Stage 5C 同输入数据：

| timing mode | v18 ms | v1 ms | v1 gain us |
|:--|--:|--:|--:|
| full single event | `0.474385` | `0.455577` | `18.808` |
| per-stage events summed | `0.490008` | `0.485421` | `4.587` |

分段 event 相对 full single event 增加：

- v18：`15.623 us`
- v1：`29.844 us`
- 对 A/B gain 的扭曲：`14.221 us`

因此 Stage 5C 的 “W/U +16.865 us、asm +75.352 us” 只能用于定位 downstream
组合状态，不能当作真实、可加的 `92.217 us` 分解。另一方面，Stage 5D 的连续 tail
单 event 复现了 `64.255 us`，所以 instrumentation 不是抵消的主因。

rocprof 相对 HIP-event duration 的扰动本轮 N/A。

## 12. 根因矩阵摘要

| candidate | status | confidence | explanation |
|:--|:--|:--|:--|
| solved 数值内容 | rejected as dominant | high | canonical bitwise input 仍保留差距 |
| downstream pointer/alignment | rejected as dominant | high | 同 consumer pointer、同 alignment 仍保留差距 |
| allocator pool | rejected as dominant | medium-high | timed region 无 allocation，图结构相同 |
| cache residency | supported, not isolated | medium | warm/大扰动压平差距；无 cache counter |
| clock/power ramp | unresolved, supported alternative | medium | 长 predecessor/扰动可能拉高频率；无 telemetry |
| dispatch gap | unresolved | low/unknown | whole-graph timeline N/A |
| predecessor occupancy/resource state | supported at operational level | high | v18 特有；no-solve/dummy/v1 均慢 |
| stage-event instrumentation | not dominant | high | continuous tail 复现抵消 |
| hidden dispatch | rejected | high | frozen graph dispatch count/order 相同 |
| downstream code-object difference | rejected | high | 同 process 同 JIT specialization；同 asm HSACO |
| solve-store physical pointer/cache set | unresolved | medium | copy-to-canonical 未让 solve 直接写同一 out |

详细证据见 `root_cause_matrix.md` 和 `root_cause_matrix.json`。

## 13. 对 25 个核心问题的回答

1. **两条 full 图除 solve 外是否一致？** 是。八个 dispatch 中只有 solve symbol/WG
   不同。
2. **W/U 与 asm 是否使用相同 symbol/code object？** 是。W/U/chunk-o 是同 process
   同 JIT function/specialization；asm HSACO hash 完全相同。JIT binary byte hash N/A。
3. **launch 数量是否相同？** 是，均为八个。
4. **W/U/asm grid、WG、LDS 是否相同？** 是。
5. **solve output shape/dtype/stride 是否相同？** 是，contiguous FP32，storage offset 0。
6. **solve output 地址是否不同？** 原始 A/B 分别预分配，地址不同。
7. **alignment/cache-set 低位是否不同？** 检查的 mod16 到 mod64KiB 均为 0；物理
   cache-set 映射 N/A。
8. **timed region 内是否有不同 allocation？** 没有。
9. **allocator pool 是否不同？** 未观察到图相关差异；不能从公开 API 得到所有内部
   allocator/cache-set 信息。
10. **消费 bitwise 相同 solved tensor 后差距是否存在？** 是，T=2048 为
    `64.555 us`。
11. **两 solve 直接写同一预分配 pointer 后差距是否存在？** N/A，尚未执行；现有控制
    是 solve 后 copy 到同一个 consumer pointer。
12. **受控大 buffer 扰动后差距是否消失？** 是，T=2048 剩 `-0.080 us`。
13. **相同 reduction prime 后差距是否消失？** 否，T=2048 仍为 `75.412 us`。
14. **warm/cold 差距？** none `68.622 us`，warm `0.120 us`，512 MiB perturb
    `-0.080 us`。
15. **W/U 动态指令是否改变？** N/A；相同 code object 已确认，动态 counter 未采集。
16. **asm 动态指令是否改变？** N/A；相同 HSACO 已确认，动态 counter 未采集。
17. **cache hit/miss counter 是否改变？** N/A，counter 查询/rocprof 受 quota 阻塞。
18. **occupancy/wave/CU/gap 是否改变？** N/A。
19. **clock/power 是否存在稳定差异？** N/A，未得到 telemetry。
20. **rocprof/分段观察能否由无 rocprof tail 复现？** 可以，连续 tail 单 event 复现
    `64.255 us`。
21. **per-stage event 扰动？** 对 v18/v1 full 分别增加 `15.623/29.844 us`，扭曲
    A/B gain `14.221 us`。
22. **约 78 us 抵消主因？** 高置信度是 predecessor-induced transient downstream
    execution state；具体 cache 与 clock 机制尚未分离。
23. **是否可能有共同原因？** 是，cache residency、频率/功耗爬升以及未关闭的 solve
    store physical pointer 可能共同作用。
24. **下一步？** 选择 A：两个 solve 直接写同一个固定预分配 out pointer；不是 fusion。
25. **需要修改 compiler/assembly 吗？** 没有证据支持，当前不需要。

## 14. Stage 5E 唯一建议

实现一个 audit-only fixed-out harness：

```text
v18 solve -------------------> solved_common
hierarchical_fp32_v1 solve --> solved_common
                                |
                                +-> unchanged W/U -> asm -> chunk-o
```

必须让 solve kernel 本身直接接收并写同一个 `solved_common.data_ptr()`，而不是先写不同
buffer 再 copy。继续保持：

- 相同 KKT 输入；
- 相同 stream；
- 相同 graph order；
- 无内部 stage event；
- 相同 downstream pointer；
- 数学和 production 不变。

如果 direct-common-out 仍保留 tail 差距，solve-store pointer/cache-set 可排除，随后只读
采集 clock telemetry 与 whole-graph cache/dispatch counters，区分 cache 与频率状态。
如果差距消失，则 pointer/cache-set 是可行动根因。

预计可恢复收益保持 `N/A`。当前可观察上界是 T=2048 约 `64.255 us` tail penalty，
但在 direct control 前不能把它承诺为 full 收益。

## 15. 复现与证据路径

命令：

`codex_qwen_bt64_downstream_state_coupling_stage5d/commands.sh`

主数据：

- `full_ab_benchmark.csv`
- `full_ab_raw_samples.csv`
- `downstream_tail_benchmark.csv`
- `downstream_tail_raw.csv`
- `canonical_data_control.csv`
- `same_pointer_control.csv`
- `cache_state_control.csv`
- `pointer_alignment.csv`
- `solved_data_statistics.csv`
- `measurement_perturbation.csv`

结构化结论：

- `root_cause_matrix.json`
- `stage5e_decision.json`
- `final_decision.json`

未执行项和原因：

- `rocprof/README.md`
- `whole_graph_trace_analysis.md`
- `clock_power_interpretation.md`
- `pytest_results.txt`

## 16. 最终状态

- `audit_only=true`
- `ready_for_stage5e=true`
- `ready_for_production=false`
- dominant operational cause：`predecessor-induced transient downstream execution state`
- exact hardware mechanism：unresolved between cache residency, clock/power ramp, and
  direct solve-store physical-address interaction
- compiler/assembly modification：不推荐

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
# Qwen gfx942 BT64 Chunk-O Ownership Stage 6B

## 结论

**O0 是一个正确、真实减少重复工作的实验，但未通过晋级门槛；Stage 6B 不接入 full graph，也不进入 Stage 6C direct-BF16 output。**

O0 将 T=2048 的 chunk-o CTA 从 2048 降到 512，和当前 vLLM 的 CTA 数相同；它还将动态
MFMA、VMEM、LDS 指令分别降为当前的 `41.1%`、`39.4%`、`38.7%`。但它为保持跨 V16
score 复用而引入 8 KiB score LDS lifetime，使 `Accum_VGPR` 从 64 增至 188，LDS block
从 27136 B 增至 33280 B，occupancy 从 17.90% 降至 8.86%。最终 T=2048 body 仅从
`0.076073 ms` 到 `0.062012 ms`，即 `1.227x`，不足 `1.5x` gate。

O1（唯一允许的第二个试验，V64 改 V32）是负结果：更多 CTA、更多 MFMA/LDS/barrier，
T=2048 反而慢于 current。所有生产路径、FP32 output staging、独立 BF16 cast、KKT、solve、
W/U、asm recurrence、compiler 和 assembly 均未修改。

## 1. 实际 Ownership

| 实现 | T=2048 CTA | CTA/chunk-head | tile | WG / waves | 说明 |
|---|---:|---:|---|---|---|
| Stage 4 current | 2048 | 8 | token16 x V16 | 256 / 4 | 一个 wave 管一个 token16 row；Q/K score 在八个 V16 CTA 重复 |
| O0 | 512 | 2 | token64 x V64 | 256 / 4 | 四 waves 管四个 token16 row；同一 CTA 复用 lower score 到四个 V16 subtile |
| O1 | 1024 | 4 | token64 x V32 | 256 / 4 | O0 的唯一变化：两个 V16 subtile |
| vLLM | 512 | 2 | token64 x V64 | 256 / 4 | 实测 autotune: `BK=32,BV=64,num_warps=4,num_stages=2` |

当前 Avelang 的 lane mapping 和 O0/O1 相同：`lane_col=lane&15` 选择 V16/K16 列，
`lane_group=lane>>4` 与 `r=0..3` 选择四行，`wave_id` 选择 token16 row。vLLM 的精确
lane fragment permutation 由 Triton block-dot lowering 生成，不能从 Python 源码可靠恢复，
因此本报告不伪造 lane-level 映射。

vLLM 的真实 Python kernel `chunk_fwd_kernel_o` 在一个 CTA 内构造 `b_o:[64,64]` 和
`b_A:[64,64]`，完成 QH、QK、causal mask、`b_A @ V-new` 并直接写最终 output；没有
partial output global tensor、atomic 或中间 global accumulation。

## 2. 为什么 current 有 4x CTA 和大量重复工作

当前 grid 是 `NT * 8 heads * 8 V16 = 2048`。一个 chunk/head 的 64x128 Q 和 K block
在八个 CTA 被读取/变换；O0 是两个 V64 CTA，因此 Q/K 读取次数从 8 降至 2。H 和 V-new
是 V-specific，128-wide 的唯一 V16 ranges 仍必须覆盖一次，并不能随 CTA 数等比例消失。

从 source-level MFMA16 tile 模型看，数值所需的工作为 inter=256、lower-QK=80、
intra=80，共 416 个 tile calls/chunk-head。Stage 4 schedule 对每个 V16 CTA 重做完整
QK，模型为 1360；O0 用 lower 10 个 score tiles 且只在 V64 级重复，模型为 496。
该模型解释复用来源，但不能与 `SQ_INSTS_MFMA` 逐项相等：最终硬件 MFMA 指令数取决于
MFMA operand lowering、静态展开和硬件计数定义。资源决策使用下表的实际 rocprof 值。

## 3. O0/O1 实现

- O0: `_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o0`，wrapper
  `qwen_gdn_chunk_o_bt64_ownership_o0`。
- O1: `_qwen_gdn_chunk_o_bf16_kernel_bt64_ownership_o1`，wrapper
  `qwen_gdn_chunk_o_bt64_ownership_o1`。

O0 先 stage Q `[64,128]`，只计算 ten lower `[16,16]` score tiles，并保存
`score_decay_bf16[4,4,16,16]`。之后四次顺序执行已有、已验证的 V16 inter/intra MFMA
microtile，output FP32 一次写回。没有改变 gate、scale、数值顺序、输出 dtype 或独立 cast。

O1 仅将 V64 改为 V32；它不是额外优化集合，也不是 autotune sweep。

## 4. 正确性

GPU pytest 结果：`108 passed in 31.54s`。

- O0/O1 对 Stage 4 FP32 staging 在 T=64/128/512/2048 的 random、zero-H、zero-V-new、
  inter-only、intra-only、small、high、cancellation 全部 `max_abs=0, mean_abs=0`。
- T=8192 random/cancellation/high 也全部 bit-exact。
- 四个 cross-token16 source 范围和所有 V16 boundary `[0:16,...,112:128]` 全部 bit-exact。
- Stage 4 对 vLLM `chunk_fwd_o` 的最大直接 body 差异为 `1.1281809e-05`；O0/O1 bit-exact
  于 Stage 4，因此并未放大该实现差异。
- 仅用于语义 smoke 的 O0/O1 full wrapper 在 T=64/512 对 current 的 `output_fp32`、public
  output、final state 都是零差异。

完整 Stage 6B full correctness matrix 是 `N/A_gate_not_met`：性能 gate 未过，按预注册规则
没有把候选作为 selected full graph 运行。这个 N/A 不是 correctness failure，也不能被当作
production correctness 通过。

未改动上游的回归也已单独执行：Stage 4 nonrecurrence KKT/W-U/chunk-o 与 hierarchical
solve 共 `36 passed in 30.82s`；冻结 asm-v0 bridge 共 `4 passed in 24.87s`。

## 5. Body Timing

相同输入、current stream、预分配 CUDA/HIP graph replay、warmup=20、repeat=100、5 sessions、
平衡顺序 `current,O0,O1,vLLM,vLLM,O1,O0,current`：

| T | current ms | O0 ms | O1 ms | vLLM ms | O0 gain vs current | O0-vLLM gap us |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.027080 | 0.034090 | 0.027320 | 0.017546 | -7.010 us | 16.544 |
| 1024 | 0.043624 | 0.036935 | 0.044506 | 0.020390 | 6.689 us | 16.545 |
| 2048 | 0.076073 | 0.062012 | 0.077395 | 0.029885 | 14.061 us | 32.127 |
| 4096 | 0.126629 | 0.113248 | 0.128391 | 0.052318 | 13.381 us | 60.930 |
| 8192 | 0.240077 | 0.192887 | 0.243682 | 0.088090 | 47.190 us | 104.797 |
| 16384 | 0.450830 | 0.384932 | 0.489127 | 0.178184 | 65.898 us | 206.748 |

| series | intercept ms | slope us/chunk |
|---|---:|---:|
| current | 0.017821 | 1.701165 |
| O0 | 0.017755 | 1.423759 |
| O1 | 0.013260 | 1.846943 |
| vLLM | 0.009919 | 0.648614 |
| current-vLLM | 0.007902 | 1.052551 |
| O0-vLLM | 0.007836 | 0.775145 |

O0 的长序列 slope 确实改善了 `0.277406 us/chunk`，但仍不满足 `<=0.65 us/chunk`。T=512
变慢表明更宽 CTA 的固定 LDS/barrier 成本没有被短文本摊销。

## 6. T=2048 rocprof 资源

| metric | current | O0 | O1 | vLLM Stage 6A full |
|---|---:|---:|---:|---:|
| median trace us | 67.060 | 52.919 | 68.202 | 14.662 |
| CTA | 2048 | 512 | 1024 | 512 |
| MFMA | 458752 | 188416 | 229376 | 81920 |
| VALU | 12918784 | 4054016 | 7108608 | 1728512 |
| SALU | 1495040 | 555008 | 860160 | 337920 |
| VMEM | 851968 | 335872 | 507904 | 71680 |
| LDS inst | 1343488 | 520192 | 712704 | 245760 |
| VGPR / AccVGPR / SGPR | 112 / 64 / 112 | 76 / 188 / 112 | 76 / 188 / 112 | 100 / 36 / 96 |
| LDS block B | 27136 | 33280 | 33280 | 0 |
| scratch B | 0 | 0 | 0 | 0 |
| occupancy | 17.897% | 8.864% | 9.271% | 9.473% |

O0 通过 CTA 4x、MFMA 2.435x、VMEM 2.537x、LDS 2.583x 的降低证明 ownership 改写确实生效；
但它未达到预设的 MFMA>=2.5x、VMEM>=3x，且其 AccVGPR/LDS/occupancy 退化限制了 latency。
O1 仅将 CTA 降为 2x，且 static/dynamic MFMA、DS、barrier 都变多，是明确负结果。

HSACO metadata 的 private segment、VGPR spill、SGPR spill 三者均为零。rocprof 的
`Accum_VGPR_Count` 不等同于 code-object `.agpr_count`，故本报告不将二者一一对应。

## 7. ISA/IR Evidence

每个 candidate 都保存了 ordinary JIT HSACO、AMDGCN disassembly、pre-link BC、disassembled
LLVM IR 和可 replay linker argv：

`codex_qwen_bt64_chunko_ownership_stage6b/{ir,isa}/{current,o0,o1}/`

O0 和 current 都有 56 个 static `v_mfma` text matches，O1 有 80；O0 的 static barrier
从 13 增到 19，O1 到 29。结合动态资源，这与 O0 的 score-cache reuse 和 O1 更密集的
V32 CTA 重复相一致。详见 `static_isa_analysis.md`；没有 compiler 或 assembly 改动。

## 8. Gate and Decision

| mandatory body gate | O0 | result |
|---|---:|---|
| T2048 speedup >= 1.50x | 1.227x | fail |
| T2048 body gap <= 25 us | 32.127 us | fail |
| gap slope <= 0.65 us/chunk | 0.775 us/chunk | fail |
| T8192/T16384 no regression | pass | pass |
| scratch/spill zero | pass | pass |

因此没有 selected variant，`full_integrated=false`，T=512..16384 full times、full gain 和
full gap slope 全部 **N/A**，而非估算值。不能把 14.061 us 的 body gain 直接声称为 full gain。

下一步不是 direct BF16 output/cast fusion，也不是 compiler/assembly：先关闭这个最多两
variant 的 ownership 分支并重新排序 Stage 6A 的剩余 body gaps。O0 的可观但不足收益应保留
为证据，不应静默替换 Stage 4 或 v24 production baseline。

## Evidence

- New code: `vllm_compare/qwen_gdn_bt64_chunko_ownership_stage6b.py`,
  `qwen_gdn_bt64_chunko_ownership_stage6b_o1.py`.
- Test: `vllm_compare/test_qwen_gdn_bt64_chunko_ownership_stage6b.py`.
- Benchmark: `vllm_compare/bench_qwen_gdn_bt64_chunko_ownership_stage6b.py`.
- HSACO dump helper: `vllm_compare/dump_qwen_gdn_bt64_chunko_ownership_stage6b_isa.py`.
- Reproduction commands, raw timing, profiler CSV, IR/ISA, decisions:
  `codex_qwen_bt64_chunko_ownership_stage6b/`.
# Qwen gfx942 BT64 Recurrence Reconciliation: Stage 6R

## 结论

**CASE B：当前 Stage 6A vLLM full graph 选择了不同且更快的 recurrence specialization。** 这不是 asm-v0 退化、也不是 rocprof 把 Avelang 计数重复三次。asm-v0 仍然忠实对应历史 FP32/WG256 Triton lineage；当前 vLLM 则是 BF16 W/U/v_new、BV32、WG128 的新 code object。

## 计数口径

最终三个显式 replay 的每 dispatch 动态 MFMA：asm-v0 `196608`，vLLM `65536`。按 32 chunks 归一化为 `6144` / `2048`；按 chunk-head 为 `768` / `256`。两边都只取最后 3 个 dispatch 的中位数，因此 3x 是真实动态工作差异。

## 当前身份与 ABI

- asm-v0 HSACO: `eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226`
- 当前 vLLM HSACO: `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`
- 历史 Triton original: `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9`
- 二者均为 88 B kernarg、相同 pointer-slot 物理布局；但语义 ABI 不同：asm W/U/v_new 为 FP32，当前 vLLM 为 BF16。
- 当前 vLLM config：`BV=32,num_warps=2,num_stages=2`，WG128，dynamic LDS 40960 B。asm-v0：WG256，dynamic LDS 57344 B。

## ISA 与资源

当前 vLLM 静态 MFMA 为 BF16 `64`、XF32 `0`；asm-v0 为 BF16 `48`、XF32 `96`。
T=2048 fresh rocprof：asm trace `145.737 us`, VGPR/AccVGPR/SGPR `128/192/80`, VMEM/LDS/barrier `91136/588928/44`；vLLM `106.278 us`, `104/160/96`, `58368/305472/32`。两边 scratch 都为 0；完整明细见 `resource_diff.csv` 与 `instruction_diff.csv`。

## 同口径 Body 延迟

| T | asm-v0 FP32 ms | vLLM actual BF16 ms | gap us | actual bridge delta |
|--:|--:|--:|--:|--:|
| 512 | 0.049554 | 0.039199 | 10.356 | -0.207% |
| 2048 | 0.154670 | 0.114570 | 40.100 | -0.052% |
| 8192 | 0.574333 | 0.413274 | 161.059 | 0.049% |
| 16384 | 1.172622 | 0.841410 | 331.212 | 0.017% |

current-vLLM bridge 在 T=512/2048/8192/16384 都对 native vLLM 的 h/v_new/final_state bit-exact；T=2048 bridge 与 native 相差 `-0.052%`，通过 <=5% gate。历史 original、rebuilt 与 asm-v0 在同一 FP32 W/U 输入上也均 bit-exact。

## 解释

Stage 6A 的 `~40 us` T=2048 gap 可以稳定复现。以本轮原生 ABI body 数据拟合，asm/vLLM/gap 的 intercept 为 `0.008181`/`0.009630`/`-0.001449` ms，slope 为 `4.524662`/`3.230977`/`1.293686` us/chunk。Stage6A 在 asm side 将 vLLM BF16 W/U 外部转换为 FP32，asm 随后执行历史 XF32-heavy WG256 body。将 W/U 都设为同一 FP32 值时，当前 vLLM source body 与 asm-v0 数值 bit-exact；但它仍不能把这当成纯 dtype 因果控制，因为输入 dtype 也可能影响 Triton 选择的代码路径。证据只支持：当前 BF16/BV32/two-wave specialization 及其相关 lowering 是主导边界，仍存在较小的代码生成/几何差异。

## 决策

不修改 asm，不修改 compiler，不在本轮继续 chunk-o。允许的唯一下一步是单独的、opt-in 的 current-vLLM BF16 recurrence bridge 全图 contract 集成实验；它不得接入 production，且必须先证明上下游 BF16 W/U/v_new 边界的完整正确性。

## 产物

- `counter_aggregation_{raw,normalized}.csv` / `counter_aggregation_audit.md`
- `current_kernels/`, `kernel_identity_comparison.*`, `isa_diff.md`, `abi_diff.md`, `dtype_layout_diff.md`, `launch_diff.md`
- `standalone_{raw,summary,slopes,correctness}.csv`, `body_native_abi_{comparison,slopes}.csv`, `resource_diff.csv`, `instruction_diff.csv`
- `bridge_probe/`, `tests/pytest_results.txt`, `root_cause_decision.*`, `next_stage_decision.*`, `final_decision.json`
# Qwen gfx942 BT64 Stage 6S: BF16 Recurrence Full-Graph Contract

## Conclusion

Stage 6S completes as Case A. Graph B connects the Stage 6R current-vLLM BF16 recurrence HSACO through explicit, opt-in boundaries. It does not replace the default path. At T=2048, the complete graph improves from 0.335679 ms to 0.309019 ms: a 26.647 us gain with paired bootstrap 95% interval [26.595, 26.700] us. This exceeds the primary 20 us gate.

The full result is not the isolated recurrence gain copied directly into the graph. The recurrence saves 40.099 us at T=2048, while the three materialized boundary casts cost 20.750 us together. At long sequence lengths Graph B remains faster and reduces the native-vLLM gap slope from 4.396726 to 3.284716 us/chunk.

No production selector, asm-v0, current-vLLM HSACO, compiler, KKT, solve, W/U, chunk-o, FP32 output staging, or final BF16 cast changed.

## New Wrapper and Contract

New experimental-only entry:

- qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(...)
- [qwen_gdn_bt64_bf16_recurrence_full_stage6s.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_bf16_recurrence_full_stage6s.py)

The contract is gfx942, B=1, Hk=4, Hv=8, K=V=128, BT=64 and T divisible by 64. The wrapper checks device, dtype, shape, contiguity, HSACO hash and current stream. Every mismatch raises. There is no pointer reinterpretation, no silent fallback and no default-selector change.

| boundary | before | recurrence side | after |
|:--|:--|:--|:--|
| W | FP32 [1,T,8,128] | BF16 | numeric FP32-to-BF16 cast |
| U | FP32 [1,T,8,128] | BF16 | numeric FP32-to-BF16 cast |
| V-new | bridge output BF16 | unchanged chunk-o needs FP32 | numeric BF16-to-FP32 cast |

The recurrence bridge code object is SHA256 632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e, symbol chunk_gated_delta_rule_fwd_kernel_h_blockdim64, grid (4,8,1), workgroup 128, dynamic LDS 40960 B. Graph A keeps the distinct historical asm-v0 SHA256 eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226.

## Actual Dispatch Graphs

| graph | count | logical dispatches |
|:--|--:|:--|
| A, current companion | 8 | cumsum, KKT, hierarchical solve, W, U, asm recurrence, chunk-o, final cast |
| B, Stage 6S | 11 | shared stages plus W cast, U cast, current-vLLM recurrence and V-new cast |
| C, native vLLM | 7 | cumsum, KKT, BF16 fill, inverse solve merge, combined W/U, recurrence, chunk-o |

Graph C is from a new direct T=2048 rocprof capture rather than inferred from Graph B. Its final four complete replay sequences each show these seven dispatches. Native chunk-o writes public BF16 output and has no standalone final cast.

## Correctness

The full matrix covers T=64, 128, 512, 1024, 2048 and 8192 with random nonzero state, zero state, high dynamic, small values, cancellation, neutral gate and multichunk feedback. It contains 12 accepted full cases.

| check | result |
|:--|:--|
| bridge vs native recurrence at identical BF16 boundary | h, V-new and final-state bit-exact |
| Graph-B output | max abs 0.001953125, below 0.0078125 |
| Graph-B final state | max abs 0.0172200203, below 0.0200000000 |
| non-default stream | pass |
| graph replay | pass |
| invalid dtype and T guard, no fallback | pass |
| captured hierarchical solve vs v18 contract | pass |

W rounding widened back to FP32 has maximum absolute error 0.0009517372 and U has 0.0145950317. Widened BF16 V-new is exact. No executed case showed threshold violation, NaN, Inf, saturation report, subnormal-specific failure or layout change.

## Boundary Body Timing

All bodies use graph replay, warmup=20, repeat=100, five sessions and the same stream. Individual cast timings are not additive because each isolated graph has its own launch/intercept cost. The all-boundary measurement is the authoritative cast total.

| T=2048 body | ms | us |
|:--|--:|--:|
| historical asm-v0 recurrence | 0.154669 | 154.669 |
| current-vLLM bridge recurrence | 0.114570 | 114.570 |
| recurrence gain | - | 40.099 |
| W FP32-to-BF16 alone | 0.014061 | 14.061 |
| U FP32-to-BF16 alone | 0.014061 | 14.061 |
| W plus U together | 0.017026 | 17.026 |
| V-new BF16-to-FP32 alone | 0.014581 | 14.581 |
| all three boundaries together | 0.020750 | 20.750 |
| diagnostic net before full | - | 19.349 |

The recurrence-body slope is 4.513539 us/chunk for asm-v0 and 3.108600 us/chunk for the bridge. The combined boundary-cast slope is 0.252201 us/chunk.

## Same-Harness Full Timing

All values are medians of five session medians in one process, with identical inputs, current stream, capture-before-timing and balanced ABBA/BCCB/ACCA replay. Profiler time is not used as latency.

| T | Graph A current ms | Graph B Stage6S ms | Graph C vLLM ms | B gain vs A us | B-vLLM gap us |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.121721 | 0.120579 | 0.100450 | 1.370 | 20.049 |
| 1024 | 0.191685 | 0.181350 | 0.129493 | 10.319 | 51.865 |
| 2048 | 0.335679 | 0.309019 | 0.188800 | 26.647 | 120.206 |
| 4096 | 0.608985 | 0.550138 | 0.324282 | 58.960 | 225.767 |
| 8192 | 1.159743 | 1.033295 | 0.604138 | 126.288 | 429.022 |
| 16384 | 2.302501 | 2.024268 | 1.182217 | 278.926 | 841.739 |

Graph-A minus vLLM slope is 4.396726 us/chunk. Graph-B minus vLLM is 3.284716 us/chunk, so Stage 6S recovers 1.112009 us/chunk. No long-text point regresses. Graph B is still slower than vLLM, so it remains an experimental candidate rather than a production promotion.

### Eager-versus-Graph Reconciliation

The 0.188800 ms native-vLLM number is a CUDA Graph replay device-schedule result. It must not be compared directly with the older approximately 0.364 ms eager public-call reports. To verify the distinction, the old Stage-2 eager HIP-event harness was rerun in the current container with the same public vLLM API, T=2048, current input contract and the same autotune patch. It produced 0.360936 ms, p10 0.353244 ms and p90 0.373635 ms.

Thus both series are real but answer different questions: eager timing includes host launch/queue gaps between the seven vLLM dispatches after the start event is recorded; graph replay enqueues the already-captured graph as one replay and removes those host-side gaps. The Stage-6S A/B/C comparison remains internally fair because all three use the same graph-replay protocol. It is not a claim that ordinary eager vLLM invocation is 0.188800 ms.

### Supplemental Eager Public-API Leaderboard

After the graph-replay audit, the primary leaderboard was rerun with direct
eager public calls for v24, Stage 6S, and native vLLM. CUDA/HIP Graph replay
was not used. Every row uses the same seeded BF16/FP32 input contract, current
stream, warmup=20, repeat=100, five sessions, and pair-balanced
ABBA/BCCB/ACCA order. Public wrappers retain their warmed cached-allocator
behaviour; this is a public-API result, not an allocation-free claim.

| T | v24 eager ms | Stage 6S eager ms | vLLM eager ms | v24 / vLLM | Stage 6S / vLLM |
|--:|--:|--:|--:|--:|--:|
| 512 | `0.346034` | `0.293796` | `0.375658` | `0.921x` | `0.782x` |
| 1024 | `0.457640` | `0.324823` | `0.389739` | `1.174x` | `0.833x` |
| 2048 | `0.617018` | `0.391782` | `0.412713` | `1.495x` | `0.949x` |
| 4096 | `1.060757` | `0.601955` | `0.527264` | `2.012x` | `1.142x` |
| 8192 | `2.163678` | `1.093706` | `0.769584` | `2.811x` | `1.421x` |
| 16384 | `4.276980` | `2.088145` | `1.243489` | `3.439x` | `1.679x` |

All 12 Avelang-vLLM correctness checks passed. The largest Stage-6S error was
output max abs `0.0009765625` and final-state max abs `0.005528152`, within
the frozen `1/128` and `0.02` thresholds.

This is the first eager measurement that makes Stage 6S directly comparable
with v24 on the primary leaderboard. Stage 6S is faster than native vLLM at
T=512, 1024, and 2048, with its best relative result `0.782x` at T=512. It
crosses behind vLLM between T=2048 and T=4096, but remains substantially
better than v24 at every measured length. A least-squares fit over the sweep
gives Stage 6S `7.340 us/chunk`, native vLLM `3.569 us/chunk`, and a
Stage-6S-minus-vLLM gap slope of `3.771 us/chunk`; v24's corresponding gap
slope is `12.425 us/chunk`.

The raw per-length sessions and exact contract are in
`eager_public_leaderboard/by_t/`. The executable harness is
`stage6s_eager_public_leaderboard.py`.

## Resource and Identity Audit

The direct Graph-B trace contains chunk_gated_delta_rule_fwd_kernel_h_blockdim64, not qwen_gdn_bt64_gfx942_asm_v0:

| recurrence | WG | VGPR | AccVGPR | SGPR | scratch | MFMA | VMEM | LDS instructions |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Graph B current-vLLM bridge | 128 | 104 | 160 | 96 | 0 | 65,536 | 58,368 | 305,472 |
| Graph A historical asm-v0 | 256 | 128 | 192 | 80 | 0 | 196,608 | 91,136 | 588,928 |

The external-module rocprof metadata reports LDS_Block_Size=0 for Graph B while the guarded Stage-6R ABI/code-object reports dynamic LDS 40960 B. The fixed bridge ABI is authoritative; this is a collector metadata limitation, not an alternative launch.

Graph B adds three cast dispatches. The mixed direct trace recognizes the expected PyTorch BF16 copy classes, but contains warmup/replay and the unchanged final output cast. Individual cast VMEM, LDS-instruction and barrier counts are N/A, recorded as such in conversion_instruction_counts.csv rather than estimated.

## Decision

Case A: retain Graph B as an opt-in experimental full path. Do not promote it to the default path.

The only next action is BF16 storage-boundary propagation: have W/U producers natively write the recurrence BF16 contract and let chunk-o consume BF16 V-new, removing the three explicit casts one at a time under this same full-graph correctness/benchmark contract. Do not change the recurrence HSACO, compiler, asm-v0, KKT, solve, output staging or final public cast in that next action.

## Validation and Artifacts

- Stage 6S, Stage 6R bridge and asm-v0 regression: 8 passed in 21.95s.
- Shared Stage-4 nonrecurrence KKT/W/U/chunk-o regression: 29 passed.
- Direct source-JIT hierarchical-solve regression: 7 failed because the active runtime binding does not export al.amdgpu.mfma_16x16x4_f32_f32. This is the pre-existing feature-export limitation that requires the immutable, hash-guarded Stage-5B solve HSACO bridge in this experiment. The Stage-6S captured-solve versus v18 contract test is included in the 8 passing tests.
- Syntax checks passed with isolated /tmp/pycache_stage6s.
- git diff --check passed.
- [Stage 6S artifacts](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_recurrence_full_contract_stage6s) contain raw samples, correctness matrices, bridge source, commands and traces.
# Qwen gfx942 BT64 Stage 6T-Eager: Fused W/U

`timing_contract = "eager_public_api"`; `cuda_graph_used = false`.

## 结论

Stage 6T 的唯一 fused W/U schedule 已完整实现、通过 Eager public API correctness 和 expanded seed 稳定性，但**不通过性能晋级门槛**。F1 确实把 long-sequence gap slope 从 `3.455` 降到 `2.930 us/chunk`，回收 `0.525 us/chunk`；但 T=2048 相对本轮 Stage 6S 是 `-9.794 us`，即回归，而不是要求的至少 +10 us 收益。根据预注册 CASE C，不接入、不启动 Stage 6U；保留代码和证据为 experimental diagnostic。

所有权威测试均为 Eager public API；未使用 capture/replay。recurrence 仍是 hash-guard current-vLLM HSACO `632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`。没有改 KKT、solve、chunk-o、V-new cast、final cast、compiler、assembly、production 或 v24。

## 实现

F0 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_fp32`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_fp32_eager`。F1 kernel: `_qwen_gdn_wu_bf16_kernel_bt64_fused_stage6t_bf16`，public API `qwen_gdn_full_bt64_stage6t_fused_wu_bf16_eager`。

两者均为 one-CTA-per-(chunk,value-head)，T=2048 为 256 CTA、WG=256；旧分离 W/U 为 4096 CTA。它们使用完全相同的 MFMA16、LDS、barrier 与 FP32 accumulation 顺序；只有最终 output store 是 F0 FP32 对 F1 BF16。F0 仍有两个 numeric W/U casts；F1 无 W/U casts 且不物化 FP32 W/U。

## Correctness

pytest `4 passed in 26.49s`。全图矩阵覆盖 T=64/128/512/1024/2048/8192、random/neutral/zero-beta/small/high-dynamic/cancellation/sparse-beta、zero/nonzero state、non-default stream；另有 T=2048 20 seeds 和 T=8192 5 seeds。99 条 public comparison 全部通过。

- 输出最大绝对误差: `0.0029296875`，阈值 `0.0078125`。
- final state 最大绝对误差: `0.015427827835083008`，阈值 `0.02`。
- F0/F1 public output 与 final state 在所有运行 case 中相等。

## 权威 Eager Full 时间

HIP event 与 wall-clock 都围绕同一次完整 public API 调用。单位 ms，前四列 HIP event，后四列 wall-clock。

| T | Stage6S | F0 | F1 | vLLM | Stage6S wall | F0 wall | F1 wall | vLLM wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.265433 | 0.255137 | 0.249769 | 0.372131 | 0.280835 | 0.270686 | 0.265022 | 0.387984 |
| 1024 | 0.292393 | 0.298161 | 0.292753 | 0.372130 | 0.307816 | 0.313804 | 0.308587 | 0.387474 |
| 2048 | 0.380383 | 0.394043 | 0.390177 | 0.426070 | 0.396642 | 0.409972 | 0.406902 | 0.442314 |
| 4096 | 0.597484 | 0.598586 | 0.588331 | 0.547390 | 0.613372 | 0.614524 | 0.604350 | 0.563694 |
| 8192 | 1.084264 | 1.054961 | 1.031687 | 0.793073 | 1.100434 | 1.071641 | 1.048126 | 0.809122 |
| 16384 | 2.079838 | 1.999439 | 1.951007 | 1.336598 | 2.097380 | 2.016750 | 1.969085 | 1.353719 |

T=2048 paired session mean gain: F0 `-13.905 us`，95% CI `[-14.718, -13.336]`; F1 `-10.780 us`，95% CI `[-12.471, -9.522]`。F1 CI 完全小于 0，明确未达门槛。wall-clock 与 HIP-event 对 F0/F1/Stage6S/vLLM 的相对方向一致。

## Slope

| implementation | intercept ms | slope us/chunk | vs-vLLM gap slope us/chunk |
|:--|--:|--:|--:|
| Stage6S | 0.160809 | 7.411 | 3.455 |
| F0 | 0.173075 | 7.067 | 3.112 |
| F1 | 0.172221 | 6.886 | 2.930 |
| vLLM | 0.308957 | 3.956 | 0.000 |

F1 的 long-T 没有回归：T8192/16384 相比 Stage6S 分别快约 52.6/128.8 us；但短中序列 T2048 回归约 9.8 us，所以全局 performance gate 失败。

## 资源诊断

| W/U path | CTA | WG | VGPR | AccVGPR | SGPR | LDS B | Scratch B | occupancy | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| old separate W+U, Stage6A context | 4096 | 256 | W52/U48 | W4/U8 | N/A | N/A | 0 | N/A | 524288 | N/A | N/A | 851968 | 1638400 |
| F0 public-profiled | 256 | 256 | 64 | 8 | 48 | 3072 | 0 | 8.523 | 524288 | 4518912 | 697344 | 491520 | 1114112 |
| F1 public-profiled | 256 | 256 | 64 | 8 | 48 | 3072 | 0 | 8.766 | 524288 | 4846592 | 696320 | 491520 | 1114112 |

F0/F1 无 scratch，未出现 v29 风格 resource cliff。F1 与 F0 MFMA/VMEM/LDS 相同，但 VALU 从 `4518912` 上升到 `4846592`；它说明单纯将 store 改为 BF16 没有把端到端 T2048 latency 转化为收益。private segment / exact spill 需要 standalone HSACO dump；本轮 Docker quota 在该补充收集前耗尽，因此标为 N/A。

## 决策

CASE C。F0/F1 都正确，F1 删除了两个 cast 和 FP32 W/U materialization，且 slope 改善超过次级 `0.30 us/chunk` 条件；但核心 T2048 gain gate 失败，不能保留为选定的 experimental public path，也不进入 Stage 6U。唯一下一动作是 **current-vLLM fused W/U golden bridge audit**，先审计 native fused W/U 的完整 ABI/中间量/ownership；本轮不创建 bridge、不修改 production。

## 证据

所有 raw samples、summary、correctness、rocprof CSV、contracts 和 commands 位于 `codex_qwen_bt64_fused_wu_eager_stage6t/`。
# Qwen gfx942 BT64 Stage 6T-Golden Audit

## 总结

本轮是严格 audit-only：没有修改 F1、Stage 6S、recurrence、KKT、solve、chunk-o、编译器、汇编或 production selector，也没有创建 vLLM W/U bridge。所有权威性能和正确性都来自完整 Eager public API，`cuda_graph_used=false`；IR、ISA、HSACO 和 PMC 只作诊断。

真实 native vLLM W/U 为 `recompute_w_u_fwd_kernel`。T=2048 正常 eager specialization 是 4 warps / 2 stages、grid `(32,8,1)`、256 CTA、WG=256、每 `(chunk,value-head)` 一个 CTA，HSACO 为 `d7cd8744d597dcc0a09ff0430770b6e97e23f7c8529240311f9624f1e0e0dfcd。

F1 将旧分离 W/U 的 CTA 数从 4096 降为 256 是真实的 launch/round-trip 改善，但总 MFMA 没有下降：F1 每 CTA 仍为 2048 条动态 MFMA，native vLLM 为 128。F1 的 16x 来自 16x16x16 几何 2x、main+residual 2x、每 wave 的 predicated lane-group fragment 4x。因此 256 CTA × 2048 = 524288；这不是 CTA 融合失败。

主根因是 **CASE C**：F1 需要 FP32 `a_solved`，native W/U 需要 BF16 `A`。layout 相同，但 solve-to-W/U 的数值 ABI 不兼容。次要因素是 F1 16x 动态 MFMA 和 BF16 scalar-store lowering。

## 实际 specialization

| T | config | grid | CTA | WG | HSACO |
|---|---|---|---|---|---|
| 512 | 4w/2s | (8,8,1) | 64 | 256 | d7cd8744d597dcc0 |
| 2048 | 4w/2s | (32,8,1) | 256 | 256 | d7cd8744d597dcc0 |
| 8192 | 2w/3s | (128,8,1) | 1024 | 128 | dee9be336292ad53 |
| 16384 | 2w/3s | (256,8,1) | 2048 | 128 | dee9be336292ad53 |

T=512/2048 共享一个 4w/2s specialization；T=8192/16384 共享一个 2w/3s specialization。跨完整 T sweep 不稳定，但 T=2048 重复 capture 保持同一 hash/config。

## 数学、ABI 与 ownership

- F1 W：`A_fp32 * beta * exp(g)` 先做 BF16 main MFMA16，再做 BF16 residual MFMA16；U 同理但没有 `exp(g)`。
- native W：`dot(A_bf16, BF16(K*beta*exp(g)))`；U：`dot(A_bf16, BF16(V*beta))`。
- native source 先执行并存储两个 U 的 BV=64 block，再执行并存储两个 W 的 BK=64 block；没有 residual dot，也没有可共享 W/U coefficient matrix。
- T=2048 两边都为 32 chunks、256 chunk-heads、256 CTA、每 chunk-head 一个 CTA、正常 WG=256/4 waves。
- K/V、beta/g、W/U 的 layout 对齐；决定性差异为 A：native BF16、F1 FP32。`direct_abi_compatible=false`，需要 dtype/upstream contract change，不需要 layout transform。

## 归一化工作与 store

| implementation | W main | W residual | U main | U residual | MFMA/CTA | MFMA/dispatch |
|---|---|---|---|---|---|---|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

F1 的 F0/F1 static MFMA、LDS 和 barrier 数相同。额外 327680 VALU 定位在 BF16 输出 epilogue：F0 是 `global_store_dword`；F1 是 `global_store_short_d16_hi`，前面有 `v_bfe_u32`、`v_add3_u32`、`v_or_b32`、`v_cmp_u_f32` 和 `v_cndmask_b32`。native vLLM 是 `buffer_store_dwordx2`。PMC 无法把 VALU 精确逐 opcode 分账，因此更细拆分为 N/A。

## 资源与 Eager 基线

F1 standalone HSACO 是 VGPR 72、AGPR 8、SGPR 35、LDS 3072 B、private 0、无 VGPR/SGPR spill。native normal T=2048 HSACO 是 VGPR 180、AGPR 16、SGPR 101、private 0、无 spill。F1 并非由 register/scratch 压力导致。Triton cache `shared=8192` 与 code-object fixed group segment=0 不一致，报告保留该事实，不作未证明解释。

| T | Stage6S ms | F1 ms | vLLM ms | F1-S us | F1/vLLM |
|---|---|---|---|---|---|
| 512 | 0.272184 | 0.262711 | 0.367967 | -9.474 | 0.714x |
| 2048 | 0.381487 | 0.397290 | 0.415096 | 15.803 | 0.957x |
| 8192 | 1.088638 | 1.038984 | 0.810765 | -49.654 | 1.281x |
| 16384 | 2.078308 | 1.955686 | 1.332881 | -122.622 | 1.467x |

所有表中的 baseline 是五个 session median 的 median，warmup=30、repeat=200、balanced order，HIP event 包住完整 eager public API，allocation 计入，不使用 graph。F1 在 T=2048 比 Stage6S 慢，8192/16384 更快，方向与旧 Stage6T 一致。

## Correctness 与唯一下一步

Eager public full correctness 覆盖 T=64..8192、zero/nonzero initial state、zero/sparse beta、neutral/high-dynamic/cancellation/small-value 与 non-default stream。public output 最大 abs `0.0029296875`，final-state 最大 abs `0.0102265477`，均在冻结阈值内。W/U 的中间差异来自 FP32/BF16 solve boundary，不能用最终 BF16 output 掩盖。

唯一下一步：**Stage 6U BF16 solved-boundary propagation**。只改变 solve-to-W/U storage boundary，并在同一 Eager full contract 下验证；不创建 golden bridge，不需要 compiler/assembly 修改，也不先做 tile sweep。

## 回归

`16 passed in 27.74s`。执行的集合包括 Stage 6T 完整 Eager public API 对照、non-default stream、Stage 6S BF16 recurrence hash/非法输入/solve contract，以及本轮静态审计检查。旧 `torch.cuda.graph` replay case 被刻意排除；审计 runner 与 F1 public path 均扫描确认没有 `CUDAGraph`、`torch.cuda.graph(...)` 或 `.replay()` 调用。

## 产物

- 审计目录：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg`
- 最终 JSON：`test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_vllm_fused_wu_golden_audit_stage6tg/final_decision.json`
# Qwen gfx942 BT64 BF16 Solved Boundary Stage 6U

## Conclusion

Stage 6U-Solved completes as **CASE A**. The selected U1 path keeps solve
arithmetic and accumulators in FP32, writes the solved matrix directly as
BF16, and feeds a main-only fused W/U kernel. It is opt-in and experimental;
Stage 6S remains the unchanged production/default path.

All authoritative correctness and latency measurements use complete Eager
public API calls. CUDA/HIP Graph capture and replay were not used. Private
kernel runs and rocprof durations are diagnostic only.

At T=2048, U1 improves over Stage 6S by `122.551 us`, with paired bootstrap
95% CI `[108.741, 136.236] us`. C0 executes `1024 MFMA/CTA` and `262144
MFMA/dispatch`, exactly half F1. Public output max error is `0.001953125` and
final-state max error is `0.0152155161`, both within the frozen contract.

## Source Audit

The hierarchical solve keeps input, shared `x/work`, recurrence and all
`mfma_16x16x4_f32_f32` accumulators in FP32. In the new producer, only the
output pointer and stores change to BF16:

- output zero/strict upper: lines 72-79;
- diagonal identity: lines 107-112;
- diagonal writeback: lines 114-124;
- lower-block writeback: lines 169-177, 218-222, 262-266 and 319-323.

These references are in
`vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`.

F1 generated the main and low residual coefficients separately and executed
W-main, W-residual, U-main and U-residual. W was fully stored before U began,
so the issue was duplicated arithmetic rather than overlapping W/U
accumulator lifetime. Exact old-source references are retained in
`f1_residual_source_map.md` and `f1_mfma_source_accounting.md`.

## P0 Producer

- Kernel: `_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`
- Wrapper: `qwen_gdn_solve_hierarchical_bt64_bf16_solved_stage6u`
- P-REF: unchanged FP32 solve followed by a numeric BF16 cast
- P0 change: final global storage dtype only
- Output: contiguous BF16 `[1,T,8,64]`
- HSACO SHA256: `4bd6ddffa7812a0d66b0919963dbda064078ff7d0cfc9d9c7f348903d7fc7c07`

P0 passed 42 producer cases at T=64/128/512/1024/2048/8192, including zero,
identity-like, high-dynamic, small, cancellation and sparse-lower inputs,
non-default stream, NaN prefill and output reuse. It is BF16 bit-exact to
P-REF: mismatch count `0`, max abs `0`.

P1 packed writeback is N/A. ISA uses scattered `global_store_short` and
`global_store_short_d16_hi`; the current lane ownership does not expose a
safe contiguous four-value store without also changing ownership/layout.

## C0 Consumer

- Kernel: `_qwen_gdn_wu_kernel_bt64_fused_bf16_solved_stage6u`
- Wrapper: `qwen_gdn_w_u_bt64_fused_bf16_solved_stage6u`
- Input: BF16 solved A/K/V, FP32 g/beta
- Output: BF16 W/U
- Launch: 256 CTAs at T=2048, WG=256
- Residual source/coefficient/MFMA: completely absent
- HSACO SHA256: `00bee81e2646ecf6d858bb92b3668e0cc4b8ff1011e4055e38c71f40c751d2cc`

The isolated T=64/512/2048 matrix passed all six W/U checks. Maximum absolute
error against the BF16-coefficient matrix reference was `0.0009765625`.

| implementation | W main | W residual | U main | U residual | MFMA/CTA | T=2048 MFMA |
|:--|--:|--:|--:|--:|--:|--:|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| C0 | 512 | 0 | 512 | 0 | 1024 | 262144 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

## Full APIs And Correctness

- U0: `qwen_gdn_full_bt64_stage6u_casted_bf16_solved_eager`
- U1: `qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager`
- U2: N/A

U1 does not materialize FP32 solved A, FP32 W or FP32 U. It has no solved
cast and no W/U cast. The recurrence HSACO remains
`632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e`.
KKT, recurrence, V-new cast, chunk-o and final output cast are unchanged.

| coverage | result |
|:--|:--|
| Full matrix T=64/128/512/1024/2048/8192 | 7/7 accepted |
| T=2048 expanded stability | 30 seeds accepted |
| T=8192 expanded stability | 10 seeds accepted |
| T=16384 smoke | 3 seeds accepted |
| U0 versus U1 | output and state bit-exact |
| Max public BF16 output abs | `0.001953125` <= `0.0078125` |
| Max FP32 final-state abs | `0.0152155161` <= `0.02` |
| Non-default stream | pass |

## Eager Public Performance

Each cell is the median of five independent session medians. Every session
uses warmup=30, repeat=200, synchronized HIP events, wall-clock confirmation,
the same current stream and position-balanced order. Each sample includes a
complete public call and its internal allocations/casts/wrapper glue.

| T | Stage 6S ms | F1 ms | U0 ms | U1 ms | vLLM ms |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.361336 | 0.448805 | 0.408086 | 0.367786 | 0.403258 |
| 1024 | 0.293695 | 0.289290 | 0.278994 | 0.267397 | 0.381506 |
| 2048 | 1.236274 | 1.142135 | 1.106523 | 1.064339 | 0.884972 |
| 4096 | 0.602694 | 0.585548 | 0.540381 | 0.537717 | 0.582104 |
| 8192 | 2.387603 | 2.426420 | 2.299091 | 2.269348 | 1.759431 |
| 16384 | 2.104422 | 1.961651 | 1.852589 | 1.844657 | 1.449531 |

The machine showed substantial cross-session clock/load variation, including
non-monotonic absolute medians at T=2048/4096. Therefore the promotion gate
uses paired samples from the same runs, not comparisons to old reports.

| T | U1 gain vs Stage 6S us | paired 95% CI us | U1 gain vs F1 us |
|--:|--:|:--|--:|
| 512 | 26.904 | [-29.576, 59.235] | 30.817 |
| 1024 | 29.277 | [26.564, 31.980] | 17.400 |
| 2048 | 122.551 | [108.741, 136.236] | 84.903 |
| 4096 | 34.294 | [-21.641, 64.345] | 36.303 |
| 8192 | 181.001 | [158.452, 203.506] | 99.289 |
| 16384 | 232.026 | [163.207, 278.647] | 97.352 |

Wall-clock medians confirm the same selected direction:

| T | Stage 6S ms | F1 ms | U0 ms | U1 ms | vLLM ms |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.406648 | 0.472201 | 0.447394 | 0.428326 | 0.431095 |
| 1024 | 0.312053 | 0.306816 | 0.297021 | 0.285915 | 0.399608 |
| 2048 | 1.291911 | 1.226600 | 1.185214 | 1.129205 | 0.950711 |
| 4096 | 0.622513 | 0.604567 | 0.558714 | 0.556476 | 0.602018 |
| 8192 | 2.410462 | 2.450797 | 2.320153 | 2.293549 | 1.783061 |
| 16384 | 2.125860 | 1.983714 | 1.874607 | 1.866695 | 1.471073 |

U1 slope is `6.771890 us/chunk`, versus Stage 6S `7.623539` and vLLM
`4.719342`. Its vLLM gap slope is `2.052548 us/chunk`, improving Stage 6S by
`0.851649 us/chunk` and exceeding the 0.5 target. U1 remains slower than vLLM
at the stable long-text points.

## Resources And ISA

These counters come from complete public API profiler entry but are
diagnostic only. Profiler trace duration is explicitly not used for latency.

| W/U | VGPR | AccVGPR | SGPR | LDS B | scratch | MFMA | VALU | SALU | VMEM | LDS inst |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| F1 | 64 | 8 | 48 | 3072 | 0 | 524288 | 4846592 | 696320 | 491520 | 1114112 |
| C0 | 68 | 12 | 32 | 3072 | 0 | 262144 | 2087936 | 316416 | 311296 | 589824 |
| native vLLM | 60 | 164 | 112 | 0 | 0 | 32768 | 2176000 | 223232 | 92160 | 75776 |

C0 code-object metadata reports private segment `0`, VGPR spills `0`, SGPR
spills `0`; ISA contains `v_mfma_f32_16x16x16_bf16` and no residual phase.
P0 similarly reports private segment/spills `0` and retains
`v_mfma_f32_16x16x4_f32`.

U1 has eight dispatches: cumsum, KKT, P0, C0, recurrence, V-new cast,
chunk-o and final cast. U0 has nine due to solved cast; Stage 6S has eleven;
native vLLM has seven. The U1 trace confirms no solved or W/U cast dispatch.

## Remaining Gap And Decision

After residual removal:

```text
C0 1024 MFMA/CTA = 4x lane-group predicate x 2x MFMA16 geometry x
                   native vLLM 128 MFMA/CTA
```

The 4x factor comes from four divergent `lane_group` call regions in each W
and U source loop. The 2x factor is ideal MFMA16 versus MFMA32 tile geometry.
U2 was not implemented because branch-free operand selection has not yet
passed an isolated lowering/resource gate; changing predicates and geometry
together would violate the one-variable rule.

U1 passes the promotion gate and is the current best **experimental** Eager
candidate. Stage 6S remains production/default. The next single action is an
isolated C0 predicate-collapse experiment that proves one wave-uniform MFMA
call after per-lane fragment selection. No compiler or assembly change is
needed for that experiment.

## Evidence

All raw samples, correctness matrices, CSV comparisons, LLVM IR, ISA,
code-object metadata and exact commands are under
`codex_qwen_bt64_bf16_solved_boundary_stage6u/`.

# Qwen gfx942 BT64 Stage 6V: C0 Predicate-Collapse

## 结论

**V0 isolated lowering 通过，V1 full eager-public promotion 不通过。**

V0 严格只改变 C0 的一个 source scheduling 变量：原先四个
`lane_group` predicated MFMA16 区域，改为每 lane 选择已打包的 A/B
fragment，再执行一次 wave-uniform MFMA16。它成功把 static/dynamic MFMA
缩小四倍，且没有 scratch 或 spill。不过该选择链将 VGPR 从 68 提高到
100，完整 U1 图在 T=2048 的收益只有约 1--4 us；高重复 paired measurement
的置信区间跨零。因此 V1 仅保留为实验代码，**不替换 U1，不修改 default
selector，也不进入下一轮 source tuning**。

本轮没有改 solve producer、BF16 solved boundary、W/U 数学、MFMA16 geometry、
CTA/workgroup、W/U BF16 输出、recurrence、chunk-o、layout、compiler 或 assembly。

## V0 Source Change

C0 在 W 和 U 的每个 `source_tile` 内有四段同构代码：

```python
if lane_group == 0:
    acc = mfma(af[0], b[0], acc)
if lane_group == 1:
    acc = mfma(af[1], b[1], acc)
if lane_group == 2:
    acc = mfma(af_next[0], b_next[0], acc)
if lane_group == 3:
    acc = mfma(af_next[1], b_next[1], acc)
```

V0 将 fragment choice 显式变为条件表达式，并把 MFMA 放在选择后：

```python
a_operand = a_frag0[0] if lane_group == 0 else (...)
b0_operand = b0_frag0[0] if lane_group == 0 else (...)
b1_operand = b1_frag0[0] if lane_group == 0 else (...)
acc0 = mfma(a_operand, b0_operand, acc0)
acc1 = mfma(a_operand, b1_operand, acc1)
```

AveLang 将 conditional expression lower 为 `arith.select`。首次尝试使用
statement-level `if` 对临时变量赋值失败：Avelang 的 `scf.if` 分支拥有隔离
scope，赋值不能形成 if 后的 SSA result，因而所有 lane 都错误地使用默认
group-0 operand。该尝试未进入任何结果。最终 V0 使用条件表达式修复，不改
fragment layout 或 MFMA operand order。

## V0 Correctness

`test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`：`3 passed`，覆盖
T=64/512/2048。V0 同时与 C0 output 和现有 BF16 W/U reference 对比。
四个 lane-group input fragment 均由上述 select path 消费。

V0 不与 C0 bit-exact：从四个 predicated MFMA 的 EXEC 行为改成完整 wave
MFMA 后，少量 BF16 最低位不同。冻结 gate 是 C0 delta <= `1e-3`，明显小于
W/U reference 的 `0.0078125` 接受阈值。

| T | W bit mismatch | U bit mismatch | W max abs vs C0 | U max abs vs C0 |
|---:|---:|---:|---:|---:|
| 64 | 1 | 1 | 1.4901e-08 | 1.4901e-08 |
| 512 | 8 | 8 | 1.2207e-04 | 4.8828e-04 |
| 1024 | 13 | 18 | 6.1035e-05 | 9.7656e-04 |
| 2048 | 18 | 24 | 3.0518e-05 | 9.7656e-04 |

## V0 ISA and Rocprof Gate

T=2048 has 256 W/U CTAs (`Grid_Size=65536`, `WG=256`). Both code objects
have zero private segment and zero VGPR/SGPR spills.

| metric | C0 predicated | V0 predicate-collapse |
|:--|--:|--:|
| static `v_mfma_f32_16x16x16_bf16` | 64 | 16 |
| static `v_cndmask_b32` | 56 | 200 |
| static `s_cbranch*` | 38 | 6 |
| code-object VGPR / AGPR / SGPR | 76 / 8 / 31 | 108 / 8 / 40 |
| code-object scratch / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |
| profiler VGPR / AccVGPR / SGPR | 68 / 12 / 32 | 100 / 12 / 48 |
| profiler LDS / scratch | 3072 B / 0 | 3072 B / 0 |
| occupancy | 8.0107% | 7.9765% |
| dynamic MFMA | 262144 | 65536 |
| dynamic MFMA per CTA | 1024 | 256 |
| dynamic LDS | 589824 | 393216 |
| dynamic VALU | 2087936 | 2414592 |
| dynamic SALU | 316416 | 54272 |
| dynamic VMEM | 311296 | 311296 |

同一 profiling driver 的 trace median 是 C0 `41.642 us`、V0 `38.598 us`
（-7.3%）。单独重复 V0 profile 得到 `41.842 us`，所以 trace 的绝对差只作
诊断，不作为 promotion 依据；关键事实是 dynamic MFMA 已精确达到
`256/CTA`，并且没有 resource cliff、scratch 或 spill。

未插 profiler 的 isolated body timing 在 T=2048 为 C0 `0.073129 ms`、V0
`0.071827 ms`（1.8%）。省掉的 MFMA 被 select/cndmask 和更高 VGPR 部分抵消。

## V1 Full Public Contract

V1 是唯一的 full integration：

```text
P0 BF16 solved -> V0 predicate-collapse fused W/U -> unchanged BF16 recurrence -> unchanged chunk-o
```

它复用 U1 的 cumsum/KKT/P0 solve、Stage 6S recurrence bridge、BF16-to-FP32
V-new boundary和 chunk-o。没有额外 dispatch、fallback 或 public contract change。

完整 eager-public correctness 已覆盖 T=64/512/2048/8192/16384、随机 nonzero
initial state。全部 finite 且通过原有阈值：output <= `1/128`，final state <=
`0.02`。

| T | V1 output max abs vs vLLM | V1 final-state max abs vs vLLM | V1 output max abs vs U1 | V1 state max abs vs U1 |
|---:|---:|---:|---:|---:|
| 64 | 4.8828e-04 | 5.7392e-03 | 0 | 0 |
| 512 | 4.8828e-04 | 4.6706e-03 | 0 | 0 |
| 2048 | 7.3242e-04 | 7.9160e-03 | 1.2207e-04 | 2.9802e-08 |
| 8192 | 9.7656e-04 | 4.6512e-03 | 1.2207e-04 | 0 |
| 16384 | 9.7656e-04 | 4.7188e-03 | 1.2207e-04 | 0 |

## V1 Eager Public Timing

All values below use the required eager public API: same process/input/current
stream, no CUDA graph, pre-warmup, five sessions, 20 warmups and 100 ABBA-
balanced repeats. They are not rocprof times.

| T | U1 ms | V1 ms | vLLM ms | V1-U1 us | V1/U1 | V1/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.249551 | 0.248410 | 0.358412 | -1.141 | 0.9954x | 0.6931x |
| 1024 | 0.273627 | 0.270582 | 0.356490 | -3.045 | 0.9889x | 0.7590x |
| 2048 | 0.364962 | 0.361397 | 0.416459 | -3.565 | 0.9902x | 0.8678x |
| 4096 | 0.541285 | 0.538461 | 0.534434 | -2.824 | 0.9948x | 1.0075x |
| 8192 | 0.969402 | 0.968219 | 0.776614 | -1.182 | 0.9988x | 1.2467x |
| 16384 | 1.834768 | 1.829000 | 1.324268 | -5.768 | 0.9969x | 1.3811x |

Slope fit is U1 `6.460055 us/chunk` and V1 `6.448760 us/chunk`，只减少
`0.011295 us/chunk`。V1 的长文本 slope 确实没有变差，但下降过小。

为检验 T=2048 stability，额外运行 nine sessions、20 warmups、200 ABBA repeats：
U1 `0.362038 ms`，V1 `0.360316 ms`，aggregate 差 `-1.722 us`。但 sample-level
paired mean gain 的 bootstrap 95% CI 是 `[-2.003, 43.375] us`，包含零。因此
“相对 U1 稳定加速”这个主 gate **不成立**。

## 决策

| gate | result |
|:--|:--|
| V0 select 后 fragment 正确、冻结误差内 | pass |
| static MFMA 缩小四倍 | pass, 64 -> 16 |
| dynamic MFMA 约 256/CTA | pass, exactly 256 |
| 无 scratch/spill/resource cliff | pass; VGPR 增加但 occupancy 基本不变 |
| V1 public correctness | pass |
| T=2048 相对 U1 稳定加速 | fail |
| 长文本 slope 有实质下降 | fail; only -0.011295 us/chunk |

**No-Go for promotion.** 保留 U1 作为 Stage 6U 的实验选择，V1 保留为独立
negative/diagnostic result。不要为了这一点微小差距继续改 C0、compiler、recurrence
或 chunk-o；下一个动作应回到已有全图审计中更大的、数据支持的结构性 gap。

## Files and Commands

Source and drivers:

- `vllm_compare/qwen_gdn_bt64_predicate_collapse_stage6v.py`
- `vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v1.py`
- `vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/profile_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py`
- `vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_eager_public.py`

Raw artifacts, HSACOs, rocprof CSVs, correctness JSON and ABBA samples:

- `codex_qwen_bt64_predicate_collapse_stage6v/`

Key reproductions:

```bash
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v0.py -s

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_predicate_collapse_stage6v_v1.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_predicate_collapse_stage6v_eager_public.py \
  --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 20 --repeat 100 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_predicate_collapse_stage6v
```
# Qwen gfx942 BT64 Stage 6W: BF16 Chunk-O Boundary

## 结论

Stage 6W 完成了一个 opt-in 的全图边界优化：BT64 chunk-o 直接读取
recurrence 的 BF16 `v_new`，并将其 FP32 累加结果直接写为 BF16 public
output。没有修改 recurrence HSACO、KKT、solve、W/U、chunk-o 的 MFMA
几何/LDS tile、compiler 或默认 selector。

相对当前 U1 BT64 图，Stage 6W 的完整 **Eager public API** sweep 在全部
`T=512..16384` 点更快，T=2048 session-median 从 `0.361818 ms` 降到
`0.351862 ms`，T=16384 从 `1.829537 ms` 降到 `1.768947 ms`。不过 T=2048
的逐调用 paired bootstrap 区间跨零，所以该结果足以保留为下一轮的实验
候选，尚不足以修改默认路径。

最重要的判断是：收益来自消除全图中两个 materialized boundary，而不是
chunk-o 核心 MFMA 算术突然更快。预分配 body 测量中，新的 chunk-o 本体在
T=2048 反而约慢 `4.27 us`；完整图仍获益，是因为删除了 `v_new` 扩 FP32 和
final output cast 两个 dispatch，以及相应的 FP32 global staging。

## 冻结的改动

旧 U1 tail：

```text
BF16 recurrence V-new
  -> torch BF16-to-FP32 cast
  -> chunk-o loads FP32 then truncates to BF16 LDS/MFMA operand
  -> FP32 output staging
  -> torch FP32-to-BF16 final cast
```

Stage 6W tail：

```text
BF16 recurrence V-new
  -> chunk-o loads BF16 directly
  -> unchanged BF16 MFMA operand path with FP32 accumulators
  -> convert only at final BF16 public-output store
```

新文件：

- `vllm_compare/qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py`
- `vllm_compare/test_qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py`
- `vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_eager_public.py`
- `vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_body.py`
- `vllm_compare/profile_qwen_gdn_bt64_bf16_chunko_stage6w.py`

唯一新增 public entry 是
`qwen_gdn_full_bt64_stage6w_bf16_chunko_eager(...)`。它复用 U1 的
BF16 solve、C0 W/U 和 Stage 6S BF16 recurrence；只替换 chunk-o storage
boundary。任何非 BF16 `v_new`、不连续、非 BT64 的 shape 或设备不匹配都会
`ValueError`，没有 fallback。

## ISA 和资源证据

两个 HSACO 都保持相同的 WG=256、grid=2048 CTA（T=2048）、MFMA16
schedule、LDS block 和零 scratch。关键 ISA 差异符合预期：

| 位置 | U1 current chunk-o | Stage 6W |
|:--|:--|:--|
| V-new byte stride | `s_lshl_b64 ..., 2` | `s_lshl_b64 ..., 1` |
| V-new global read | `global_load_dword` | `global_load_ushort` |
| public output store | `global_store_dword` | `global_store_short_d16_hi` |
| MFMA mnemonic | `v_mfma_f32_16x16x16_bf16` | 相同 |

T=2048 rocprof 的动态 instruction counts 保持同一计算主体：

| metric | current | Stage 6W | 变化 |
|:--|--:|--:|--:|
| WG / Grid work-items | 256 / 524288 | 256 / 524288 | 相同 |
| VGPR / AccVGPR / SGPR | 112 / 64 / 112 | 112 / 64 / 112 | 相同 |
| LDS / scratch | 27136 B / 0 | 27136 B / 0 | 相同 |
| MFMA | 458752 | 458752 | 相同 |
| VALU | 12918784 | 12918784 | 相同 |
| SALU | 1495040 | 1449984 | -45056 |
| VMEM | 851968 | 851968 | instruction count 相同 |
| LDS instructions | 1343488 | 1343488 | 相同 |

VMEM 是指令数而不是传输字节数，所以 BF16 read/store 未必让该计数下降。

**rocprof trace caveat：** 在这个短 kernel 上，带 PMC 的 rocprof 把 Stage
6W 的 occupancy 报成 `0.81%`、trace 报成约 `1498 us`，而 current 为
`17.75%`、`66.86 us`。这和无 profiler、预分配 HIP-event body 的 `~0.09 ms`
以及完整 Eager 图相矛盾，且资源/动态计数并未显示 resource cliff。因此这组
trace 不满足低扰动使用门槛，只保留作 collector perturbation 证据，不能用于
性能因果结论。

## 数值正确性

独立 chunk-o 在 `T=64/512/2048` 对照旧路径
`chunk_o(v_new_bf16.float()).to(bf16)`，均为 BF16 bit-exact：

| T | int16 mismatch | max abs | mean abs |
|--:|--:|--:|--:|
| 64 | 0 | 0 | 0 |
| 512 | 0 | 0 | 0 |
| 2048 | 0 | 0 | 0 |

完整图也与 U1 public output bit-exact，final state bit-exact。相对 native
vLLM 的 full contract：

| T / case | output max abs | final-state max abs | threshold |
|:--|--:|--:|:--|
| 64 random + state | 5.24521e-4 | 4.82208e-3 | 1/128, 0.02 |
| 512 high-dynamic + state | 1.953125e-3 | 1.74136e-2 | pass |
| 2048 neutral-gate, zero state | 1.953125e-3 | 1.63818e-2 | pass |
| 8192 cancellation + state | 3.81470e-6 | 2.62512e-5 | pass |

The no-fallback guard for FP32 `v_new` also passed.

## 预分配 Chunk-O Body

这些数只测预分配 input/output 的 kernel launch，不包括两个被删除的 cast。
它们说明 direct BF16 global boundary 的 kernel body 没有形成更快的 MFMA
body，且小幅落后；这不否定全图优化。

| T | current ms | Stage 6W ms | current / W1 |
|--:|--:|--:|--:|
| 512 | 0.036154 | 0.041822 | 0.8645x |
| 1024 | 0.055202 | 0.057185 | 0.9653x |
| 2048 | 0.086869 | 0.091136 | 0.9532x |
| 4096 | 0.137304 | 0.141430 | 0.9708x |
| 8192 | 0.250192 | 0.251434 | 0.9951x |
| 16384 | 0.468576 | 0.473244 | 0.9901x |

## 权威 Eager Full Benchmark

所有下面数据均是同一 public API contract，`cuda_graph_used=false`，相同
输入/stream，ABBA order，warmup=20、repeat=100、5 session。它们是 Stage 6W
的性能判定；ISA/rocprof 只作诊断。

| T | U1 ms | Stage 6W ms | W1 gain vs U1 | W1 / vLLM |
|--:|--:|--:|--:|--:|
| 512 | 0.250191 | 0.223572 | 26.619 us, 1.1191x | 0.6190x |
| 1024 | 0.275829 | 0.267397 | 8.432 us, 1.0315x | 0.7308x |
| 2048 | 0.361818 | 0.351862 | 9.956 us, 1.0283x | 0.8581x |
| 4096 | 0.539100 | 0.523217 | 15.884 us, 1.0304x | 0.9801x |
| 8192 | 0.968759 | 0.936471 | 32.289 us, 1.0345x | 1.2119x |
| 16384 | 1.829537 | 1.768947 | 60.590 us, 1.0343x | 1.3401x |

线性拟合的 U1 slope 是 `6.439639 us/chunk`，Stage 6W 是
`6.252842 us/chunk`，回收 `0.186797 us/chunk`；native vLLM 的对应 slope
仍为 `3.926166 us/chunk`。

额外的 9-session / warmup=30 / repeat=200 confirmation：

| T | U1 ms | Stage 6W ms | aggregate gain |
|--:|--:|--:|--:|
| 2048 | 0.361998 | 0.353285 | 8.713 us (2.47%) |
| 8192 | 0.970882 | 0.940497 | 30.385 us (3.13%) |

两点的 session median 均支持正收益；不过逐调用 paired bootstrap 因 eager
调度/顺序噪声区间跨零，故不将其描述为达到默认 selector promotion gate 的统计
确定性收益。

## 决策

保留 Stage 6W 作为正确、正向但尚未提升默认的 opt-in 图候选。它完成了已知的
BF16 storage-boundary 传播方向，且不需要改 recurrence assembly。

下一步不应继续微调相同的 chunk-o arithmetic：预分配 body 已显示该本体没有
可观收益。应对完整 Eager 图做一次严格的 boundary-dispatch accounting，确认
removed cast dispatch 在公共 API 计时中的稳定贡献；若该贡献经过更强的
order-balanced confirmation 仍显著，再将 Stage 6W 作为 U1 的候选 tail，并只
选择一个新的、数据支持的 upstream global-intermediate 消除动作。

## 复现

```bash
cd /workspace/project/avelang
python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_bf16_chunko_boundary_stage6w.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_eager_public.py \
  --T 512 1024 2048 4096 8192 16384 --sessions 5 --warmup 20 --repeat 100 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_chunko_boundary_stage6w

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_bf16_chunko_stage6w_body.py \
  --T 512 1024 2048 4096 8192 16384 --warmup 20 --repeat 100
```

## 后续严格确认

本报告中的 5-session sweep 与 9-session confirmation 是候选收益线索，不是 baseline
promotion 结论。后续完成的 12-session randomized-Williams clustered Eager confirmation
在 T=2048 和 T=8192 都未通过旧的 block/session/nested CI 共同正下界门槛；GPU process
审计同时发现非独占 context、累计 eviction 和数百毫秒级 outlier。这些历史数据不能否定
早期正向收益。后续使用同一共享环境内随机 Williams block 的新鲜 8-session 重测确认：W1
相对 U1 在 T=2048 快 `8.994 us`（95% CI `[7.246,10.545]`），T=8192 快 `27.352 us`
（`[25.398,29.163]`），HIP event、wall-clock 与 nested sensitivity 一致。因此 W1 已晋级为
当前 Avelang experimental baseline；结论范围是 paired shared-environment Eager 排名。完整
数据、协议与下一候选在 `qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md`。
# Qwen gfx942 BT64 Stage 6W: Clustered Eager Confirmation And Intermediate Audit

## 结论

在 2026-07-21 的新鲜、共享环境严格配对 Eager 重测后，Stage 6W / W1 晋级为当前
**Avelang experimental baseline**。该晋级的范围是同一共享 GPU、同一 stream、随机
Williams block 交错下的相对 public-API 排名；它不是跨宿主机的绝对延迟宣称。

冻结的 Stage 6W 路径删除了尾部的 `v_new BF16 -> FP32` cast 和最终
`FP32 output -> BF16` cast；此前小样本的中心趋势显示正收益。但按本轮预先设定的
历史严格 Eager public-API gate 没有通过；但该样本已经显示出数十至上千毫秒的异常
GPU 延迟和外部 context eviction，无法用来判断预期只有 10--60 us 的 Stage 6W 收益。
新鲜的成对重测没有复现这些长尾，且在 T=2048 和 T=8192 均稳定确认 W1 快于 U1。

本轮不修改任何 kernel。新增加的只是确认脚本和静态 intermediate-accounting 脚本。

`KKT FP32 a -> solve` 的 producer-consumer global intermediate 消除仍只是已审计的
候选。它尚未实现；现在可以开展 source/DAG/CTA ownership 可行性审计，但不与其他 kernel
改动并行实现。

## 冻结对象

对比三条完整 Eager public API：

| 名称 | 路径 |
|:--|:--|
| U1 | `qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager` |
| W1 / Stage 6W | `qwen_gdn_full_bt64_stage6w_bf16_chunko_eager` |
| vLLM | `chunk_gated_delta_rule` public API |

固定合约为 gfx942、`B=1,Hk=4,Hv=8,K=V=128,BT=64`、BF16 `q/k/v`、FP32
`g/beta/initial_state`、非零 initial state、同一输入 seed 和同一 current stream。
测量使用一个完整 public API call 周围的 HIP event；没有 CUDA Graph capture/replay。

Stage 6W 代码、recurrence HSACO、KKT、solve、fused W/U 与 compiler 均未改变。

## 2026-07-21 共享环境成对重测

用户确认 U1、W1 与 vLLM 在同一个共享 GPU 环境中按随机顺序交错运行时，外部负载是共同
条件，适合用于相对排名。本次新鲜重测显式使用 `--allow-shared-gpu`，仍保留 preflight
context 和 eviction 记录，并将结论范围写为 `paired_shared_environment`。

每个 T 有 8 个 process-isolated session，每 session 有 8 个完整 Williams block；即每个
T 有 64 个配对 block、1,152 次完整 public-API call。主统计量为每 session 的 paired
median HIP-event gain；block/nested CI 是 sensitivity analysis。HIP event 与 wall-clock
均在每次同一 public API call 周围记录。

| T | U1 session-median mean ms | W1 session-median mean ms | W1 相对 U1 gain us | primary HIP CI us | wall CI us | nested CI us | W1 相对 vLLM |
|--:|--:|--:|--:|:--|:--|:--|:--|
| 2048 | `0.358282` | `0.348283` | `8.994` | `[7.246, 10.545]` | `[7.138, 10.517]` | `[5.671, 10.819]` | 快 `56.685 us`，`0.8607x` |
| 8192 | `0.966426` | `0.939173` | `27.352` | `[25.398, 29.163]` | `[26.100, 29.373]` | `[26.015, 30.168]` | 慢 `172.933 us`，`1.2259x` |

两个长度下 8/8 session 的 HIP 与 wall-clock paired median 均为正，U1-vs-W1 的 primary
gate 全部通过。T=2048 的 vLLM session-median mean 是 `0.404651 ms`，所以本 harness 中
W1 也快于 vLLM；T=8192 的 vLLM 是 `0.766289 ms`，W1 仍落后。因此 W1 是当前最佳
**Avelang** experimental baseline，并非在所有长度都全面超过 vLLM。

## 历史污染样本与统计设计

每个 `T` 独立执行 12 个 process-isolated sessions。每个 session 的两个 warmup
Williams blocks 不计时；随后执行 6 个计时 block。一个完整 block 含六个随机化的
Williams order：

```text
U1 W1 vLLM    W1 vLLM U1    vLLM U1 W1
U1 vLLM W1    vLLM W1 U1    W1 U1 vLLM
```

因此每个实现均出现于每个相对位置，并以相同次数相邻于另两个实现。每个 block 有
18 次完整 API call；每个 `T` 有 72 个配对 block 和 1,296 次计时 call。脚本保存：

- 每 call 的事件时间、session、block、order、position 和时间戳；
- 每 block 的 `U1 - W1`、`vLLM - W1` 配对差值；
- 每 session 的配对差值均值与中位数；
- session 开始/结束的 clock、温度、utilization、容器进程快照与 UTC 时间。

收益定义为 `reference - W1`，正值才代表 Stage 6W 更快。每个比较报告三种
95% bootstrap CI：i.i.d. block、session mean、nested session/block cluster。

这次历史运行的升级门槛是 U1 对 W1 在两个 `T` 的所有三类 CI 下界均大于 0 us。它把
三个相关聚合方式作为并列 gate，偏保守；更重要的是，环境污染已使该历史样本不适合判断。

## 严格确认结果

### U1 对 Stage 6W

| T | sessions | blocks | block mean / median us | block CI us | session CI us | nested CI us | gate |
|--:|--:|--:|--:|:--|:--|:--|:--|
| 2048 | 12 | 72 | `1842.964 / 160.419` | `[-1464.229, 4776.261]` | `[123.018, 3779.325]` | `[-2056.890, 5245.996]` | fail |
| 8192 | 12 | 72 | `-13.598 / 32.208` | `[-102.877, 54.876]` | `[-118.115, 51.297]` | `[-160.278, 70.893]` | fail |

T=2048 的正均值由少数几十毫秒以上的 outlier 主导，nested CI 穿过零；T=8192 的
三类 CI 也跨零。它们只能说明这批非独占样本不足以确认早期正向趋势，不能否定该趋势。

### vLLM 对 Stage 6W

这不是 Stage 6W 的升级 gate，但用于说明完整 public-API 相对关系。

| T | block mean / median us | block CI us | session CI us | nested CI us |
|--:|--:|:--|:--|:--|
| 2048 | `54881.531 / 373.085` | `[18983.948, 97085.539]` | `[-10276.580, 163247.214]` | `[-11884.085, 162116.710]` |
| 8192 | `-826.126 / -177.623` | `[-1219.237, -479.902]` | `[-1796.383, -182.098]` | `[-1768.867, -182.313]` |

T=2048 和 T=8192 都处在同一个受污染环境中，不能用于 W1/U1/vLLM 的 Eager 排名。
它们都不是此前 CUDA Graph replay 或 body benchmark 的替代品。

## 环境有效性审计

本轮脚本在每 session 的开始与结束保存 `amd-smi metric -g 0 --json` 和容器内 `ps`
快照。测量外的只读 `amd-smi process -g 0 --json` 还发现 5 个 GPU context，名称均为
`N/A`；其中一个 context 的累积 `evicted_time` 为 `90,332 ms`。这表明 GPU 不是独占
测量环境，且容器内 `ps` 看不到所有 GPU context。

原始 event 分布也确实异常宽：T=2048 的 U1/W1/vLLM 单次 event 中位数约为
`11.404/11.270/11.649 ms`，最小值约为 `0.335/0.333/0.312 ms`，最大值分别达到
`445.222/390.990/1276.906 ms`。这不是 BT64 算子的正常计算差异，不能据此宣称
Stage 6W 的实际 gain 消失或反转；它只足以阻止在该环境中晋级。

Eager public API 是此项目的权威口径，故 API 内 allocation、cast、dispatch 与返回对象
构造都应保留在计时中。caller-owned intermediate preallocation 只适用于另一个纯设备
pipeline 诊断口径，**不是** W1 Eager ranking 或晋级的前提。

2026-07-21 的重新 preflight 在任何 benchmark child/session 启动前观察到一个外部 GPU
context（PID `83597`，名称 `N/A`，累计 `evicted_time=100656 ms`），因而正确中止，未
产生第二批污染数据。它确认当前机器仍不满足微秒级确认的独占性前提。

更新后的确认脚本现在默认先执行 exclusive-GPU preflight；只有 preflight 看到零个已有
GPU context 才允许开始 session。`--allow-shared-gpu` 则允许当前已采用的严格配对相对
排名，并在产物中标记该范围。主统计量是 **session-level paired median HIP-event gain**
的 bootstrap CI；block 与 nested-cluster CI 仅作 sensitivity analysis。脚本还记录
wall-clock paired gain，并要求其 session-level 方向和 HIP event 一致。

## Stage 6W 更新后的完整图

Stage 6W 现在是 6 个 dispatch：

```text
cumsum -> g_cumsum FP32
KKT -> a FP32
solve -> a_solved BF16
fused W/U -> w_bf16 + u_bf16
immutable recurrence HSACO -> h_bf16 + v_new_bf16 + final_state
chunk-o -> public output BF16
```

已删除的两个 dispatch 是 `v_new BF16 -> FP32` 与最终 `FP32 output -> BF16` cast。

### 全局 intermediate accounting

`write+read` 对单消费者表示一次完整 materialization round trip；多消费者的
`g_cumsum` 使用所有消费者读流量。

| tensor | dtype | T=2048 / T=8192 | producer -> consumer | single use | write+read traffic at 2048 / 8192 | 直接写 consumer 格式 | 额外 dispatch |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `g_cumsum` | FP32 | 0.0625 / 0.25 MiB | cumsum -> KKT,W/U,recurrence,chunk-o | no, 4 | 0.3125 / 1.25 MiB | yes | no |
| `a` | FP32 | 4 / 16 MiB | KKT -> solve | yes | 8 / 32 MiB | yes | no |
| `a_solved_bf16` | BF16 | 2 / 8 MiB | solve -> fused W/U | yes | 4 / 16 MiB | yes | no |
| `w_bf16` | BF16 | 4 / 16 MiB | fused W/U -> immutable recurrence | yes | 8 / 32 MiB | yes | no |
| `u_bf16` | BF16 | 4 / 16 MiB | fused W/U -> immutable recurrence | yes | 8 / 32 MiB | yes | no |
| `h_bf16` | BF16 | 8 / 32 MiB | immutable recurrence -> chunk-o | yes | 16 / 64 MiB | yes | no |
| `v_new_bf16` | BF16 | 4 / 16 MiB | immutable recurrence -> chunk-o | yes | 8 / 32 MiB | yes | no |
| `final_state` | FP32 | 0.5 / 0.5 MiB | recurrence -> public optional output | public | n/a | n/a | no |
| `output_bf16` | BF16 | 4 / 16 MiB | chunk-o -> public output | public | n/a | n/a | no |

## 唯一登记的下一实验

选择 `KKT FP32 a -> solve` 的 producer-consumer global intermediate 消除实验。

- 它在 T=2048 为 4 MiB，一次 write+read 是 8 MiB；T=8192 为 16 MiB 和 32 MiB。
- 两端均为 Avelang source kernel，未跨 immutable recurrence HSACO ABI。
- `h_bf16`、`v_new_bf16`、`w_bf16` 和 `u_bf16` 虽然流量更大，但其边界跨越当前不可变的
  recurrence ABI；本轮不触碰它们。
- `g_cumsum` 有四个消费者，不能把它当作单边 materialization 消除；`a_solved_bf16`
  仅有一半 `a` 的流量。

这里的含义是 KKT producer 与 solve consumer 的 private/tiled handoff 或融合，**不是**
再增加一个 BF16 cast。`a` 在 T=2048 是 4 MiB，不能假定可以整体私有化；后续只能先做
source/DAG/CTA ownership 可行性审计，不能默认融合一定会获益。独占-GPU W1 confirmation
之前不得实现该实验或叠加其他优化。

## 产物与复现

确认脚本：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/confirm_qwen_gdn_bt64_bf16_chunko_stage6w_eager.py \
  --T 2048 8192 --sessions 8 --warmup-blocks 2 --blocks-per-session 6 \
  --bootstrap-samples 10000 --out-dir <out-dir>
```

默认模式会在 session 开始前拒绝任何已有 GPU context，用于绝对延迟声明。
`--allow-shared-gpu` 启用严格配对的共享环境相对排名，并在 summary 中明确写入
`promotion_scope=paired_shared_environment`。静态图审计：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6w_intermediates.py \
  --T 2048 8192 --out-dir <out-dir>
```

实际原始数据位于：

- `codex_qwen_bt64_stage6w_cluster_confirm_t2048/`
- `codex_qwen_bt64_stage6w_cluster_confirm_t8192/`
- `codex_qwen_bt64_stage6w_paired_shared_retest/`
- `codex_qwen_bt64_stage6w_cluster_confirmation_and_intermediate_audit/`
# Qwen gfx942 BT64 Stage 6X-KS X0: KKT-Solve Handoff Ownership Audit

## Scope And Decision

This is the zero-source-change feasibility audit for **Stage 6X-KS: KKT-Solve
Handoff Elimination**.  It audits the current Stage 6W/W1 graph only:

```text
cumsum -> KKT FP32 a -> hierarchical FP32 solve / BF16 store -> fused W/U
```

No kernel, compiler, recurrence HSACO, default selector, or production path is
changed by X0.  Eager public API remains the formal ranking contract;
standalone body timings and rocprof are diagnostic gates only.

**X0 decision: proceed to X1 only.**  The current KKT and solve have compatible
per-`(chunk, value_head)` ownership and exactly compatible logical `a` layout.
X1 can therefore test a one-CTA KKT without changing the global handoff.  X2
is still conditional: source-level shared-memory aliasing/lifetime reuse must
be demonstrated rather than assumed.

## Frozen Contract

| item | value |
|:--|:--|
| target | gfx942 / MI300 |
| tensor shape | `B=1, Hk=4, Hv=8, K=V=128, BT=64` |
| KKT inputs | BF16 `k`; FP32 cumsum `g` and `beta` |
| KKT current output | FP32 `a`, contiguous `[1,T,8,64]` |
| solve current input | that same FP32 `a` layout |
| solve current output | BF16 `a_solved`, contiguous `[1,T,8,64]` |
| current experimental full path | Stage 6W / W1 |
| formal timing | complete Eager public API, no CUDA Graph replay |

At `T=2048`, `a` is 4 MiB; the producer write plus the solve read is 8 MiB.
At `T=8192`, it is 16 MiB and 32 MiB respectively.  It is the largest
source-native, one-use upstream boundary remaining after Stage 6W.

## 1. Current KKT Ownership

Source: `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`,
`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`.

The launch is:

```text
grid = num_chunks * 8 value_heads * 16 token16 tiles
workgroup = 64 threads = 1 wave
```

For one `(chunk, value_head)`, `tile_id in [0, 15]` maps as:

```text
row_tile = tile_id // 4
col_tile = tile_id % 4
row_base = 16 * row_tile
col_base = 16 * col_tile
```

The current kernel launches all 16 matrix tiles.  Only ten lower-or-diagonal
tiles satisfy `row_tile >= col_tile` and execute K staging plus MFMA.  The six
strict-upper CTA instances perform no MFMA; they only write their 16x16 output
region as zero.  This is intentional: it materializes the full `[64,64]`
layout expected by existing consumers while retaining strict-lower math.

### KKT Formula And Mask

For local token row `t` and source column `s` in the same BT64 chunk, current
writeback is:

```text
a[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t] - g[s])     when s < t
a[t,s] = 0                                               when s >= t
```

The MFMA produces a complete 16x16 dot tile.  The strict-lower/causal rule is
applied only at FP32 writeback by `source_offset < token_offset`.  Therefore
diagonal 16x16 tiles compute the full dot product but retain only their own
strict lower triangle; their diagonal and upper entries are written as zero.

### Current Per-Tile Work

Each active tile stages:

```text
row_k_bf16[16,128] = 4 KiB
col_k_bf16[16,128] = 4 KiB
```

It accumulates four `batch128` iterations, each containing two
`mfma_16x16x16_bf16_f32` operations.  Thus:

```text
8 dynamic BF16 MFMA / active 16x16 tile
10 active tiles / 64x64 matrix
80 dynamic BF16 MFMA / (chunk, value_head)
```

The Stage 4 T=2048 rocprof total of 20,480 MFMA instructions exactly matches
`32 chunks * 8 heads * 80`.  The current KKT resource tuple was `WG=64`,
`LDS=8192 B`, `VGPR=20`, `AccVGPR=4`, `scratch=0`.

The ten active current CTAs re-read the K inputs independently.  Ignoring
cache effects, their staged global K traffic is `10 * (4 KiB + 4 KiB) =
80 KiB` per matrix.  X1 can instead stage the chunk's whole
`K[64,128]` once, which is 16 KiB, while preserving the same 80 MFMA
operations and exact per-tile reduction order.

## 2. Current Solve Ownership And Inputs

Source: `vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`,
`_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`.

The solve launch is already the desired coarse ownership:

```text
grid = num_chunks * 8 value_heads
workgroup = 256 threads = 4 waves
CTA = exactly one (chunk, value_head) 64x64 matrix
```

It consumes the exact KKT layout:

```text
a[0, chunk_start + row, head_idx, col]
```

No transpose, packing conversion, or dtype conversion occurs between the KKT
global store and solve load.  A fused handoff can therefore place the KKT
FP32 tile into an LDS `a_lds[row,col]` with the same row-major indices and
replace only the source of the solve's `a_frag` reads.

### Which A Blocks Solve Uses

Partitioning the 64x64 matrix into four 16x16 blocks, solve uses all strict
lower entries and no diagonal/upper `A` input values:

| input block | use count in the solve DAG |
|:--|--:|
| four diagonal strict-lower regions | initialize the four diagonal inverses |
| `A21`, `A32`, `A43` | level 1 |
| `A31`, `A32` | level 2a |
| `A42`, `A43` | level 2b |
| `A41`, `A42`, `A43` | level 3 |

The strict-lower matrix has `64 * 63 / 2 = 2016` FP32 semantic elements.
The current full output allocation has 4096 FP32 locations because KKT also
writes zeros for its diagonal and upper locations.  X2 should initially
retain all 4096 FP32 LDS locations: it preserves layout, avoids introducing
packed-lower address arithmetic, and keeps the experiment scoped to handoff
elimination.  Packed lower-triangle storage is an explicitly separate future
experiment, not an X2 addition.

### Current Solve LDS And DAG

The proven hierarchical solve uses:

```text
x[7,16,16] FP32       = 7 KiB
work[16,16] FP32      = 1 KiB
total                 = 8 KiB
```

`x` retains four diagonal inverse blocks and three level-1 lower blocks.
Slots are reused as the block DAG advances; `work` is first used for diagonal
row snapshots and later as one 16x16 product workspace.  This source schedule
has already passed correctness with `scratch=0`, `VGPR=44`, `AccVGPR=4`, and
`LDS=8192 B`.  At T=2048 it dynamically executes 64 FP32 MFMA16x16x4
instructions per matrix.

## 3. X1 Ownership: One CTA KKT, Still Global FP32 A

X1 uses the exact solve CTA mapping without changing the handoff:

```text
program_id -> chunk_idx = program_id // 8, head_idx = program_id % 8
WG256 -> wave_id = tid // 64
wave_id 0..3 owns row_tile 0..3
```

The proposed X1 staging is one BF16 shared tile:

```text
k_all_bf16[64,128] = 16 KiB
```

All four waves cooperatively stage it once.  Then each wave iterates the four
compile-time `col_tile` values.  For `row_tile >= col_tile`, it executes the
same eight-MFMA 16x16 dot sequence as current KKT and applies the same
writeback mask/decay.  For upper tiles it stores zero exactly as current KKT.
All writes retain the current global FP32 `a[1,T,8,64]` ABI.

| property | current KKT | X1 candidate |
|:--|:--|:--|
| CTA ownership | one 16x16 tile | one complete 64x64 chunk/head |
| grid per matrix | 16 WG64 CTAs | 1 WG256 CTA |
| active BF16 MFMA / matrix | 80 | 80 target |
| K LDS | 8 KiB per active CTA | 16 KiB per CTA |
| global `a` store | FP32 full 64x64 | unchanged |
| global `a` handoff | present | present |
| strict-upper handling | zero-store CTA | zero-store wave/tile |

X1 has a real risk: at `T=2048`, the KKT grid drops from 4096 total CTAs to
256 CTAs.  The fivefold reduction in theoretical K staging traffic can lose
to reduced inter-CTA parallelism, especially at T=64/128.  That is why X1 is
a hard gate rather than a presumed prerequisite for X2.

## 4. X2 LDS Lifetime And Capacity Audit

X2 would change only the KKT-to-solve boundary:

```text
KKT FP32 tile results -> a_lds[64,64] FP32 -> existing FP32 solve DAG
                                          -> BF16 a_solved global store
```

Capacity accounting is:

| region | bytes | KKT phase | solve phase |
|:--|--:|:--:|:--:|
| `a_lds[64,64]` FP32 | 16 KiB | live | live |
| KKT `k_all_bf16[64,128]` | 16 KiB | live | dead after final KKT tile |
| solve `x + work` FP32 | 8 KiB | unused | live |

There are two possible static-LDS outcomes:

1. **Correct reuse design:** reuse the 16 KiB K staging allocation after KKT
   as the solve's 8 KiB `x + work` area. Peak explicit LDS is 32 KiB:
   `a_lds 16 KiB + reusable scratch 16 KiB`.
2. **Naive separate allocations:** if the source declares independent K,
   `a_lds`, `x`, and `work` shared arrays, compiler allocation can retain all
   of them. Static LDS becomes 40 KiB even though K data is semantically dead
   before solve begins.

X0 does **not** claim that lexical phase order automatically aliases two
`al.make_shared` arrays.  Existing source uses `al.view` for byte-preserving
packed views, but X0 did not find a prior validated FP32-to-BF16 workspace
reinterpretation pattern for this exact use.  A future X2 implementation must
either demonstrate a single explicitly typed reusable backing store or accept
and measure the 40 KiB conservative case.  It must not call the memory reused
until HSACO resource metadata proves it.

Even the conservative 40 KiB is below a 64 KiB workgroup LDS budget, but it
can reduce resident workgroups and must be checked together with VGPR,
AccVGPR, scratch/private segment, spills, and occupancy.  The required
barriers are already substantial in solve.  X2 needs one additional
phase-publication barrier after the complete KKT `a_lds` write; it should not
add a barrier per output element or per KKT tile beyond the staging schedule.

## 5. Gate Definitions

### X1 Gate

Proceed from X1 to X2 only if all hold:

- KKT FP32 output matches current KKT at the frozen tolerance, with the first
  mismatch emitted on failure;
- solve-after-KKT matches current solve, proving layout compatibility;
- `scratch=0`, `private_segment=0`, and no VGPR/SGPR spills;
- dynamic MFMA is 80 per `(chunk,head)` matrix, not increased by predicate
  lowering or duplicated staged work;
- LDS/VGPR/AccVGPR show no resource cliff;
- isolated KKT is not materially slower than current KKT across T=64 through
  8192.  Any short-T launch-parallelism loss must be quantified rather than
  hidden by T=2048-only reporting.

### X2 Gate

Only after X1 passes:

- compare `current KKT -> current Stage 6U solve -> BF16 a_solved` against
  fused KKT+solve for T=64,128,512,2048,8192 with random, high-dynamic,
  cancellation, neutral-gate, non-default stream, output reuse, and NaN
  prefill cases;
- target BF16 bit exactness; if it fails, stop to identify numerical order
  changes rather than widening a tolerance silently;
- verify resource metadata, code-object spill fields, ISA MFMA mix, and
  barrier count;
- compare W1, X2, and native vLLM through the same complete Eager public API
  protocol at T=512,1024,2048,4096,8192,16384 with paired/Williams ordering,
  HIP event plus wall-clock, and cluster-aware confidence intervals.

## 6. X0 Answer To The Main Questions

| question | X0 answer |
|:--|:--|
| Does KKT calculate all 16 tiles? | It launches all 16; exactly 10 run MFMA, six strict-upper CTAs write zeros only. |
| How is diagonal strict-lower applied? | Full MFMA dot tile followed by `source_offset < token_offset` FP32 writeback mask. |
| Does solve need a layout conversion? | No. It reads the exact row-major KKT `[row,source]` layout. |
| Can `a` be CTA-local? | Yes, one 64x64 FP32 matrix is 16 KiB per chunk/head, not the whole T tensor. |
| Can KKT and solve LDS be reused? | Semantically yes; source-level static aliasing is not yet proven. |
| Is resource risk acceptable? | X1 has a 16 KiB K tile and is a justified test. X2 has a 32 KiB reuse target or 40 KiB conservative case and requires profiler proof. |
| Is a full fusion justified now? | No. X1 must first establish that the 16-CTA to one-CTA KKT ownership does not lose more parallelism than it saves. |

## Evidence

- `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`
- `vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`
- `qwen_gfx942_bt64_nonrecurrence_stage4_report.md`
- `qwen_gfx942_bt64_hierarchical_solve_stage5b_completion_report.md`
- `qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md`
- `codex_qwen_bt64_stage6w_cluster_confirmation_and_intermediate_audit/stage6w_intermediate_accounting.csv`
# Qwen gfx942 BT64 Stage 6X-KS: KKT-Solve Handoff Elimination

## 状态

Stage 6X-KS 已完成 X0、X1 和 X2 的源码实现、直接正确性、预分配 body
benchmark、HSACO 检查、T=2048 rocprof，以及随后补齐的完整 Eager public-API
正式 sweep。X2 删除了 KKT 到 solve 的 FP32 全局矩阵边界，并保持相对于 Stage 6W
链的 bit-exact 语义。

**X2 已晋级为新的 Avelang BT64 experimental baseline。** 这不是从 body
benchmark 推断出来的结论：正式 Eager public API 在每个 T 独立进程中执行，5 个
session、50 个 paired Williams block、每实现 300 次调用，且 HIP event 与
wall-clock 均同向。T=2048 与 T=8192 的 event cluster CI 下界都大于零；long-text
slope 也从 W1 的 `6.341 us/chunk` 降至 X2 的 `5.694 us/chunk`。

人工正式测量的结构化汇总存于
`codex_qwen_bt64_kkt_solve_handoff_stage6x_manual_confirmation/`。旧的 Docker
产物目录由容器 `nobody` 所有，故没有覆盖其中的原始 JSON；本报告以这份明确标记为
`user_manual_formal_run` 的确认记录作为晋级证据。

本阶段没有修改 recurrence HSACO、W/U、chunk-o、vLLM 或既有 production
baseline。

## X0：源码、ownership 与 LDS 生命周期审计

### 现有 KKT

当前 BT64 KKT 是
`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`：

- launch：`16 * num_chunks * 8` CTA，WG=64；
- 一个 CTA 对应一个 `(chunk, value-head, token16 row-tile, token16 col-tile)`；
- 每个 64x64 matrix 共有 16 个 tile；其中 10 个下三角或对角 tile 做 dot，6 个
  上三角 tile 只写零；
- active tile 以 8 次 `mfma_16x16x16_bf16_f32` 完成 K=128 reduction；故每个
  `(chunk, head)` 有 `10 * 8 = 80` 次动态 BF16 MFMA；
- 数学与 output layout 保持：

```text
a[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t]-g[s])  if s < t
         0                                          otherwise
```

`a` 的 global layout 是 FP32 `[1,T,8,64]`，每一个 `(chunk, head)` 对应其中
一行 64-wide matrix。Stage 6U solve 正好按同一 `(chunk, head)` 使用 WG256/4
waves 消费这一块；它只需要下三角，但现有 ABI 仍为完整 64x64 row-major matrix。

### 容量结论

每 CTA 的 A matrix 为 `64*64*4 = 16 KiB`。X1 只需一次性 stage
`K[64,128] BF16 = 16 KiB`。X2 的保守实现同时分配：

| LDS 对象 | 大小 | 生命周期 |
|---|---:|---|
| `k_all_bf16[64,128]` | 16 KiB | KKT phase |
| `a_lds[4,16,64] FP32` | 16 KiB | KKT 完成到 solve 完成 |
| Stage6U `x[7,16,16]` + `work[16,16]` | 8 KiB | solve phase |
| 合计 | 40 KiB | 保守、无 alias |

本轮没有假设 Avelang 可以安全地将 BF16 K staging 和 FP32 solve work 做类型
重解释 alias。40 KiB 在 gfx942 CTA LDS 容量内；是否值得做 32 KiB lifetime reuse
是后续独立 micro-experiment，而非本阶段的正确性前提。

## X1：1 CTA / chunk-head KKT

源码：
`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py`

`_qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1` 采用 WG256。四个 waves 分别
拥有 4 个 token16 row tile，并循环四个 column tile。严格上三角不做 dot 但仍写零；
因此 a 的 layout、mask 和数值顺序保持不变。X1 仍写原 global FP32 `a`，仅验证
ownership、并行度与资源。

### X1 正确性

`test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` 已在 gfx942 执行：

- KKT-only：T=64/128/512/2048，random 与 high_dynamic；全部 FP32 bit-exact；
- X1 KKT 后接当前 Stage6U solve：T=64/512/2048，random 与 cancellation；全部
  BF16 bit-exact。

### X1 KKT body

预热 5、repeat 20、HIP-event body；仅供 ownership gate，不是 Eager full timing。

| T | current KKT ms | X1 KKT ms | current/X1 |
|---:|---:|---:|---:|
| 64 | 0.035473 | 0.031447 | 1.128x |
| 128 | 0.035333 | 0.032369 | 1.092x |
| 512 | 0.034832 | 0.029624 | 1.176x |
| 1024 | 0.035192 | 0.031106 | 1.131x |
| 2048 | 0.046910 | 0.032529 | 1.442x |
| 8192 | 0.170874 | 0.038117 | 4.483x |

T=2048 rocprof 显示 X1 保持 KKT 的 `20,480` MFMA，但把重复 tile staging/address
工作大幅降低。代价是 WG256 的资源和 occupancy；它没有 scratch。

| metric | current KKT | X1 KKT |
|---|---:|---:|
| grid work-items | 262,144 | 65,536 |
| workgroup | 64 | 256 |
| LDS | 8 KiB | 16 KiB |
| VGPR / AccVGPR / SGPR | 20 / 4 / 32 | 84 / 20 / 112 |
| scratch | 0 | 0 |
| MFMA | 20,480 | 20,480 |
| VALU | 1,565,696 | 639,488 |
| SALU | 187,904 | 93,184 |
| VMEM | 210,944 | 79,872 |
| LDS instructions | 184,320 | 47,104 |
| OccupancyPercent | 8.991% | 3.646% |
| median trace | 40.961 us | 9.815 us |

因此 X1 gate 通过：CTA 数从 16 降为 1 并未造成不可接受的并行度损失，且没有
scratch/spill cliff。

## X2：CTA-local KKT + hierarchical solve

### 实现

同一文件中的
`_qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2`：

```text
K/Beta/G
  -> KKT FP32 (CTA-local a_lds[4,16,64])
  -> Stage6U FP32 block-triangular solve in the same CTA
  -> BF16 a_solved global output
```

它不创建也不接收 global FP32 `a`。`a_lds` 的逻辑行布局是 `[row_block,
row_in_block,column]`；该布局与 solve 的 four-wave ownership 一致。

实现中发现了一个 Avelang fragment-layout 要点：`mfma_16x16x4_f32_f32` 的 A/B
operand 必须为 `vector<1xf32>`。初版错误地将 view 的最后一个 unit dimension
也索引掉，得到标量并触发 `MFMA operands must be vector types`。最终使用：

```python
a_lds_frag = al.view(a_lds, al.f32,
    al.make_layout((4, 16, 64, 1), (1024, 64, 1, 1)))
rhs = a_lds_frag[row_block, row_in_block, column]  # 保留最后的 1
```

这与现有 `x_frag` 的用法一致，直接将 LDS fragment 作为 FP32 MFMA operand。

### X2 直接正确性

已执行的 Stage6X KKT/solve matrix：

- current KKT -> current Stage6U solve vs X2：T=64/128/512/2048；
  random、high_dynamic、cancellation；全部 BF16 bit-exact；
- X1 KKT 与 current KKT：T=64/128/512/2048；全部 FP32 bit-exact；
- full Stage6W vs Stage6X：T=64/128/512/2048/8192，random、high_dynamic、
  cancellation、neutral_gate，且分别有/无 initial state；共 40 cases，public
  BF16 output 和 FP32 final-state 均 bit-exact。

补齐的接口回归：

- X2 caller-owned BF16 output 的 NaN prefill + 两次 reuse：通过；
- non-default stream T=64/2048：`2 passed`；
- 主 KKT/solve correctness matrix：`29 passed`。

这些测试证明 X2 相对 Stage6W 保持精确语义及接口行为；它们不把“与 Stage6W
bit-exact”偷换成“与 vLLM 完全 bit-exact”。完整输出/final-state 的跨实现接受范围仍
沿用 Stage 6S/6U 的冻结 contract。

### 预分配 KKT+solve body

直接 launch 到 caller-owned outputs，warmup=5、repeat=20。current 是原 KKT
kernel 加 Stage6U solve；X2 是一 kernel。全部 X2 BF16 bit-exact。

| T | current KKT+solve ms | X2 ms | speedup |
|---:|---:|---:|---:|
| 512 | 0.046850 | 0.034591 | 1.354x |
| 2048 | 0.056384 | 0.034691 | 1.625x |
| 8192 | 0.161080 | 0.076193 | 2.114x |

该趋势符合 handoff 消除的预期：收益随 chunk 数增长。但这仍是 standalone body，不能
替代 full Eager gate。

### X2 T=2048 rocprof / ISA

| metric | X2 |
|---|---:|
| grid work-items / WG | 65,536 / 256 |
| LDS block | 40 KiB |
| VGPR / AccVGPR / SGPR | 100 / 164 / 112 |
| scratch | 0 B |
| MFMA | 36,864 |
| VALU / SALU | 1,280,512 / 206,336 |
| VMEM / LDS inst | 90,112 / 285,696 |
| OccupancyPercent | 6.362% |
| median trace | 16.505 us |

MFMA 数为 `20,480` 次 KKT BF16 MFMA 加 `16,384` 次 solve FP32 MFMA，正好符合
融合的两阶段工作量；没有通过减少 solve 数学换取收益。HSACO 中可见：

- `v_mfma_f32_16x16x16_bf16` 静态 32 条；
- `v_mfma_f32_16x16x4_f32` 静态 56 条。

X2 的 VGPR/AccVGPR/LDS 均高于 X1，但 scratch 为零，且 trace 仍明显低于旧 KKT
单阶段 trace。资源 gate 因此为通过，但 40 KiB LDS / AccVGPR=164 意味着后续若尝试
buffer reuse，必须再次检查 occupancy，而不是假定 LDS 更少一定更快。

## Full Eager public API

全图入口为：

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py`

```text
cumsum -> X2 fused KKT+solve -> existing BF16 W/U
       -> immutable BF16 recurrence -> existing Stage6W BF16 chunk-o
```

权威计时脚本：

`vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py`

它固定输入、current stream、Eager public API（`cuda_graph_used=false`），对每个
block 打乱六种 Williams 三路顺序，记录 HIP event 和 wall-clock，并按 session/block
配对做 cluster bootstrap。

### 完整正式 Eager confirmation

每个 T 以独立进程运行。每个进程固定相同的 `q/k/v/g/beta/initial_state`、current
stream 和 Eager public API；compile、module load 与首次 allocation 在计时前 warmup。
随后执行 5 个 session，每个 session 运行 10 个 timed paired Williams block，block 的
起始顺序随机化。每实现共 300 个完整调用。收益为 `W1 - X2`，正值代表 X2 更快。

| T | W1 median ms | X2 median ms | paired mean gain us | event 95% CI us | X2 相对 W1 |
|---:|---:|---:|---:|:---|---:|
| 512 | 0.221349 | 0.199697 | 22.123 | [19.720, 24.647] | 快约 10.9% |
| 1024 | 0.262049 | 0.233467 | 26.070 | [23.592, 28.667] | 快约 10.9% |
| 2048 | 0.348378 | 0.319154 | 26.020 | [23.304, 28.563] | 快约 8.4% |
| 4096 | 0.518070 | 0.479532 | 40.465 | [37.439, 43.506] | 快约 7.4% |
| 8192 | 0.935690 | 0.852166 | 82.823 | [80.432, 85.154] | 快约 8.9% |
| 16384 | 1.773194 | 1.595190 | 177.540 | [176.004, 179.024] | 快约 10.0% |

T=1024/2048 的 HIP 与 wall-clock CI 都完全为正，且每个长度的多数 session 都为正；
T=4096、8192、16384 的长文本收益也稳定扩大。这满足预注册 gate：correctness、stream、
NaN/reuse、无 scratch/spill、T=2048 和 T=8192 paired CI 下界大于零、两类计时器方向
一致且 slope 不恶化。

对 T=1024--16384 的 event median 按 chunk 数拟合：W1 为 `6.341 us/chunk`，X2 为
`5.694 us/chunk`，因此回收 `0.646 us/chunk` 或约 `10.2%`。这说明收益不仅来自少一个
dispatch，也来自随 chunk 增长而消失的重复 K staging、地址计算和 FP32 `a` global
write/read。

### 同批 native vLLM 对比

| T | X2 ms | vLLM ms | X2/vLLM | 结论 |
|---:|---:|---:|---:|:---|
| 1024 | 0.233467 | 0.358813 | 0.651x | X2 快 |
| 2048 | 0.319154 | 0.401397 | 0.795x | X2 快 |
| 4096 | 0.479532 | 0.522356 | 0.918x | X2 快，gain CI [39.710, 45.394] us |
| 8192 | 0.852166 | 0.765016 | 1.114x | X2 慢 |
| 16384 | 1.595190 | 1.235115 | 1.292x | X2 慢 |

vLLM 的拟合 slope 为 `3.688 us/chunk`，仍低于 X2 的 `5.694 us/chunk`。因此当前准确
结论是：这批同口径 Eager 测试中 X2 在 `T <= 4096` 快于 vLLM，在 `T >= 8192` 稳定落后。
4096--8192 之间存在粗略 crossover，但没有把线性插值值宣称为 dispatch policy。

## 产物与复现

| 产物 | 作用 |
|---|---|
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | X1/X2 kernels 和 direct-out API |
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | X2 full Eager graph |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | KKT/solve/reuse gates |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | 40-case full bit-exact matrix |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_stream.py` | non-default-stream gate |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | preallocated X1/X2 body benchmark and HSACO capture |
| `vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | rocprof driver |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py` | formal Williams Eager benchmark |
| `codex_qwen_bt64_kkt_solve_handoff_stage6x/` | JSON, HSACO, rocprof raw artifacts |

关键命令：

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py -s

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --T 512 2048 8192 --warmup 5 --repeat 20

/opt/rocm/bin/rocprofv3 --kernel-trace --pmc \
  SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2 \
  -d test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_kkt_solve_handoff_stage6x/rocprof_x2 \
  -o stage6x_x2 -f csv -- python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --implementation x2 --T 2048 --warmup 2 --repeat 5

# Run once per T in a fresh process after execution access is restored.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py \
  --T 2048 --sessions 5 --warmup-blocks 3 --blocks 10 \
  --out-dir /tmp/stage6x_eager_t2048
```

## 决策与下一步

版本状态：

| 版本 | 状态 |
|:--|:--|
| production/default | 不变 |
| U1 / Stage 6U | 历史 experimental baseline |
| W1 / Stage 6W | 上一 experimental baseline |
| X1 | 成功的 one-CTA KKT 组件/诊断 |
| X2 / Stage 6X | **当前 Avelang BT64 experimental baseline** |

冻结 X2；不立即压缩 `40 KiB` LDS，也不做 alias/reuse、packed-lower 或
recurrence--chunk-o fusion。唯一下一步是 **Stage 6Y updated full-gap audit**：以 X2
的五-dispatch 图重新比较 cumsum、fused KKT+solve、fused W/U、recurrence 和 chunk-o
对 native vLLM 的 standalone body slope、资源、真实 dispatch identity 及 global
intermediate accounting。只有该审计确认最大的可恢复缺口后，才选择一个新的 graph 或
kernel 实验。
# Qwen GDN Next Decision After Stage 6X-KS

X1 与 X2 的 source/body gates、NaN/reuse 和 non-default-stream gates 均通过。X2 在
同一 CTA 内保留 FP32 KKT matrix，直接执行现有 hierarchical FP32 solve，删除 global
FP32 `a` 的 allocation、store、read 和一个 dispatch；它对 Stage6W 为 full bit-exact。

随后补齐的正式 Eager public-API confirmation 使用每个 T 独立进程、5 sessions、50
paired Williams blocks、每实现 300 calls、HIP event 与 wall-clock。X2 对 W1 在
T=512/1024/2048/4096/8192/16384 都稳定更快；关键 gate 为：

| T | X2 相对 W1 paired gain | 95% event CI |
|---:|---:|:---|
| 2048 | +26.020 us | [23.304, 28.563] us |
| 8192 | +82.823 us | [80.432, 85.154] us |
| 16384 | +177.540 us | [176.004, 179.024] us |

W1 的长文本 slope 为 `6.341 us/chunk`，X2 降至 `5.694 us/chunk`，回收约 10.2%。
资源没有 scratch/spill cliff。因此：

```text
X2 / Stage 6X = new Avelang BT64 experimental baseline
W1 / Stage 6W = previous experimental baseline
production/default = unchanged
```

X2 不是所有长度都击败 native vLLM：同口径 Eager 下 X2 在 T<=4096 快于 vLLM，
在 T>=8192 因 vLLM `3.688 us/chunk` 的更低 slope 而落后。故不更改 production
selector，也不从两个点插值得出精确 crossover policy。

## 唯一下一步：Stage 6Y updated full-gap audit

冻结 X2 的五-dispatch 图：

```text
cumsum -> fused KKT+solve -> fused W/U -> recurrence -> chunk-o
```

Stage 6Y 是 measurement-only：重新审计 X2 与 vLLM 的实际 dispatch graph、每个
逻辑 body 的 T/chunk slope、资源/ISA 与所有剩余 global intermediate。它必须先产生
当前图的证据，才可以在以下候选中选择**一个**动作：`a_solved_bf16 -> W/U` 的 source
native handoff，或 immutable recurrence -> chunk-o 边界的后续设计。不得根据旧 Stage
6W 的六-dispatch账本，直接开始 LDS alias、packed lower、recurrence fusion 或其它
kernel 改动。
# Qwen gfx942 BT64 Stage 6Z Z0: Native Chunk-O Specialization Audit

## Scope And Result

Z0 is complete and passes its audit gate. No Avelang chunk-o implementation
was created before the native capture completed. The captured native vLLM
public API uses `chunk_fwd_kernel_o`, but it does not use one immutable
specialization across sequence lengths.

| T | BK | BV | workgroup | stages | grid | CTA | metadata LDS | scratch |
|---:|---:|---:|---:|---:|:---|---:|---:|---:|
| 2048 | 32 | 64 | 256 / 4 waves | 3 | `(2,32,8)` | 512 | 24576 B | 0 B |
| 8192 | 32 | 64 | 128 / 2 waves | 2 | `(2,128,8)` | 2048 | 12288 B | 0 B |

The tile/ownership is stable: one CTA computes `[BT=64, BV=64]` for one
chunk and one value head. There are two CTA per chunk-head because `V=128`.
The operational specialization changes at long sequence: native picks two
waves/two stages for the `T=8192` fresh capture. Per the Stage 6Z rule, the
one allowed Z1 candidate is based on this long-text ownership, not a sweep.

The raw machine-readable record is
[`z0_native_specializations.json`](codex_qwen_bt64_stage6z_native_chunko/z0_native_specializations.json).

## Exact Native Source And Mapping

The captured installed native source is
[`chunk_o.py`](codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/chunk_o.py).
Its launch grid is:

```python
grid = (triton.cdiv(V, BV), NT, B * H)
i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
```

Thus `i_v` selects a V64 block, `i_t` selects a BT64 chunk, and `i_bh`
selects a value head. This proves the CTA interpretation from source, rather
than inferring it from the T=2048 trace's 512 CTA count.

Each CTA keeps two FP32 register accumulator tiles:

```text
b_o [64,64] = q @ h^T
b_A [64,64] = q @ k^T
```

The `K=128` reduction uses four `BK=32` stages. It then applies decay and
the causal lower mask to `b_A`, converts that operand to BF16, performs the
`[64,64] @ [64,64]` score-times-V-new dot, and converts the final FP32 output
to BF16 at the global store. There is no global score tensor and no FP32
output staging.

## What Repeats, And What Does Not

| Item per chunk-head | Stage6W frozen | Native vLLM |
|:--|:--|:--|
| CTA | 8 x V16 | 2 x V64 |
| Q/K score evaluations | 8 V blocks x 4 source V16 tiles | 2 V64 blocks |
| Q/K score repeats | 8x | 2x |
| H elements | V16 partition, each H element once overall | V64 partition, each H element once overall |
| V-new elements | V16 partition, each V-new element once overall | V64 partition, each V-new element once overall |
| score global tensor | none | none |
| output boundary | BF16 direct | BF16 direct |

The key recoverable work is not H/V-new traffic. It is the Stage6W repeated
Q/K score preparation: it evaluates the same token-token score calculation
once for each of eight V16 blocks, while native evaluates it once for each
of two V64 blocks.

## Native LDS And Lifetime Evidence

The external-module rocprof trace reports `LDS_Block_Size=0`; that field is
not trustworthy here. The selected T=2048 TTGIR contains five local
allocations, three local deallocations, and the AMDGCN contains 72 `ds_read`,
40 `ds_write`, and 10 `s_barrier` instructions. Triton metadata records
24576 B of shared memory. T8192 similarly records 12288 B and 56/36
DS read/write instructions.

The important TTGIR ordering is:

1. Allocate and use Q/K/H source shared buffers for the K reduction.
2. Deallocate those source buffers.
3. Allocate BF16 score operand and V-new local buffers for the score-times-V
   dot.

That phase separation avoids O0's unfavorable simultaneous live region of
wide source staging, long-lived score storage, and multiple V16 accumulators.
It is the property Z1 must preserve, not merely its V64 CTA count.

## Captured Resources And ISA

| Item | T2048 | T8192 |
|:--|---:|---:|
| MFMA mnemonic | `v_mfma_f32_32x32x8_bf16` | same |
| static MFMA32 | 40 | 80 |
| static MFMA16 | 0 | 0 |
| buffer load/store | 14 / 4 | 28 / 8 |
| ds read/write | 72 / 40 | 56 / 36 |
| static barriers | 10 | 11 |
| profiler final WG, VGPR, AccVGPR | 256, 100, 36 | 128, 28, 196 |
| scratch | 0 B | 0 B |

The T2048 PMC attempt is retained only as a diagnostic: rocprof caused a new
autotune run and emitted several alternative specializations, so its dynamic
counters are not attributed to the exact Z0 final code object. It remains
under `native/T2048/rocprof_pmc/`, marked non-authoritative rather than
silently merged into this table.

## Z0 Decision

All Z0 prerequisites hold: final code objects and IR exist at both required
lengths; ownership and actual global/LDS lifetime are explicit; the native
schedule needs no compiler or recurrence ABI change; and Avelang has an
already-validated `mfma_32x32x8_bf16_f32` source primitive.

**Decision: GO to one Z1 prototype only.** It will fix `BT64/BV64/BK32`, use
the long-text V64 ownership, and retain a single fixed Avelang `WG=256`
four-wave implementation. It will not revive O0, perform a tile sweep, or
enter the full graph until isolated correctness, resource, and body gates all
pass.

## Evidence Paths

- T2048 selected source/IR/ISA/HSACO:
  `codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/`
- T8192 selected source/IR/ISA/HSACO:
  `codex_qwen_bt64_stage6z_native_chunko/native/T8192/trace_capture/selected/`
- Native trace captures:
  `codex_qwen_bt64_stage6z_native_chunko/native/T2048/rocprof_trace/` and
  `codex_qwen_bt64_stage6z_native_chunko/native/T8192/rocprof_trace/`
- Reproducible capture script:
  [`capture_qwen_gdn_bt64_stage6z_native_chunko.py`](../../vllm_compare/capture_qwen_gdn_bt64_stage6z_native_chunko.py)
# Qwen gfx942 BT64 Stage 6Z: Native-Style Chunk-O Result

## Final Decision

**No-Go. Stage 6Z stops at Z1.** The new kernel is correct and improves the
isolated body, but its captured ISA contains **41 static `s_barrier`**
instructions. The Stage 6Z hard bound is `barrier < 19`, so Z2 integration,
Eager full benchmarking, promotion, and selector changes are all forbidden.

## Z0 Native Evidence

Z0 captured final native vLLM `chunk_fwd_kernel_o` source, TTIR, TTGIR,
LLVM IR, AMDGCN, HSACO, metadata, and trace in fresh public-API processes.

| native final selection | T2048 | T8192 |
|:--|--:|--:|
| tile | BT64, BV64, BK32 | BT64, BV64, BK32 |
| workgroup / stages | 256 / 3 | 128 / 2 |
| CTA | 512 | 2048 |
| CTA per chunk-head | 2 | 2 |
| source metadata LDS | 24576 B | 12288 B |
| scratch | 0 B | 0 B |
| ISA MFMA | `v_mfma_f32_32x32x8_bf16` | same |

Native source maps `program_id(0/1/2)` to V block, token chunk, and value
head. One CTA owns a `[64,64]` output tile. It evaluates Q/K score twice per
chunk-head for two V64 blocks. Frozen Stage6W evaluates that score eight times
for eight V16 blocks. H and V-new remain partitioned across V, so this does
not claim to remove their traffic.

TTGIR deallocates Q/K/H source buffers before allocating score and V-new
operands. That phase separation was the important design constraint, not CTA
count alone. The detailed audit is
[`qwen_gfx942_bt64_stage6z_z0_native_chunko_audit.md`](qwen_gfx942_bt64_stage6z_z0_native_chunko_audit.md).

## Z1 Implementation

New source:
[`qwen_gdn_bt64_native_chunko_stage6z.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z.py)

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
q/k/v-new/h: BF16; g: FP32; accumulator: FP32; output: BF16
BT64, BV64, BK32, WG256, two CTA per chunk-head
```

The four waves own four `[row32,value32]` output quadrants. One 16 KiB phase
buffer serially holds Q/H staging, then the two score halves, then V-new
transpose. The two score halves are placed in distinct physical buffer halves
so Q/K staging for source-half 1 cannot overwrite source-half 0 score.

Two narrow correctness repairs were made and no schedule sweep occurred:

| issue | repair |
|:--|:--|
| score-owner-only barrier was undefined | every wave stages and joins barriers; owner waves alone update score accumulators |
| second Q/K stage overwrote first score half | second Q/K stage uses the phase-buffer upper half, later reused for V-new |

This avoids O0's long-lived V16 accumulator groups and adds neither a global
score tensor nor FP32 output staging.

## Isolated Correctness

Reference: frozen Stage6W chunk-o at the identical BF16 boundary.

| T | BF16 mismatch elements | max abs | mean abs |
|---:|---:|---:|---:|
| 64 | 1 | 9.313e-10 | 1.421e-14 |
| 128 | 2 | 7.451e-09 | 5.862e-14 |
| 512 | 34 | 1.526e-05 | 1.749e-10 |
| 2048 | 133 | 1.526e-05 | 1.344e-10 |
| 8192 | 306 | 1.526e-05 | 5.953e-11 |

All maxima are below frozen `1/128`. The small mismatch count is MFMA16 versus
MFMA32 reduction-order rounding, not a relaxed threshold. The suite also
passes zero V-new, caller-owned NaN-prefilled output reuse, and invalid FP32
V-new rejection: **7 passed in 10.29 s**.

## Isolated Body Signal

These are caller-owned diagnostics, not formal Eager public ranking. Each
implementation and T ran in its own process with warmup 10 and 100 samples.

| T | Stage6W CTA | Stage6W | Z1 CTA | Z1 | Z1 speedup |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 0.092237 ms | 512 | 0.075592 ms | 1.220x |
| 8192 | 8192 | 0.252154 ms | 2048 | 0.196411 ms | 1.284x |

The two-point body slope is 1.666 us/chunk for Stage6W and 1.259 us/chunk for
Z1. Thus the ownership change recovers about 0.407 us/chunk. This positive
signal does not override the resource gate.

## Resource Gate

| item, T2048 | Stage6W | Z1 | O0 hard bound | result |
|:--|---:|---:|---:|:--|
| dynamic MFMA | 458752 | 81920 | N/A | reduced |
| dynamic VALU | 12918784 | 6626304 | N/A | reduced |
| dynamic VMEM | 851968 | 540672 | N/A | reduced |
| dynamic LDS instructions | 1343488 | 770048 | N/A | reduced |
| profiler AccVGPR | 64 | 172 | `<188` | pass |
| LDS block | 27136 B | 28672 B | `<33280 B` | pass |
| scratch | 0 B | 0 B | `0 B` | pass |
| code-object spill | 0/0 | 0/0 | `0/0` | pass |
| static MFMA32 | N/A | 32 | N/A | correct geometry |
| static MFMA16 | N/A | 0 | N/A | absent |
| **static barriers** | N/A | **41** | **`<19`** | **FAIL** |

Z1 HSACO has `VGPR=164`, `AGPR=32`, `SGPR=46`, `LDS=28672 B`, private segment
zero, and zero VGPR/SGPR spills. rocprof reports `VGPR_Count=4`, inconsistent
with HSACO, so code-object metadata is authoritative for static VGPR while
rocprof remains the source of collected AccVGPR and PMC values.

The 41 barriers come from safe CTA-wide K32 staging, two score halves, and
V-new transpose. This source schedule cannot reproduce native Triton's 10 to
11 barrier pipeline.

## Stop Record And Reproduction

The following are intentionally absent: Z2 full wrapper, X2 integration,
Eager full benchmark, paired bootstrap, promotion, selector change, and Z1b
tile/workgroup sweep. Stage6W/X2 are unchanged.

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z.py -s
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_native_chunko_stage6z.py \
  --implementation z1 --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/z1/profile
```

Raw selected native IR/ISA, Z1 HSACO/ISA, body JSON, and rocprof CSV/JSON live
under `codex_qwen_bt64_stage6z_native_chunko/`.
# Stage 6Z Chunk-O 实验复盘

## 结论

本轮没有把 Z1 接入 X2 full graph。不是因为它不正确，也不是因为它 body 不快；它正确且
body 有收益。但 HSACO 生成了 41 个静态 barrier，超过开始前冻结的 19 个上限，所以必须
停止，不能用 isolated 速度绕过资源风险。

## 为什么先做 Z0

Stage6Y 说明长文本 chunk-o 是最大的可恢复差距。旧 Stage6W 的一个 CTA 只处理 V16：

```text
一个 chunk-head 有 8 个 V16 CTA。
每个 CTA 都重新读取 Q，并重新计算 token-token score。
```

因此可以考虑做宽 V ownership，但不能只看到 native CTA 更少就复活 O0。O0 历史上有
AccVGPR 188、LDS 33280 B、19 barrier，收益有限。Z0 先抓真实 native source/IR/ISA。

## Z0 学到的真实结构

native `chunk_fwd_kernel_o` 的真实 ownership：

```text
CTA = [BT64 token, BV64 value, 一个 value head, 一个 chunk]
每个 chunk-head 有 2 个 CTA
BK32，BF16 MFMA32，FP32 accumulator，BF16 直接输出
```

T2048 选择 4-wave/3-stage；T8192 选择 2-wave/2-stage。tile 保持一致，只有 pipeline
参数不同。Z1 遵守规则，只采用长文本的 V64/BK32 ownership，不做两套实现。

关键不是 V64 本身，而是 native TTGIR 的 LDS 生命周期：先放 Q/K/H 做 K reduction，
结束后再放 score 和 V-new。它不会让 source tile、score tile、多个 V accumulator 长期
同时活着。

## Z1 如何实现

四个 wave 覆盖四个 `[row32,value32]` 输出象限。一个 16 KiB phase buffer 被分时复用：

1. Q/H K32 staging，累积 inter-state。
2. source-half 0 score 写入前半 LDS；source-half 1 的 Q/K 使用后半 LDS，避免覆盖。
3. score 完成后，后半 LDS 改装 V-new transpose；最后做 score times V-new。

出现过两次确定性错误，且只修了明确根因：

| 问题 | 原因 | 修复 |
|:--|:--|:--|
| 初始 NaN/大误差 | 只有两个 wave 进入了 workgroup barrier | 所有 wave 均参与 staging/barrier，owner wave 才累积 score |
| intra 大误差 | source-half 1 Q/K 覆盖了 source-half 0 score | second half 改用 phase buffer 上半区 |

没有换 tile、没有换 WG、没有改 compiler、没有加 fallback。

## 结果为什么仍然是 No-Go

数值正确：T64 到 T8192 最大误差最多 `1.526e-5`，小于 `1/128`；zero V-new、output
reuse、invalid dtype 都通过。

body 也正确变快：

| T | Stage6W | Z1 | Z1 加速 |
|---:|---:|---:|---:|
| 2048 | 0.092237 ms | 0.075592 ms | 1.220x |
| 8192 | 0.252154 ms | 0.196411 ms | 1.284x |

但是资源 gate 不是只看 latency：

```text
scratch = 0，spill = 0，AccVGPR = 172，LDS = 28672 B，均通过。
static s_barrier = 41，要求 < 19，失败。
```

所以不创建 Z2 full API，不跑 Eager public 排名，也不改 X2。这个反例说明 CTA 和 MFMA
数量下降不等于 source-level schedule 已经适合 promotion；同步形状同样是硬资源。
# Next Decision After Stage 6Z Native Chunk-O

## Decision

**Close Stage 6Z at Z1. Keep Stage6X X2 as the Avelang BT64 experimental
baseline and keep v24 as production/default.**

Z0 completed its native vLLM audit and Z1 passed isolated correctness plus
showed a positive body effect. However, Z1's captured ISA contains 41 static
`s_barrier` instructions, violating the Stage 6Z hard resource gate
`barrier < 19`. The experiment must not proceed to Z2 full integration.

| gate | result |
|:--|:--|
| Z0 final native specializations captured | pass |
| Z1 BF16 isolated correctness T64 to T8192 | pass, max abs <= 1.526e-5 |
| Z1 zero V-new/reuse/rejection checks | pass |
| Z1 scratch/spill | pass, zero |
| Z1 AccVGPR / LDS | pass, 172 / 28672 B |
| Z1 barrier | **fail, 41 >= 19** |
| Z1 T2048/T8192 isolated body | positive, 1.220x / 1.284x |
| Z2 full graph | not created |

## What The Evidence Means

The native-style V64 ownership is a real source of recoverable work: it
reduces Stage6W's Q/K score repetition from eight V16 blocks per chunk-head
to two V64 blocks and reduces caller-owned body slope by about 0.407
us/chunk in this prototype. But the Avelang Z1 expression implements that
schedule with an unsafe resource shape: source staging and score/V phases
need 41 static barriers. The correct action is to preserve this evidence, not
to accept the body win and create a full graph with an unbounded long-text
risk.

## Explicit Non-Actions

- No Stage6Z Z2 full wrapper or Eager public benchmark.
- No change to X2, Stage6U W/U, immutable recurrence, compiler, or selector.
- No Z1b tile/workgroup sweep and no revival of O0.
- No production/default promotion.

The next optimization must be chosen by a new, separately audited decision.
It cannot be a continuation of this failed barrier envelope under a different
name.
# Qwen gfx942 BT64 Stage 7A: Chunk-O Barrier Provenance And Phase Audit

## 结论

**Case C: 关闭当前 Stage 6Z source-native chunk-o 路线。**

Stage 7A 证明了两件同时成立的事：

1. Z1 的 `41` 个静态 `s_barrier` 不是 AMDGPU backend 额外保守插入的。
   它们在 source 显式 `al.syncthreads()` 的循环展开后已经存在于 pre-link
   LLVM，并且 pre-LTO assembly 与最终 HSACO ISA 都是同一个 `41`。
2. 某些局部 barrier 在最小 repro 中确实可删且 bit-exact；但按规则只做的
   一次完整 Z1 phase-compaction 在 `T=8192` 立刻失去 bit exactness。

因此，不能把最小 repro 的“same-lane fragment”结论直接推广到 full CTA MFMA
pipeline。对**当前** Avelang source schedule 来说，这些 phase boundary 是实际
需要的同步形状。不能继续删 barrier、不能改 tile/WG 扫描、不能接入 full graph。

Stage 6Z 保持 No-Go，Stage6W/X2/v24/default selector 均未改变。

## 冻结边界

本轮没有修改：

- Stage6Z Z1 kernel；
- Stage6W、Stage6X X2、v24、default selector；
- BT64/BV64/BK32、WG256、CTA mapping、MFMA32、dtype 或数学；
- recurrence ABI、full graph、compiler、LLVM/AMDGPU RA、assembly、vLLM source。

唯一新增 full-kernel source 是一个**未晋级的失败实验**：

[`qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py)

它只删除 lane-private `frag_words` 同步并把 score-half sync 延迟到已有的
V-new producer-to-consumer barrier。它未通过 correctness gate，不能使用。

## 1. 精确 Barrier 链

审计对象是冻结 Z1：

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
```

生成的四层 artifact：

| 层 | artifact | barrier 数 | 结论 |
|:--|:--|--:|:--|
| AveLang source | `qwen_gdn_bt64_native_chunko_stage6z.py` | 10 个语法 site | 全部是显式 `al.syncthreads()` |
| AveLang IR 语义 | `lib/IR/builtin_module.cc` | 1:1 | `syncthreads()` 直接构造 `gpu::BarrierOp` |
| pre-link LLVM | `z1_prelink.ll` | 41 | `fence release -> llvm.amdgcn.s.barrier -> fence acquire` |
| pre-LTO assembly | `z1_prelink.s` | 41 | 与 LLVM barrier ordinal 相同 |
| final HSACO ISA | `z1_final.isa` | 41 | 逐 ordinal 与 pre-LTO 链对齐 |

因此本轮能严谨地说：**没有看到 compiler/backend 额外创建 barrier。** 它做的是
保留 source barrier 并对 `al.range` 的部分循环展开。

初始 MLIR 也尝试在独立子进程导出，但当前 Docker binding 的 `get_mlir()` 触发
segmentation fault（return code `-11`）。这个调试 API 限制不会影响 LLVM/HSACO 的
确证：同一 AST source 的 LLVM、pre-LTO assembly 与 freshly captured final HSACO 都
完成且数目一致。它意味着 source-line DebugLoc 没有可用的 MLIR 打印 artifact，不能
声称拥有 MLIR source location 到 ISA PC 的逐条 debug metadata 映射。

完整 41 条 PC ledger 在：

- [`z1_barrier_ledger.md`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [`z1_barrier_ledger.json`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.json)
- [`audit_summary.json`](codex_qwen_bt64_chunko_barrier_stage7a/audit_summary.json)

ledger 使用已验证的同序 `source schedule -> LLVM call -> assembly -> final ISA`
ordinal 对齐，而不是伪造不存在的 DebugLoc。

## 2. 41 个 Barrier 的 source provenance

Z1 source 有 10 个同步 site。对 `T=2048, WG256` 的 exact specialization，静态
展开/保留结果如下：

| source line | site | static count | dynamic context | hazard | 初始分类 |
|--:|:--|--:|:--|:--|:--|
| 90 | `A.stage_qh` | 4 | 4 个 inter K32 stage | CTA Q/H producer -> MFMA consumer RAW | 必要 |
| 96 | `A.pack_frag` | 8 | 4 stage x 2 kt | `frag_words` pack -> load | 待验证 |
| 103 | `A.reuse_frag` | 8 | 4 stage x 2 kt | MFMA 后下一 fragment reuse | 待验证 |
| 125 | `B.stage_qk` | 2 | 2 个 source half 的 K-stage loop body，各动态 x4 | CTA Q/K producer -> owner MFMA RAW | 必要 |
| 131 | `B.pack_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | `frag_words` pack -> load | 待验证 |
| 139 | `B.reuse_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | MFMA 后 fragment reuse | 待验证 |
| 153 | `B.serialize_score` | 2 | 每个 score half 一次 | score write -> later score/V consume | 待验证 |
| 164 | `C.stage_v` | 1 | V-new transpose | score/V producer -> intra MFMA RAW | 必要 |
| 172 | `C.pack_frag` | 4 | 2 half x 2 kt | score/V fragment pack -> load | 待验证 |
| 179 | `C.reuse_frag` | 4 | 2 half x 2 kt | MFMA 后 fragment reuse | 待验证 |

总数为：`4 + 8 + 8 + 2 + 4 + 4 + 2 + 1 + 4 + 4 = 41`。

这也解释了表面矛盾：source 只有 10 行 barrier，但它不是 10 个静态 ISA barrier。
Phase A 的 `k_stage=4` 被展开；Phase B 保留了动态 K loop body；Phase C 的两个
score half/两个 fragment 被展开。

## 3. 与 Native vLLM 的阶段对齐

native selected `chunk_fwd_kernel_o` 在这次重新读取的 T2048 selected AMDGCN artifact
有 `11` 个 lexical `s_barrier`（此前 Z0 汇总的 `10` 是旧统计口径；Stage 7A 使用同一
selected file直接计数）。它的 Python source 没有 `tl.barrier()`；这些 barrier 是 Triton
local-memory/dot pipeline lowering 的结果，不能对 Python 行号做虚假的一对一归因。

| native phase | native source | Z1 对应 phase | 核心差异 |
|:--|:--|:--|:--|
| Q/K/H K32 load + two dots | `chunk_o.py:93-113` | A + B 的 source stage/fragment sequence | native 用 compiler-managed local operand pipeline；Z1 手动把 fragment 反复存入/读出 CTA LDS |
| decay/mask/score BF16 operand | `115-125` | B score serialize | native score 保持 local dot operand；Z1 将两个 score half 显式序列化到 `phase` |
| V load + score-times-V + BF16 store | `127-138` | C V transpose/intra | native TTGIR 释放 Q/K/H local buffers 后再分配 score/V local buffers；Z1 以 CTA-wide phase boundaries 保护共享复用 |

native T2048 TTGIR 有 5 个 local allocation、3 个 local deallocation；这就是它能以
11 个 barrier 完成多阶段局部 pipeline 的直接 evidence。Z1 的 41 个 barrier 不能仅用
“所有权变成 BV64”消掉。

## 4. 最小 Repro

新增 Qwen-free repro：

- [`repro_qwen_bt64_chunko_barrier_stage7a.py`](repro_qwen_bt64_chunko_barrier_stage7a.py)
- [`profile_qwen_bt64_chunko_barrier_stage7a.py`](profile_qwen_bt64_chunko_barrier_stage7a.py)

所有模式都是 WG256、shared BF16、MFMA32（A/C）且 zero scratch/spill。

| experiment | comparison | barrier | output | 解释 |
|:--|:--|--:|:--|:--|
| A | per-lane fragment pack without/with extra barrier | 1 / 2 | bit-exact | 单独的 lane-private write 后 barrier 可去掉 |
| B | score lower half write, then disjoint upper half write | 2 / 1 | bit-exact | 在没有中间 consumer 时，可合并为最终 consumer 前一条 barrier |
| C | all CTA stage, one owner wave MFMA | 1 | finite | owner wave 不意味着 source stage barrier 可以去掉 |

原始 JSON/ISA/readobj：

- [`minimal_repros.md`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)
- [`minimal_repros.json`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.json)

这些 repro 的价值是限定因果：它们证明 A/B 的 barrier 在**最小独立内存关系**中不是
必需的；它们没有证明完整 MFMA pipeline 在不同 wave 进度、反复 MFMA issue 与 LDS reuse
下也可安全删除。

## 5. 唯一允许的 Local Fix 与失败

依据 A/B，实施了唯一一个 phase-scheduling experiment：

```text
删除 A/B/C 的 frag_words pack/reuse barriers
删除两个 B.serialize_score barriers
保留 A.stage_qh、B.stage_qk、C.stage_v
```

理论上它会让静态 barrier 从 `41` 降至 `7`，且没有改动 tile、WG、MFMA、layout、dtype 或
math。该候选第一个完整 Z1 correctness case（T8192）即失败：

| comparison | result |
|:--|:--|
| phase-compact vs frozen Z1 | `bit_exact = false` |
| max abs | `0.00206613541` |
| gate | 失败，要求 bit-exact |

在该失配 kernel 后同一 pytest process 继续编译下一 specialization 时，HIP report 了
memory access fault 并 abort。该 abort 不用于归因；唯一可靠的 stop fact 是更早出现的
T8192 numerical mismatch。没有继续收集该无效候选的 body、rocprof 或 full graph 数据。

为什么最小 repro 不足以放行？最可能是完整 kernel 中 `frag_words` 的 reuse 不只是普通
“同一 lane store 后同一 lane load”：它夹在跨 wave 的 LDS source read、MFMA issue、下一
round LDS overwrite 和非锁步 wave progress 中。一个 barrier 可能同时充当 schedule-wide
phase boundary。当前一次修复同时移除了多类同步，Stage 7A 规则禁止再逐个恢复/扫组合，
所以不能把责任精确归给某一条 barrier。

## 决策

这不是 Case B：LLVM/pre-LTO/final ISA 都未显示额外 compiler-inserted barrier；问题不是
generic backend hazard analysis 平白增加了同步。

也不是可以继续的 Case A：唯一允许的 source compaction 未通过 full Z1 exactness。

所以是 **Case C for the current Avelang source schedule**。停止 Stage 6Z pure-source
native chunk-o 路线，不做 barrier subset sweep、tile/WG sweep、Z2 或 full integration。

后续如果追求端到端性能，下一条独立路线可以是已捕获 native `chunk_fwd_kernel_o` HSACO 的
external-kernel bridge，并明确标记 external integration；它不是 Avelang source kernel
优化。若坚持纯 Avelang source，按此前 Stage 6Z 排序转向 W/U runner-up gap，预期收益较小。

## 复现

```bash
cd /workspace/project/avelang

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_bt64_chunko_barrier_stage7a.py \
  --T 2048 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_bt64_chunko_barrier_stage7a.py \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a \
  --warmup 5 --repeat 20
```

The phase-compact test is intentionally skipped in ordinary collection because
its first full-kernel exactness gate already failed; the source is retained as
a documented failed experiment rather than a candidate baseline.
# Qwen GDN Next Decision After Stage 7A Chunk-O Barrier Audit

## Decision

**Close the Stage 6Z pure-Avelang native chunk-o source route.** Do not create
Z2, do not run full-graph timing, and do not promote the Stage 7A
phase-compact candidate.

The frozen Stage6Z Z1 kernel has 41 static barriers. Stage 7A established an
exact count-preserving chain:

```text
explicit al.syncthreads source schedule
  -> 41 pre-link LLVM barrier calls
  -> 41 pre-LTO assembly barriers
  -> 41 final HSACO s_barrier instructions
```

The backend did not create a hidden surplus of barriers. A one permitted
source scheduling change removed the candidate lane-private and early
score-half barriers, but failed the first full-Z1 exactness case at T8192:

```text
phase-compact vs Z1: max_abs=0.00206613541, bit_exact=false
```

The standalone barrier repro remains valuable evidence: a local per-lane
fragment barrier and a disjoint score/V store barrier can each be removed in
isolation. The full pipeline disproves treating those local facts as a global
license to erase the phase boundaries.

## What Remains Frozen

- Stage6W/X2 and v24/default selector;
- Stage6Z Z1 source and its No-Go status;
- BT64/BV64/BK32/WG256 mapping;
- recurrence ABI and all full paths;
- compiler, LLVM/AMDGPU RA, assembly, and vLLM source.

## Next Single Direction

Choose one independent objective, not both:

1. **End-to-end diagnostic:** integrate the captured native vLLM `chunk-o`
   HSACO behind a strict external-kernel bridge and measure the recoverable
   full-graph gap. It must be labeled external integration, not an Avelang
   source-kernel result.
2. **Pure Avelang source:** leave chunk-o closed and move to the previously
   ranked W/U runner-up gap. Its expected upside is smaller than native
   chunk-o replacement.

Do not reopen Stage6Z with barrier subset sweeps, alternate tile/WG shapes,
or a full-graph "just to see" integration.

## Evidence

- [Stage 7A report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_chunko_barrier_provenance_stage7a_report.md)
- [41-entry barrier ledger](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [minimal repro results](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)
# Triton vs Avelang v28 Lowering and AccVGPR Report

## 1. Summary Conclusion

Two profiling tasks were run for the fixed Qwen GDN chunk_delta_h/chunk_gdr shape:

- `B=1`, `T=2048`, `Hk=4`, `Hv=8`, `K=128`, `V=128`
- BF16 `k`, FP32 `w/u/g`
- chunk size `64`

Main findings:

1. Avelang v28 high `Accum_VGPR_Count` is not caused by `h` store, `vn` store, or decay. Those ablations keep AccVGPR high, and even increase it from `212` to `220`.
2. `pred_only` is low AccVGPR (`40`) and `update_only` is moderate (`136`). The high AccVGPR appears only when pred and update are both present in one kernel.
3. Evidence suggests the v28 AccVGPR jump is caused by combined pred/update accumulator lifetime and reuse pressure around the persistent `h1/h2` state fragments, not by output stores.
4. Triton selected config is `BV=32`, `num_warps=2`, `num_stages=2`.
5. Triton is much faster, but not because it has clearly lower VGPR/AccVGPR. Forced selected-config Triton has `VGPR=128` and `AccVGPR=200` without initial state, or `AccVGPR=224` with initial state/final state. Avelang v28 full has `VGPR=52`, `AccVGPR=212`.
6. The stronger Triton-vs-Avelang evidence is instruction/traffic shape: v28 has much higher MFMA, SALU, VMEM, and LDS instructions, plus larger workgroup/grid work-items and a nonzero LDS block allocation.

So the Triton comparison weakens the narrow claim that v28 is slow simply because Avelang has worse VGPR/AccVGPR allocation than Triton. It supports a broader lowering/geometry-cost hypothesis: Avelang v28 expresses a Triton-like `h1/h2` recurrence, but lowers it into substantially more MFMA and memory/LDS work.

## 2. v28 Ablation Counter Table

Normal benchmark command:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  python profile_v28_accvgpr_ablation.py --T 2048 --variant all --warmup 10 --repeat 30
'
```

Normal benchmark latency:

| variant | latency ms |
|:---|---:|
| `full_v28` | 0.604137 |
| `no_h_store` | 0.578940 |
| `no_vn_store` | 0.577078 |
| `no_decay` | 0.572050 |
| `pred_only` | 0.243642 |
| `update_only` | 0.388157 |

rocprof counters:

| variant | trace us | WG | grid work-items | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_v28` | 577.338 | 256 | 8192 | 36864 | 0 | 52 | 212 | 112 | 327680 | 2758784 | 614912 | 466944 | 1015808 | 1.2819 |
| `no_h_store` | 558.570 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2751744 | 614656 | 401408 | 1015808 | 1.2851 |
| `no_vn_store` | 526.662 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2471808 | 614912 | 401408 | 1015808 | 1.2797 |
| `no_decay` | 544.909 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2176896 | 605952 | 397312 | 1015808 | 1.2749 |
| `pred_only` | 215.420 | 256 | 8192 | 16384 | 0 | 64 | 40 | 112 | 65536 | 1033472 | 79744 | 196608 | 327680 | 1.2433 |
| `update_only` | 341.588 | 256 | 8192 | 20480 | 0 | 16 | 136 | 112 | 262144 | 751232 | 539136 | 165888 | 688128 | 1.2732 |

## 3. v28 AccVGPR Interpretation

Answers to the requested questions:

1. Does `pred_only` already have high AccVGPR?

No. `pred_only` has `AccVGPR=40`, far below `full_v28=212`.

2. Does `update_only` already have high AccVGPR?

Not at the v28 full level. `update_only` has `AccVGPR=136`, close to the older v23/v24 range and far below `212`.

3. Does `full_v28` become high only when pred and update are both present?

Yes. The high value appears in `full_v28` and in variants that keep both pred and update (`no_h_store`, `no_vn_store`, `no_decay`).

4. Do `h`/`vn` stores extend accumulator lifetime?

The evidence says no. Removing `h` store or `vn` store does not reduce AccVGPR. Both variants report `AccVGPR=220`, slightly higher than `full_v28=212`.

5. Does decay extend accumulator lifetime?

The evidence says no. `no_decay` also reports `AccVGPR=220`.

6. Is `AccVGPR=212` caused by unavoidable `h1/h2` state fragments, or by poor accumulator reuse between `pred_acc` and `update_acc`?

Evidence suggests the jump is caused by the combined pred/update region and accumulator lifetime/reuse pressure. The persistent state fragments alone are not enough to reach `212`: `update_only` still has persistent state/update and lands at `136`; `pred_only` lands at `40`. The jump appears when v28 stages persistent state for pred, computes pred partials, creates `v_decay`, and then performs update in the same kernel.

This is still Avelang-side evidence, so the careful conclusion is: evidence suggests the high AccVGPR comes from combined pred+update lowering/lifetime, not from stores or decay.

## 4. vLLM Triton Config and Kernel Identification

Direct vLLM workload:

- Python wrapper: `vllm.model_executor.layers.fla.ops.chunk_delta_h.chunk_gated_delta_rule_fwd_h`
- Triton kernel: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- Regex used for final profiling: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

Autotune discovery before forced profiling:

```text
patched_vllm_rocm_autotune_configs=12->8,disabled_num_stages=4
selected cache entry:
BV=32,num_warps=2,num_stages=2
```

For final counters, the profiling script forced only this selected config:

```text
--force-bv 32 --force-num-warps 2 --force-num-stages 2
```

This avoids rocprof mixing counters from autotune candidate kernels.

Normal direct chunk_delta_h latency:

| Triton case | initial_state | output_final_state | latency ms |
|:---|:---:|:---:|---:|
| selected forced | false | false | 0.174199 |
| selected forced | true | true | 0.181049 |

## 5. vLLM Triton vs Avelang v28 Counter Table

Closest comparison uses vLLM with `initial_state=True`, `output_final_state=True`, because v28 full loads/stores final state in this benchmark. The no-initial-state vLLM result is also included because it was requested first.

| kernel | case | trace us | WG | grid work-items | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Avelang v28 | full_v28 | 577.338 | 256 | 8192 | 36864 | 0 | 52 | 212 | 112 | 327680 | 2758784 | 614912 | 466944 | 1015808 | 1.2819 |
| Triton | no init, no final | 138.767 | 128 | 4096 | 0 | 0 | 128 | 200 | 96 | 97280 | 1600896 | 137600 | 77760 | 293824 | 0.6046 |
| Triton | init + final | 142.111 | 128 | 4096 | 0 | 0 | 128 | 224 | 112 | 98304 | 1637824 | 141120 | 78848 | 298432 | 0.6078 |

Derived ratios, Avelang v28 full vs Triton init+final:

| metric | Avelang / Triton |
|:---|---:|
| trace median | 4.06x |
| MFMA | 3.33x |
| VALU | 1.68x |
| SALU | 4.36x |
| VMEM | 5.92x |
| LDS inst | 3.40x |

## 6. Interpretation

### a. Does Triton have much lower AccVGPR than Avelang v28?

No, not under the selected-config counters.

- Triton no-init: `AccVGPR=200`, slightly lower than Avelang v28 `212`.
- Triton init+final: `AccVGPR=224`, higher than Avelang v28 `212`.

This weakens the narrow hypothesis that v28 is slow because Avelang simply uses much more accumulator register file than Triton.

### b. If Triton also has high AccVGPR but is faster, what explains the speed difference?

The strongest counter evidence is instruction and memory/LDS footprint:

- v28 uses `327680` MFMA vs Triton init+final `98304`.
- v28 uses `466944` VMEM vs Triton init+final `78848`.
- v28 uses `1015808` LDS instructions vs Triton init+final `298432`.
- v28 uses `614912` SALU vs Triton init+final `141120`.
- v28 has `LDS_Block_Size=36864`, while Triton reports `0`.
- v28 launches with workgroup `256` and grid work-items `8192`; selected Triton uses workgroup `128` and grid work-items `4096`.

So Triton is faster despite similar/high AccVGPR because its generated kernel does substantially less work in the profiled counters.

### c. If Triton counters cannot be extracted reliably

They were extracted, but an important methodology issue was found: a normal rocprof run with the autotuner enabled mixes candidate configs into the trace/counter CSV. The report therefore uses forced selected config counters (`BV=32,num_warps=2,num_stages=2`) for the final comparison.

## 7. Exact Commands

Syntax check:

```bash
env PYTHONPYCACHEPREFIX=/tmp/profile_v28_vllm_pycache \
  python3 -m py_compile \
  avelang/test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py \
  avelang/test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py
```

v28 normal benchmark:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  python profile_v28_accvgpr_ablation.py --T 2048 --variant all --warmup 10 --repeat 30
'
```

v28 rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  for variant in full_v28 no_h_store no_vn_store no_decay pred_only update_only; do
    outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_${variant}
    /opt/rocm/bin/rocprofv3 \
      --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      --kernel-include-regex chunk_gdr \
      -d ${outdir} \
      -o v28_accvgpr_${variant} \
      -f csv \
      -- python test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py \
           --T 2048 --variant ${variant} --warmup 2 --repeat 5
  done
'
```

vLLM selected-config normal benchmark:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
    --T 2048 --warmup 10 --repeat 30 \
    --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

vLLM selected-config rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_forced
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    -d ${outdir} \
    -o vllm_triton_chunk_delta_h_forced \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
         --T 2048 --warmup 2 --repeat 5 \
         --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

vLLM selected-config init+final benchmark and rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
    --T 2048 --warmup 10 --repeat 30 \
    --with-initial-state --output-final-state \
    --force-bv 32 --force-num-warps 2 --force-num-stages 2 &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_init_state_forced
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    -d ${outdir} \
    -o vllm_triton_chunk_delta_h_init_state_forced \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
         --T 2048 --warmup 2 --repeat 5 \
         --with-initial-state --output-final-state \
         --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

## 8. File Paths

Scripts:

- `test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py`
- `test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py`

Report:

- `test/examples/linear_attention/vllm_compare/triton_vs_avelang_v28_lowering_and_accvgpr_report.md`

rocprof outputs:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_full_v28`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_h_store`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_vn_store`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_decay`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_pred_only`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_update_only`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_forced`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_init_state_forced`

# Qwen GDN v29 Pred-Only MFMA32 Resource Audit Report

## Summary

v29 MFMA32 pred-only has been fully tested in three forms:

1. k-split v29 pred-only
2. token-split no-reduce v29 pred-only
3. token-split stage-all v29 pred-only

All versions pass BF16-level correctness against the torch reference and generate the intended 32x32 BF16 MFMA path. However, all three are slower than the existing v28/v24-style pred/chunk_gdr direction.

Therefore, v29 MFMA32 pred-only is a no-go for now. Do not implement v29_update_only.

## Results

| Variant | T=2048 latency | Main issue |
|---|---:|---|
| k-split v29 pred-only | 0.482717 ms | AccVGPR=204, high VALU/SALU |
| token-split v29 pred-only | 0.524720 ms | VGPR=128, AccVGPR=168 |
| token-split stage-all v29 pred-only | 0.626070 ms | AccVGPR=216, VALU=4.99M, SALU=345K |

## Stage-all rocprof

| Metric | Value |
|---|---:|
| trace avg | 626.386 us |
| Workgroup_Size | 128 |
| Grid_Size | 4096 |
| LDS_Block_Size | 24576 |
| Scratch_Size | 0 |
| VGPR_Count | 48 |
| Accum_VGPR_Count | 216 |
| SGPR_Count | 112 |
| SQ_INSTS_MFMA | 32768 |
| SQ_INSTS_VALU | 4992320 |
| SQ_INSTS_SALU | 345152 |
| SQ_INSTS_VMEM | 198656 |
| SQ_INSTS_LDS | 165888 |

## Diagnosis

The original hypothesis was that k-split v29 was slow mainly because of cross-wave LDS reduction. The token-split version removed cross-wave reduction but became slower. Therefore, cross-wave reduction is not the dominant bottleneck.

The stage-all version shortened ordinary VGPR live range and reduced VGPR count from 128 to 48, but AccVGPR increased to 216 and VALU/SALU exploded. Therefore, the deeper issue is the source-level 32x32 accumulator path itself: accumulator unpack, high-level address generation, and Avelang lowering around the 32x32 fragment are too expensive.

## Decision

Do not implement v29_update_only.

Do not continue full_v29 based on this MFMA32 pred-only path.

Keep v29 as evidence that source-level MFMA32 works, but current high-level schedule/lowering is not performance-viable.

## Next Direction

Return to the v24 production baseline.

Recommended next optimization directions:

1. chunk_o optimization
2. w_u optimization
3. targeted raw_buffer_store_x4/vectorized store integration in existing v24/v28-style kernels
4. small stage-level optimizations rather than another BT64/BV32 rewrite

# L6 MIR / Regalloc Audit

## Summary

This audit adds late LLVM/MIR/register-allocation evidence for the two existing L6 variants:

- `L6_baseline_current_update`
- `L6_subtile16_stage_full_update_like`

No new source variant was added, and no full Qwen kernel was modified.

The main result is:

- The real Avelang-produced hsaco for `L6_baseline_current_update` contains the problematic high AGPR copy region, including `v_accvgpr_write_b32 a100..a131` and later matching reads.
- The `L6_subtile16_stage_full_update_like` hsaco does not contain high AGPR writes/reads; its explicit AGPR copies stay at `a0..a15`.
- Both variants have the same dynamic MFMA shape count in the relevant kernel: 8 `mfma_32x32x8_bf16` and 32 `mfma_16x16x16_bf16`.
- MIR after `virtregrewriter` reports no VGPR spills and no scratch reservation, so this is not a memory-spill issue. It is register allocation using AGPR copies under pressure.
- The high `a100..a131` values in baseline are ordinary address/data temporaries around global `k` loads and LDS staging, not required MFMA accumulator state.

The strongest classification is:

```text
primary cause:
  broad k_all_t / kall_vec shared-view address lowering,
  plus update-side lifetime pressure,
  causing LLVM/AMDGPU RA to park ordinary VGPR temporaries in AGPRs

not primary:
  update MFMA intrinsic alone
  pred/v_decay value alone
  scratch spill
```

## Artifact Paths

New dump script:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_mir_regalloc_artifacts.py`

Generated artifacts:

- `test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/`

Important files per variant:

- `lowered_optimized.ll`
- `llc_structure.s`
- `stop_after_amdgpu-isel.mir`
- `stop_after_finalize-isel.mir`
- `stop_after_greedy.mir`
- `stop_after_virtregrewriter.mir`
- `stop_after_post-RA-sched.mir`
- `print_after_greedy.txt`
- `print_after_virtregrewriter.txt`

Prior hsaco/ISA evidence reused for final binary behavior:

- `test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/hsaco/`

## Dump Status

Available:

- Optimized LLVM IR from Avelang codegen.
- AMDGPU MIR at `amdgpu-isel`, `finalize-isel`, `greedy`, `virtregrewriter`, `prologepilog`, `postrapseudos`, and `post-RA-sched`.
- Final structured `llc` assembly from the dumped LLVM IR.
- Real hsaco objdump ISA from the Avelang JIT path.

Not available:

- Initial Avelang/ROCDL MLIR text. The current Docker build hits an MLIR printer assertion while dumping initial MLIR:

```text
llvm::dyn_cast ... Assertion `dyn_cast on a non-existent value' failed
```

- Dedicated LLVM live-interval pressure logs. The optimized LLVM build accepted machine dumps, but did not produce a useful live interval/register pressure stream from the attempted flags. We therefore rely on MIR metadata, physical register assignment, and final ISA.

## Counter Context

From the previous L6 profiling pass:

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_current_update` | `34.371` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile16_stage_full_update_like` | `19.189` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_update_mfma_no_pred_dependency` | `10.175` | 96 | 80 | 0 | 20480 | 2048 | 69504 | 8192 | 7168 |
| `L6_update_mfma_minimal_frag` | `3.164` | 12 | 4 | 0 | 1024 | 256 | 4096 | 2432 | 512 |

The baseline-to-subtile improvement keeps the same MFMA count but reduces trace, LDS block, VMEM, LDS instructions, and AccVGPR.

## ISA AGPR Summary

Measured from real hsaco objdump ISA:

| artifact | max AGPR in ISA | `v_accvgpr_write/read` | high AGPR writes | high AGPR reads | MFMA32 | MFMA16 |
|:---|---:|---:|---:|---:|---:|---:|
| baseline hsaco | 131 | 155 / 155 | 51 | 51 | 8 | 32 |
| subtile hsaco | 15 | 32 / 32 | 0 | 0 | 8 | 32 |
| baseline `llc` from dumped LLVM | 15 | 32 / 32 | 0 | 0 | 8 | 32 |
| subtile `llc` from dumped LLVM | 15 | 32 / 32 | 0 | 0 | 8 | 32 |

The `llc` route is still useful for MIR structure and source comments, but it does not reproduce the exact high physical AGPR numbering from the Avelang JIT hsaco. The final hsaco remains the authoritative source for the `a100..a131` symptom.

## Baseline High AGPR Region

The problematic baseline hsaco region writes high AGPRs while computing pointer/address values. Example:

```asm
v_lshl_add_u64 v[16:17], v[16:17], 0, v[2:3]
v_accvgpr_write_b32 a101, v17
...
v_accvgpr_write_b32 a100, v16
...
v_accvgpr_write_b32 a103, v15
v_accvgpr_write_b32 a102, v14
...
v_lshlrev_b64 v[12:13], 10, v[12:13]
v_lshl_add_u64 v[12:13], s[12:13], 0, v[12:13]
v_lshl_add_u64 v[12:13], v[12:13], 0, v[2:3]
v_accvgpr_write_b32 a107, v13
v_accvgpr_write_b32 a106, v12
...
v_accvgpr_write_b32 a131, v13
v_accvgpr_write_b32 a130, v12
```

These instructions are surrounded by address-generation operations such as:

- `v_lshrrev_b32`
- `v_lshlrev_b32`
- `v_lshlrev_b64`
- `v_lshl_add_u32`
- `v_lshl_add_u64`
- `v_sub_u32`
- `v_and_b32`

The later readback region uses the high AGPR values as pointers for global loads and LDS stores:

```asm
v_accvgpr_read_b32 v12, a100
v_accvgpr_read_b32 v13, a101
global_load_ushort v2, v[12:13], off
v_accvgpr_read_b32 v12, a98
s_waitcnt vmcnt(0)
ds_write_b16 v12, v2
...
v_accvgpr_read_b32 v12, a130
v_accvgpr_read_b32 v13, a131
global_load_ushort v2, v[12:13], off
v_accvgpr_read_b32 v12, a128
s_waitcnt vmcnt(0)
ds_write_b16 v12, v2
```

This strongly indicates the high AGPR values are address/data temporaries feeding global `k` loads and LDS staging, not MFMA accumulator outputs.

## MIR Evidence

After `virtregrewriter`, both variants report no scratch/spill fallback:

| variant | LDS size | occupancy metadata | hasSpilledVGPRs | vgprForAGPRCopy | scratchReservedForDynamicVGPRs |
|:---|---:|---:|:---|:---|---:|
| baseline | 45056 | 1 | false | empty | 0 |
| subtile | 32768 | 1 | false | empty | 0 |

The update MFMA region in MIR uses normal low accumulator registers:

```mir
%1259:vreg_64_align2 = DS_READ_B64_gfx9 %85, 0 ...
%1260:vreg_64_align2 = DS_READ_B64_gfx9 %86, 0 ...
%1890:areg_128_align2 =
  V_MFMA_F32_16X16X16BF16_1K_e64 %1259, %1260, 0 ...
...
GLOBAL_STORE_DWORD ... %1890.sub0 ...
```

The source comments on those loads point to:

```text
%ir.669
%ir.invariant.gep63
%ir.gep64.*
%ir.gep68.*
%ir.gep72.*
%ir.gep76.*
```

These are exactly the generic shared-memory addresses produced by the staged update inputs. In the baseline they come from the broad `k_all_t[128,BT]` / `kall_vec` path; in the subtile variant they come from narrower fixed subtile staging.

## LLVM IR Evidence

The optimized LLVM IR shows large addrspace(3) workgroup globals in baseline:

```llvm
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_0 =
  internal unnamed_addr addrspace(3) global [16384 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_1 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_2 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_3 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_4 =
  internal unnamed_addr addrspace(3) global [4096 x i8] undef
```

The IR also contains long runs of generic `getelementptr` chains into addrspace(3), for example:

```llvm
%247 = getelementptr i8, ptr addrspace(3) %222, i32 %.idx17
%248 = getelementptr float, ptr addrspace(3) %247, i32 %221
%249 = getelementptr float, ptr addrspace(3) %248, i32 %213
...
%298 = getelementptr i8, ptr addrspace(3) %261, i32 3168
%299 = getelementptr float, ptr addrspace(3) %298, i32 %221
%300 = getelementptr float, ptr addrspace(3) %299, i32 %213
```

This matches the broad shared-view lowering pattern rather than a compact fixed-layout fragment load.

## Classification Of `a100..a131`

### kall_vec shared-view address lowering

Supported as the primary culprit.

Evidence:

- The high AGPR write region is dominated by address generation.
- The readback region feeds `global_load_ushort` and `ds_write_b16`.
- Baseline uses broad shared K staging and generic `kall_vec` view lowering.
- Subtile reduces the broad shared view and removes high AGPR copies entirely in hsaco.
- MFMA counts are unchanged between baseline and subtile.

### update B operand fragment construction

Contributing, but not the direct high-AGPR instruction source.

Evidence:

- The later update MFMA reads from shared K fragments.
- MIR comments tie the update B operand loads to generic shared GEPs.
- However, the explicit `a100..a131` hsaco writes are mostly pointer/address temporaries before the actual MFMA16 update region.

So the issue is best phrased as:

```text
the update B operand path inherits expensive generic shared K view lowering
```

not:

```text
mfma_16x16 update intrinsically needs high AGPR
```

### pred/v_decay temporary

Secondary pressure source, not the direct high-AGPR source.

Evidence:

- `L6_update_mfma_no_pred_dependency` reduces AccVGPR from 264 to 80, so pred/v_decay lifetime pressure matters.
- But the high `a100..a131` writes are address-generation values, not pred/v_decay arithmetic values.
- The lifetime marker experiment did not change counters, likely because the marker is erased too early.

### generic RA spill/use-AGPR fallback

Supported, with a specific caveat.

It is not scratch spilling:

- `hasSpilledVGPRs: false`
- `Scratch_Size: 0`
- `scratchReservedForDynamicVGPRs: 0`

It is register allocation using AGPR as an overflow/parking class for ordinary VGPR temporaries under pressure. The values being parked are address/data temporaries around global load and LDS staging.

## Should Fixed-Layout Qwen K Fragment Lowering Live In AveLang/MLIR Or LLVM/AMDGPU?

It should live in the AveLang / MLIR / Avelang-AMDGPU lowering layer, before LLVM register allocation.

Reason:

- The problematic structure is visible before LLVM as broad addrspace(3) shared buffers and generic GEP chains.
- LLVM/AMDGPU can only allocate the temporaries it receives; by the time RA runs, the high-level meaning of `k_all_t[128,BT] -> kall_vec -> update B operand` is mostly lost.
- The subtile experiment shows that changing the source/lowering shape removes high AGPR copies without changing the MFMA math.

LLVM/AMDGPU may still need a late lifetime or scheduling primitive later, but fixed-layout K fragment lowering is more naturally and safely handled before LLVM.

## Should Ordinary Address Temporaries Be Restricted From AGPR?

As a diagnostic guard: yes, this is worth exploring.

As the main fix: no, not yet.

A blanket restriction could force more values into VGPRs, increase VGPR pressure, or create real scratch spills. The better primary fix is to avoid generating the large number of ordinary address temporaries in the first place.

Recommended policy:

- First implement fixed-layout K fragment lowering.
- Then optionally add an assertion/debug mode that flags address/pointer temporaries copied through AGPR, so future regressions are visible.
- Only consider a real RA restriction after measuring its effect on VGPR pressure and scratch.

## Next Minimal Compiler Patch

Add a narrow AveLang/MLIR lowering pattern or helper for the Qwen update K fragment.

Target pattern:

```text
k_all_t[128, BT] staged in shared
kall_vec = view(k_all_t, i32, ...)
kall_vec[base_k + lane_col, token_pack]
feeding mfma_16x16x16_bf16_f32 update B operand
```

Lower it to a fixed-layout packed fragment load:

```text
for each update tile:
  compute constant-ish LDS offsets for base_k + lane_col and token pack
  emit compact DS_READ_B64 / packed load sequence
  feed vreg_64_align2 directly to MFMA16 B operand
```

Constraints:

- Preserve current source semantics and output ABI.
- Do not materialize full transposed `k_all_t[128,BT]` when only one 16-column K subtile is consumed.
- Avoid generic shared-view GEP ladders in the update B path.
- Keep this as a specific AMDGPU/Qwen fragment helper first, not a broad lifetime system.

If this patch lands and reduces the L6 baseline toward `L6_subtile16_stage_full_update_like` without scratch, then the same helper should be applied to the production Qwen chunk_gdr lowering path.

## Exact Commands

Artifact dump:

```bash
cd /workspace/avelang
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_mir_regalloc_artifacts.py
```

Key inspection commands:

```bash
rg -n "v_accvgpr_write_b32|v_accvgpr_read_b32|v_mfma|a1[0-9][0-9]|a[89][0-9]" \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/hsaco/*.isa

rg -n "ldsSize|occupancy|hasSpilledVGPRs|scratchReservedForDynamicVGPRs|vgprForAGPRCopy" \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/*/stop_after_virtregrewriter.mir

rg -n "DS_READ|DS_WRITE|GLOBAL_LOAD|V_MFMA|V_ACCVGPR|%ir\\." \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/L6_baseline_current_update/stop_after_virtregrewriter.mir
```

# Compiler Lifetime Boundary Patch Report

## Summary

Implemented a minimal Avelang lifetime-boundary marker:

```python
al.end_lifetime(x, ...)
al.discard(x, ...)
```

The marker is visible in AveLang IR and protected from early DCE by a conservative memory effect.  The current lowering is marker-only: it is erased at AveLang-to-memref lowering and does not yet become LLVM `lifetime.end` or a backend-visible AMDGPU liveness boundary.

## Compiler Files Changed

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.h`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/IR/builtin_module.h`
- `lib/IR/builtin_module.cc`
- `lib/Dialect/AveLang/Transforms/lower_ave_lang_to_memref_pass.cc`
- `python/avelang/language/core.py`
- `python/avelang/language/__init__.py`
- `python/avelang/runtime/jit.py`

## Op Design

`ave.end_lifetime`:

- variadic operands;
- no results;
- verifier requires at least one operand;
- has `MemoryEffects::Write` on the default resource;
- no numerical semantics.

Python DSL entry points:

- `al.end_lifetime(...)`
- `al.discard(...)`

The alias exists so future source can use either wording without another compiler patch.

Python frontend fix:

- `end_lifetime` and `discard` are now exported from `avelang.language`.
- Both stubs are marked `__avelang_builtin__ = True`.
- The JIT dependency collector again ignores functions marked `__avelang_builtin__`, so these DSL stubs are not mistaken for ordinary Python functions that must be decorated with `@avelang.jit`.

## Lowering Behavior

Current behavior:

1. Python frontend emits `cf::EndLifetimeOp`.
2. The op survives early AveLang IR cleanup because it is not pure and advertises a side effect.
3. `EndLifetimeLoweringPattern` erases it in `lower_ave_lang_to_memref_pass.cc`.

This is not yet a real lifetime end for LLVM or AMDGPU register allocation.

Why no stronger lowering was added in this patch:

- The relevant shared/private allocas are hoisted intentionally before/around memref lowering.
- No existing explicit lifetime/dealloc infrastructure was found in the Avelang pass path.
- Lowering to `memref.dealloc` for workgroup/private allocas would be semantically risky.
- A real private lifetime path needs a later pass after memref pointers are materialized.
- A real shared-memory reuse path needs a conservative non-overlap allocator or local rewrite pass.

## Isolated L6 Test Files

Added:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py`

The marker is inserted after `v_decay_t` staging and synchronization:

```python
al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
al.syncthreads()
```

It does not mark `v_decay_t`, K staging buffers, or state values needed by update.

Variants:

- `L6_baseline_no_lifetime`
- `L6_with_end_lifetime_after_vdecay`
- `L6_subtile_no_lifetime`
- `L6_subtile_with_end_lifetime_after_vdecay`

## Build Result

Docker ROCm build command:

```bash
docker exec ac739c57a0bf sh -lc \
  'cd /workspace/project/avelang && cmake --build build-vllm-rocm722 --target _avelang_bindings -j 16'
```

Result:

```text
_avelang_bindings.cpython-312-x86_64-linux-gnu.so linked successfully
```

## Syntax Checks

Local syntax checks passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_lifetime python3 -m py_compile \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py

PYTHONPYCACHEPREFIX=/tmp/pycache_lifetime python3 -m py_compile \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py
```

## Profiling Status

Completed in the Docker ROCm environment after fixing two integration issues:

1. `al.end_lifetime` was missing from the Python DSL export surface.
2. The Docker workspace loaded `python/_avelang_bindings...so`, which was stale.  The freshly built `build-vllm-rocm722/python/_avelang_bindings...so` was copied into `python/`.

Smoke:

```bash
PYTHONPATH=/workspace/project/avelang/python:/opt/avelang/python \
PYTHONDONTWRITEBYTECODE=1 HIP_LAUNCH_BLOCKING=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_lifetime_boundary.py \
  --variant all --seed 20260621 --warmup 1 --repeat 2 --json
```

Result: all four variants ran and produced finite sink checksums.

Profile:

```bash
PYTHONPATH=/workspace/project/avelang/python:/opt/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_mfma32_l6_lifetime_boundary.py \
  --warmup 5 --repeat 20 --rocprof-warmup 2 --rocprof-repeat 5
```

Generated:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/l6_lifetime_boundary_profile_report.md`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary/`
- `test/examples/linear_attention/rocprof_outputs/qwen_mfma32_l6_lifetime_boundary_hsaco/`

Key counters:

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_no_lifetime` | `34.491` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_with_end_lifetime_after_vdecay` | `34.291` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile_no_lifetime` | `19.229` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_subtile_with_end_lifetime_after_vdecay` | `19.269` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |

Deltas:

- Broad-K marker: trace `-0.200 us`, AccVGPR `0`, Scratch `0`.
- K-subtile marker: trace `+0.040 us`, AccVGPR `0`, Scratch `0`.

Because isolated L6 did not improve materially, the full Qwen lifetime-boundary experiment was not created or run.

## Interpretation

At this point the patch proves the frontend/dialect hook can be built and used from source, but it does not produce a compiler-resource improvement.

The unchanged AccVGPR/VGPR/instruction counters confirm the marker is erased too early to affect LLVM/AMDGPU register allocation.  The next compiler step should make the marker survive later or lower it to a real lifetime/scheduling primitive.  Source-level `al.end_lifetime(...)` alone should not be promoted to full Qwen v29.

## Next Required Commands

Do not create the full Qwen lifetime-boundary copy from this marker-only patch.

Next compiler action:

- implement a late lifetime-boundary lowering that survives beyond AveLang-to-memref lowering, or
- lower private values to a real LLVM lifetime/scheduling primitive after memref pointer materialization, then rerun the same L6 profile.
# Qwen MFMA32 L6 Lifetime Boundary Profile Report

## Summary

This report profiles the minimal `al.end_lifetime(...)` marker placed after `v_decay_t` staging and before update MFMA.

- Broad-K baseline lifetime-marker delta: trace `-0.08 us`, AccVGPR `0`, Scratch `0`.

- Broad-K memref-only lifetime-marker delta: trace `-32.2080 us`, AccVGPR `-264.0000`, Scratch `0`.

- K-subtile lifetime-marker delta: trace `-0.16 us`, AccVGPR `0`, Scratch `0`.

- K-subtile memref-only lifetime-marker delta: trace `-17.0650 us`, AccVGPR `-168.0000`, Scratch `0`.


## Marker Placement

```python
# after v_decay_t has been written and synchronized
al.end_lifetime(pred_acc, pred_partial, state_bf16, w_bf16, state_vec, w_vec)
al.syncthreads()
# update MFMA begins here
```
The marker does not end the lifetime of `v_decay_t`, K staging buffers, or any state needed by update.

## Smoke/Finite Checks

| variant | latency_ms | finite | checksum | conclusion |
|:---|---:|:---:|---:|:---|
| L6_baseline_no_lifetime | 0.0471095 | True | 129497 | valid |
| L6_with_end_lifetime_after_vdecay | 0.046509 | True | 129497 | valid |
| L6_memref_only_end_lifetime | 0.025658 | True | 0 | valid |
| L6_subtile_no_lifetime | 0.034692 | True | 129497 | valid |
| L6_subtile_with_end_lifetime_after_vdecay | 0.034852 | True | 129497 | valid |
| L6_subtile_memref_only_end_lifetime | 0.025398 | True | 0 | valid |

## Rocprof Counters

| variant | trace_median_us | VGPR_Count | Accum_VGPR_Count | SGPR_Count | Scratch_Size | LDS_Block_Size | SQ_INSTS_MFMA | SQ_INSTS_VALU | SQ_INSTS_SALU | SQ_INSTS_VMEM | SQ_INSTS_LDS | OccupancyPercent |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_no_lifetime | 34.3710 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.512611 |
| L6_with_end_lifetime_after_vdecay | 34.2910 | 128.0000 | 264.0000 | 112.0000 | 0 | 45056 | 5120 | 182144 | 11200 | 22528 | 28672 | 0.50609 |
| L6_memref_only_end_lifetime | 2.1630 | 8.0000 | 0 | 16.0000 | 0 | 0 | 0 | 576.0000 | 384.0000 | 2048 | 0 | 0.0791932 |
| L6_subtile_no_lifetime | 19.3490 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.423275 |
| L6_subtile_with_end_lifetime_after_vdecay | 19.1890 | 96.0000 | 168.0000 | 112.0000 | 0 | 32768 | 5120 | 120128 | 11264 | 16384 | 21504 | 0.421452 |
| L6_subtile_memref_only_end_lifetime | 2.2840 | 8.0000 | 0 | 16.0000 | 0 | 0 | 0 | 576.0000 | 384.0000 | 2048 | 0 | 0.0689527 |

## Deltas

| comparison | trace_delta_us | AccVGPR_delta | VGPR_delta | Scratch_delta | MFMA_delta | VALU_delta | VMEM_delta | LDS_delta |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline + marker vs baseline | -0.08 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| baseline + memref-only marker vs baseline | -32.2080 | -264.0000 | -120.0000 | 0 | -5120 | -181568 | -20480 | -28672 |
| subtile + marker vs subtile | -0.16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| subtile + memref-only marker vs subtile | -17.0650 | -168.0000 | -88.0000 | 0 | -5120 | -119552 | -14336 | -21504 |

## Static ISA Counts

| variant | v_mfma_f32_32x32x8_bf16 | v_mfma_f32_16x16x16_bf16 | global_load | global_store | buffer_load | buffer_store | ds_read | ds_write | s_barrier | s_waitcnt | v_add | v_add3 | v_lshl | v_lshl_add | v_or | v_bfe | max_vgpr_index_static_best_effort | max_acc_index_static_best_effort |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L6_baseline_no_lifetime | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_with_end_lifetime_after_vdecay | 8.0000 | 32.0000 | 144.0000 | 64.0000 | 2.0000 | 2.0000 | 88.0000 | 152.0000 | 6.0000 | 180.0000 | 192.0000 | 72.0000 | 722.0000 | 401.0000 | 166.0000 | 73.0000 | 255.0000 | 131.0000 |
| L6_memref_only_end_lifetime | 0 | 0 | 0 | 32.0000 | 2.0000 | 2.0000 | 0 | 0 | 0 | 9.0000 | 6.0000 | 0 | 2.0000 | 1.0000 | 0 | 0 | 11.0000 |  |
| L6_subtile_no_lifetime | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_subtile_with_end_lifetime_after_vdecay | 8.0000 | 32.0000 | 96.0000 | 64.0000 | 2.0000 | 2.0000 | 80.0000 | 104.0000 | 6.0000 | 131.0000 | 208.0000 | 73.0000 | 410.0000 | 242.0000 | 116.0000 | 73.0000 | 223.0000 | 15.0000 |
| L6_subtile_memref_only_end_lifetime | 0 | 0 | 0 | 32.0000 | 2.0000 | 2.0000 | 0 | 0 | 0 | 9.0000 | 6.0000 | 0 | 2.0000 | 1.0000 | 0 | 0 | 11.0000 |  |

## Conclusion

Interpretation belongs in `compiler_lifetime_boundary_patch_report.md`, which combines these counters with the compiler patch behavior.
# Full v29 K-Fragment Rewrite Regression: Root Cause

## Question

Why does the persistent K-fragment producer-consumer rewrite improve isolated
L6 but make full v29 BT64 chunk_gdr slower, with AccVGPR 384, scratch 736 B,
and higher VMEM?

This is not another audit of broad K staging. That result was already known.
The focus is why the successful L6 rewrite does not compose with the complete
recurrence.

## Result

Primary classification: Category E.

The full regression is caused by the rewrite B-fragment generic vector-load
path interacting with the real full-v29 live set:

- two MFMA32 pred tiles with 16-f32-element accumulators;
- state_bf16, w_bf16, and pred_partial live through correction;
- full BV by BT v-decay staging;
- state update/writeback; and
- the num_chunks equals 32 loop.

The reduced ladder proves the outer loop and state writeback are necessary
pressure amplifiers but insufficient alone to create scratch. Full v29 is the
first tested configuration with the real MFMA32 pred accumulator/live
structures plus rewritten generic B loads; it crosses the allocation
threshold and spills.

This is not Category A or B:

- static global-load count is unchanged: 198 in original and rewrite;
- the pass logs one matched producer, one erased broad producer store, and one
  inserted replacement producer;
- dynamic MFMA and LDS instruction counts are unchanged; and
- there is no evidence that old and new K staging coexist.

The extra dynamic VMEM is consistent with spill traffic, not duplicate K
loading: 399360 to 601984 while static global loads remain 198.

## Original Versus Rewrite At T=2048

| Metric | original full v29 | rewrite full v29 |
|:---|---:|---:|
| normal chunk_gdr ms | 0.837143 | 1.338669 |
| rocprof trace median us | 830.974 | 1302.635 |
| workgroup / grid work-items | 128 / 4096 | 128 / 4096 |
| LDS block bytes | 61440 | 61440 |
| scratch bytes | 0 | 736 |
| VGPR | 128 | 128 |
| AccVGPR | 264 | 384 |
| SGPR | 112 | 112 |
| MFMA | 294912 | 294912 |
| VALU | 4977280 | 3180992 |
| SALU | 810496 | 567808 |
| VMEM | 399360 | 601984 |
| LDS instructions | 1242304 | 1242304 |

The rewrite is bit-exact relative to original v29 for h and final_state at
T=512, 1024, and 2048. Original v29 has a separate nonzero-W
reference-correctness issue; it is not caused by rewrite.

## Artifact-First Diff

Pass debug output for exact full v29 at T=2048:

    persistent_ops_seen=4
    consumers=4
    producer_for_depth=1
    direct_outer_for_upper=32
    broad_producer_stores_erased=1
    compact_producer_loops=1
    cloned_scalar_reloads=3
    unrewritten=0

The L6 anchor has the same producer depth and three cloned scalar reloads,
but direct outer loop upper is 2, not 32.

The current full experimental pass uses a diagnostic full-width 128 by 64
replacement tile to preserve the BT64 fragment coordinate system. The broad
producer loop is erased; eraseDeadSharedChain removes the old
view/reinterpret-cast/allocation chain after its final use.

| Static HSACO metric | original | rewrite |
|:---|---:|---:|
| global/buffer/flat loads | 198 | 198 |
| ds_read plus ds_write | 492 | 498 |
| explicit v_accvgpr_write_b32 | 16 | 16 |
| explicit high writes a100 or greater | 0 | 0 |
| max explicit AGPR write index | 3 | 3 |

The historical a100 through a131 observation was not reproduced in the
current exact HSACO pair. This does not contradict rocprof AccVGPR 384:
that counter measures allocated accumulator registers, not only explicit
write moves. The JIT path does not currently expose a full MIR/virtreg dump,
so spill-vreg attribution cannot be made more specific than the HSACO and
rocprof evidence.

## Reduced Full-Loop Ladder

All variants are finite. R0 is one window; R1 through R4 use 32 BT64 windows.
The reduced ladder omits the real MFMA32 pred schedule, which is the remaining
full-only feature.

| Variant | trace us | VGPR | AccVGPR | scratch | VMEM | MFMA | LDS inst | static global loads | max static AGPR |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 isolated rewrite | 11.777 | 92 | 84 | 0 | 5376 | 4096 | 13312 | 80 | 7 |
| R1 plus BT64 window loop | 314.006 | 124 | 132 | 0 | 172032 | 131072 | 425984 | 80 | 3 |
| R2 plus pred/v-decay-like live values | 396.429 | 64 | 136 | 0 | 180224 | 131072 | 425984 | 84 | 3 |
| R3 plus state update/writeback | 406.684 | 72 | 192 | 0 | 182272 | 131072 | 459968 | 116 | 3 |
| R4 full-loop skeleton | 398.132 | 72 | 192 | 0 | 182272 | 131072 | 459968 | 116 | 3 |

R1 is the first regression boundary. Its dynamic MFMA, LDS, and VMEM counts
are 32 times R0 because it runs 32 windows; its static global-load count stays
80. This is ordinary long-loop amplification, not duplicate K loads or
duplicated compact staging per window.

R3 and R4 reach AccVGPR 192 without scratch. Full v29 adds the actual MFMA32
pred schedule and its 16-element accumulator mapping, state_bf16, w_bf16,
and pred_partial. That missing live region combines with the rewritten update
fragment path to produce 384 AccVGPR and scratch.

## Scalar Reloads And Placement

The pass clones exactly three private scalar reloads: thread id, key head, and
token-window base. L6, R0, and R1 all clone the same three values. R1 has no
scratch, so Category C is excluded as the primary cause.

The replacement allocation is inside the direct outer loop. It is a genuine
lifetime concern, but not sufficient to explain the spill: R1 keeps that
shape with no scratch.

## One Minimal Fix Attempt

The one allowed local candidate hoisted the replacement compact alloca before
direct outer loops with static upper at least four, leaving the stage loop at
the producer site. It preserved full-v29 semantics but failed:

| Metric | original rewrite | hoisted candidate |
|:---|---:|---:|
| normal T=2048 ms | 1.338669 | 1.335064 |
| trace us | 1302.635 | 1301.974 |
| AccVGPR | 384 | 384 |
| scratch bytes | 736 | 736 |
| VMEM | 601984 | 601984 |

The candidate was reverted. L6 remains on its existing short-loop placement.

## Exact Next Patch Proposal

Do not create another Qwen source variant. Keep the persistent B-fragment
operation through GPU lowering and lower it directly to a fixed packed LDS
read feeding MFMA16, rather than replacing it with a generic dynamically
indexed vector.load. First validate that late fragment operation in a reduced
repro containing the real MFMA32 pred accumulator schedule. The acceptance
gate is removal of scratch and a material reduction from AccVGPR 384.

## Artifacts

- repro_qwen_kfrag_full_loop_regression.py
- profile_qwen_kfrag_full_loop_regression.py
- rocprof_outputs/qwen_full_kfrag_rewrite_regression_audit/
# Full v29 LTO MIR Audit and Pred Streaming Experiment

## Scope

This report answers two narrow questions without modifying a production
baseline:

1. What exactly creates the full-v29 K-fragment rewrite's `736 B` scratch and
   `Accum_VGPR_Count=384` report?
2. Can one source-level MFMA32 accumulator serialization change reduce the
   pred epilogue's register/LDS pressure while preserving original-v29
   semantics?

The audited kernel is the full v29 K-fragment rewrite experiment, not v24:

`_qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32`.

## Exact LTO Debug Chain

The initial JIT LLVM IR is not sufficient here: final AMDGPU register
allocation occurs inside ROCm `ld.lld` full LTO. The Avelang backend now
records a replayable linker command when this variable is set:

```bash
AVELANG_AMDGPU_LINK_DEBUG_DIR=<directory>
```

The backend change is in `lib/Target/AMDGPU/amdgpu_backend.cc`. It preserves
the JIT pre-link bitcode, final code object, and replayable argv. The replay
script appends the exact LTO-plugin flags:

```text
-plugin-opt=save-temps
-plugin-opt=-print-before=greedy
-plugin-opt=-print-after=greedy
-plugin-opt=-print-after=virtregrewriter
-plugin-opt=-print-after=prologepilog
```

Artifacts:

- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/linked.hsaco.0.5.precodegen.bc`
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/post_lto_precodegen.ll`
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_07.mir`:
  spill-producing post-greedy MIR
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_08.mir`:
  post-virtregrewriter MIR
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_18.mir`:
  later physical-register MIR containing the high-AGPR copies

The reproducible driver is
`replay_qwen_v29_lto_mir.py` in this directory.

## Exact Scratch Location

The fresh T=2048 profile of the full K-fragment rewrite is:

| metric | value |
|:--|--:|
| trace median | `1302.317 us` |
| scratch | `736 B` |
| VGPR | `128` |
| AccVGPR | `384` |
| SGPR | `112` |
| LDS block | `61440 B` |
| MFMA | `294912` |
| VMEM | `601984` |

The final code object independently reports:

```text
.private_segment_fixed_size: 736
.vgpr_spill_count: 190
.sgpr_spill_count: 25
.agpr_count: 256
```

The exact post-greedy machine section has:

| MIR operation | count | virtual-register classes |
|:--|--:|:--|
| `SI_SPILL_AV32_SAVE` | `70` | `vgpr_32` |
| `SI_SPILL_AV64_SAVE` | `60` | `vreg_64_align2`, `av_64_align2` |

Thus the VGPR spill-word count is exactly:

```text
70 * 1 + 60 * 2 = 190
```

This matches `.vgpr_spill_count: 190`; it is the direct MIR explanation for
the `736 B` private segment. Representative spilled virtual registers are
`%6842:vreg_64_align2`, `%6848:vreg_64_align2`, `%6854:vreg_64_align2`, and
the scalar sequence `%7554` through `%7863:vgpr_32`. The full list is in
`exact_lto_postra/summary.json` and `kernel_section_07.mir`.

`Accum_VGPR_Count` is a profiler resource metric, not a one-to-one virtual
register ID. The direct mapping available from MIR is physical allocation:
the final physical section contains high AGPR copies beginning at
`$agpr100` and reaching `$agpr254` (`166` high-AGPR copy lines). The sequence
is preceded by broad `V_LSHL_ADD_U64` address formation and followed by
fragment copies feeding MFMA paths. The post-LTO LLVM IR contains repeated
`<2 x i64>`/`<2 x i32>` extraction chains in the same generic vector-style
lowering region.

This is strong evidence that the rewrite's generic dynamic vector/address
path crosses the AGPR/VGPR allocation threshold. It does not prove that every
high AGPR is a K fragment alone: the exact machine output shows the combined
address/fragment/live-region composition, including pred and update state.

## One Pred Accumulator Serialization Experiment

New experimental kernel:

`vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_pred_epilogue_streaming_exp.py`

Only the pred MFMA32 epilogue changes. The original stores `pred_acc[16]`
directly to a permuted shared `pred_partial[2,32,32]` tile. The experiment
stores the same values in a lane-major `pred_acc_serial[2,64,16]` tile, with
the same 8 KiB LDS footprint. The correction epilogue uses the inverse
MFMA32 layout to reload the two K-half partials and generate `v_decay`.

No pred schedule, update recurrence, K staging, chunk size, or public
`h/final_state` interface changes.

### Semantic Check Against Original v29

| T | h max abs | final-state max abs |
|--:|--:|--:|
| 64 | `0` | `0` |
| 512 | `0` | `0` |
| 1024 | `0` | `0` |
| 2048 | `0` | `0` |

The experiment is bit-exact relative to original v29. This does not resolve
the separate known nonzero-W v29-vs-reference recurrence issue.

### Normal Timing

| T | original v29 ms | serialized epilogue ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.235409` | `0.222250` | `1.0592x` |
| 1024 | `0.437590` | `0.430279` | `1.0170x` |
| 2048 | `0.835021` | `0.819657` | `1.0187x` |

### T=2048 rocprof

| metric | original | serialized epilogue | change |
|:--|--:|--:|--:|
| trace median us | `827.169` | `785.347` | `-5.06%` |
| VGPR | `128` | `128` | `0` |
| AccVGPR | `264` | `256` | `-8` |
| scratch | `0 B` | `0 B` | `0` |
| LDS block | `61440 B` | `61440 B` | `0` |
| MFMA | `294912` | `294912` | `0` |
| VALU | `4977280` | `4961792` | `-15488` |
| SALU | `810496` | `810688` | `+192` |
| VMEM | `399360` | `399360` | `0` |
| LDS instructions | `1242304` | `1164480` | `-77824` |
| occupancy percent | `0.64499` | `0.64514` | effectively unchanged |

## Conclusion

The LTO debug chain now proves the `736 B` scratch is real register spilling:
`190` VGPR spill words are visible in exact post-greedy MIR. It also exposes
the high `$agpr100..$agpr254` copy region that accompanies the generic
vector/address lowering under the full pred+update live region.

The one allowed source experiment is a positive but small result. Compact
lane-major serialization eliminates the permuted accumulator-unpack path,
keeps original-v29 semantics bit-exact, reduces AccVGPR by `8`, and improves
T=2048 trace by `5.06%`. It does not eliminate the deeper full-composition
pressure or make v29 a production candidate. Keep v24 as the production
baseline; use this as evidence that a future design should keep MFMA32 output
serialization compact rather than expanding it through generic permuted
epilogue lowering.

## Commands

```bash
# Capture an exact Avelang LTO input while compiling the full rewrite.
export AVELANG_AMDGPU_LINK_DEBUG_DIR=\
  /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_link_replay

# Replay the captured ROCm LTO command with pre/post RA dumps.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py \
  --argv-file test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_link_replay/amdgpu-link-0.argv.txt \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_v29_pred_epilogue_streaming.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_pred_epilogue_streaming.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

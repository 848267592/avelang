# Qwen GDN gfx942 BT64：Stage 4 到 Stage 6Y 完整优化实验复盘

这是一份完整的中文实验档案，不是结果摘要。它保留 Stage 4--6S 的详细历史，
并用真实已经完成的 Stage 6T、6T-Golden、6U、6V、6W、6X 和 6Y 结果替换了
旧版“未来计划”。读者可以从中看到：问题如何缩小、每段源码为什么改、失败如何被
记录、哪些数字可以比较，以及什么时候必须停止猜测。

**必须先读的口径更正：**

```text
正式 correctness 与性能排名：完整 Eager public API
timing_contract = eager_public_api
cuda_graph_used = false

Graph replay / private body / isolated HSACO / rocprof trace：仅诊断，不进正式排行榜
```

当前状态：v24 BT16 production/default 不变；W1 是上一 BT64 experimental baseline；
X2/Stage 6X 是当前 **Avelang BT64 experimental baseline**，未改 production selector。
正式 paired Eager confirmation 中，X2 对 W1 在 T=512--16384 都稳定更快；T=2048 gain
为 `26.020 us`（HIP 95% CI `[23.304,28.563] us`），T=8192 为 `82.823 us`
（`[80.432,85.154] us`），slope 从 `6.341` 降至 `5.694 us/chunk`。X2 在该批 T<=4096
快于 vLLM，T>=8192 仍落后，所以不能声称全面超过 vLLM。

本文把 Stage 6R 所捕获的 ABI/config/code-object 统称为 **current-vLLM specialization**；
它与历史 asm-v0 不是同一个 recurrence specialization。本文的唯一后续登记项是
**Stage 6Z native BT64 chunk-o ownership**：Stage 6Y 已证明 chunk-o 有
`+1.286 us/chunk` 的主要 X2-vLLM body slope 差；它尚未实现，不能被误读成已经融合。

原始旧报告已原样备份为
[report_final_before_stage6w_update.md](report_final_before_stage6w_update.md)。具体旧结论如何修正，见
[修订日志](report_final_revision_log.md)和[事实冲突记录](report_final_fact_corrections.md)。源码、测试、数据和报告的路径见[证据索引](report_final_evidence_index.json)。

---

## 1. 先理解整个 Qwen GDN pipeline

这条路径不是一个单独 kernel，而是多个阶段串起来：

```text
g cumsum
  -> KKT
  -> solve
  -> W/U
  -> chunk_gdr recurrence
  -> chunk_o
  -> BF16 public output
```

本次 Stage 4 的核心原则是：

```text
只优化非 recurrence 的高级 Avelang 代码
保持 gfx942 asm recurrence 完全不变
```

固定目标形状：

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

最终使用的 Stage 4 源文件是：

[qwen_gdn_bt64_nonrecurrence_mfma_v2.py](../vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py)

---

## 2. 为什么要做 Stage 4

### 2.1 Stage 2 的情况：功能正确，但上游和下游是 scalar fallback

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

### 2.2 Stage 3 的情况：已经 MFMA 化，但 BT64 ownership 还不够好

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

## 3. 实验总策略

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

## 4. KKT：从单线程 scalar 到 BT64 native MFMA

### 4.1 原来的代码和问题

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

### 4.2 Stage 4 KKT-S0 的代码变化

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

### 4.3 为什么先选这个方案

原因不是单纯“MFMA 越多越好”，而是：

- v24 BT16 KKT 已经证明数学和 layout 是可靠的；
- v20 BT32 KKT 已经证明 token matrix 可以 MFMA 化；
- 16x16 tile 的 fragment mapping 在仓库中已有稳定范例；
- 不需要引入新的 32x32 fragment layout 风险；
- causal mask 可以在 MFMA 后安全处理。

### 4.4 结果

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

## 5. W/U：从重复 token16 MFMA 到 BT64 四 wave ownership

### 5.1 W/U 的数学

W 和 U 的共同形式是一个 chunk-local 小矩阵乘：

```text
A_w[t,s] = a_solved[t,s] * beta[s] * exp(g[s])
A_u[t,s] = a_solved[t,s] * beta[s]

W = A_w @ K
U = A_u @ V
```

输出为 FP32，供 asm recurrence 使用。

### 5.2 Stage 3 之前的主要问题

历史 v14/v24 的 MFMA16 microkernel 是为 BT16 设计的。直接把它扩展到
BT64，容易形成：

```text
4 个独立 token16 ownership
每个 tile 都重复读取 A / K / V / g / beta
每个 tile 都单独做 correction 和 store
```

这种方案虽然出现了 MFMA，却没有充分利用 BT64 内部的共享数据。

### 5.3 W/U-S0：先保守地改 ownership

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

### 5.4 W/U-S0 的第一版：保留 scalar residual correction

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

### 5.5 失败尝试：删除 correction

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

### 5.6 W/U-S1：把 residual correction 也 MFMA 化

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

### 5.7 W/U-S1 结果

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

## 6. chunk-o：从多个小 tile 到一个 BT64 cooperative CTA

### 6.1 chunk-o 的计算结构

chunk-o 需要计算两类贡献：

```text
inter-state contribution:
    q @ H

intra-chunk causal contribution:
    score(q,k,g) @ V_new
```

然后把两者相加，得到输出。

### 6.2 Stage 2/Stage 3 的问题

Stage 2 的 generic chunk-o 是单线程 scalar fallback：

- workgroup=1；
- MFMA=0；
- 需要大量 q/k/vn/h global load；
- private scratch；
- T=2048 trace 超过 11 ms。

Stage 3 已经使用 MFMA16，但仍然保留比较分散的 token16 tile ownership，
跨 tile 的 Q/K/H/V-new 复用不充分。

### 6.3 chunk-o-S0 的代码变化

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

### 6.4 失败尝试：合并 inter/intra accumulator

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

### 6.5 chunk-o-S0 结果

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

## 7. solve：为什么审计了但没有重写

### 7.1 当前 solve 是什么

v18 solve 的数学是 lower-triangular recurrence：

```text
M[row,col] = M0[row,col]                         if col >= row
M[row,col] = M0[row,col]
                + sum_i<row M0[row,i] * M[i,col]  if col < row
a_solved[row,col] = M[row,col] + diagonal_identity
```

v18 使用 128-thread workgroup，把一个 row 内的 column/update 并行化，
但 row dependency 仍然必须顺序推进。

### 7.2 审计结果

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

## 8. full pipeline 的最终结果

### 8.1 完整延迟

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

### 8.2 frozen correctness

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

### 8.3 资源和 ISA

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

## 9. 哪些方案最终保留，哪些方案放弃

### 保留

1. KKT：BT64 拆成 4x4 token16 tile，64-thread MFMA16。
2. W/U：256-thread BT64 ownership，主 MFMA + residual MFMA。
3. chunk-o：256-thread cooperative CTA，共享 Q/K/H/V-new，inter/intra accumulator 分离。
4. v18 solve：不改，作为当前正确 baseline。
5. asm recurrence：不改，直接复用冻结版本。

### 放弃

1. W/U 只保留 BF16 main MFMA：速度快但数值错误。
2. W/U 保留 scalar residual：正确，但 T=2048 只有约 `1.084x`，不够好。
3. chunk-o 合并 inter/intra accumulator：误差约 `2e-2~3.4e-2`。
4. 继续改 v29 accumulator/lifetime/compiler lowering：不属于本轮 BT64 高级代码目标，且已有 resource cliff 证据。
5. 修改 asm recurrence：当前 asm 是稳定且高效的冻结组件，没有理由在本轮冒险。
6. 直接用 vLLM full wrapper：违反实验边界，不能作为 Avelang 候选实现。

---

## 10. 最终瓶颈和下一步

需要区分两个概念：

### 整体最大 stage

T=2048 时，冻结 asm recurrence 约 `0.2015 ms`，是单个 stage 中最大的。
但它是本轮明确禁止修改的 asm kernel，而且已经是稳定实现。

### 当前可继续优化的最大高级 stage

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

## 11. 如何重现实验

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

## 12. 一句话总结

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

# 续篇：Stage 5 到 Stage 6S 的完整实验链、失败原因与推理方法

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

## 13. Stage 4 之后，我们为什么先优化 solve

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

## 14. Stage 5A：solve 根因审计

### 14.1 为什么做这个实验

v18 solve 正确、无 scratch、无 spill，但 body 明显慢。这里有两种完全不同的可能：

```text
可能 A：算法结构本身串行，应该换算法；
可能 B：算法合理，只是编译器 lowering 或寄存器分配不好。
```

如果是 A，修改 compiler 没有意义；如果是 B，贸然换数学结构又会增加正确性风险。

所以 Stage 5A 的目标不是“让 solve 变快”，而是把这两个可能分开。

### 14.2 审计发现

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

### 14.3 关键测量

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

### 14.4 结论

Stage 5A 的结论是：

> solve 的主因是算法调度结构，不是 AMDGPU RA 或普通 lowering 失误。

因此唯一合理的下一步是：

```text
高层 AveLang 实现
FP32 BT64 4×16 hierarchical block inverse
一个 256-thread / 4-wave CTA
使用现有 FP32 MFMA primitive
```

### 14.5 这一步教会我们的东西

遇到慢 kernel 时，不要先问“怎样减少几条指令”，而要先问：

```text
当前依赖图是否天然串行？
参考实现是否使用了不同的数学分解？
```

Stage 5A 避免了一次错误的 compiler/assembly 优化支线。

---

## 15. Stage 5B：hierarchical FP32 solve v1

### 15.1 为什么做

Stage 5A 已经证明 v18 的问题是 63-row 串行依赖。因此 Stage 5B 不再微调 v18，
而是实现一个独立的层次化 FP32 solve：

- BT64 拆成 4×4 个 16×16 block；
- 一个 chunk/head 对应一个 CTA；
- workgroup 256，4 waves；
- block product 使用 `mfma_16x16x4_f32_f32`；
- accumulator 保持 FP32；
- 明确限制 LDS、scratch 和 spill。

### 15.2 实现原则

最重要的不是“使用 MFMA”本身，而是缩短依赖链：

```text
v18：
row0 -> row1 -> row2 -> ... -> row63

hierarchical：
4个对角块并行/局部处理
-> 三层有限的block DAG
```

这把 63 层依赖压缩成少量 block-level 阶段。

### 15.3 结果

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

### 15.4 为什么这次算成功

它同时满足三项：

1. 数学结构改变有明确依据；
2. 正确性没有依赖放宽阈值；
3. 性能和资源都朝正确方向变化。

它没有修改 compiler 或 assembly，说明高级代码足以表达这个 block DAG。

---

## 16. Stage 5C：把新 solve 接回 Stage 4 full graph

### 16.1 为什么 standalone 成功后还必须做 full integration

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

### 16.2 正确性

- 新集成测试：`6 passed`；
- 原 Stage 4 默认路径：`29 passed`；
- public output 和 final state 均在冻结阈值内；
- v1 solve 资源为 `VGPR=44`、`AccVGPR=4`、`Scratch=0`。

### 16.3 性能结果

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

### 16.4 第一轮分段 event 诊断

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

## 17. Stage 5D：downstream state-coupling 审计

### 17.1 为什么做

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

### 17.2 做过的控制实验

#### A. 冻结两条执行图

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

#### B. 连续 tail 单 event

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

#### C. Canonical-data 控制

两种 solve 都执行，但丢弃输出；downstream 始终消费同一份 bitwise-identical
canonical tensor。

T=2048：

```text
after v18：约 0.2675 ms
after v1： 约 0.3321 ms
差距：     约 64.6 us
```

因此 solved 数值差异不是主因。

#### D. 同 consumer pointer 控制

两种 solve 的结果 copy 到同一个 canonical consumer buffer，copy 放在计时区间外。

差距仍约 `68 us`，说明 downstream 可见 pointer 和 alignment 不是主因。

但这一步仍未关闭“两个 solve 自身写入不同物理输出地址”这一变量，因此后来还需要
Stage 5E。

#### E. Warm 与大工作集扰动

T=2048：

```text
正常连续执行：差距约 68.6 us
先运行一次相同 tail：差距约 0.12 us
512 MiB workload 扰动：差距约 -0.08 us
reduction prime：差距仍约 75 us
```

这说明差距属于一种可被较长 GPU workload 重置的瞬态状态。

### 17.3 得到什么结论

高置信度结论：

> v18 solve 会留下一个对后续 tail 有利的瞬态执行状态；v1、no-solve 和短 dummy
> predecessor 不会。

但没有足够证据把它唯一归因于：

- L2/TCC/TCP cache；
- 短时频率/功耗爬升；
- runtime queue pacing；
- CU/wave 状态。

### 17.4 为什么这个“没有精确根因”的实验仍有价值

它排除了很多看似合理但错误的修复：

- 不应因为输出误差小就怪数值；
- 不应随意修改 W/U；
- 不应修改 recurrence 汇编；
- 不应增加 dummy warmup；
- 不应根据插 event 的分段数字做加法。

负结果的价值在于收缩问题空间。

---

## 18. Stage 5E：direct common-output pointer

### 18.1 为什么做

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

### 18.2 实现和正确性

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

### 18.3 关键结果

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

### 18.4 资源和 profiler 结果

- downstream 动态指令数相同；
- TCP 总访问数相同；
- TCC 最大相对差约 `0.232%`；
- 没有发现可解释 64 us 的 cache 流量差；
- 粗粒度 clock/power telemetry 没有稳定分叉；
- profiler 本身会放大 dispatch gap。

### 18.5 结论

> output pointer/address 被排除为主因。

此时只剩“前驱诱发的瞬态 GPU 状态”这一操作层结论，但精确硬件机制仍未识别。

这一步失败在“没有找到可修复的 pointer 问题”，但成功关闭了一个重要变量。

---

## 19. Stage 5F：低扰动可观测性 Go/No-Go

### 19.1 为什么做

Stage 5D/5E 已经证明现象真实，但没有分离：

- cache residency；
- runtime pacing；
- clock/power；
- CU/wave 状态。

继续 profiler 之前必须先确认：

> profiler 本身是否足够低扰动，能观察一个约 63 us 的效应？

如果 profiler 会改变该效应，任何 timestamp 和 counter 因果分析都不可信。

### 19.2 预注册 gate

只有同时满足以下条件才允许继续：

- 不修改 graph；
- graph 内不插 event；
- latency 扰动足够小；
- A/B penalty 扭曲足够小；
- 能提供 cache、dispatch 或短时状态信息。

### 19.3 实测结果

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

### 19.4 决策

Stage 5F 正式宣布：

```text
Stage 5 transient-state root-cause branch closed
```

没有继续 PMC replay、PC sampling 或更多 profiler 花样，因为这会违反预注册 stop rule。

### 19.5 这一步最重要的学习点

优化研究中必须允许：

> “现有工具无法可靠归因，因此停止。”

这不是放弃，而是防止无限追逐不可观测机制。一个严格的 No-Go
通常比一个听起来合理但无证据的“L2 根因”更有科研价值。

---

## 20. 为什么后来感觉“离 vLLM 越来越远”

这里必须解释测量口径变化。

### 20.1 Avelang 实际一直在变快

按当时各阶段自己的测量环境：

```text
Stage 2：约 16.04 ms
Stage 3：约 1.09 ms
Stage 4：约 0.476 ms
Stage 5C：约 0.454 ms
Stage 6A CUDA Graph：约 0.336 ms
```

Avelang 并没有因为优化而变慢。

### 20.2 为什么 gap 看起来扩大

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

### 20.3 Stage 6A 的关键 slope

```text
Avelang：8.771643 us/chunk
vLLM：   4.365784 us/chunk
gap：    4.405859 us/chunk
```

所以真正问题是：

> Avelang 每处理一个 BT64 chunk 仍做了更多结构性工作。

### 20.4 后续统一规则：Eager public API

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

## 21. Stage 6A：严格 full-graph gap audit

### 21.1 为什么做

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

### 21.2 同口径 full 结果

当时采用 CUDA Graph replay 作为结构化对比口径：

| T | Avelang | vLLM | gap |
|--:|--:|--:|--:|
| 512 | `0.122823 ms` | `0.100930 ms` | `21.893 us` |
| 1024 | `0.191845 ms` | `0.128952 ms` | `62.893 us` |
| 2048 | `0.335539 ms` | `0.188900 ms` | `146.639 us` |
| 4096 | `0.607022 ms` | `0.324142 ms` | `282.880 us` |
| 8192 | `1.156198 ms` | `0.602416 ms` | `553.783 us` |
| 16384 | `2.302501 ms` | `1.179092 ms` | `1123.409 us` |

### 21.3 实际 dispatch 图

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

### 21.4 中间 dtype 差异

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

### 21.5 各 stage body gap

| stage | T=2048 gap | gap slope |
|:--|--:|--:|
| cumsum | `+6.850 us` | `+0.001 us/chunk` |
| KKT | `+18.828 us` | `+0.878 us/chunk` |
| solve | `-14.902 us` | `-0.100 us/chunk` |
| W/U | `+36.214 us` | `+0.990 us/chunk` |
| recurrence | `+39.980 us` | `+1.261 us/chunk` |
| chunk-o | `+46.169 us` | `+1.196 us/chunk` |
| cast | `+15.483 us` | `+0.069 us/chunk` |

### 21.6 为什么当时先选 chunk-o

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

## 22. Stage 6B：chunk-o ownership O0/O1

### 22.1 当前 ownership

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

### 22.2 O0：token64 × V64

#### 为什么做

目标是把 CTA 降到与 vLLM 一致，同时让同一 CTA 复用 lower score。

O0：

- T=2048 CTA `2048 -> 512`；
- 4 waves 管 4 个 token16 row；
- CTA 内缓存 4×4 的 lower score tiles；
- output FP32 staging 和 final cast 不变。

#### 正确性

O0 对 current Stage 4：

- random、zero-H、zero-V_new；
- inter-only、intra-only；
- high/small/cancellation；
- cross-token16；
- 所有 V16 boundary；
- T=8192；

均 bit-exact。

#### 工作量改善

T=2048：

| metric | current | O0 |
|:--|--:|--:|
| CTA | 2048 | 512 |
| MFMA | 458,752 | 188,416 |
| VMEM | 851,968 | 335,872 |
| LDS inst | 1,343,488 | 520,192 |

这证明 ownership 假设成立：O0 真实减少了重复工作。

#### 为什么仍然失败

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

#### 这个失败告诉了我们什么

> 减少理论工作量不等于减少 latency。

必须同时检查：

- fragment lifetime；
- AccVGPR；
- LDS 容量；
- barrier；
- occupancy。

O0 不是“方向完全错误”，而是“工作复用成功，但资源组织不足以把收益完全兑现”。

### 22.3 O1：token64 × V32

#### 为什么做

O1 是唯一允许的第二个变量，只把 V64 改为 V32，试图：

- 减小单 CTA accumulator；
- 缩短部分 lifetime；
- 提高 occupancy。

#### 结果

O1 反而产生：

- 1024 CTA；
- 更多 MFMA；
- 更多 LDS；
- 更多 barrier；
- T=2048 慢于 current；
- 长序列 slope 更差。

因此 O1 是明确负结果。

### 22.4 最终决策

- O0/O1 都不接 full；
- Stage 4 current 保持不变；
- 不继续 O2/O3 tile sweep；
- 不修改 compiler 或 assembly；
- 暂不进入 direct-BF16 chunk-o。

### 22.5 学习总结

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

## 23. Stage 6R：recurrence baseline reconciliation

### 23.1 为什么突然回头核对 recurrence

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

### 23.2 Counter aggregation 审计

结果确认 `196608 / 65536` 是同口径、每个 recurrence dispatch 的动态 MFMA：

```text
按 chunk：     6144 / 2048
按 chunk-head：768 / 256
```

因此 3x 差异真实存在，不是 replay 聚合错误。

### 23.3 当前两份机器代码

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

### 23.4 同口径 body

| T | asm-v0 FP32 | current vLLM BF16 | gap |
|--:|--:|--:|--:|
| 512 | `0.049554 ms` | `0.039199 ms` | `10.356 us` |
| 2048 | `0.154670 ms` | `0.114570 ms` | `40.100 us` |
| 8192 | `0.574333 ms` | `0.413274 ms` | `161.059 us` |
| 16384 | `1.172622 ms` | `0.841410 ms` | `331.212 us` |

### 23.5 Isolated actual-vLLM bridge

Codex 提取当前 vLLM HSACO，建立 isolated external bridge：

- T=512/2048/8192/16384；
- h/v_new/final_state 对 native vLLM bit-exact；
- T=2048 性能差约 `-0.052%`；
- hash、grid、WG、LDS 和 ABI 都有 guard。

### 23.6 结论

CASE B：

> asm-v0 没有退化，它仍忠实对应历史 FP32/WG256 specialization；当前 vLLM
> 选择了新的 BF16/WG128 specialization。

因此正确下一步不是重写 asm，而是测试：

```text
当前更快的 BF16 recurrence bridge
能否接入 Avelang FP32 上下游
```

---

## 24. Stage 6S：BF16 recurrence full-contract integration

### 24.1 为什么不能直接替换 recurrence

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

### 24.2 三条图

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

### 24.3 Correctness

- recurrence bridge 对 native recurrence：
  `h/v_new/final_state` bit-exact；
- public output max abs：`0.001953125`；
- final state max abs：`0.0172200203`；
- 均低于冻结阈值；
- non-default stream、graph replay、invalid guard 均通过。

注意 final state 已接近 `0.02` 门槛，所以后续 BF16 propagation 必须扩大 seed 稳定性测试，
不能随意放宽阈值。

### 24.4 Body 成本

T=2048：

```text
old recurrence：154.669 us
new recurrence：114.570 us
body gain：      40.099 us

W/U/v_new 三个边界 cast 合计：20.750 us
```

这些 isolated body 数不能简单相减当作 full gain，但能解释为什么 full 只能保留部分收益。

### 24.5 Full 结果

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

### 24.6 决策

Stage 6S 是成功候选：

- 保留为 opt-in；
- 不改默认路径；
- 不改 recurrence HSACO；
- 不改 compiler/assembly；
- 下一步传播 BF16 storage boundary，消除三次显式 cast。

### 24.7 学习总结

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

## 25. 从 Stage 6S 到 Stage 6W：为什么接下来不是再改旧 asm

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

### 25.1 本阶段统一的公开 API 合同

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

### 25.2 为什么不能继续用 CUDA Graph 当正式排名

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

## 26. Stage 6T-Eager：融合 W/U 的 F0 与 F1

### 26.1 问题

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

### 26.2 F0：只测试 fusion，仍写 FP32

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

### 26.3 F1：同 schedule，只把最终存储改为 BF16

F1 唯一相对 F0 的语义边界是 output dtype：

```text
F0: FP32 W/U global store -> numeric cast -> BF16 recurrence input
F1: BF16 W/U global store -> BF16 recurrence input
```

因此 F0/F1 的差异才可以归因于 cast/materialization，而不能归因于不同 tile。
F1 不是把 accumulator 改成 BF16；MFMA 的 accumulator 和中间计算仍按原计划保持 FP32。

### 26.4 Correctness：先分开证明，再接 full

F0/F1 分别与原 W/U 参考比较 W、U，再比较完整 public output/final state。测试与
driver 为：

- [fused W/U source](../vllm_compare/qwen_gdn_bt64_fused_wu_eager_stage6t.py)
- [correctness test](../vllm_compare/test_qwen_gdn_bt64_fused_wu_eager_stage6t.py)
- [Eager benchmark](../vllm_compare/bench_qwen_gdn_bt64_fused_wu_stage6t_eager_public.py)

不通过 numeric contract 时，禁止用 full output “似乎还行”作为替代证据。

### 26.5 Eager public 结果

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

### 26.6 为什么少 3840 CTA 仍未直接获胜

资源证据：

| W/U path | CTA | WG | VGPR | AccVGPR | LDS | scratch | MFMA | VALU | VMEM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical separate W+U | 4096 | 256 | W52/U48 | W4/U8 | N/A | 0 | 524288 | N/A | 851968 |
| F0 | 256 | 256 | 64 | 8 | 3072 B | 0 | 524288 | 4518912 | 491520 |
| F1 | 256 | 256 | 64 | 8 | 3072 B | 0 | 524288 | 4846592 | 491520 |

CTA 与 VMEM 流量下降，但 BF16 epilogue 引入额外 VALU，fusion 还改变了 work per CTA
与 launch scheduling。故不能只用 “CTA 4096 -> 256” 预测 T=2048 一定快。

### 26.7 Stage 6T 决策

F0/F1 是有效的 source 与 Eager 实验，但不是新 baseline。最需要知道的不是“是否再调
F1”，而是 native vLLM fused W/U 在数值 ABI、MFMA 预算和 ownership 上究竟做了什么。
因此下一步是 Golden audit，不是立即 tile sweep。

详细证据见 [Stage 6T report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_fused_wu_eager_stage6t_report.md)。

---

## 27. Stage 6T-Golden：真实 vLLM fused W/U 的数学、ABI 与 MFMA 预算

### 27.1 为什么要做 Golden Audit

看到 F1 的 MFMA 或 CTA 仍远高于 vLLM 时，不能凭函数名猜 native kernel 的行为。需要
实际捕获 Triton source、TTIR/TTGIR、HSACO、autotune specialization 与 kernel trace，
核对：

- CTA 对应哪个 logical unit；
- solved A 的 dtype；
- W/U 是否同一 dispatch；
- 是否有 residual；
- tile/warp/stage；
- static/dynamic MFMA、LDS/barrier、资源与输出 layout。

### 27.2 Golden Audit 的关键发现

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

### 27.3 为什么不能直接桥接 native W/U

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

## 28. Stage 6U：BF16 solved boundary，P0 producer 与 C0 consumer

### 28.1 P0 的最小改变：FP32 compute，不再 FP32 storage

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

### 28.2 P0 correctness gate

P-REF 是原 FP32 solve 后执行 numeric BF16 cast。P0 覆盖 T=64/128/512/1024/2048/8192，
zero、identity-like、high-dynamic、small、cancellation、sparse-lower、NaN prefill、output reuse
与 non-default stream，共 42 producer cases。结果是 BF16 bit-exact：mismatch `0`、max abs `0`。

P1 packed x4 global store 没有实现。当前 lane ownership 不能安全地给每 lane 连续四个元素；
ISA 是 scattered `global_store_short`/`global_store_short_d16_hi`。它是 N/A，不是失败后悄悄
fallback。

### 28.3 C0：改变 consumer 数值 ABI，删除 residual

C0 接收 BF16 solved A/K/V，输出 BF16 W/U。它不再生成 residual coefficient，不再执行
residual MFMA。P0 的 bit-exact producer gate 使这个改变有数学依据，而不是“看起来 BF16
更快”就删修正项。

| implementation | W main | W residual | U main | U residual | MFMA/CTA | T2048 MFMA |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 512 | 512 | 512 | 512 | 2048 | 524288 |
| C0 | 512 | 0 | 512 | 0 | 1024 | 262144 |
| native vLLM | 64 | 0 | 64 | 0 | 128 | 32768 |

C0 isolated T=64/512/2048 W/U max abs 相对 BF16-coefficient reference 为 `0.0009765625`。

### 28.4 U0/U1 full graph 与 correctness

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

### 28.5 U1 Eager gain、资源和限制

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

### 28.6 Stage 6U 的推理结论

这一步是整个路线的重要转折：

```text
不是先压低 accumulator precision，
而是证明 precision 只需在 global storage boundary 变化；
然后把该事实传播到 consumer，移除重复的 correction schedule。
```

详见 [Stage 6U report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_bf16_solved_boundary_stage6u_report.md)。

---

## 29. Stage 6V：四个 predicated MFMA region 到一次 uniform MFMA

### 29.1 假设

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

### 29.2 V0 的写法与首次失败

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

### 29.3 V0 correctness 与 ISA/resource gate

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

### 29.4 V1 full integration

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

## 30. Stage 6W：chunk-o 直接消费 BF16 V-new，直接写 BF16 public output

### 30.1 问题：两次明显却未被消除的 storage boundary

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

### 30.2 真实 kernel 改动

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

### 30.3 正确性：为什么能 bit-exact

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

### 30.4 ISA 与资源：证明“身体”没有被偷换

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

### 30.5 private body 变慢为何不否定 full 图收益

预分配 body 只包 chunk-o kernel，不包括删掉的 casts：

| T | U1 current body | W1 body | W1 变化 |
|---:|---:|---:|---:|
| 512 | 0.036154 | 0.041822 | 慢 |
| 2048 | 0.086869 | 0.091136 | 慢 4.267 us |
| 8192 | 0.250192 | 0.251434 | 慢 |
| 16384 | 0.468576 | 0.473244 | 慢 |

这精确回答了一个常见误区：W1 的收益不是“BF16 MFMA 更快”，而是删除整个 public graph
里的两个 boundary dispatch/staging。用 body 反驳 full 改善是口径错误。

### 30.6 初始 Eager sweep：正向候选，不足以晋级

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

## 31. Stage 6W clustered confirmation：从历史污染样本到当前成对共享环境基线

### 31.1 为什么小样本 Eager 不够

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

### 31.2 历史 12-session run 为什么不能判定 W1

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

### 31.3 新鲜 paired shared-environment retest

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

### 31.4 历史 W1 与 vLLM：按倍数表述，不能只看绝对微秒

同一批：

| T | W1 | vLLM | 关系 |
|---:|---:|---:|---:|
| 2048 | 0.348283 ms | 0.404651 ms | W1 是 `0.8607x` vLLM，快 `56.685 us` |
| 8192 | 0.939173 ms | 0.766289 ms | W1 是 `1.2259x` vLLM，慢 `172.933 us` |

这就是准确的当前结论：短/中 T 在该同批共享环境可领先，长文本仍有明显 slope gap；
不是“vLLM 产生两个真性能”，而是不同 harness/session/specialization/GPU 环境测到的
不同样本。只有同一公开 API contract 的 paired comparison 才能做该表中的倍数结论。

### 31.5 W1 历史 status（已被 X2 取代）

```text
W1 = previous Avelang experimental baseline
scope = paired_shared_environment
production/default = unchanged
exclusive GPU absolute confirmation = 尚未获得
```

完整统计、历史污染说明和图 accounting 见
[Stage 6W clustered confirmation report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md)。

---

## 32. Stage 6W 后的完整 intermediate accounting：下一步为什么是 KKT -> solve

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

## 33. 完整实验账本：从 Stage 4 到 W1

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

## 34. 失败实验、N/A 与环境限制如何记录

### 34.1 数学/正确性失败

- Stage4 no-correction：错误删除 residual；不可以通过放松 tolerance 解决。
- Stage5B initial diagonal identity order：单位阵太早加入会重复第一条 sub-diagonal；修正计算顺序。
- Stage5B X42 offset：只含 A43 的 identity-diagonal case 定位为 output offset bug，而非 MFMA mapping。
- Stage6V statement `if`：SSA/scope 失败；要改为 conditional expression/select。

### 34.2 性能/资源失败

- accumulator merge：缩短源级变量数不保证降 live range；
- O0/O1：少 CTA/少 score 重复，却增加 AccVGPR/LDS，occupancy 下降；
- F0/F1：少 dispatch/CTA，却可能引入 fusion lifetime 与 BF16 epilogue VALU；
- V1：MFMA 4x 少，却被 cndmask/VGPR 与全图其他成本抵消；
- W1 body：BF16 boundary body 不是更快的 MFMA body，但 full 仍获益。

### 34.3 环境/工具 N/A

- `P1 packed solve writeback`：没有安全 x4 lane packing，N/A；
- U2：没有实施，不应被写成 predicate-collapse 之后的版本；
- Stage5F cache/clock：profiler 低扰动 gate 失败，N/A；
- historical v29 exact artifact：报告存在但当前 worktree 无原始 artifact directory，证据索引注明 availability note；
- shared GPU：绝对延迟确认受外部 context 限制，但 paired shared ranking 的结论范围明确保留。

**N/A 既不是 pass 也不是 fail。** 它指出当前问题尚没有可信测量或安全实现，不允许拿旧数据补空格。

---

## 35. 当前状态、严谨表述和下一步

### 35.1 可以说什么

1. Stage 4 消除了 BT64 上游/下游 scalar fallback；
2. Stage 5 将 solve 根因定位为依赖图，并完成 FP32 MFMA16x4 hierarchical solve；
3. current-vLLM recurrence 与旧 asm-v0 是不同 specialization；
4. BF16 solved boundary 使 U1 大幅减少 residual 工作和 boundary dispatch；
5. predicate collapse 精确降了 MFMA，但没有 full promotion；
6. W1 删除尾部两个 materialized boundary，并曾在严格 paired shared environment 下稳定快于 U1；
7. Stage 6X X2 现已将 KKT 与 solve handoff 收进同一 CTA，在新的完整 Eager confirmation 中于 T512--16384 全部稳定快于 W1，因而取代 W1 成为当前 Avelang experimental baseline；
8. X2 在同批 T<=4096 快于 vLLM，但 T>=8192 仍慢于 vLLM；X2/vLLM 的 per-chunk slope 仍为 5.694/3.688 us。

### 35.2 不能说什么

1. 不能说 X2 已经是 production/default；
2. 不能说 X2 全序列超过 vLLM；
3. 不能用 Graph replay/body/trace 代替 Eager leaderboard；
4. 不能把 old asm-v0 叫 current vLLM kernel；
5. 不能把 `a_solved_bf16 -> W/U` 或 recurrence--chunk-o handoff fusion 写成已经完成；
6. 不能把 historical polluted CI 说成 W1 或 X2 无收益；
7. 不能从没有 scratch 推出没有 AGPR/VGPR pressure。

### 35.3 唯一下一候选

在不动 recurrence HSACO、v24、default selector、compiler/RA 或多个 kernel 的前提下：

```text
先做 Stage 6Z 的 native BT64 chunk-o ownership X0 audit，
再实现一个单变量 isolated chunk-o prototype。
```

标准仍不变：correctness 先行；只改一类边界；Eager public API paired result 决定是否晋级；
ISA/rocprof 只解释，不替代性能；发现资源 cliff 或测量不可信时停止。

---

## 36. 文档、证据与复现入口

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
- [Stage 6X report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_kkt_solve_handoff_stage6x_report.md)
- [Stage 6Y updated gap audit](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_stage6y_updated_full_gap_audit.md)

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

## 37. Stage 6X-KS：KKT--Solve Handoff Elimination

### 37.1 为什么 X2 是高价值但风险受控的实验

Stage 6W 的图仍有一个明确且完全位于 Avelang source 范围内的单消费者边界：

```text
KKT -> FP32 a global -> hierarchical solve -> BF16 a_solved
```

对于每个 `(chunk, value-head)`，`a` 是一个 `64 x 64` FP32 矩阵，即 16 KiB。T=2048
时完整 `a` 为 4 MiB，producer store 加 consumer read 为 8 MiB；T=8192 时是 32 MiB。
它和 `h_bf16/v_new_bf16` 不同，不跨 immutable recurrence HSACO ABI。因此 Stage 6X
没有碰 recurrence、W/U、chunk-o、compiler 或 default selector，只把这个 handoff 放进
一个 CTA 做验证。

### 37.2 X0：先确认 ownership，而不是直接写融合大 kernel

旧 KKT 使用 `16 CTA / (chunk, head)`、WG64：每个 CTA 只负责一个 `16 x 16` token tile。
其中 10 个 lower/diagonal tile 做 8 次 BF16 MFMA16 K reduction，6 个 strict-upper tile
写零。solve 已经是 `1 CTA / (chunk, head)`、WG256、四 waves 的 hierarchical FP32
block-triangular 结构。

X0 的关键容量结论：不需要把全序列 `a` 放入 LDS；每个 CTA 只保留本 chunk/head 的
`64 x 64 x FP32 = 16 KiB`。保守实现同时保留：K staging 16 KiB、`a` 16 KiB、solve
`x/work` 8 KiB，总共 40 KiB LDS。这在 gfx942 的 CTA 容量内，但不能因此假定 alias
或 lifetime reuse 会更快。

### 37.3 X1：只改 KKT ownership，仍保留 global a

X1 使用 `1 CTA / chunk-head`、WG256；四 waves 分别拥有四个 token16 row tile，循环
四个 column tile，输出仍是旧 FP32 `a` layout。它把 K 的 staging 从 tile 级重复工作
改为 CTA 级共享，但不消除 KKT--solve handoff。

| T | old KKT ms | X1 KKT ms | old/X1 |
|---:|---:|---:|---:|
| 512 | 0.034832 | 0.029624 | 1.176x |
| 2048 | 0.046910 | 0.032529 | 1.442x |
| 8192 | 0.170874 | 0.038117 | 4.483x |

在 T=2048，MFMA 数仍为 20,480；但 VALU 从 1,565,696 降至 639,488，VMEM 从 210,944
降至 79,872，LDS 指令从 184,320 降至 47,104。WG256 的资源增加到 VGPR/AccVGPR/SGPR
为 84/20/112，LDS 16 KiB，且 occupancy 降低；关键是 scratch 仍为零、trace 从
40.961 us 降至 9.815 us。X1 证明 ownership 转换本身没有并行度 cliff。

### 37.4 X2：在同一 CTA 内完成 KKT + solve

X2 的数据流为：

```text
K/Beta/G
  -> KKT FP32 into CTA-local a_lds[4,16,64]
  -> existing hierarchical FP32 solve
  -> BF16 a_solved global output
```

它删除 `a` allocation、global write、global read 和独立 solve dispatch。X2 没有减少
数学工作：T=2048 的动态 MFMA 是 20,480 次 KKT BF16 MFMA 加 16,384 次 solve FP32
MFMA，总计 36,864。其资源为 40 KiB LDS、VGPR/AccVGPR/SGPR 100/164/112、scratch 0、
trace 16.505 us。高 AccVGPR 是已知成本，但没有 spill；因此它必须由完整图而不是只由
isolated body 决定去留。

### 37.5 正确性：为何可以称为语义保持

X1 KKT 在 random/high-dynamic、T64/128/512/2048 上 FP32 bit-exact；X1 接旧 solve 在
random/cancellation 上 BF16 bit-exact；X2 在 random/high-dynamic/cancellation 上 BF16
bit-exact。full Stage6W-vs-X2 矩阵覆盖 T64/128/512/2048/8192、random/high-dynamic/
cancellation/neutral gate、零和非零 initial state，public BF16 output 与 FP32 final
state 都 bit-exact。补齐的 NaN prefill/reuse 和 non-default stream 为 `2 passed`。

这里的 exact 是 X2 对冻结 Stage6W 语义的精确保持，不是声称它与 native vLLM 的每个
intermediate 必然 bit-exact。

### 37.6 完整 Eager 结果与晋级理由

权威口径是 uncaptured Eager public API，而非 CUDA Graph replay、standalone body 或
rocprof trace。每个 T 独立进程、5 sessions、50 paired Williams blocks、每实现 300
calls，并同时记录 HIP event 与 wall-clock。

| T | W1 ms | X2 ms | X2 对 W1 gain | 95% CI |
|---:|---:|---:|---:|:---|
| 512 | 0.221349 | 0.199697 | 22.123 us | [19.720, 24.647] us |
| 1024 | 0.262049 | 0.233467 | 26.070 us | [23.592, 28.667] us |
| 2048 | 0.348378 | 0.319154 | 26.020 us | [23.304, 28.563] us |
| 4096 | 0.518070 | 0.479532 | 40.465 us | [37.439, 43.506] us |
| 8192 | 0.935690 | 0.852166 | 82.823 us | [80.432, 85.154] us |
| 16384 | 1.773194 | 1.595190 | 177.540 us | [176.004, 179.024] us |

这些 CI 的下界全部为正。W1/X2 slope 为 6.341/5.694 us/chunk，X2 回收 0.646
us/chunk，即约 10.2%。因此 X2 是新的 **Avelang BT64 experimental baseline**；W1
退为上一 experimental baseline；production/default 完全不变。

### 37.7 与 native vLLM 的真实位置

| T | X2 ms | vLLM ms | X2/vLLM |
|---:|---:|---:|---:|
| 1024 | 0.233467 | 0.358813 | 0.651x |
| 2048 | 0.319154 | 0.401397 | 0.795x |
| 4096 | 0.479532 | 0.522356 | 0.918x |
| 8192 | 0.852166 | 0.765016 | 1.114x |
| 16384 | 1.595190 | 1.235115 | 1.292x |

X2 在本批 T<=4096 Eager 测试中快于 vLLM；T>=8192 落后。拟合 vLLM slope 为 3.688
us/chunk，对比 X2 的 5.694 us/chunk，剩余差距是约 2.006 us/chunk。这个事实决定了
下一步不能继续为 X2 省几个 KiB LDS，而应先执行 Stage 6Y 的新五-dispatch gap audit。

人工正式结果以 JSON 形式归档在
`compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_kkt_solve_handoff_stage6x_manual_confirmation/`；
旧 Docker 产物目录由 `nobody` 所有，保留不改以维护原始证据完整性。

---

## 38. Stage 6Y：X2 新五-dispatch图的差距审计

### 38.1 为什么旧 Stage 6A/6W 数据不能继续决定新动作

X2 已删除 `KKT -> global a -> solve` 与 standalone solve dispatch。继续用旧图的
kernel 数、intermediate accounting 或 body slope 选下一步，会把已删除的流量又当成
瓶颈。因此 Stage 6Y 不改 kernel，先冻结 X2 的真实图：

```text
cumsum -> fused KKT+solve -> fused W/U -> recurrence -> chunk-o
```

静态账本中 `a` 已完全消失。最大的 remaining source-native one-use tensor 是
`a_solved_bf16`（T2048 2 MiB，write+read 4 MiB），但它与 X2 的 40 KiB LDS/
AccVGPR=164 live region 相邻，不能仅凭流量直接融合。最大的总体边界 `h_bf16` 与
`v_new_bf16` 跨 immutable recurrence ABI，也不能不经审计直接修改。

### 38.2 当前 body sweep 如何定位长文本 gap

新的 body runner 以每个 T 独立 Docker process、3 session、5 warmup、20 ABBA repeat
测量当前 wrapper 边界。它是诊断工具，不累计为 full latency。所有六个长度的线性拟合：

| logical body | X2 slope | vLLM slope | X2-vLLM |
|:--|--:|--:|--:|
| cumsum | .008 | .008 | .000 us/chunk |
| KKT+solve | .348 | .082 | +.266 us/chunk |
| W/U | .504 | .247 | +.257 us/chunk |
| recurrence | 3.126 | 3.144 | -.018 us/chunk |
| chunk-o | 1.713 | .427 | **+.1.286 us/chunk** |

T16384 的 X2/vLLM chunk-o body 为 0.481958/0.153148 ms，差约 329 us；这已大于
full X2-vLLM 约 360 us 的绝大部分。W/U 是次级差距；KKT+solve 的 fixed-cost收益在
长文本转为小幅 slope 劣势；recurrence 使用相同 HSACO symbol，不是当前 kernel 目标。

### 38.3 真实 trace 的 ownership 证据

T2048 最终稳定重放中，X2 是五 dispatch，vLLM 是七 dispatch，后者仍有 KKT、BF16 fill
和 solve merge 三步。这消除了“X2 因 kernel 数多而慢”的解释。相反，chunk-o 资源显示：

| chunk-o | WG | global grid | CTA | LDS | VGPR/AccVGPR |
|:--|--:|:--|--:|--:|:--|
| X2 | 256 | 524288 x 1 x 1 | 2048 | 27136 B | 112/64 |
| vLLM | 256 | 512 x 32 x 8 | 512 | 0 B | 100/36 |

X2 同样的 workgroup 下启动四倍 CTA，并使用 27 KiB LDS。此证据与 `1.286 us/chunk`
chunk-o slope deficit 一致，但不把相关性说成唯一硬件因果。

### 38.4 下一步只选一个：Stage 6Z

Stage 6Z 是 **native BT64 chunk-o ownership redesign**。先只做 native
`chunk_fwd_kernel_o` 的 source/trace X0 ownership audit，再写一个 isolated chunk-o
prototype；不同时改 recurrence、KKT+solve、W/U、compiler、assembly 或 default
selector。prototype 的 correctness、resource、Eager full gates 必须独立通过。W/U
明确退为 runner-up。

---

## 39. Stage 6Z：native-style chunk-o ownership 的严格停止结果

Stage 6Z 先实际捕获 native `chunk_fwd_kernel_o` 的 final source/TTIR/TTGIR/LLVM
IR/AMDGCN/HSACO。native 的真实 tile 是 BT64/BV64/BK32，一个 CTA 对一个
`[chunk,value-head,V64]`，每个 chunk-head 两个 CTA；Stage6W 则是八个 V16 CTA。
因此 native 每个 chunk-head 只重复两次 Q/K score，而 Stage6W 重复八次。

Z1 用一个新的 isolated V64/BK32 MFMA32 kernel 重建该 ownership。它通过 T64 到 T8192
正确性，最大误差不超过 `1.526e-5`，并通过 zero-V-new、caller-owned reuse 和 invalid
dtype gate。caller-owned body 也有实际改善：T2048 `0.092237 -> 0.075592 ms`（1.220x），
T8192 `0.252154 -> 0.196411 ms`（1.284x）。

但它不能晋级。捕获的 HSACO 有 41 个静态 `s_barrier`，超过 Stage 6Z 预注册上限
`barrier < 19`；其余资源虽正常（scratch/spill=0，AccVGPR=172，LDS=28672 B），这个单一
hard gate 已足以禁止 Z2 full integration。Stage6Z 因此关闭在 Z1：不创建 full wrapper、
不跑 Eager 排名、不更改 X2 或 selector。详细记录见
`compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_stage6z_native_chunko_report.md`。

---

## 40. Stage 7A：chunk-o barrier provenance 与 phase-boundary 审计

Stage 7A 不改变 production 或 full graph，先把 Z1 的 41 个 barrier 串回源头。结果是：

```text
10 个显式 al.syncthreads() source site
  -> 41 个 pre-link LLVM barrier call
  -> 41 个 pre-LTO assembly s_barrier
  -> 41 个 final HSACO s_barrier
```

它不是 LLVM/AMDGPU backend 额外添加的同步。差异来自 Phase A 的 K-stage 展开、Phase B
每个 source half 的保留 K-loop body，以及 Phase C 的 half/fragment 展开。native Triton
T2048 selected artifact 只有 11 个 lexical barrier，并通过 compiler-managed local buffer
lifetime 而非手工 `frag_words` phase 来表达流水。

最小 repro 证明两条局部命题：lane-private fragment write 后的 barrier 可以在独立 MFMA
repro 中去掉；score lower-half write 与不重叠 upper-half write 中间的 barrier 可以合并。
两组 output 都 bit-exact。但唯一允许应用到完整 Z1 的 phase-compaction 尝试在 T8192
首个 case 就失去 bit exactness（max abs `0.00206613541`），所以不能把 local proof 推广到
full CTA MFMA pipeline。后者存在跨 wave progress、MFMA issue 和 LDS reuse 的组合 phase
边界。

结论是 Case C：关闭当前 Stage6Z pure-Avelang native chunk-o 路线；不进行 barrier subset
sweep、tile/WG sweep、Z2 或 full integration。之后若追求端到端收益，可另立 external
native-HSACO bridge 诊断路线；若坚持 pure Avelang，则转向 W/U runner-up。完整 ledger、
最小 repro 与失败记录见
`compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_chunko_barrier_provenance_stage7a_report.md`。

# Qwen GDN standalone chunk-o 优化完整复盘

## 结论先行

截至当前实验记录，**单独 chunk-o 的最佳纯 Avelang 版本是 Stage 6Z Z5B：
`direct-Q-cache-consumer`**。

源码入口：

```text
avelang/chunko_z5b_minimal.py
```

原始源码入口：

```text
test/examples/linear_attention/vllm_compare/
qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py
```

它是 standalone isolated kernel，不是完整 Qwen GDN。完整算子的最佳组合曾
命名为 X2+Z5B，但 X2 还包含 current-vLLM recurrence external HSACO；本报告
只回答 chunk-o 本身，所以不能把 X2 的 full-graph latency 混成 chunk-o latency。

Z5B 的核心变化只有一个：

```text
Z5A:
global Q once -> dedicated Q LDS cache -> old phase Q rows -> phase_vec -> MFMA

Z5B:
global Q once -> dedicated Q LDS cache -> Q fragment -> MFMA
```

没有改 K/H/V-new/g/output producer，没有改 MFMA geometry、WG、数学、BF16 ABI、
K32 累加顺序，也没有改 allocator/RA 或 production selector。

## 1. chunk-o 在完整算子中的位置

Qwen GDN 的完整逻辑可以抽象为：

```text
cumsum -> KKT -> solve -> W/U -> recurrence -> chunk-o -> public BF16 output
```

chunk-o 消费：

- BF16 `q/k/v_new`；
- BF16 chunk state `h`；
- FP32 cumulative gate `g`；

并产生 BF16 public output。它计算两个部分：

```text
inter = (q * scale) @ H_chunk^T
intra = causal(exp(g_target - g_source) * (q @ k^T)) @ v_new
out = exp(g_target) * inter + intra
```

这里的 `H_chunk` 是 chunk 入口 state，`v_new` 是 recurrence 产生的 BF16
新 value。Z5B 本身不负责 recurrence，也不负责 W/U/solve。

## 2. 为什么 chunk-o 成为主瓶颈

Stage 6A 的 full-graph audit 把逻辑阶段拆开后发现，早期 Avelang full path 的
generic chunk-o 有明显的标量地址、global round-trip 和低并行度问题。Stage 2
profiling 里 generic chunk-o 曾出现约 `11.761 ms` 的 T=2048 trace，远高于
其他阶段。这个数字属于早期 full pipeline generic wrapper，不能拿来当 Z5B
现在的 body latency；它的作用是说明为什么必须单独拆 chunk-o。

随后我们把问题从“整个算子慢”收敛成：

1. chunk-o 的 tile ownership 是否和 native 一致；
2. Q/K/H/V-new 是否被重复 materialize；
3. MFMA 前的 LDS/fragment feeding 是否产生多余指令；
4. 长文本下每个 chunk 的 slope 是否持续扩大。

Stage 6W 又固定了 BF16 boundary：chunk-o 直接读取 BF16 `v_new`，并直接把
FP32 accumulator round 成 BF16 output。这样后续 chunk-o 对比不再把两个尾部
FP32 cast dispatch 混进 kernel 本体。

## 3. 先审 native Triton，而不是猜它

Stage 6Z Z0 从 current-vLLM 的真实 selected `chunk_fwd_kernel_o` 抓取了
TTIR/TTGIR/LLVM/ISA/HSACO 和源代码。Triton 源码在本包：

```text
triton/chunk_o.py
```

native 的基本 ownership 是：

```text
program_id(0) -> V block
program_id(1) -> BT=64 chunk
program_id(2) -> value head
```

在 T=2048 的同形状诊断中，native 使用 `BK=32、BV=64、WG=256`，一个 CTA
负责 `[64 token, 64 value]` 输出 tile，每个 chunk-head 有两个 CTA。长文本
selector 可能切换到 WG128/2 waves/2 stages，所以必须每个 T 记录实际 selector，
不能用 T=2048 的配置插值到所有长度。

Triton 源码的关键结构是：

```python
b_o = tl.zeros([BT, BV], dtype=tl.float32)
b_A = tl.zeros([BT, BT], dtype=tl.float32)
for i_k in range(cdiv(K, BK)):
    b_q = tl.load(block_q)
    b_k = tl.load(block_k)
    b_h = tl.load(block_h)
    b_o += tl.dot(b_q, tl.trans(b_h))
    b_A += tl.dot(b_q, b_k)

b_A = causal_mask(exp(g_target - g_source) * b_A)
b_v = tl.load(block_v)
out = exp(g_target) * b_o + tl.dot(b_A.to(b_v.dtype), b_v)
tl.store(block_o, out.to(bfloat16))
```

这不是“抄汇编”。我们使用 Triton 源码、TTIR/TTGIR 和 ISA 作为 mapping
oracle；Triton 编译器负责把 `tl.dot` 变成 MFMA 和 LDS/VMEM schedule。
ISA 只是机器证据，不是 Avelang 源码。

## 4. Avelang chunk-o 优化时间线

### 4.1 Z0：native mapping audit

Z0 没有先写 Avelang kernel，而是先确认 native 的真实选择：

- BT64/BK32/BV64 的 tile 结构；
- T=2048 与长文本可能使用不同 WG/stage；
- native 有 local operand allocation/deallocation；
- source operand 阶段结束后会释放 shared buffer，再进入 score-times-V 阶段；
- native 通过较紧的 local/dot operand layout 避免保留大范围 phase buffer。

结论：剩余差距不是 MFMA 指令型号错了，而是 operand feeding、ownership、
phase lifetime 和 global/LDS materialization 的组合问题。

### 4.2 Z1：native-style chunk-o 初版

Z1 固定 BT64/BV64/BK32/WG256，采用 Avelang 的 MFMA32 和显式 phase LDS。它
通过 correctness，但有约 41 条静态 `s_barrier`，并且有额外的
`frag_words` lane-private fragment LDS round trip。

这轮学到的不是“barrier 越少越好”，而是：必须先知道每条 barrier 保护什么。
直接批量删 barrier 没有证据基础，也会破坏跨 wave 的 producer-consumer RAW/WAR。

### 4.3 Z2：phase-aware、删除 frag_words

Z2 删除了：

```text
phase_vec -> frag_words LDS -> barrier -> frag_words read -> MFMA
```

改成 lane 直接将已经读到的 packed word view 成 MFMA operand。Z2 相对 Z1：

- correctness 通过；
- dynamic MFMA 数学工作不变；
- Accum_VGPR 大幅下降；
- LDS/VALU/SALU 下降；
- body 变快；
- 但仍有约 27 条静态 barrier，未通过预注册 barrier gate。

因此 Z2 是重要的中间学习版本，但不是最终最佳版本。

### 4.4 Z3：WG128 长文本候选

Z3 尝试让两个 wave 分工处理一个 tile，以贴近 native 长文本配置。首次版本
在 T=8192/16384 暴露了 Phase-B K dead overread，后来修复了地址边界并通过
correctness。修复后的性能仍没有在长文本稳定超过 Z2，且没有理由替代已经
更快的 Z5B。Z3 也证明了：改变 WG/ownership 不是免费的，必须分别审计 grid、
address、phase coverage 和 selector。

### 4.5 Z4：Q residency ladder

Z4 把 Q 的三个 producer pass 作为独立变量拆成三臂：

- Z4A：只改 Q load width；
- Z4B：只把一个 Q slice 从 3 个 consumer 共享到 2 个；
- Z4C：把一个 Q slice 共享给 3 个 consumer。

Z4C 能表达 full-Q reuse，但把 `inter_acc、score_acc0、score_acc1` 拉进共同
K32 loop，造成 accumulator 同时 live，资源增长明显。它是很有价值的反例：
“global load 少”不等于“kernel 变快”，过长的 accumulator live range 会抵消
收益。

### 4.6 Z5A：dedicated full-Q LDS cache

Z5A 从 fixed Z2 分叉，新增完整 Q cache：

```text
[64,128] BF16 = 16 KiB
```

Q 在每个 CTA 内只从 global producer 一次，然后 Phase A 的 Q*H、Phase B 两个
Q*K consumer 共享这块 cache。Z5A 保持 Z2 原来的 phase-separated accumulator
顺序，因此没有重演 Z4C 的三 accumulator overlap。

代价是 shared footprint 从约 16 KiB 增到约 32 KiB，但 measured body 明显
变快。来自正式 5-session 数据的 session-median：

| T | fixed Z2 | Z5A | Z5A 相对 Z2 |
|--:|--:|--:|--:|
| 2048 | 0.077555 ms | 0.066138 ms | 约快 14.7% |
| 8192 | 0.177143 ms | 0.160178 ms | 约快 9.6% |

Z5A 首次证明 Q residency 是有效方向，但它还把 Q cache 当前 slice 复制到
旧 phase Q rows，再通过 `phase_vec` 喂 MFMA，动态 LDS 仍偏高。

### 4.7 Z5B：direct-Q-cache-consumer

Z5B 是从最终 Z5A 源码分叉的唯一变化：删除 Q cache 到旧 phase Q rows 的
republish；H/K/V-new 仍使用旧 phase 区域，Q 直接从 dedicated cache 形成
fragment。

源码中最重要的两处是：

```python
q_words = q_cache_vec[...]
h_words = phase_vec[...]
```

以及 score 阶段：

```python
q_words = q_cache_vec[...]
k_words = phase_vec[...]
```

这样既保留了 Q 的跨 consumer residency，又不把 Q 再写一遍 LDS。

### 4.8 Z6G：g residency stable/ideal

Z6G-S 把 64-token FP32 g 放进简单 shared cache；Z6G-I 尝试用更接近 native
的 typed residency。两臂都通过 correctness，但没有稳定超过 Z5B。这个结果说明
“还有多个 g consumer”是真实 provenance 事实，却没有证明 g cache 是最大的
latency 控制杆。不能只根据 modeled VMEM 数选择 winner。

### 4.9 Z7AB：更大范围的 Q/H/K fusion

Z7AB 尝试将 Q/H/K producer、phase 和 consumer 统一到更大的 superphase。它
可以减少某些逻辑 producer，但会增加 layout/ownership/address 工作，整体没有
稳定超过 Z5B。它说明 source fusion 的收益必须和 register lifetime、LDS phase
以及实际 schedule 一起测。

### 4.10 Z8W：compiler waterfall-free lowering A/B

Z8W 是一次很干净的 same-source compiler lowering 实验。高层 source schedule、
MFMA、LDS 和数学不变，只改变 raw-buffer 动态 offset 从 scalar `soffset` 转到
VGPR-compatible `vindex`，避免 lane-divergent address 触发 waterfall loop。

机器工作明显改善：T=2048 每 CTA 的 VMEM 从 2464 降到 448，SALU 从 5000 降到
840，VALU 从 11114 降到 6990。但 body T=2048 比 Z5B 慢约 0.727 us，T=8192
慢约 6.347 us。因此它是 compiler correctness/machine-work 的成功，不是
standalone performance winner。

这轮给编译器团队的准确结论是：AveLang lowering 确实能造成可观的机器额外工作，
并且同源 lowering A/B 能改变最终 ISA；但“机器指令少”仍不自动等于端到端更快。

### 4.11 Z9S：critical-path scheduler

Z9S 继续尝试 load/wait/MFMA 依赖调度。它能让机器图改变，但没有得到跨长度
稳定正收益。原因是 Z5B 的主要差距不只是某个 waitcnt，而是完整 operand feeding
和 CTA 内工作密度。于是 scheduler 微调没有资格替代 Z5B。

### 4.12 P1/P2/P3/P4 与 C17-C26

后续 block-dot generalization 和 full physical plan 主要回答“能不能把
producer-layout-shared-consumer 抽象成通用 compiler infrastructure”：

- BDV2/P1/P2/P3/P4 证明 block-dot 可以从 K/H 逻辑 block 统一接管 producer、
  shared placement、MFMA-B consumer，并产生不同 LLVM/MIR/ISA/HSACO；
- C17-C19 进一步减少部分 global/materialization 工作；
- C20 正式冻结 C19，T=2048 C19 为 `0.095181 ms`，Z5B 为 `0.066899 ms`，
  C19 是 Z5B 的约 `1.4228x`，所以 No-Go；
- C25 把 H-ready/K-pending 的依赖顺序保留到 ISA，但 T=2048、8192、16384
  都慢于 Z5B；
- C26 修正了 native MFMA/CTA 归一化错误，确认 Z5B 和 native 同一 logical
  unit 的动态 MFMA 都是 320，不能再用错误的 160/320 差异解释性能。

这些实验说明通用 compiler infrastructure 是有价值的，但它们没有产生比
Z5B 更快的 standalone chunk-o。因此本包不把 C19/C25/C26 伪装成最佳 kernel。

## 5. Z5B 的完整 contract 与机器证据

### 5.1 Source contract

| 项目 | Z5B |
|:--|:--|
| target | gfx942 / wave64 |
| tile | BT64 / BV64 / BK32 |
| workgroup | 256 threads / 4 waves |
| CTA | 2 CTA per chunk-head |
| dtype | q/k/v_new/h/output BF16，g FP32，accumulator FP32 |
| MFMA | `v_mfma_f32_32x32x8_bf16` |
| shared | 约 32 KiB：16 KiB Q cache + phase area |
| output | caller-owned contiguous BF16 |
| private/spill | 0 / 0 |

### 5.2 T=2048 dynamic PMC，同形状 WG256 对照

下表来自 fresh rocprof capture，按 CTA 归一化。它们是动态计数，不是 ISA
静态行数：

| 指标 / CTA | Z5B | native WG256 diagnostic | Z5B/native |
|:--|--:|--:|--:|
| MFMA | 160 | 160 | 1.00x |
| VMEM | 672 | 140 | 4.80x |
| LDS | 672 | 480 | 1.40x |
| VALU | 7,072 | 3,376 | 2.09x |
| SALU | 768 | 660 | 1.16x |
| profiler VGPR | 76 | 100 | 0.76x |
| profiler Accum_VGPR | 100 | 36 | 2.78x |
| profiler SGPR | 112 | 96 | 1.17x |
| shared footprint | 32,768 B | 24,576 B | 1.33x |
| scratch | 0 | 0 | - |

`Accum_VGPR` 是 profiler resource metric，不能和 code-object AGPR 数直接等同。
Z5B 的 code-object metadata 是 `VGPR/AGPR/SGPR=104/32/28`；native 不同
artifact 层曾报告过 metadata discrepancy，因此表中只把 native shared 作为
diagnostic footprint，不能把 collector 的 `LDS_Block_Size=0` 当成 native 无 LDS。

### 5.3 Z5B 静态机器图

Z5B exact-LTO/ISA archive 中可见的主要静态 family 约为：

| family | count | 解释边界 |
|:--|--:|:--|
| `global_load_ushort` | 112 | BF16 窄 load family，不能单凭 mnemonic 分 Q/K/H/V |
| `global_load_dword` | 80 | FP32/scalar family，不能单凭 mnemonic 分 g/地址辅助 |
| `global_store_short_d16_hi` | 16 | BF16 output store |
| `ds_write_b16` | 112 | producer phase 的窄 BF16 store |
| `ds_write_b16_d16_hi` | 32 | packed word 的另一半 |
| `ds_read_b128` | 56 | packed fragment/local read |
| `s_barrier` | 32 | static lexical count，不是动态同步次数 |
| `v_mfma_f32_32x32x8_bf16` | 56 | static lexical count，PMC 才是 160/CTA |

因此 Z5B 的 remaining gap 不能写成“多了 532 条某种 load”。532 是 aggregate
dynamic VMEM gap；现有 ledger 能较强地证明 Q duplicate 已消除，g 有多个
consumer role，K/H/V-new 仍存在 generic producer/layout feeding 成本，但不能
把 672 精确分摊到六类 operand。

## 6. 正式 standalone body 性能

### 6.1 主 benchmark 口径

正式数据来自 fresh-process、current HIP stream、no Graph、caller-owned output、
warmup=10、repeat=50、5 sessions、轮换 arm 顺序。计时边界是 standalone body，
compile/autotune/allocation 在计时外。

本最小包另外在 `ljd_qwen_vllm_avelang_rocm722` Docker 中做过一次 `T=64`、
`warmup=1/repeat=1` smoke。两臂均成功编译、运行并产生 finite 输出；Triton
实际选择记录为 `BK=32/BV=32/num_warps=4/num_stages=3/WG256`。该文件是
`benchmark/smoke_T64_not_for_ranking.json`，只证明交接包可运行，不改变下面的
正式性能排名。

### 6.2 Z5B、Z5A、Z2、native

主 T=2048/T=8192 结果，单位 ms，为 5 个 session median 的中位数：

| T | fixed Z2 | Z5A | Z5B | native selected | Z5B/native |
|--:|--:|--:|--:|--:|--:|
| 2048 | 0.077555 | 0.066138 | **0.065618** | 0.042643 | **1.54x** |
| 8192 | 0.177143 | 0.160178 | **0.156473** | 0.090755 | **1.72x** |

Z5B 相对 Z5A：

- T=2048 平均 session median 约快 `0.66 us`；
- T=8192 五个 session 全部更快，平均约快 `4.36 us`；
- endpoint slope 从 Z5A `0.980 us/chunk` 降到 Z5B `0.946 us/chunk`。

Z5B 相对 native：

- T=2048 仍约为 native 的 `1.54x`；
- T=8192 仍约为 native 的 `1.72x`；
- native 的 per-chunk slope 约 `0.501 us/chunk`，Z5B 约 `0.946 us/chunk`。

这里的 `1.54x/1.72x` 是 body 倍数，不是 full Qwen GDN public API 倍数。

### 6.3 另一次完整长度 source sweep

C20 的 7-session source sweep 还保存了下面一组独立 session median。它用于
看长度趋势，不与上表的 5-session 数字逐项覆盖：

| T | chunks | Z5B | native selected | Z5B/native |
|--:|--:|--:|--:|--:|
| 512 | 8 | 0.052358 | 0.037696 | 1.39x |
| 1024 | 16 | 0.052758 | 0.038337 | 1.38x |
| 2048 | 32 | 0.066498 | 0.042963 | 1.55x |
| 4096 | 64 | 0.101651 | 0.055983 | 1.82x |
| 8192 | 128 | 0.157655 | 0.090314 | 1.74x |
| 16384 | 256 | 0.275529 | 0.140650 | 1.96x |

长文本 native selector 可能切换 WG/warp/stage；所以这张表的 native 是每个
T 的实际 selected body，不是一个固定 code object。正式复现时以 benchmark
输出中的 `triton_selected_config` 为准。

## 7. Z5B 为什么是最优，而不是“计数最低”的版本

后续实验中出现过比 Z5B 低的局部 counter：

- C19 的 VMEM/VALU 比 Z5B 低；
- Z8W 的 waterfall-free A/B 把 VMEM/SALU/VALU 大幅压低；
- C25 把一部分 H/K issue 依赖保留到了最终 ISA。

但它们都没有稳定的 body latency 胜出。原因可能包括：

1. 机器工作分布改变后，等待/依赖链变长；
2. 通用 physical plan 引入了额外的 address/layout 或 phase management；
3. shared footprint、register lifetime 和 occupancy 改变了实际调度；
4. standalone latency 由 critical path 决定，不是所有指令数量线性相加；
5. 某些优化只减少了一个 counter，却没有消除真正的 serialized producer-consumer。

所以最终排序规则是：

```text
correctness -> 同口径 fresh body latency -> 长文本 slope -> machine evidence
```

而不是：

```text
先看 VMEM/LDS 少，再宣布性能最好
```

## 8. 对编译器团队应该如何表述

证据支持以下较精确的结论：

1. AveLang 能正确表达 BF16、MFMA32、dedicated Q residency 和 direct shared
   consumer；Z5B 不是“硬件做不到”。
2. Z5B→Z8W 的 same-source lowering A/B 改变了 LLVM/MIR/ISA，并删除了
   raw-buffer scalar `soffset` 引起的 waterfall 机器路径。这是明确的 lowering
   证据。
3. block-dot 的 full-scope generalization 也能产生不同的 LLVM/MIR/ISA/HSACO，
   说明 compiler infrastructure 可以控制 producer/shared/consumer 表达。
4. 但是现有数据不足以把 Z5B 与 native 的全部 `VMEM=672 vs 140`、
   `VALU=7072 vs 3376` 都归罪于单一 compiler pass。剩余差距是高层 ownership、
   shared/dot layout、通用 lowering 和 schedule 的组合。
5. 最干净的 compiler proof 是同一 high-level source/schedule、只切换 lowering，
   并保留到 final ISA；Z8W满足“机器图不同”，但因为 Z8W latency没有超过 Z5B，
   不能把它写成最终性能成功。

“改汇编”不能等价解决这个问题。ISA/HSACO 可以作为 external-kernel bridge
证明硬件上限或作为诊断 control，但它绕过了 AveLang 的 source/IR/lowering，
不能证明 AveLang compiler 已经能生成同等代码。提交给 compiler 团队的修复应
落在通用 operand/address/layout lowering，而不是 Qwen-specific 手写汇编。

## 9. 给后来优化者的实验方法

### 第一步：先确认比较对象

- Z5B 固定 WG256；
- native 每个 T 先走真实 public selector，再记录实际 BK/BV/warps/stages；
- 不把 native WG128 与 Z5B WG256 混成同一 shape；
- body benchmark 与 full Eager benchmark 分开。

### 第二步：只改一个数据流变量

例如只做：

- Q cache 直接 consumer；或
- K packet 的 typed producer；或
- g residency；或
- 一个 raw-buffer operand 的 lowering。

不要同时改 Q/K/H、WG、BV、MFMA geometry、barrier、RA 和 pipeline，否则无法
解释收益来源。

### 第三步：correctness 先行

至少覆盖：

```text
T=64/512/1024/2048/4096/8192/16384
BF16 byte-exact 或明确冻结误差范围
finite
caller-owned output
zero V-new
NaN-prefilled output
```

任意长度非法访问或输出未完全覆盖，不能进入性能排名。

### 第四步：同时保存四层机器证据

```text
source -> lowered LLVM -> exact-LTO MIR -> final ISA/HSACO
```

再用 rocprof 采动态 MFMA/VMEM/LDS/VALU/SALU。静态 ISA 只能回答“机器图是否
改变”，不能冒充动态工作量。

### 第五步：用跨长度和 fresh process 决定 winner

至少 T=2048、8192，最好补 T=512/1024/4096/16384；每个 arm 做多个 fresh
process session，轮换顺序。只有短文本、长文本都稳定快，才升级 baseline。

## 10. 文件地图

### 本最小包

| 文件 | 作用 |
|:--|:--|
| `avelang/chunko_z5b_minimal.py` | 最小、可读的纯 Avelang Z5B source |
| `triton/chunk_o.py` | current-vLLM Triton chunk-o 源码 |
| `triton/metadata.yaml` | dtype/shape/selector 说明 |
| `benchmark/bench_chunko_z5b_vs_triton_minimal.py` | fresh-process body benchmark |
| `reports/06_z5b_direct_q_cache.md` | Z5B 正式结果 |
| `reports/07_z5b_remaining_vmem_ledger.md` | remaining machine gap |
| `reports/00_chunk_o_optimization_complete_report_cn.md` | 本总复盘 |

### 完整原始证据

仍在仓库原位置：

```text
codex_qwen_bt64_stage6z_native_chunko/
codex_qwen_bt64_stage6z_z5b_machine_stage1/
codex_qwen_bt64_stage6z_z5b_rocprof/
codex_qwen_bt64_stage6z_z5b_t2048_bench.json
codex_qwen_bt64_stage6z_z5b_t8192_bench.json
```

其中包含 lowered LLVM、pre-LTO AMDGCN、exact-LTO MIR、ISA、HSACO、readobj、
rocprof CSV/JSON。最小包不复制这些大文件，是为了让别人先读懂 source 和报告；
需要审查机器细节时再按上面的路径回看。

### 与 compiler 修改的关系

Z5B 本身不是一个新的 compiler patch；它使用当前 Avelang branch 已有的：

- `al.amdgpu.mfma_32x32x8_bf16_f32` intrinsic 注册和 lowering；
- `al.make_shared` / `al.view` / BF16 shared load/store lowering；
- AMDGPU backend 的 code-object/replay debug 能力。

后续 Z8W、block-dot、C17-C26 才专门扩展或切换了通用 compiler lowering。它们的
报告副本在 `reports/10`、`reports/15`、`reports/16` 以及原始 Stage6Z 工件中。
因此不要说“Z5B 依赖一段 Z5B 专用汇编”；Z5B 是 Avelang source kernel，Triton
只作为 native 对照。

## 11. 最终状态

```text
standalone pure Avelang best: Z5B
native Triton diagnostic: faster, about 1.54x at T=2048 and 1.72x at T=8192
Z5B correctness: full recorded matrix passed
Z5B scratch/spill: zero
Z5B production promotion: No, remains experimental-only
X2+Z5B: separate full-operator combination, not this package
```

下一位优化者最值得研究的是 remaining operand feeding/layout 的通用表达，尤其
让 K/H/V-new 共享更接近 Triton 的 typed blocked/dot operand pipeline；但必须
从 Z5B 分叉、保留它的 correctness 和 fresh body benchmark，不能直接把 C19/C25
的较低 counter 当作成功答案。

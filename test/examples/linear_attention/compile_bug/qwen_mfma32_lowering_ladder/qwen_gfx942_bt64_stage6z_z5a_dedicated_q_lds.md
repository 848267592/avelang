# Qwen gfx942 BT64 Stage 6Z Z5A：Dedicated Full-Q LDS Cache

## 结论

Z5A 完成了从修复后 fixed Z2 分叉的 dedicated full-Q LDS cache 实验。
在当前 gfx942 Docker 环境中，Z5A 通过了 correctness、finite、caller-owned
output 和 NaN 预填充检查；T=2048 和 T=8192 的 fresh-process body benchmark
均显示稳定收益。因此按预注册决策树归为 **Case A**：用约 16 KiB 额外
LDS 容量换掉重复 Q global producer，晋级为新的 **Stage 6Z isolated
research baseline**。

这不是 production 或 X2 晋级。Z5A 没有接入 X2、selector、Eager public
API、recurrence HSACO、allocator/RA 或 production dispatch。

最重要的结果如下：

| 项目 | fixed Z2 | Z5A | 变化 |
|:--|--:|--:|--:|
| Q source producer pass | 3 | 1 | -2 |
| code-object LDS | 16,384 B | 32,768 B | +16 KiB |
| code-object VGPR | 168 | 132 | 机器捕获值，不等同 profiler 字段 |
| code-object AGPR | 32 | 32 | 0 |
| profiler VGPR | 88 | 100 | +12 |
| profiler Accum_VGPR | 32 | 76 | +44 |
| scratch/private segment | 0 / 0 B | 0 / 0 B | 无 spill |
| dynamic MFMA | 160/CTA | 160/CTA | 0 |
| dynamic VMEM | 928/CTA | 672/CTA | -27.59% |
| dynamic LDS | 928/CTA | 1,440/CTA | +55.17% |
| dynamic VALU | 11,400/CTA | 7,136/CTA | -37.40% |
| dynamic SALU | 1,072/CTA | 768/CTA | -28.36% |
| T=2048 body | 0.078056 ms | 0.067720 ms | -13.24% |
| T=8192 body | 0.176743 ms | 0.161119 ms | -8.84% |

Z5A 的主要代价是 LDS 和寄存器资源增加，occupancy 从 `15.925%` 降到
`14.841%`。但是在 T=8192 的五个 paired session 中每一个差值都为负，
说明这个 occupancy 变化没有抵消 Q reload 消除带来的收益。

## 实验边界

固定不变：

- gfx942、wave64、BT64、BV64、BK32；
- WG256，两个 CTA/chunk-head；
- BF16 Q/K/V-new/H/output，FP32 g；
- MFMA32 geometry 和 K32 accumulation order；
- causal mask、数学、global IO、caller-owned output contract；
- fixed Z2 的 inter -> score half 0 -> score half 1 -> intra accumulator 顺序；
- 不修改 X2 immutable recurrence HSACO；
- 不修改 allocator/RA、selector、production dispatch；
- 不使用 double buffer、private Q array 或旧 broad/compact-K 路线。

实验源文件：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5a_dedicated_q_lds.py`

固定 Z2 对照源文件：

`test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z2_phase_aware.py`

## Z5A 高级代码改变

fixed Z2 在 Phase A 和两个 Phase B score half 中都从 `q` global tensor
生产当前 Q K32 slice，因此 Q producer 在源码中出现三条独立路径。
Z5A 增加一块逻辑形状为 `[64, 128] BF16` 的 Q cache，物理上按四个
`[64, 32]` stage 存放，总大小：

```text
64 * 128 * sizeof(bf16) = 16 KiB
```

最终实现把 Q cache 和原 Z2 phase buffer 放到同一个 32 KiB shared
allocation 的两个不重叠区域：

```text
shared rows [0,   256): dedicated Q cache, 16 KiB
shared rows [256, 512): original Z2 phase buffer, 16 KiB
```

Q cache 在 Phase A 之前由 CTA 生产一次。之后：

```text
Q global
  -> scaled BF16
  -> dedicated Q LDS cache, one producer path
  -> phase-local Q slice for current K32 stage
  -> original Z2 phase_vec MFMA consumer
```

Phase A、Phase B0、Phase B1 都只能从 Q cache 取得 Q；没有再次访问 kernel
Q pointer。为了保持 Z2 的 accumulator phase separation，当前 K32 Q slice
会从 dedicated cache 复制到原 phase 区域，然后继续使用原 `phase_vec`
operand feeding。这个选择增加了 LDS read/write，但避免了把三套 accumulator
合并到同一个 loop。

在最终源文件中，kernel Q pointer 的直接访问只有一处：

```text
q[0, chunk_start + row, key_head_idx, k_stage * BK + col]
```

对应源文件约第 111 行。Phase A/B 的 Q consumer 不再有 `q[...]` 访问。

## 实现过程与失败的中间机器图

这些不是额外性能 arms，而是同一个 Z5A 候选为满足机器审计而进行的
源表示修正。最终结论只使用最后一次 stage5 artifact。

### Stage 1：直接 persistent Q cache consumer

最初的实现使用独立 Q cache 和 `q_cache_vec`，Phase A/B 直接从 persistent
cache 取 MFMA words。结果：

- correctness 在旧版本通过；
- LDS 为 32 KiB；
- static MFMA32 为 56，而 fixed Z2 为 20；
- static barrier 为 28，而 fixed Z2 为 9。

这表明直接把 Q cache 作为新 MFMA consumer path，会改变 LTO 对原有循环
和条件分支的处理，不能把这份机器图当成“只删除 Q reload”的干净结果。

### Stage 2：stage-major Q cache physical encoding

把 Q cache 改成物理 `[4*64, 32] BF16` stage-major layout，试图复用 Z2
的 `(256, 4, 4) i32` word view。static MFMA 仍为 56，说明仅修改 cache
物理行序不能恢复 Z2 的 loop graph。

### Stage 3：cache-fill stage barrier

在四个 Q cache fill stage 之间加入显式 workgroup barrier，作为编译器阶段
边界。结果 static MFMA 仍为 56，barrier 增到 32；它不是有效的性能修复。

### Stage 4：一个 32 KiB allocation

把 Q cache 和 phase 改为同一 32 KiB allocation 的两个不重叠区域，避免
两个独立 shared global。第一次 view 失败是 AveLang `memref.cast` 要求
source/target 总 byte size 相同；随后改为完整 32 KiB view，逻辑索引限制
仍保证两个区域不重叠。static MFMA 仍为 56。

### Stage 5：恢复 Z2 phase operand feeding

最终版本在每个 K32 stage 中把 Q cache 的当前 slice 复制到 phase 的 Q
区域，Phase A/B 再使用 Z2 原来的 `phase_vec`。这恢复了 phase-separated
数学和正确性，但机器图仍保留 56 条 lexical MFMA 和 32 条 static
barrier。动态 PMC 显示实际执行的 MFMA 仍为 160/CTA，因此 static lexical
展开和 dynamic arithmetic work 必须分开报告。

## Same-source / lowering identity 证据

本实验不是同一 source 的 generic/specialized lowering A/B；它是从 fixed
Z2 高级 kernel 分叉出的 source-level dedicated-cache candidate。故不能把
它声称为“只改变 lowering”的纯 compiler A/B。它能证明 Q producer lifetime
和全局流量的效果，但不能单独证明所有收益都来自 lowering。

### Source identity

- Z2 源码在 Phase A、Phase B0、Phase B1 各有一个 Q global producer path。
- Z5A 源码中 `q[...]` 只有一个 producer path。
- Z5A 没有增加 Q private array，也没有把 Q cache 做成 global output。
- `inter_acc` 在 score accumulator 创建前完成；`score_acc` 仍按 source half
  串行创建和消费。

### Lowered LLVM

最终 lowered LLVM：

`codex_qwen_bt64_stage6z_z5a_machine_stage5/lowered_llvm.ll`

关键可检查事实：

1. kernel 参数 `%0` 只在 Q cache fill 区域形成 global Q GEP/load。
2. 后续 Q 读取是 `addrspace(3)` shared load/store，不再从 `%0` 形成 Q
   global GEP。
3. dedicated Q cache 与 phase 是同一个 32 KiB addrspace(3) global，地址
   区域通过行偏移分离。
4. MFMA call site 仍是同一 `mfma_f32_32x32x8bf16` intrinsic family。

`rg` 对最终 LLVM 的结果是：Q pointer `%0` 的 bfloat load 只有一个 source
region；其它 Q-stage loads 使用 addrspace(3)。

### MIR / ISA

最终 exact-LTO 和 llc 工件：

```text
codex_qwen_bt64_stage6z_z5a_machine_stage5/exact_lto/
codex_qwen_bt64_stage6z_z5a_machine_stage5/llc_mir/
codex_qwen_bt64_stage6z_z5a_machine_stage5/final_isa.s
```

final ISA lexical count：

| instruction family | fixed Z2 | Z5A |
|:--|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 20 | 56 |
| `ds_read*` | 20 | 152 |
| `ds_write*` | 88 | 240 |
| global load family | 136 | 192 |
| global store family | 16 | 16 |
| `s_barrier` | 9 | 32 |
| `v_add` | 187 | 203 |
| `v_lshl_add` | 159 | 138 |

这些是最终 ISA 的 lexical/static counts，不能当作 dynamic PMC。Z5A 的
static MFMA 增长是 LTO/ISA 展开现象；rocprof 的 dynamic `SQ_INSTS_MFMA`
在两者都为 81,920 total、即 160/CTA。报告保留这个差异，不能把 56
伪写成 20，也不能把 static 56 直接当成执行了 56 个 MFMA/CTA。

### HSACO metadata / spill

最终 Z5A HSACO：

```text
SHA256: 976857f75eca9cb1a8351277529b33e576686511a347d1c3e0ce0317cf4aaa65
```

最终 Z5A code-object metadata：

| field | value |
|:--|--:|
| VGPR | 132 |
| AGPR | 32 |
| SGPR | 28 |
| group segment | 32,768 B |
| private segment | 0 B |
| VGPR spill | 0 |
| SGPR spill | 0 |
| wavefront | 64 |
| workgroup | 256 |

exact-LTO `summary.json` 中所有 greedy/virtregrewriter/prologepilog sections
的 `av32_spill_saves`、`av64_spill_saves`、`spill_virtual_registers` 均为
零。Z5A 与 Z2 的 HSACO hash 不同，说明不是复用了 Z2 code object。

## Correctness gate

测试脚本：

`test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z5a.py`

Docker fresh run：

```text
11 passed in 13.12s
```

覆盖：

| 检查 | 结果 |
|:--|:--|
| Z5A vs fixed Z2 BF16 byte-exact | pass |
| T=64 | pass |
| T=512 | pass |
| T=1024 | pass |
| T=2048 | pass |
| T=4096 | pass |
| T=8192 | pass |
| T=16384 | pass |
| finite output | pass |
| T=64 zero-V + NaN output | pass |
| T=8192 zero-V + NaN output | pass |
| T=16384 zero-V + NaN output | pass |

因此没有在 correctness 失败后继续使用错误结果做性能结论。

## T=2048 dynamic PMC

rocprofv3 工件：

```text
codex_qwen_bt64_stage6z_z5a_rocprof_stage5/
```

T=2048 有 `32 * 8 * 2 = 512` 个 CTA。下面是总计数除以 512 得出的
per-CTA dynamic value；不是从 static ISA 推导：

| metric | fixed Z2 total | Z2/CTA | Z5A total | Z5A/CTA | 变化 |
|:--|--:|--:|--:|--:|--:|
| MFMA | 81,920 | 160 | 81,920 | 160 | 0 |
| VMEM | 475,136 | 928 | 344,064 | 672 | -27.59% |
| LDS | 475,136 | 928 | 737,280 | 1,440 | +55.17% |
| VALU | 5,836,800 | 11,400 | 3,653,632 | 7,136 | -37.40% |
| SALU | 548,864 | 1,072 | 393,216 | 768 | -28.36% |

资源和 trace：

| metric | fixed Z2 | Z5A |
|:--|--:|--:|
| profiler VGPR | 88 | 100 |
| profiler Accum_VGPR | 32 | 76 |
| profiler SGPR | 112 | 112 |
| LDS block | 16,384 B | 32,768 B |
| Scratch | 0 | 0 |
| OccupancyPercent | 15.9251 | 14.8410 |
| rocprof trace median | 55.362 us | 41.402 us |

rocprof trace 带有工具扰动，只作为动态机器工作辅助；正式 latency 使用
HIP event 的 no-Graph body benchmark。

## T=2048 fresh-process body benchmark

benchmark 脚本：

`test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z5a.py`

口径：caller-owned isolated body、预分配 output、当前 HIP stream、no Graph、
warmup=10、repeat=50、5 fresh processes、rotating order。

第二轮稳定复测为正式 T=2048 结果；第一轮原始样本仍保留，但后两 session
出现 0.8--1.0 ms 的外部 GPU contention，不能用于晋级判断。

| arm | median of 5 session medians |
|:--|--:|
| fixed Z2 | 0.078056 ms |
| Z4C diagnostic | 0.075392 ms |
| Z5A | 0.067720 ms |

Z5A 相对 fixed Z2：

```text
absolute gain = 10.3355 us
relative gain = 13.2411%
all five paired differences < 0
```

第二轮每个 paired difference（Z5A - Z2）：

```text
-12.2780 us, -9.6745 us, -10.2750 us, -11.3970 us, -9.7745 us
```

## T=8192 长文本复测

正式结果文件：

`codex_qwen_bt64_stage6z_z5a_t8192_bench_stage5.json`

五个 session 全部稳定：

| arm | median of 5 session medians |
|:--|--:|
| fixed Z2 | 0.176743 ms |
| Z4C diagnostic | 0.192486 ms |
| Z5A | 0.161119 ms |

Z5A 相对 fixed Z2：

```text
absolute gain = 15.6235 us
relative gain = 8.8397%
all five paired differences < 0
```

paired Z5A - Z2：

```text
-17.9470 us, -15.6030 us, -14.7215 us, -16.7845 us, -12.8790 us
```

以 32 chunks 和 128 chunks 两个端点计算的 body slope：

| arm | endpoint slope |
|:--|--:|
| fixed Z2 | 1.027984 us/chunk |
| Z5A | 0.972901 us/chunk |

Z5A 的端点 slope 约下降 `5.36%`。这不是完整长文本拟合，只是两个已测
端点之间的 diagnostic slope；后续若需要 publication-quality slope，需在
同口径增加更多 T 点。

## Q reload 到底减少了什么

可以确认的事实：

1. source-level Q global producer 从 3 条变成 1 条。
2. LLVM 中从 kernel Q pointer 产生的 Q bfloat load 只在 cache fill 区域
   出现；后续 Q 使用来自 addrspace(3)。
3. dynamic MFMA 不变，说明收益不是减少数学工作。
4. dynamic VMEM 从 928 降到 672/CTA，和 Q producer elimination 方向一致。
5. VALU/SALU 也下降，说明重复 Q address/scale/producer 的一部分机器工作
   被消除。
6. LDS 从 928 增到 1,440/CTA，因为本实现把 Q cache 的当前 K32 slice
   再发布到原 Z2 phase layout，以维持 Z2 accumulator/operand graph。

不能从当前 counters 单独断言每一条 VMEM 都属于 Q，也没有 transaction-byte
counter 可用。因此更严格的表述是：source/LLVM provenance 证明 Q global
producer pass 减少，PMC 证明总 VMEM 同向下降；不能把 `256 VMEM/CTA` 的
全部差值硬分配给某一种 Q load width。

## 与预注册 Case 对照

### Case A：成立

- Q global producer pass：3 -> 1；
- VMEM：928 -> 672/CTA；
- dynamic MFMA：160 -> 160/CTA；
- scratch/spill：仍为 0；
- T=2048：稳定约 13.24% faster；
- T=8192：稳定约 8.84% faster；
- 长文本 paired 差值全部同方向。

因此 Z5A 晋级为 Stage 6Z isolated research baseline。

### Case B：没有发生

32 KiB LDS 确实让 occupancy 和 Accum_VGPR 变差，但没有导致 T=8192 latency
回退。故不能把本轮归类为 LDS residency cliff。

### Case C：没有发生

LLVM Q pointer provenance 没有显示 Q 被重新从 global materialize；Q cache
实际位于 addrspace(3)，且动态 VMEM 下降。故不是 compiler 完全忽略 Q
cache 的结果。

## 仍然存在的风险和限制

1. **static ISA 图不够紧凑。** Z5A final ISA 有 56 条 lexical MFMA 和
   32 条 static barrier，而 fixed Z2 是 20/9。dynamic MFMA 相同，但 code
   size 和静态展开差异仍是后续 compiler/source 审计对象。
2. **寄存器压力上升。** profiler Accum_VGPR 32 -> 76，VGPR 88 -> 100；
   当前尚未出现 spill，因此本轮仍通过资源 gate，但没有声称资源完全不变。
3. **LDS 工作增加。** Q cache -> phase 的再发布不是免费操作；本轮收益来自
   Q global reload 消除和随之减少的 VALU/SALU，不能推广成任意 persistent
   cache 都会更快。
4. **只测 isolated body。** 本轮没有 Eager public API，也没有 full X2 graph；
   Z5A 不能宣称为 Qwen production kernel 或超过 vLLM full operator。
5. **当前环境存在过 contention。** 首轮 T=2048 的后两 session 被保留并
   标记为无效噪声；正式结论使用第二轮全稳定复测和 T=8192 复测。

## 决策

`Z5A = new isolated Stage 6Z research baseline`。

允许的后续方向只有：在不改变 Z5A Q residency 语义的前提下，单独审计并
降低其额外 LDS republish/static MFMA expansion；或者建立明确的 full-graph
接入实验计划。禁止把它直接接入 X2，除非另行通过 full graph correctness、
Eager public API 和预注册 gate。

本轮没有实现 Z5A selector，也没有修改 production dispatch。

## 复现命令

```bash
cd /workspace/project/avelang

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z5a.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/dump_qwen_gdn_bt64_stage6z_z5a_machine_artifacts.py \
  --variant z5a --T 2048 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_z5a_machine_stage5 \
  --skip-initial-mlir

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z5a.py \
  --T 2048 --sessions 5 --warmup 10 --repeat 50

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_stage6z_z5a.py \
  --T 8192 --sessions 5 --warmup 10 --repeat 50
```


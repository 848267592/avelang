# Qwen gfx942 Stage 6Z：BDV2-P3 Packed Operand Reuse / Wide LDS Consumer

## 1. 本轮结论

本轮完成了一个 experimental-only 的通用 compiler lowering 实验：在同一份
`al.amdgpu.block_dot_bf16_f32` full-scope source 上，把相邻的两个
`<4xbf16>` MFMA operand fragment 组织成一个 128-bit LDS load，再从这个
`vector<8xbf16>` 中拆出 low/high 两个 fragment，分别供两个相邻的 MFMA32
consumer 使用。

结果分成两个层面：

1. **机器表示层成功。** P3 的 final HSACO 与 P2 不同，P3 的静态 `ds_read`
   从 P2 的 104 条恢复到 56 条，和 P1 的宽读取机器图一致；P3 的 LLVM/MLIR
   中存在一个 `vector<8xbf16>` 的 LDS load，并有两个显式 slice user。
2. **性能层没有晋级。** P3 与 P1 收敛到相同的最终 machine graph，T=2048 和
   T=8192 都稳定慢于 Z5B。P3 只删除了 P2 自己引入的 packed-load/LDS
   物化惩罚，没有删除 BDV2 full-scope 共有的 affine/layout VALU 工作。

因此本轮正式决策为：

```text
Z5B       = Stage 6Z isolated performance baseline
P3        = correctness PASS + representation PASS + performance No-Go
P3 infra  = 保留为通用 block_dot compiler infrastructure / regression evidence
production/X2 = 不修改
```

P3 没有新建 Qwen/chunk-o public op，没有修改 production selector、X2
recurrence HSACO、allocator/RA、WG、MFMA geometry、ownership、数学或 ABI。

---

## 2. 实验问题与冻结边界

### 2.1 要回答的问题

P2 已经证明 first-class MFMA operand 可以从 GPU outlining 后保留到专用 late
materialization，并最终影响 HSACO。但 P2 使用每个 fragment 的 B64/i64 LDS
读取：

```text
one <4xbf16> fragment
    -> one i64 / B64 LDS read
    -> one MFMA operand
```

这个表示把本来相邻的 operand load 拆散了，T=2048 动态 LDS 从 P1/P3 的
`464/CTA` 增加到 P2 的 `592/CTA`。P3 的唯一问题是：

```text
two adjacent <4xbf16> fragments
    -> one aligned <8xbf16> / B128 LDS read
    -> low/high <4xbf16> slices
    -> two MFMA consumers
```

本轮不重新研究 producer，不重新设计 layout，也不改变 MFMA 数量。它只检查
“一个 packed operand load 是否可以服务两个 MFMA consumer”。

### 2.2 冻结的 source contract

P3、P2、P1 和 Z5B 使用同一份 full-scope Qwen chunk-o source，冻结：

- gfx942、wave64；
- BT64、BV64、BK32；
- WG256，每 chunk-head 两个 CTA；
- BF16 Q/K/H/V-new/output ABI，FP32 `g` 和 accumulator；
- dedicated full-Q LDS cache；
- phase-separated accumulator 顺序：`inter_acc -> score_acc0 ->
  score_acc1 -> intra_acc`；
- causal mask、K32 accumulation order、MFMA32 geometry；
- global layout、caller-owned output；
- K/H 通过同一个 `LogicalBlockLayoutPlan` 和同一个通用
  `block_dot_bf16_f32` contract；
- Q/K/H/V-new/g/output producer、barrier phase 和 source schedule；
- no private large tile、no double buffer、no global V-new reload。

选择器只改变 late operand materialization：

```text
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=none
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p2_first_class
AVELANG_BLOCK_DOT_OPERAND_PRESERVATION=p3_packed_reuse
```

P3 的 Python kernel source 与 P2 相同，source SHA256 为：

```text
aee02e0309152ab34893836720c38266c6519c57499ef640db3e4ff48152047e
```

这里的 source hash 是 source file hash，不冒充 pre-branch MLIR hash。当前
runtime binding 的 `get_mlir()` 在 initial snapshot 路径会 SIGSEGV，因此 P3
机器 dump 使用 `--skip-initial-mlir`，初始 MLIR hash 记录为 unavailable。

---

## 3. P1、P2、P3 的关系

### 3.1 P1：typed fragment 表示，但最终收敛

P1 已经把 consumer 的高层请求从旧的：

```text
LDS <8xbf16>
    -> extractelement/insertelement
    -> <4xbf16>
    -> MFMA32
```

改成更直接的 `<4xbf16>` typed fragment 表示。LLVM 文本中的
`extractelement/insertelement` 减少，但 AMDGPU/LTO 后最终与 BDV2-S 收敛到：

```text
HSACO SHA256 = d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a
```

P1 的 T=2048 动态机器工作为：

```text
MFMA/VMEM/LDS/VALU/SALU = 160/448/464/8410/780 per CTA
```

这组高 VALU/layout 工作是 P3 没有修改的共有成本。

### 3.2 P2：first-class operand 保留到 late lowering

P2 在 GPU outlining 后创建内部：

```text
ave.gpu.amdgpu_block_dot_mfma_operand
```

它保存 operand role、K/H source role、logical shape、physical encoding、
fragment mapping、目标 MFMA 等信息。P2 的 late lowering 使用：

```text
addrspace(3) pointer
    -> volatile llvm.load i64, align 8
    -> bitcast i64 -> vector<4xbf16>
    -> existing MFMA32 intrinsic
```

P2 机器图因此真正不同，但代价是静态 `ds_read=104`、动态 LDS
`592/CTA`，并且 T=2048 body 比 Z5B、P1 都慢。

### 3.3 P3：packed pair reuse

P3 只对 P2 的 operand consumer 做成对分组。对一个 consumer group，late
lowering 生成：

```text
%packed = llvm.load ptr addrspace(3), align 16 : vector<8xbf16>
%low    = vector.extract_strided_slice %packed [0] : vector<4xbf16>
%high   = vector.extract_strided_slice %packed [4] : vector<4xbf16>
%acc0   = MFMA32(..., %low,  ...)
%acc1   = MFMA32(..., %high, %acc0)
```

A operand 和 B operand 都按这个 pair 形态生成；K/H 的 `source_role` 仍只由
通用 planner metadata 区分。P3 没有把两个 MFMA 合并成一个 MFMA，也没有
改变每个 MFMA 的数值输入或 K32 次序。

---

## 4. 编译器实现

### 4.1 修改位置

实现集中在：

```text
lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc
```

主要 helper：

```text
usePackedOperandReusePlan()       line 147
emitPackedLdsLoad128()             line 1710
splitPackedFragmentPair()          line 1765
emitPackedConsumerGroup()          line 1778
P3 branch in matchAndRewrite()     line 1841
```

这些行号是本轮编译后的 source 位置附近的定位，不是 ISA PC。P3 仍复用
`LogicalBlockLayoutPlan`、`AMDGPUBlockDotMfmaOperandOp` 和现有 MFMA lowering，
没有引入 Qwen 专用地址表或新的 public dialect op。

### 4.2 P3 的 lowering 伪代码

```text
if preservation == p3_packed_reuse:
    for each consumer pair (word0, word1):
        a_packed = load_lds_b128(aligned_a_address)
        b_packed = load_lds_b128(aligned_b_address)

        a0, a1 = split(a_packed, low4, high4)
        b0, b1 = split(b_packed, low4, high4)

        acc0 = mfma32(a0, b0, acc)
        acc1 = mfma32(a1, b1, acc0)
else if preservation == p2_first_class:
    load each fragment as i64/B64
    bitcast each load to vector<4xbf16>
    issue the same MFMA sequence
```

这里的 `split` 是 vector slice，不是四个 BF16 scalar load，也不是
`ds_bpermute`。P3 没有增加 LDS barrier；它只改变每组 operand 的 load
宽度和 SSA grouping。

### 4.3 同源证据的限制

严格的 source-level code path 相同，P3/P2 的 source SHA 相同；但是当前
binding 的 initial `get_mlir()` 会崩溃，所以不能声称“pre-branch MLIR hash
完全相同”。能确认的证据层次是：

| 层次 | P2/P3 状态 | 证据 |
|:--|:--|:--|
| Python/source | 相同 | source SHA 相同 |
| post GPU outlining | 同一 block-dot source，P3 selector 分叉 | `post_gpu_outlining.mlir` |
| operand materialization MLIR | 不同 | P2 B64/i64；P3 B128/vector<8> |
| lowered LLVM | 不同 | P3 有 b128 group load/slice |
| pre-LTO AMDGCN | 不同 | P3 有宽 LDS read path |
| exact-LTO replay | 都成功 | return code 0 |
| final HSACO | 不同于 P2；与 P1 收敛 | SHA 如下 |

不能把 unavailable 的 initial MLIR 补写成“已验证 hash”。

---

## 5. MLIR、LLVM、MIR、ISA 证据

### 5.1 P3 post-materialization MLIR

P3 artifact：

```text
codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized/
  ir/post_block_dot_operand_materialization.mlir
```

关键片段的语义是：

```mlir
%82 = llvm.load %81 {alignment = 16,
        avelang.block_dot.first_class_lds_b128_group}
     : !llvm.ptr<3> -> vector<8xbf16>
%89 = llvm.load %88 {alignment = 16,
        avelang.block_dot.first_class_lds_b128_group}
     : !llvm.ptr<3> -> vector<8xbf16>
%90 = vector.extract_strided_slice %82
     {offsets = [0], sizes = [4], strides = [1]}
%91 = vector.extract_strided_slice %82
     {offsets = [4], sizes = [4], strides = [1]}
%94 = mfma32(..., %90, ...)
%95 = mfma32(..., %91, %94)
```

K/H 两处都存在同样的 `consumer_group = packed_b128_pair` metadata。这个
证据说明两个 MFMA consumer 共享一个 packed SSA load；它不只是把单个
`llvm.load i64` 改名。

### 5.2 P2/P3 final code object identity

| arm | HSACO SHA256 | code VGPR | code AGPR | SGPR | LDS | private | spill |
|:--|:--|--:|--:|--:|--:|--:|--:|
| P2 | `f612f23968d1e9b9e205b1daf75a70cef2d9016d56f44df24fef6d3ae20c7947` | 132 | 48 | 30 | 32768 B | 0 | 0 |
| P3 | `d17483b933a7abf68b34b0e193aa4e3e5d9f93f48a12cfa8d84b285c9091af5a` | 132 | 48 | 30 | 32768 B | 0 | 0 |

P3 的 HSACO 与 P2 不同，说明 P3 没有在 P2 的 late lowering 分叉后收敛
回去；但 P3 与 P1/BDV2-S 的 HSACO 相同，说明 P3 最终回到了此前已有的
P1 packed LDS machine shape，而不是产生一个新 winner。

### 5.3 Static ISA counts

这些是 final ISA lexical counts，只用于确认机器图和读取宽度，不能替代
rocprof 动态计数：

| arm | MFMA32 | global load | global store | ds_read | ds_write | s_barrier |
|:--|--:|--:|--:|--:|--:|--:|
| P2 | 56 | 140 | 16 | 104 | 92 | 44 |
| P3 | 56 | 140 | 16 | 56 | 92 | 44 |
| P1/BDV2-S | 56 | 140 | 16 | 56 | 92 | 44 |

P3 的 final ISA 关键 family 是 `ds_read_b128`；P2 的额外读取来自
`ds_read_b64`。P3 没有减少 static MFMA，也没有改变 global producer graph。

Exact-LTO artifacts：

```text
codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized/exact_lto/
  kernel_section_00.mir ... kernel_section_19.mir
  linked.hsaco.0.5.precodegen.bc
  replay_argv.json
  summary.json
```

`machine_summary.json` 记录 code object 为 `VGPR=132, AGPR=48, SGPR=30`、
LDS `32768 B`、private `0`、VGPR/SGPR spill `0`。Profiler T=2048 的
`VGPR=88`、`AccumVGPR=88` 是运行时资源计数，不能与 code-object
`VGPR=132/AGPR=48` 直接等同；P3 报告保留两套指标并明确区分。

---

## 6. Correctness

P3 使用 fresh process correctness driver，输入、Z5B reference、caller-owned
output 和 BF16 比较口径不变。结果如下：

| T | P3 vs Z5B BF16 output | finite | max abs |
|--:|:--:|:--:|--:|
| 64 | PASS / byte-exact | PASS | 0 |
| 512 | PASS / byte-exact | PASS | 0 |
| 1024 | PASS / byte-exact | PASS | 0 |
| 2048 | PASS / byte-exact | PASS | 0 |
| 4096 | PASS / byte-exact | PASS | 0 |
| 8192 | PASS / byte-exact | PASS | 0 |
| 16384 | PASS / byte-exact | PASS | 0 |

额外的 caller-owned output、zero-V-new 和 NaN-prefilled output 检查：

| T | byte-exact | finite | 结果 |
|--:|:--:|:--:|:--|
| 64 | 是 | 是 | PASS |
| 8192 | 是 | 是 | PASS |
| 16384 | 是 | 是 | PASS |

P3 没有依赖放宽误差、跳过 output 检查或改变 NaN prefill 语义。正确性原始
JSON 在：

```text
codex_qwen_bt64_stage6z_bdv2_p3_correctness/T*.json
```

旧能力回归也已重新运行：

```text
23 passed in 89.49s
```

覆盖 P3 full-scope 静态 contract、direct-K64 generic/specialized、BV32 typed
operand 和 Stage6S BF16 recurrence。P3 相关静态测试单独为 `7 passed`。

---

## 7. T=2048 动态 PMC 与资源

P3 使用 fresh profiler capture，按 512 个 CTA 归一化。动态计数不能从 static
ISA lexical count 推出。

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | profiler VGPR | profiler AccVGPR | occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 76 | 100 | 14.498718% |
| P1-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.634712% |
| P2-S | 160 | 448 | 592 | 8474 | 780 | 88 | 88 | 14.880388% |
| P3-S | 160 | 448 | 464 | 8410 | 780 | 88 | 88 | 14.940439% |
| native WG256 diagnostic | 160 | 140 | 480 | 3376 | 660 | native artifact | native artifact | native artifact |

P3 相对 P2 的变化：

```text
MFMA       160 -> 160 / CTA
VMEM       448 -> 448 / CTA
LDS        592 -> 464 / CTA
VALU       8474 -> 8410 / CTA
SALU       780 -> 780 / CTA
```

因此 P3 确实删除了 P2 的额外 LDS work，但 P3 的最终 dynamic PMC 与 P1
完全一致。相对于 Z5B，P3 仍然承担 BDV2 full-scope 的高 VALU/layout
成本：`8410 - 7072 = 1338 VALU/CTA`。剩余差距不是 P2 的 load width
问题。

---

## 8. 正式 body benchmark

### 8.1 口径

本轮使用的是 isolated body diagnostic，不是 public Eager API 最终排名：

- caller-owned preallocated output；
- 相同 current HIP stream；
- no Graph；
- compile/module load/allocation 在计时外；
- warmup=10、repeat=50；
- 7 个 fresh-process sessions；
- rotating arm order；
- 每个 worker 只负责一个 fresh process；
- 以 HIP event body median 为主。

原始结果：

```text
codex_qwen_bt64_stage6z_bdv2_p3_bench_T2048_sessions7.json
codex_qwen_bt64_stage6z_bdv2_p3_bench_T8192_sessions7.json
```

### 8.2 T=2048

下表是 7 个 session median 的 median；括号内为 7 个 session median 的
算术平均，单位 ms：

| arm | median-of-session-medians | mean-of-session-medians | 相对 Z5B |
|:--|--:|--:|--:|
| Z5B | `0.067460500` | `0.067420214` | `1.0000x` |
| P1-S | `0.070625000` | `0.070628000` | `1.0469x` |
| P2-S | `0.072828002` | `0.072836786` | `1.0796x` |
| P3-S | `0.070084002` | `0.069855286` | `1.0389x` |
| native | `0.042723501` | `0.045756429` | `0.6331x` |

P3-Z5B 的 7 个 paired HIP-event 差值全部为正：

```text
1.742, 2.624, 2.184, 2.363, 3.065, 2.804, 2.263 us
```

算术平均约为 `+2.435 us`。P3 相对 native 的 median ratio 是
`1.6404x`；native 的一个约 0.063 ms session outlier 被保留在原始 JSON，
但主表使用 median-of-session-medians。

### 8.3 T=8192

| arm | median-of-session-medians | mean-of-session-medians | 相对 Z5B |
|:--|--:|--:|--:|
| Z5B | `0.157334000` | `0.157428215` | `1.0000x` |
| P1-S | `0.168270000` | `0.168310215` | `1.0696x` |
| P2-S | `0.176522501` | `0.176585213` | `1.1219x` |
| P3-S | `0.167669497` | `0.168209999` | `1.0657x` |
| native | `0.091196001` | `0.091144071` | `0.5795x` |

P3-Z5B 的 7 个 paired 差值全部为正，范围为约 `+9.734` 到 `+13.680 us`，
平均约 `+10.75 us`。P3 相对 native 的 median ratio 是 `1.8386x`。

### 8.4 两点 endpoint slope

用 T=2048 和 T=8192 的 median-of-session-medians 拟合两点 slope，chunk 数
从 32 增加到 128：

| arm | endpoint slope |
|:--|--:|
| Z5B | `0.936182 us/chunk` |
| P3-S | `1.016516 us/chunk` |
| native | `0.504922 us/chunk` |

P3 的 slope 比 Z5B 高约 `0.080333 us/chunk`，约 `8.58%`。P3 没有在 T=2048
或 T=8192 提供稳定正收益，因此按预注册规则不运行条件性的 T=16384
performance；T=16384 correctness 已通过。

---

## 9. 为什么 P3 机器变好但性能没有变好

### 9.1 P3 做掉了什么

P3 删除的是 P2 自己引入的这部分机器工作：

```text
两个独立 B64 fragment LDS read
    -> 两个独立 fragment materialization
```

恢复为：

```text
一个 B128 LDS read
    -> low/high slice
    -> 两个 MFMA consumer
```

所以 P2 的 dynamic LDS `592/CTA` 回到 P3 的 `464/CTA`，static
`ds_read=104` 回到 `56`。这说明 P3 的 packed operand reuse 在编译器表示和
最终 ISA 层确实生效。

### 9.2 P3 没做掉什么

P3 没有改变：

- BDV2 full-scope producer ownership；
- K/H 的 affine index 计算；
- LDS physical address/swizzle 计算；
- generic layout/fragment feeding 的 VALU；
- accumulator phase schedule；
- VMEM producer graph；
- WG256、MFMA 数学、barrier 和 output path。

因此 P3 的 final PMC 与 P1 相同：`8410 VALU/CTA`。而 Z5B 是
`7072 VALU/CTA`。P3 只是从 P2 回到已有 P1 的 machine shape，无法消除
这 `1338/CTA` 的 full-scope VALU 负担。

### 9.3 对 lowering 的窄结论

本轮支持：

> AveLang compiler 可以在同一通用 block-dot contract 下保留一个 packed
> operand group，并把它物化成一条 B128 LDS read，供两个 MFMA consumer 使用。

本轮不支持：

> 只要把 MFMA operand packed，就能解决 Z5B/native 的主要性能差距。

如果继续性能研究，下一候选应针对 BDV2/P1 共有的 VALU provenance 或
representation-preserving planner 优化，不能再枚举 P2/P3 的 load width
变体。

---

## 10. 最终决策

| gate | 结果 |
|:--|:--|
| T=64/512/1024/2048/4096/8192/16384 correctness | PASS |
| caller-owned / zero-V / NaN-prefill | PASS |
| MFMA 数学工作不变 | PASS，160/CTA dynamic |
| P2 的 B64 LDS penalty 消失 | PASS，592 -> 464 LDS/CTA |
| scratch/private/spill | PASS，0 |
| final HSACO 与 P2 不同 | PASS |
| P3 稳定优于 Z5B，T=2048 | FAIL |
| P3 稳定优于 Z5B，T=8192 | FAIL |

正式结论是 **Case B：机器表示修复成功，性能不晋级**：

1. Z5B 继续作为 Stage 6Z isolated performance baseline。
2. P3 作为通用 `block_dot_bf16_f32` compiler infrastructure 保留。
3. P3 不建立 selector，不接 X2，不接 production。
4. 不继续做 P3 的更多 load width 或 packed pair 变体。
5. 若目标是性能，下一候选应针对 BDV2/P1 共有的 VALU/layout/address
   materialization，而不是再改 P2/P3 的 B64/B128 物化。

---

## 11. 产物与复现命令

### 11.1 机器产物

```text
codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized/
  ir/post_gpu_outlining.mlir
  ir/pre_block_dot_operand_materialization.mlir
  ir/post_block_dot_operand_materialization.mlir
  ir/preopt_llvm.ll
  ir/postopt_llvm.ll
  exact_lto/kernel_section_*.mir
  exact_lto/linked.hsaco.0.5.precodegen.bc
  final_isa.s
  pre_lto_amdgcn.s
  bdv2_specialized.hsaco
  machine_summary.json
```

### 11.2 Correctness

```bash
python3 check_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --arm specialized --planner bdv2_p1_affine \
  --preservation p3_packed_reuse --T 64 512 1024 2048 4096 8192 16384
```

zero-V/NaN-prefill 使用同一 driver 的 special-case 参数，原始结果在
`codex_qwen_bt64_stage6z_bdv2_p3_correctness/`。

### 11.3 Machine dump

```bash
python3 dump_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope_machine.py \
  --variant specialized --planner bdv2_p1_affine \
  --preservation p3_packed_reuse --T 2048 \
  --out-dir codex_qwen_bt64_stage6z_bdv2_p3_machine_specialized \
  --skip-initial-mlir
```

`--skip-initial-mlir` 是当前 binding 崩溃限制的显式记录，不是 correctness
或 machine evidence 的放宽。

### 11.4 Benchmark

```bash
python3 bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 2048 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_p1_specialized bdv2_p2_specialized \
         bdv2_p3_specialized native \
  --out codex_qwen_bt64_stage6z_bdv2_p3_bench_T2048_sessions7.json

python3 bench_qwen_gdn_bt64_stage6z_block_dot_v2_full_scope.py \
  --T 8192 --sessions 7 --warmup 10 --repeat 50 \
  --arms z5b bdv2_p1_specialized bdv2_p2_specialized \
         bdv2_p3_specialized native \
  --out codex_qwen_bt64_stage6z_bdv2_p3_bench_T8192_sessions7.json
```

上述 benchmark 均为 body diagnostic。它们不替代后续若要做的 Eager public
API 比较，也不允许把 private body 结果写成 production readiness。

---

## 12. 给 compiler 团队的最短结论

P3 是一个干净的 same-source late-lowering 证据：

```text
same source / same source schedule / same MFMA math
P2: fragment-by-fragment B64 LDS reads
P3: paired B128 LDS read + two explicit fragment slices
```

它让最终 ISA 从 `ds_read=104` 回到 `56`，动态 LDS 从 `592/CTA` 回到
`464/CTA`，并保持 correctness、scratch=0、spill=0。P3 最终与 P1 收敛，
而 P1/P3 都仍比 Z5B 慢，说明：

```text
P2 的 per-fragment packed materialization 是可修复的表示问题；
BDV2/P1 的剩余性能问题不是单纯 B64->B128 的 load-width 问题。
```

这条结论支持继续研究通用 planner 的 address/layout/fragment representation，
但不足以支持修改 RA、添加 Qwen 特例或接入 X2。

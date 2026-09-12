# Qwen Direct-K64 BV32 剩余数据流成本分解 Ladder

## 1. 结论

本轮按预注册顺序完成了两个 isolated controls，且没有修改 production
dispatch、allocator/RA、MFMA geometry、BV/CTA mapping、旧 broad-K/compact-K
或 full-v29 的 nonzero-W pred 路径。

固定实验对象是已经通过 ownership 对齐的 direct-K64 update suffix：

```text
BT=64, BV=32, WG=128, 32 CTA, 2 waves/CTA
K/V-new=BF16, g/state=FP32, H=BF16, final_state=FP32
direct K[0:64] / K[64:128]
v_mfma_f32_32x32x8_bf16
specialized block_dot_bf16_f32 scheduler
```

结果分成两条很明确的结论。

1. **A: 预计算 V-decay 不是主导成本。** 把 `g` load、`exp(g_last-g)`、
   `V-new * decay` 和 BF16 round 放到计时外，T=2048 只从 `0.431541 ms`
   降至 `0.427275 ms`（`0.99%`）。它虽然删除约 `30.4%` VALU，但没有
   明显改变总 body 时间。因此不应把剩余 native 差距主要归因于 decay。
2. **B: K/V operand 的 scalar global-to-LDS producer 是主要可恢复成本。**
   在相同 high-level source、pre-branch MLIR、BV32 ownership、LDS tile、
   barrier、MFMA、persistent H1/H2 和数学下，仅把 contiguous K/V 的 global
   producer 改成 typed BF16x8 vector load 后，T=2048 从 `0.430039 ms` 降至
   `0.258745 ms`，为 `1.662x` / `-39.83%`。native W=0 diagnostic control
   的差距从 `3.59x` 缩至 `2.16x`。

这个 B 结果是明确的 Avelang lowering 正结果，但不能据此声称整个 remaining
gap 都是 compiler 问题：native control 仍包含 fused-pred 工作，且拥有更大的
fused pipeline。当前证据只支持更窄的结论：**在冻结的 Avelang BV32 schedule
内，scalar K/V operand local-load lowering 确实制造了大部分可恢复 VMEM 和一大
部分长文本斜率。**

下一步才允许进入单独预注册的 C：typed operand 基线上测试一个 pipeline/fusion
control。不得同时改 double buffering、CTA ownership、MFMA、RA 或旧 K 路线。

## 2. 冻结边界与比较口径

本报告的 benchmark 都是 preallocated isolated update-suffix body，不是 full
forward，也不是 eager public API。每个 arm 使用相同随机 K/V-new/g/initial
state、同一 ABI、同一 stream、warmup=5、repeat=20。长基准因 Docker command
通道约 30 秒会截断连续输出，采用五个独立 fresh-process session；每个 session
中 three-arm 顺序按 `scalar -> typed -> native`、`typed -> native -> scalar`、
`native -> scalar -> typed` 循环。每个 arm 的数值是该 session 20 次 HIP-event
sample 的 median，表中为五个 session median 的 median。

native 一律指现有 `current Triton W=0` recurrence control。它与 Avelang 使用
相同 BF16 K/V-new boundary 和 BT64/WG128/32 CTA，但它仍有 fused-pred 工作，
不是 pure-update-only 时间。因此所有 native 倍数仅作诊断量尺，不能当作严格
compiler-only speedup。

| 固定项 | 值 |
|:--|:--|
| Avelang source | `repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py` |
| high-level update op | `al.amdgpu.block_dot_bf16_f32(...)` |
| launch | grid `(32,1,1)`，WG `(128,1,1)` |
| LDS operands | A `[1,32,32]` BF16，B `[32,32]` BF16，4096 B |
| CTA ownership | 1 CTA / value-head / V32；wave 0 负责 K[0:64]，wave 1 负责 K[64:128] |
| MFMA | gfx942 `v_mfma_f32_32x32x8_bf16`；T=2048 每 CTA 1024、全 grid 32768 |
| 保持不变 | state layout、K32 accumulation order、barrier、输出 layout、数学、dtype |

## 3. A: Precomputed V-decay Control

### 3.1 问题与唯一变量

原 scalar BV32 producer 在每个 V element 做：

```text
load g[token, head]
decay = exp(g_last - g)
load BF16 V-new[token, head, value]
round_bf16(f32(V-new) * decay)
store A-LDS[value, token]
```

控制 arm 在 kernel 外预先计算：

```python
v_decay = (v_new.float() * exp(g_chunk_last - g)).to(torch.bfloat16)
```

kernel 中只从同样的 source-V operand 读取该 BF16 `v_decay` 并写入相同 A-LDS
tile。persistent state 的 `exp(g_last)` 仍在 kernel 内，因为它是 state carry
数学的一部分，不能混入本控制变量。

新增 source API 是
`al.amdgpu.block_dot_bf16_f32_precomputed_vdecay(...)`；它仍生成相同 dedicated
`AMDGPUBlockDotBF16F32Op`，只带
`avelang.block_dot.precomputed_vdecay` attribute。late pass 的 K stage、A/B LDS
布局、barrier、MFMA consumer、ownership 都未变化。

### 3.2 正确性

比较的是 direct-K64 update reference，而非未解决的 full-v29 nonzero-W reference。

| T | H max abs | final-state max abs | 状态 |
|--:|--:|--:|:--|
| 64 | `0` | `5.72204590e-06` | pass |
| 512 | `0.25` | `1.52587891e-05` | pass |
| 2048 | `0.5` | `4.57763672e-05` | pass |

`pytest -q test_qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay.py -s`
报告 `3 passed in 19.45s`。

### 3.3 五-session body timing

| T | scalar BV32 ms | precomputed V-decay ms | gain | native W=0 ms |
|--:|--:|--:|--:|--:|
| 512 | `0.131836` | `0.130113` | `1.31%` | `0.037656` |
| 1024 | `0.224112` | `0.222490` | `0.72%` | `0.068021` |
| 2048 | `0.431541` | `0.427275` | `0.99%` | `0.120058` |

### 3.4 T=2048 PMC 与 ISA

| metric | scalar | precomputed | delta |
|:--|--:|--:|--:|
| trace median | `409.088 us` | `403.119 us` | `-1.46%` |
| VGPR (rocprof) | 64 | 32 | `-32` |
| AccVGPR | 160 | 160 | 0 |
| scratch / spill | 0 B / 0 | 0 B / 0 | unchanged |
| MFMA | 32768 | 32768 | unchanged |
| VALU | 1,798,464 | 1,251,584 | `-30.41%` |
| SALU | 214,208 | 214,272 | effectively unchanged |
| VMEM | 274,944 | 266,752 | `-2.98%` |
| LDS | 229,376 | 229,376 | unchanged |

precomputed ISA 保持 32 个 MFMA32、无 MFMA16、17 个 barrier、无 scratch/spill；
`v_exp` 从 scalar 的 5 降为 1。剩余 exp 是 persistent-state scale。即使 ALU
明显减少，global operand、LDS 和同步没有减少，故 wall-clock 收益很小。

**A 的结论：** 不应继续单独微调 `g`、decay 或 BF16 conversion；它们不是
remaining 3.59x gap 的主解释。

## 4. B: Typed K/V Operand Local-Load Same-Source A/B

### 4.1 设计

本控制刻意不改变 block-dot schedule。`AVELANG_BLOCK_DOT_LOWERING` 在两臂都
固定为 `specialized`，仅新增：

```text
AVELANG_BLOCK_DOT_OPERAND_LOWERING=scalar | typed_vector
```

所以这不是旧的 generic-vs-specialized scheduling A/B；这是已经冻结 specialized
BV32 schedule 内部的 **operand producer A/B**。

| arm | global-to-LDS producer | LDS consumer |
|:--|:--|:--|
| scalar | 128 threads x 8 scalar rounds；每个 K/V BF16 用标量 global load，逐元素写入 LDS | 不变：`vector.load BF16x8` from A/B LDS -> fragment extract -> MFMA32 |
| typed_vector | 每 thread 一次 contiguous `vector<8xbf16>` K 或 V global load；必要时一个 g scalar 对整段 Vx8 做 vector decay；再把 8 lane 写入同一 LDS 地址 | 完全不变 |

每个 tile 的 1024 个 K 或 V values、K/V global offsets、A `[1,32,32]` 与 B
`[32,32]` LDS shapes、17 barriers、两 K32 columns、两个 K64 halves 和 32 MFMA32
静态指令均保持不变。typed-vector 不是对 Triton memdesc 的完整语言级复刻；它是
最小的 typed contiguous local-load lowering control，用于隔离 scalar address/operand
构造。

实现位置：

| 文件 | 变更 |
|:--|:--|
| `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc` | 新增 `OperandStagingKind::{Scalar,TypedVector}`，以及只替换 producer 的 typed BF16x8 staging |
| `bench_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py` | fresh-process、rotating-order body harness |
| `profile_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py` | preallocated rocprof target |
| `test_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py` | scalar/typed 对 direct reference 的 T=64/512/2048 correctness coverage |

### 4.2 同源证明链

两臂均在 fresh JIT process 编译。pre-branch snapshot 的 hash 完全相同：

```text
pre_kfrag_branch.mlir
8675f9e514cd6f07b981c5f978212fdc90965d7bb9b59fba5bf30d682be331f2
```

| layer | scalar SHA-256 | typed-vector SHA-256 | 解读 |
|:--|:--|:--|:--|
| pre-kfrag branch MLIR | 相同 | 相同 | source / high-level op 相同 |
| post-kfrag rewrite MLIR | 相同 | 相同 | block-dot 未分叉 |
| post-block-dot lowering MLIR | `abc2dc36...` | `0d704660...` | 唯一预期分叉：producer expansion |
| pre-opt LLVM | `c38d4fa6...` | `f58b588b...` | typed vector form 保留到 LLVM |
| post-opt LLVM | `d6c7ed97...` | `56692676...` | LLVM 未将两臂重新收敛 |
| final ISA | 不同 | 不同 | final machine code 保留该差异 |

`post-block-dot lowering` 两臂都有 17 个 `gpu.barrier` 和 17 个 MFMA32
call；typed arm 的 `vector.load` 为 24，scalar 为 18，是预期的 typed producer
形态，而不是多做 MFMA 或新增同步。

### 4.3 正确性

两 arm 分别对同一 direct-K64 update reference 测试。二者的误差与原 BV32
control 相同，且没有 NaN/Inf。

| T | scalar H / final max abs | typed H / final max abs | 状态 |
|--:|:--|:--|:--|
| 64 | `0` / `5.72204590e-06` | `0` / `5.72204590e-06` | pass |
| 512 | `0.25` / `1.52587891e-05` | `0.25` / `1.52587891e-05` | pass |
| 2048 | `0.5` / `4.57763672e-05` | `0.5` / `4.57763672e-05` | pass |

这里的 tolerance 是已有 BF16 H snapshot 与 FP32 persistent-state reference
contract。当前测试证明两个 arm 都符合该 contract；本轮没有把“同一 reference
error”夸张称为单独 dump 的跨进程 bitwise proof。

### 4.4 正式 body benchmark

| T | chunks | scalar ms | typed ms | scalar / typed | typed / native W=0 | native ms |
|--:|--:|--:|--:|--:|--:|--:|
| 512 | 8 | `0.131896` | `0.086349` | `1.528x` | `2.286x` | `0.037776` |
| 1024 | 16 | `0.223893` | `0.132156` | `1.694x` | `1.943x` | `0.068021` |
| 2048 | 32 | `0.430039` | `0.258745` | `1.662x` | `2.158x` | `0.119898` |

完整 five-session 原始 medians（单位 ms）：

| T | scalar sessions | typed sessions | native sessions |
|--:|:--|:--|:--|
| 512 | `0.130593, 0.132257, 0.131896, 0.133960, 0.130694` | `0.086648, 0.085968, 0.086349, 0.086048, 0.086969` | `0.037816, 0.037619, 0.037776, 0.037776, 0.037496` |
| 1024 | `0.223012, 0.222451, 0.223893, 0.224254, 0.225875` | `0.132097, 0.133899, 0.132077, 0.133358, 0.132156` | `0.068021, 0.068021, 0.068061, 0.068141, 0.068021` |
| 2048 | `0.429658, 0.432021, 0.428376, 0.430039, 0.431441` | `0.259405, 0.257623, 0.258745, 0.255639, 0.262350` | `0.120499, 0.119898, 0.120178, 0.113128, 0.119898` |

每个 paired session 的 `scalar - typed` 都为正：

| T | paired gain range | median paired gain |
|--:|--:|--:|
| 512 | `43.725–47.912 us` | `45.548 us` |
| 1024 | `88.551–93.719 us` | `90.915 us` |
| 2048 | `169.091–174.400 us` | `170.253 us` |

端点 slope（T=512 到 T=2048）：

| arm | body slope | gap-to-native slope |
|:--|--:|--:|
| scalar | `12.423 us/chunk` | `9.001 us/chunk` |
| typed | `7.183 us/chunk` | `3.761 us/chunk` |
| native W=0 | `3.422 us/chunk` | n/a |

typed producer 收回 `5.239 us/chunk` body slope，或 `58.2%` 的 scalar-to-native
gap slope；但仍剩 `3.761 us/chunk`，因此它不是终点。

### 4.5 T=2048 rocprof

每次 matching dispatch 的 grid 是 4096 global work-items，即 32 CTA x WG128。
下表 counter 在同 arm 的 matching dispatch 中恒定；trace 是最后五个 matching
dispatch 的 median。

| metric | scalar | typed-vector | change |
|:--|--:|--:|--:|
| trace median | `401.316 us` | `228.580 us` | `-43.04%` |
| grid / WG | `4096 / 128` | `4096 / 128` | same |
| LDS block | 4096 B | 4096 B | same |
| scratch | 0 B | 0 B | same |
| MFMA | 32768 | 32768 | same |
| MFMA / CTA | 1024 | 1024 | same |
| VALU | 1,798,464 | 1,636,032 | `-9.03%` |
| SALU | 214,208 | 214,272 | effectively same |
| VMEM | 274,944 | 102,912 | `-62.57%` |
| LDS instructions | 229,376 | 229,376 | same |
| OccupancyPercent median | `0.639763` | `0.626177` | small reduction |
| rocprof VGPR count | 64 | 4 | collector-dependent; see note |
| rocprof AccVGPR count | 160 | 164 | `+4` |
| rocprof SGPR count | 48 | 48 | same |

`rocprof` 的 VGPR number 与 HSACO symbol metadata 不应混为同一单位：code object
symbol reports scalar `num_vgpr=144`, typed `num_vgpr=120`，two arms both
`num_agpr=32` and private segment size 0. 因此本报告不把 `64 -> 4` 解读为真实
“只有 4 个 physical VGPR”；它是 collector metadata 的分类/归一化表现。可靠且
一致的资源结论是：**两臂均无 private scratch、无 spill，typed 没有 resource
cliff。** exact LTO MIR 同样没有任何 `SI_SPILL_AV32_SAVE` 或
`SI_SPILL_AV64_SAVE`。

### 4.6 ISA 与 MIR

| static ISA item | scalar | typed-vector | 解读 |
|:--|--:|--:|:--|
| `v_mfma_f32_32x32x8_bf16` | 32 | 32 | MFMA schedule identical |
| `v_mfma_f32_16x16x16_bf16` | 0 | 0 | no fallback |
| `s_barrier` | 17 | 17 | synchronization identical |
| `ds_read_b128` | 32 | 32 | MFMA operand consumer identical |
| `ds_write_b16` | 96 | 96 | LDS producer volume identical |
| `global_load_ushort` | 96 | 0 | scalar narrow loads eliminated |
| `global_load_dwordx4` | 8 | 20 | contiguous BF16x8 loads materialized |
| total buffer/global loads | 114 | 30 | `-73.68%` static instruction sites |

typed ISA contains actual `global_load_dwordx4` examples such as:

```text
global_load_dwordx4 v[82:85], v[4:5], off
global_load_dwordx4 v[86:89], v[4:5], off offset:32
```

Exact captured-LTO replay has no spill saves in either arm. The first
pre-greedy section is 2552 lines for scalar and 2392 for typed; post-
prologue/epilogue is 2471 versus 2311 lines. This supports the instruction
and VMEM result without claiming a one-to-one line-count-to-latency mapping.

## 5. 诊断

### A 回答了什么

V-decay arithmetic 的 ALU cost 可以显著降低，但因为 scalar operand global
loads、LDS producer/consumer 和 synchronization 不变，整体时间只改善约 1%。
因此剩余 gap 的大头不是 exp 本身，也不是单纯把 BF16 conversion 移出 kernel 就能
解决。

### B 回答了什么

在同一个 source schedule 下，当前 scalar global-to-LDS expansion 是明确的性能
问题。typed BF16x8 local load 不改变 MFMA、LDS store volume或 barrier，却减少
VMEM 62.6%、T=2048 trace 43.0%、正常 body 39.8%。这说明 remaining difference
中有大块来自 operand 构造和地址形成，而非 MFMA32 primitive 或 BV32 ownership
本身。

同时，typed arm 仍是 native W=0 的 2.16 倍，且它的 VMEM、SALU、LDS instruction
并未完全接近 native fused pipeline。因此剩余差距仍可能来自：

- K/V stage 与 MFMA 之间没有 prefetch/overlap；
- current block-dot 仍执行 scalar scatter 到 LDS，而 native 可拥有更紧凑的
  operand pipeline；
- native fused-pred、V-decay/recurrence 的整体 schedule 与 isolated suffix
  不同；
- persistent-state output/store 的 dataflow 仍不同。

不能根据本轮结果回头修改 RA，也不能声称 full-v29 compact-K 已被修复。实验没有
接触 full-v29 pred live region，且 full-v29 nonzero-W reference correctness 仍是
独立未解决问题。

## 6. 下一步决策

**A/B 已完成，允许但尚未实现 C：Pipeline/fusion control。** C 必须以 typed-vector
arm 为固定 baseline，只改一个明确的重叠控制变量，例如 next-token-half K/V
prefetch 的 double buffer。它必须保持本报告中的 BV32/32-CTA/WG128/MFMA32、typed
operand load、LDS shapes、state layout 和 output math；不得顺手改变 BV、CTA、MFMA
或 allocator。若 C 未在 T=2048 与长文本 slope 同时提供正收益，应停止 pipeline
line，而不是重新尝试旧 broad/compact-K。

## 7. 文件、证据与复现

源码与 harness：

- `lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`
- `test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay.py`
- `test/examples/linear_attention/vllm_compare/test_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_vdecay_ladder.py`
- `test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py`
- `test/examples/linear_attention/vllm_compare/profile_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py`

生成证据根目录：

```text
test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_vdecay_ladder/
test/examples/linear_attention/rocprof_outputs/qwen_block_dot_bv32_typed_operand_ab/
```

关键复现命令：

```bash
cmake --build /tmp/avelang-build-kfrag-qwen-rocm722 \
  --target _avelang_bindings -j 16

export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test_qwen_gdn_direct_k64_block_dot_bv32_precomputed_vdecay.py -s
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  bench_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py \
  --T 512 1024 2048 --warmup 5 --repeat 20 --sessions 5 --json

AVELANG_BLOCK_DOT_OPERAND_LOWERING=typed_vector \
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM \
        SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex block_dot_bv32_coop -d <out> -o counters -f csv -- \
  python3 profile_qwen_gdn_direct_k64_block_dot_bv32_typed_operand_ab.py \
    --operand typed_vector --T 2048 --warmup 2 --repeat 5
```

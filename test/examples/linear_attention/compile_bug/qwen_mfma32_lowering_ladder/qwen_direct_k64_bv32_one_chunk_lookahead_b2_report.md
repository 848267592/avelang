# Qwen Direct-K64 BV32 B2: 单 LDS Bank 一 Chunk Lookahead 实验

## 结论

**B2 正确，但性能 No-Go。**

本轮按 current Triton 的“提前发起下一 chunk 的 global load、当前 chunk
继续计算、当前 LDS operand 消费完后覆盖同一 bank”这一大方向，实现了一条
experimental-only 的 Avelang Direct-K64 BV32 full-recurrence B2。它满足：

- 不拆 token32；
- 保持 `BT=64`、`BV=32`、`WG=128`、32 CTA、two-wave cooperative ownership；
- 保持 BF16 `K/W/U/H/V-new`、FP32 `g/state/final_state` ABI；
- 保持 P1 的 nonzero-W pred MFMA32 mapping、C0 persistent typed block-dot、
  K32 accumulation order，以及每 CTA MFMA 工作；
- 不分配第二套 W/K LDS bank；
- 不把 `V-new` 写到 global 后再读取；
- 不改 production selector、external HSACO、allocator/RA、旧 broad/compact-K，
  或 block-dot layout 选择。

但 “单 LDS bank” 不等于 “零额外 live storage”。为保护尚未消费的下一
chunk W/K，B2 将完整的 next `W[2,64,64]` 与 `K[2,64,64]` 分散保存在每个
thread 的 `w_next[64]` / `k_next[64]` 本地数组中，直到当前 pred 和两个
update K-half 结束才覆盖 LDS。这个 source-visible register window 使最终
LTO 出现 `209` 个 VGPR spill words、`840 B` private segment，并把 T=2048
body 从 B0 的 `0.676045 ms` 退化到 `1.084111 ms`（`+60.36%`）。

因此，本轮否定的是**用当前 Avelang source 形式将整块 next W/K 保存在
thread-local register window 的 software lookahead**，不是否定 Triton 存在
流水线，也不是否定未来以低层 typed async/load-to-LDS 表示来实现该流水线。
不能在 B2 上继续加入 ping-pong、double buffer 或更远 lookahead；那些操作只会
扩大已经溢出的 live window。

## 1. 背景与冻结范围

B0 是已通过 P0/P1/P2 correctness ladder 的 native full-recurrence baseline：
single kernel 内顺序执行所有 BT64 chunk，FP32 state 作为 feedback carrier，
并在 pred/update 间严格执行：

```text
pred FP32
  -> corrected FP32
  -> V-new BF16
  -> FP32(V-new) * decay
  -> V-decay BF16
  -> Direct-K64 update
  -> FP32 state feedback
```

B2 的唯一新变量是 next-chunk global load lookahead 加 single-bank tail
overwrite。它不是 full graph / Eager public API 排名；所有性能数字均为
preallocated recurrence body 的诊断数据。

在实现前的 Triton 静态审计见：

- [qwen_current_triton_pred_vnew_update_pipeline_audit.md](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_current_triton_pred_vnew_update_pipeline_audit.md)

审计确认 native code 存在 next-chunk global load 的提前发起、W/K LDS reuse、
以及 V-new 直连 update 的数据流；但 TTGIR/ISA 无法为每个高层值恢复完整
element provenance，报告中已将不可恢复项标为 N/A。因此 B2 是受约束的
Avelang feasibility 实验，而不是声称已逐指令复制 Triton 的调度器。

## 2. B2 的精确实现

新 source：

- [repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py)

关键的单 bank / local-window 定义在 [B2 source](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:146)。

```text
LDS: state_bf16, W bank, pred_partial, V-decay, K bank, g
local: w_next[64], k_next[64], u_current[16], u_next[16]
```

### 2.1 Prologue

每个 thread 从 chunk 0 global memory 读取其负责的 W/K/U，写入唯一的 W/K LDS
bank；K 明确以 C0 consumer 所需的 `[K, token]` 存储。相关代码在
[B2 source](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:165)。

初版曾将 K 以 `[token, K]` 写入，导致 T=64 final state 最大误差约 `0.022`，
超过 `0.02` 门槛。已将 prologue 与 tail overwrite 同时修正为
`k_stage[k_half, local_k, token_off]`；这是实验 source layout 错误，非
MFMA 或 recurrence 数学问题。

### 2.2 Steady state

每个 chunk 的实际顺序是：

```text
1. H BF16 snapshot + state_bf16 stage
2. 读取 next W 到 w_next（如果存在 next chunk）
3. 当前 P1 pred MFMA32，W 从当前 W LDS bank 读取
4. corrected -> BF16 V-new -> BF16 V-decay
5. 读取 next U / g / K 到 u_next / g LDS / k_next
6. 当前 C0 preloaded-K K0 + K1 Direct-K64 update
7. 当前 W/K bank 变死后，将 w_next/k_next 覆盖回同一 bank
8. u_current <- u_next；FP32 state 作为下一 chunk feedback
```

对应代码：next W 在 [214](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:214)，
next U/g/K 在 [271](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:271)，
preloaded-K update 在 [293](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:293)，
tail overwrite 在 [312](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py:312)。

注意：为保留现有 P1 correctness mapping，B2 在所有当前 token 的 corrected/V-decay
都产生后才读 next U/g/K；因此它没有假称实现了一个更激进的 pred 内部异步
load/compute overlap。B2 实测的是可由现有 source 语义安全表达的一 chunk lookahead。

### 2.3 防止 preloaded-K 意外回退为 global K producer

为使 B2 的 update 真正消费当前 K LDS bank，而不是 block-dot lowering 在内部又
重新做 source-K global load，增加了 narrow experimental intrinsic：

```text
al.amdgpu.block_dot_bf16_f32_staged_vdecay_preloaded_k(...)
```

它只给既有 C0 block-dot 标记 `preloaded_k`：lowering 跳过
`emitPersistentTypedBlockStageB`，而 MFMA32 geometry、K32 order、state layout 和
消费者 local-load 均不变。相关实现位于：

- [amdgpu_module.cc](/home/jiandongliu/project/avelang/lib/IR/Intrinsics/amdgpu_module.cc)
- [AveLangOps.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/IR/AveLangOps.cc)
- [lower_qwen_block_dot_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc)

它未进入 production dispatch，也没有引入自动 layout 选择。

## 3. Correctness

先运行 correctness，全部通过后才执行 benchmark。运行时 W 保持 nonzero，未使用
W=0 逃过 pred gate。

| T | P2 host microscope | device-contract reference | finite |
|---:|:---:|:---:|:---:|
| 64 | 全部核心 buffer byte-equal | 通过 | 是 |
| 128 | `h/pred/V-new/V-decay/state/final` 全部 byte-equal | 通过 | 是 |
| 512 | 同上 | 通过 | 是 |
| 2048 | 同上 | 通过 | 是 |

这里的 byte-equal 是 B2 对 P2 host feedback microscope；它证明单 kernel 的
feedback 顺序、BF16 V-new boundary 和 C0 update 没有因 lookahead 改变。对 device
contract reference 不要求 bit-exact，因为 reference 的 FP32/MFMA execution order
不同；T=2048 的最大误差仍在既有门槛内：H `2.4414e-4`、pred FP32
`2.1949e-5`、V-new/V-decay `2.4414e-4`、state/final `4.033e-5`。

完整逐 chunk 数字保存在：

- [b2_lookahead_correctness.json](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/b2_lookahead_correctness.json)

结论：B2 通过 correctness；性能退化不能归因于错误计算、global V-new handoff
或改变 update 数学。

## 4. Same-work 检查

| 项目 | B0 | B2 | 结论 |
|:--|--:|--:|:--|
| BT/BV/WG/CTA | 64 / 32 / 128 / 32 | 相同 | 冻结 |
| Pred/update MFMA32 静态数 | 48 | 48 | 相同 |
| T=2048 dynamic MFMA | 65,536 | 65,536 | 相同 |
| 静态 barrier | 13 | 13 | 相同 |
| V-new global store | 有 | 有 | ABI 保持 |
| V-new global reload | 无 | 无 | B2 未引入 |
| update K source | existing C0 | current preloaded LDS K bank | B2 不回退 source-K reload |

静态 ISA 的 MFMA、barrier 计数来自 B0/B2 body disassembly；动态 MFMA 来自
同一 `rocprofv3` T=2048 capture。所有 body benchmark buffer 均在计时外预分配，
且 `emit_audit=False`，只保留 ABI 所需的 H、V-new、final state store。

## 5. 机器资源与 ISA

### 5.1 HSACO metadata 和 exact LTO MIR

| 指标 | B0 | B2 | 变化 |
|:--|--:|--:|--:|
| HSACO AGPR count | 204 | 256 | +52 |
| HSACO VGPR count | 460 | 512 | +52 |
| fixed LDS | 36,864 B | 53,504 B | +16,640 B |
| private segment | 0 B | 840 B | +840 B |
| VGPR spill count | 0 | 209 | +209 words |
| SGPR spill count | 0 | 0 | 0 |

B2 exact LTO replay的 post-greedy MIR 显示：

```text
SI_SPILL_AV32_SAVE = 35
SI_SPILL_AV64_SAVE = 87
35 * 1 + 87 * 2 = 209 VGPR spill words
```

这与 code object 的 `.vgpr_spill_count: 209` 和
`.private_segment_fixed_size: 840` 完全吻合。故 B2 scratch 不是 rocprof 估计值，
而是最终 LLVM/AMDGPU RA 的真实 spill 结果。

证据位置：

- [B2 metadata](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/body_hsaco/b2_body_metadata.txt)
- [B0 metadata](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/b0_body_hsaco/b0_body_metadata.txt)
- [B2 post-greedy MIR](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/exact_lto_postra/kernel_section_07.mir)

该 MIR 没有提供“某一个物理 spill 只属于 `w_next`”的单变量 debug attribution；
但 B2 相对 B0 新增长 live 变量正是 `[64]` W/K lookahead 与 `[16]` U lookahead，
而 spill 精确出现在其编译结果中。这足以证明 source-level materialized lookahead
超出资源预算，不能将它描述成随机计时噪声。

### 5.2 静态 ISA

| 指令类别 | B0 | B2 | 变化 |
|:--|--:|--:|--:|
| `v_mfma_f32_32x32x8_bf16` | 48 | 48 | 0 |
| `global_load_ushort` | 80 | 288 | +208 |
| `global_load_dwordx4` | 16 | 8 | -8 |
| `ds_write_b16` | 208 | 336 | +128 |
| `ds_write_b128` | 8 | 8 | 0 |
| `ds_read_b32` | 28 | 49 | +21 |
| `ds_read_b128` | 40 | 40 | 0 |
| `s_barrier` | 13 | 13 | 0 |

B2 没有增加 MFMA 或 barrier，却为 next W/K staging 引入显著更多 scalar global
loads 与 LDS writes。这与 source 中每 thread 完整保存 W/K lookahead 的实现一致。

### 5.3 T=2048 rocprof PMC

每种 kernel 捕获 8 个匹配 dispatch，表为每 dispatch 平均值；trace latency 仅作
instrumented corroboration，HIP-event body benchmark 才是权威 latency。

| 指标 | B0 | B2 | 变化 |
|:--|--:|--:|--:|
| trace median | 689.745 us | 1,089.459 us | +57.95% |
| VGPR | 128 | 128 | 0 |
| Accum_VGPR | 336 | 384 | +48 |
| SGPR | 112 | 112 | 0 |
| LDS block | 36,864 B | 53,760 B | +16,896 B |
| scratch | 0 B | 840 B | +840 B |
| occupancy | 0.645573 | 0.625234 | -3.15% |
| MFMA | 65,536 | 65,536 | 0 |
| VMEM | 315,904 | 641,824 | +103.17% |
| VALU | 2,865,536 | 3,006,880 | +4.93% |
| SALU | 161,216 | 177,152 | +9.89% |
| LDS inst | 495,616 | 535,552 | +8.06% |

`Accum_VGPR` 是 profiler resource metric，不等于 metadata 的 AGPR count；报告中
两者特意分开记录。PMC 的 VMEM 接近翻倍，与 static global scalar load 增长、寄存器
spill 和 lookahead source work 三者一致。

原始 CSV：

- [B0 counters](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/rocprof_b0_t2048/b0_counters_counter_collection.csv)
- [B2 counters](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/rocprof_b2_t2048/b2_counters_counter_collection.csv)

## 6. Fresh-process body benchmark

方法：同一 BF16 current-vLLM ABI 输入、预分配输出、current HIP stream、非 graph
capture、warmup=5、repeat=20、5 个 fresh-process session，Williams-like rotating
order。数值为 session medians 的中位数。

| T | chunks | B0 ms | B2 ms | B2 vs B0 | direct current Triton ms | external bridge ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | 0.157354 | 0.245926 | 1.563x | 0.074191 | 0.040220 |
| 1024 | 16 | 0.357832 | 0.557949 | 1.559x | 0.102873 | 0.067561 |
| 2048 | 32 | 0.676045 | 1.084111 | 1.604x | 0.153729 | 0.115972 |
| 8192 | 128 | 2.778029 | 4.248433 | 1.529x | 0.463509 | 0.418762 |

线性拟合：

| body | intercept ms | slope us/chunk |
|:--|--:|--:|
| B0 | -0.006397 | 21.763744 |
| B2 | 0.042355 | 33.150944 |
| direct current Triton | 0.059026 | 3.232957 |
| external current-vLLM bridge | 0.027066 | 3.147636 |

B2 相比 B0 的 slope 增加 `52.32%`。在 T=2048，B2 为 direct Triton 的
`7.05x`，而 B0 为 `4.40x`。这是 diagnostic body comparison，不能与 full
Eager public API 排名混用，但足以给本轮 hard stop 作出判断。

完整 raw sessions 与 checks：

- [b2_body_benchmark.json](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/b2_body_benchmark.json)

## 7. 为什么 Triton 有 lookahead 而 B2 反而更慢

Triton 的静态图表明它有更宽的 load/MFMA/LDS 调度窗口；但 B2 的实现方式是将整块
next operands 先物化成一般的 source local arrays，再在 LDS bank 可以覆盖时写回。
这造成以下差异：

1. `w_next[64]` 与 `k_next[64]` 对每 thread 都跨 pred、corrected、K0 update、
   K1 update 存活；它们不是“尚未消费的一条 load”，而是大量可分配 value。
2. 为保持输入一致，B2 仍显式执行 next W/K global loads 与 tail LDS stores；没有
   使用硬件异步 LDS transaction 或 compiler-owned load-to-LDS token。
3. 当前 shared bank 虽只有一套，但 B2 为了让 pred 从 W LDS 消费而明确 materialize
   W/K bank，LDS footprint提高；本地完整 block 与 persistent state/accumulators 同时
   存活后，RA 跨越了 spill cliff。
4. 结果是 VMEM 增加超过一倍，且更多工作没有被 overlap 隐藏，反而以 spill 和低
   occupancy 形式暴露出来。

所以本轮不能得出“software pipelining 没有价值”的泛化结论。更精确的结论是：

> 在当前 AveLang source/local lowering 下，整块 next-chunk operands 的
> register-materialized lookahead 不是可行表示。

## 8. 决策与停止条件

**B2 不晋级为 native full-recurrence baseline；B0 保持该角色。**

B2 达到 correctness、同 MFMA、同 barrier、single LDS bank、无 V-new reload 等
语义门槛，但违反全部性能安全门槛：scratch 非零、MIR spill 非零、资源 cliff、所有
长度均退化且 slope 变差。

因此下一步禁止：

- 在 B2 上增加第二套 LDS bank、ping-pong、double buffering 或更远 prefetch；
- 以 B2 继续作 phase-lifetime 微调；
- 把 B2 接入 production 或 external HSACO path。

若未来重新研究 Triton-matched pipeline，最小前置问题不是再写一个 source schedule，
而是设计并在 isolated repro 证明一个可延续到 late AMDGPU lowering 的 typed
`global-load -> LDS-bank` streaming/transaction representation：它必须不先把完整
W/K block 持有为普通 thread-local values，并且要同样接受 LTO MIR spill 与 body
benchmark gate。那是独立 compiler/pipeline project，不是 B2 的下一小步。

## 9. 复现

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:./test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py \
  --T 64 128 512 2048 --seed 20260801 --json

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead.py \
  --T 512 1024 2048 8192 --warmup 5 --repeat 20 --sessions 5 --json

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_direct_k64_bv32_full_sequence_b2_lookahead_kernel \
  -d test/examples/linear_attention/rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/rocprof_b2_t2048 \
  -o b2_counters -f csv -- \
  python3 test/examples/linear_attention/vllm_compare/run_qwen_gdn_direct_k64_bv32_full_sequence_body.py \
    --variant b2 --T 2048 --warmup 2 --repeat 5
```

Exact post-RA replay command and artifacts are in
`rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b2_lookahead/exact_lto_postra/`.

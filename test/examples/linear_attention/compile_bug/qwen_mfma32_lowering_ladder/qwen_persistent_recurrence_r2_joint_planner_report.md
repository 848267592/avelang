# Qwen Persistent Recurrence R2 Joint Planner

## 当前状态

`gfx942_bt64_bv32_joint_v2` 已完成为一个 experimental-only 的完整
full-recurrence 候选，并已经通过 nonzero-W correctness、exact-LTO MIR 和
T=2048 PMC 审计。正式 fresh-process body benchmark 已在单一 Docker 进程中
启动；本报告写入时，执行环境拒绝后续 Docker 读取操作，因此**不能将尚未读取的
benchmark JSON 写成性能结论**。R2 尚未晋级为 baseline。

已确认的事实是：R2 不再只是 R1 的命名或 metadata 分叉。它的计划、source
stage 时序、late MLIR、LTO MIR、ISA 与 HSACO 均不同；但它尚未完成 Triton 式的
K producer-to-typed-dot-operand 端到端布局。因此，本轮是一个真实但未完成的
R2 performance candidate，而不是“已经完全复刻 Triton pipeline”的声称。

## 范围

R2 固定并保留：gfx942、BT64、BV32、WG128、32 CTA、two-wave cooperative
ownership、BF16 `K/W/U/H/V-new`、FP32 `g/state/final_state`、P0 nonzero-W
pred mapping、BF16 V-new round-trip、Direct-K64 MFMA32、C0 persistent typed
block-dot、K32 accumulation order 与 FP32 loop-carried feedback。

没有修改 production selector、external HSACO、allocator/RA、旧 broad-K/compact-K、
MFMA geometry、D0-P layout、prefetch ping-pong 或 double buffering。

相关 source：

- `vllm_compare/repro_qwen_gdn_persistent_recurrence_r2_joint_v2.py`
- `vllm_compare/repro_qwen_gdn_persistent_recurrence_r2.py`
- `vllm_compare/run_qwen_gdn_persistent_recurrence_r2_body.py`
- `vllm_compare/bench_qwen_gdn_persistent_recurrence_r2.py`

编译器入口：

- `Transforms/qwen_recurrence_schedule_plan.h`
- `Transforms/qwen_persistent_recurrence_pass.cc`
- `Transforms/lower_qwen_k64_pipeline_stage_pass.cc`

## R2 的统一计划

planner 在 block-dot lowering 前看见完整 recurrence region，并产生唯一计划：

```text
schedule             = gfx942_bt64_bv32_joint_v2
shared_encoding      = joint_v2_rotating_typed_bank
dot_operand_encoding = rotating_shared_typed_dot
next_chunk_stage     = one_chunk_ahead_interleaved_tail_commit
```

对每个 steady-state chunk，实际 source 及 planned MLIR 的次序为：

```text
next W0/W1 global packet stage
  -> current W bank / pred MFMA32
  -> corrected / BF16 V-new / BF16 V-decay
  -> next K0/K1 global packet stage
  -> current K bank / K0,K1 Direct-K64 update MFMA32
  -> same-bank tail commit of next W/K packets
```

这与 R1 的区别是 K stage 不再和 W stage 一起在 pred 前发出；它被放在 pred 与
当前 update 之间。tail commit 仍发生在当前 consumer 结束之后，未引入第二整套
W/K LDS bank，也没有 `w_next`、`k_next` 或 `u_next` thread-local tile。

## Producer、Shared 与 Consumer 的真实实现边界

R2 的 W 链已发生实质改变：late stage lowering 将 BF16x8 global vector packet 直接
写为 typed LDS `vector.store`，共享索引为 pred 所需的 `[half, token, feature]`。
该 op 带有：

```text
avelang.qwen.joint_v2.typed_lds_packet = bf16x8
avelang.qwen.joint_v2.placement        = distributed_register_packet
```

K 的 load ownership 和时序也已由同一 planner 管理：它是 BF16x8 global packet，且
在 pred 后、update 前发起并在 tail same-bank commit。但 K consumer 仍是 C0 的
`[half, feature, token]` preloaded-K Direct-K64 fragment consumer；为保持既有
correctness，它的 final LDS commit 仍需要 consumer-major scalar transpose。

因此，R2 尚未把 K 也改造成 Triton 的 rotating typed shared-to-dot fragment layout。
这不是遗漏在报告里的实现细节，而是本轮与 native Triton 仍存在的主要结构差距。
名称 `rotating_shared_typed_dot` 表达了 plan 的目标 encoding 和 verifier contract；
它不应被解读为 K consumer 已获得一个 first-class typed fragment。

## 正确性

机器可读结果位于：

`rocprof_outputs/qwen_persistent_recurrence_r2_joint_v2/r2_correctness.json`。

已运行 T=64、128、512、2048 的 nonzero-W matrix。每个长度都满足：

- R2 对 R1 的 H、raw FP32 pred、BF16 pred、V-new、V-decay、per-chunk state 和
  final state byte-exact；
- R2 对 P2 host feedback microscope 的上述张量 byte-exact；
- 所有结果 finite；
- device-contract comparison 保持在已冻结 BF16/FP32 gate 内。

例如 T=2048，R2 相对独立 device-contract reference 的最大绝对误差为：H
`0.000244140625`、raw pred FP32 `2.2758264e-05`、V-new/V-decay
`0.000244140625`、final state `4.5960769e-05`。这些不是 R2-vs-R1 差异；R2-vs-R1
仍严格为零。

## LTO、MIR、ISA 与资源

R2-only T=2048 capture 的 HSACO SHA256 为：

```text
e131ab4590852b22afa76eea417dc5de106dd47c13aac3f805e650dad547e630
```

R1 对应 HSACO 为：

```text
6a8a35aef57255bfdbde58131087df96cafcabdc98ae791862e255d16ac744d8
```

两者不同。R2 exact-LTO replay 导出了 greedy 前、greedy 后、virtregrewriter 与
prologepilog 的 20 个 kernel MIR section。所有 section 的
`SI_SPILL_AV32_SAVE` 与 `SI_SPILL_AV64_SAVE` 均为零；最终 physical-register MIR
无虚拟寄存器。这排除了 R2 通过 spill 换取时序变化的情况。

R2 T=2048 PMC 的同一 dispatch 值如下。PMC trace 只用于机器工作量和资源，不能用
来代替 latency。

| metric | R1 | R2 | R2 - R1 |
|:--|--:|--:|--:|
| MFMA | 65,536 | 65,536 | 0 |
| VMEM | 202,240 | 202,240 | 0 |
| VALU | 2,236,352 | 2,227,392 | -8,960 (-0.40%) |
| SALU | 163,008 | 163,008 | 0 |
| LDS instructions | 385,536 | 385,536 | 0 |
| LDS block | 53,248 B | 53,248 B | 0 |
| VGPR | 128 | 128 | 0 |
| Accum_VGPR | 192 | 192 | 0 |
| SGPR | 112 | 112 | 0 |
| scratch | 0 B | 0 B | 0 |
| occupancy | about 0.642 | about 0.642 | effectively 0 |
| MIR AV32/AV64 spill | 0 / 0 | 0 / 0 | 0 |

R2 static ISA has 48 `v_mfma`, 13 `s_barrier`, 70 `ds_read`, 232 `ds_write`,
6 `buffer_load`, 6 `buffer_store`, 73 `global_load`, 128 `global_store` and
109 `s_waitcnt` occurrences. The static LDS read/write total is 302, matching
the R1 order of magnitude. This is direct evidence that the schedule moved
real code but has not yet removed the dominant dynamic K operand materialization.

For context, current Triton at the same T=2048 recurrence control has VMEM
58,368, LDS 305,472 and LDS capacity 40,960 B. R2 therefore does not close the
main R1-to-Triton data-path gap merely by moving K stage issue after pred.

## 正式性能状态

规定的 R2 body benchmark uses T=512/1024/2048/8192, warmup=10, repeat=50,
five fresh-process sessions, rotating order, preallocated tensors, current HIP
stream, HIP events and no Graph. Arms are B0, R1, R2, direct current Triton and
the external bridge.

The completed benchmark is retained in
`codex_qwen_persistent_recurrence_r2_joint_v2/body_benchmark.json`.

| T | B0 ms | R1 ms | R2 ms | direct Triton ms | external bridge ms |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.156953 | 0.139408 | 0.139868 | 0.072307 | 0.040059 |
| 1024 | 0.357251 | 0.283862 | 0.280477 | 0.101611 | 0.067541 |
| 2048 | 0.677647 | 0.530668 | 0.523418 | 0.152687 | 0.117275 |
| 8192 | 2.779050 | 2.127642 | 2.091168 | 0.462989 | 0.419383 |

The fitted slopes are B0 `21.773886`, R1 `16.539720`, R2 `16.236898`, direct
Triton `3.241736`, and the external bridge `3.151886` us/chunk. R2 is slightly
slower than R1 at T=512, but improves R1 by 7.250 us at T=2048 and 36.474 us at
T=8192. Its long-sequence slope is 1.83% lower than R1. This is a body-only
measurement, not an Eager public-API ranking or a production promotion.

## 决策

R2 passes the semantic and no-spill gates and has a measured long-sequence body
improvement over R1. It is retained as the historical native body winner among
these placement-only schedules. It is not a production baseline: its dynamic
VMEM/LDS work remains materially above direct Triton, and the 16.236898
us/chunk slope remains about 5.01x the direct-Triton slope.

The R2 result closes the earlier bookkeeping gap but does not change the root
cause: moving W/K issue points alone does not remove the dominant K operand
materialization cost. Later experiments must preserve the same full-recurrence
correctness contract and demonstrate a machine-level reduction, rather than
adding schedule names or changing register allocation.

# Qwen Persistent Recurrence R3 Full Typed Operand Pipeline

## 结论

R3 已完成为一个真实、正确且可审计的 full-recurrence 机器图：
`gfx942_bt64_bv32_joint_v3`。它不是 metadata 或 isolated probe。它改变了
完整 persistent recurrence 中 W/K 的 producer ownership、K 的跨 lane 重排、
typed shared commit、pred/update operand feeding，以及 one-chunk-ahead 的
same-bank tail commit；改变从 MLIR 保留到 LLVM、MIR、ISA 和不同的 HSACO。

但它**不能晋级为新的 native full-recurrence baseline**。R3 在 T=8192 的
body latency 比 R2 小 `17.226 us`，斜率改善 `2.05%`，但 T=512/1024/2048
分别比 R2 慢 `20.329/17.426/27.301 us`。机器审计给出了明确原因：为了把
token-owned BF16x8 K packet 转成 K-row-owned MFMA fragment，当前正确的
通用实现展开成 `4 word planes x 8 tokens` 的跨 lane 传输，最终
静态 ISA 有 `512` 条 `ds_bpermute_b32`。这使 LDS、VALU、VMEM 和 AccVGPR
资源显著上升，抵消了 long-sequence lookahead 的小收益。

R3 因而是一个重要的反证：保留 typed operand 语义本身不够；若 typed
producer-to-fragment bridge 仍需要逐 word 的 cross-lane transpose，就不能接近
current Triton 的实际数据路径。此轮不改 production selector、external HSACO、
allocator/RA、MFMA geometry 或旧 broad/compact-K 实验路径。

## 范围和冻结契约

- 目标：gfx942，BT64、BV32、WG128、32 CTA、two-wave cooperative ownership。
- 数据类型：K/W/U/H/V-new 为 BF16；g、loop-carried state、final state 为 FP32。
- 数学：保留 P0 nonzero-W pred mapping、BF16 V-new round trip、FP32 feedback、
  Direct-K64 / `v_mfma_f32_32x32x8_bf16` 和既有 K32 accumulation order。
- 禁止项均保持禁止：不改 production dispatch、RA、LDS-layout sweep、ping-pong、
  full-v29 路径或 external HSACO body。
- 对照：B0、R1 joint-v1、R2 joint-v2、R3 joint-v3、direct current Triton、
  current-vLLM external bridge。

R2 的正式历史 benchmark 也已补读并修正到
`qwen_persistent_recurrence_r2_joint_planner_report.md`。R2 的原始 JSON 在
`codex_qwen_persistent_recurrence_r2_joint_v2/body_benchmark.json`。

## R3 实现

### 一个完整 plan

`QwenRecurrenceSchedulePlan` 新增：

```text
schedule:          gfx942_bt64_bv32_joint_v3
shared encoding:   joint_v3_full_typed_rotating_bank
dot encoding:      full_typed_rotating_dot
next stage:        one_chunk_ahead_interleaved_tail_commit
distributed tile:  bv32_two_wave
```

planner 在 block-dot lowering 之前处理完整 recurrence region，而非先分别生成
pred/update plan。它同时标注 W0/W1 与 K0/K1：

```text
typed BF16x8 global packet
  -> distributed register packet
  -> rotating shared bank
  -> typed MFMA32 operand
  -> current consumer release
  -> same-bank next tail commit
```

在 `post_recurrence_joint_planner.mlir` 中，R3 recurrence op 明确带有
`schedule = "gfx942_bt64_bv32_joint_v3"`、`stage_operands = "w0,w1,k0,k1"`、
`vnew_boundary = "bf16_round_trip"`、`feedback = "fp32_loop_carried"` 和
`next_chunk_stage = "one_chunk_ahead_interleaved_tail_commit"`。

### W 和 K 的联合数据路径

W 使用已有的 typed BF16x8 packet 到 shared 的 vector commit。K 不再走 R2 的
consumer-major scalar `memref.store` transpose commit。R3 中每个 lane 先读取一条
连续的 BF16x8 global K vector；lane/group arithmetic 选择 token block 与 feature
block，再用固定 subgroup 8x8 mapping 建构 K-row 的 BF16x8 packet，最后以
`vector.store vector<8xbf16>` 写入 K-major rotating shared bank。映射公式在
`emitJointV3RotatingKPackets` 中表达，没有把 4096 个元素地址写死在 pass 内。

这个改变在 `post_joint_v1_stage_lowering.mlir` 中可见：K commit 是标有
`joint_v3.typed_lds_packet = "bf16x8_krow"` 和
`joint_v3.shared_layout = "rotating_k_major_dot_operand"` 的 vector store。
所以旧的 K scalar shared-store **commit path 已被替换**。

不过这不等于机器代码完全没有 scalar work。正确的 source-lane/destination-row
关系要求 destination 在 shuffle 后选择 word pair；R3 以四个固定 i32 word plane
对每个 token 做 `gpu.shuffle IDX`。LLVM 保留为 `llvm.amdgcn.ds.bpermute`，而不是
在优化中退化回旧的 scalar LDS transpose。这个设计保证正确，却是本轮成本来源。

## 多层机器证据

完整原始产物位于
`codex_qwen_persistent_recurrence_r3_full_typed_operand_pipeline/`。

| 层次 | R3 证据 |
|---|---|
| planner MLIR | `ir/post_recurrence_joint_planner.mlir` 有 full typed rotating plan、W/K stage 与 current/next tags。 |
| stage-lowered MLIR | `ir/post_joint_v1_stage_lowering.mlir` 有 BF16x8 K-row vector store，而不是旧 K scalar commit。 |
| LLVM | `ir/preopt_llvm.ll` 与 `ir/postopt_llvm.ll` 都含 `llvm.amdgcn.ds.bpermute` 和 MFMA32 intrinsic。 |
| pre/post-greedy MIR | `exact_lto/kernel_section_07.mir` 有 `DS_BPERMUTE_B32`；`kernel_section_08.mir` 已为物理寄存器 MIR。 |
| ISA | `r3.isa` 有 `ds_bpermute_b32` 和 `v_mfma_f32_32x32x8_bf16`。 |
| code object | `hsaco/_qwen_gdn_persistent_recurrence_r3_joint_v3_kernel.hsaco`，SHA256 `9087df78d48da7c94f2687feeac33b42a994052607a4808f8e173932ef89587a`。 |

这证明 R3 产生的不是 B0/R1/R2 的同一机器图。full-LTO replay 覆盖 pre-greedy、
post-greedy、virtregrewriter 与 prologepilog；所有 section 的
`SI_SPILL_AV32_SAVE` 和 `SI_SPILL_AV64_SAVE` 均为零，最终 MIR 无虚拟寄存器。

## 正确性

命令使用 runtime nonzero W，而不是 W=0 gate。T=64/128/512/2048 全部 finite 且
通过。与 P2 host microscope 比较，H、raw pred、BF16 pred/V-new/V-decay、每 chunk
state 和 final state 都 byte-exact。与独立 device-contract reference 相比，R3 的
差异来自冻结的 FP32 MFMA reduction order，仍在既有阈值中。

| T | R3 vs P2 | device-contract H max abs | device-contract final-state max abs |
|---:|:---:|---:|---:|
| 64 | byte-exact | 0 | 3.7253e-09 |
| 128 | byte-exact | 0 | 3.7253e-09 |
| 512 | byte-exact | 3.0518e-05 | 4.3772e-07 |
| 2048 | byte-exact | 2.4414e-04 | 4.8446e-05 |

R3 继续遵守 BF16 boundary：`v_new = bf16(corrected)`，update 使用
`fp32(v_new_bf16)`，并在 ABI 输出一次 V-new；不存在将该 ABI output global store
后再作为 update input reload 的路径。

## T=2048 机器工作

下表为同一 full recurrence 的 rocprof PMC。R1/R2 是先前相同 contract 的结果；
R3 是本轮直接采集。动态 MFMA 完全相同，因此差异不能归因为减少数学工作。

| metric | R1 | R2 | R3 | R3 vs R2 |
|---|---:|---:|---:|---:|
| MFMA | 65,536 | 65,536 | 65,536 | 0 |
| VMEM | 202,240 | 202,240 | 205,696 | +1.71% |
| LDS inst | 385,536 | 385,536 | 807,936 | +109.56% |
| VALU | 2,236,352 | 2,227,392 | 4,338,880 | +94.80% |
| SALU | 163,008 | 163,008 | 163,072 | +0.04% |
| LDS block | 53,248 B | 53,248 B | 53,248 B | 0 |
| rocprof VGPR / AccVGPR | 128 / 192 | 128 / 192 | 128 / 384 | pressure up |
| rocprof Scratch_Size | 0 B | 0 B | 28 B | collector field up |

R3 HSACO itself reports `.private_segment_fixed_size: 0`、`.vgpr_spill_count: 0` 和
`.sgpr_spill_count: 0`，full-LTO MIR 也没有 spill save/reload。因此 rocprof 的
`Scratch_Size=28` 和 `Accum_VGPR_Count=384` 应记录为 collector resource fields，
不能误写成已由 MIR 证实的 private-memory spill。它们与真正出现的 high-pressure
machine graph 一致，但没有推翻 code-object 的零 spill 事实。

ISA 静态计数进一步定位代价：`40` MFMA32、`512 ds_bpermute_b32`、`36 ds_write_b128`、
`72 ds_write_b16`、`32 ds_read_b128`、`16 ds_read_b32` 和 `11 s_barrier`。R3 确实
保留了 typed LDS packet 和 MFMA feeding，但跨-lane word-plane unpack 引入的 512 条
bpermute 远大于任何 producer-side packed-store 收益。

## 正式 Fresh-Process Body Benchmark

口径：T=512/1024/2048/8192、预分配、current HIP stream、warmup=10、repeat=50、
每点 5 个 fresh-process session、rotating palindromic six-arm order、HIP event、
不使用 CUDA/HIP Graph。这是 recurrence body diagnostic，不是 Eager public-API 排名。

| T | B0 ms | R1 ms | R2 ms | R3 ms | direct Triton ms | external bridge ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.156633 | 0.139027 | 0.138206 | 0.158535 | 0.073069 | 0.040561 |
| 1024 | 0.356710 | 0.283361 | 0.278854 | 0.296280 | 0.102753 | 0.067941 |
| 2048 | 0.676044 | 0.529728 | 0.521134 | 0.548435 | 0.154329 | 0.117855 |
| 8192 | 2.776326 | 2.125679 | 2.090427 | 2.073201 | 0.467214 | 0.420765 |

| implementation | fitted slope, us/chunk |
|---|---:|
| B0 | 21.755263 |
| R1 | 16.526947 |
| R2 | 16.245862 |
| R3 | 15.912311 |
| direct Triton | 3.269839 |
| external bridge | 3.159895 |

R3 相对 R2 在 T=512/1024/2048 分别退化 14.71% / 6.25% / 5.24%，只在 T=8192
改善 0.82%。它的斜率比 R2 小 2.05%，但仍是 direct Triton 的 4.87x。这说明
one-chunk lookahead 带来一点长序列 amortization；它不足以覆盖 bpermute 重排的
固定和每-chunk成本。

## 决策

1. R2 的正式 benchmark 已补齐：R2 是 placement-only 历史 native body winner，
   但不具备 production promotion 条件。
2. R3 通过 full nonzero-W correctness、完整 LTO/MIR audit 和 no-MIR-spill gate，
   也确实形成了与 B0 不同的 full-recurrence machine graph。
3. R3 未通过性能/资源 gate，不能成为新的 native full-recurrence baseline，也不应
   进入 Eager public API 排名或 production selector。
4. 这不是 RA 失败，也不是 MFMA 数量问题。当前 V3 的失败点是正确的 K-row
   typed fragment 生成仍要使用 512 个 word-plane `ds_bpermute`。继续围绕同一个
   R3 producer/shared/consumer layout 做微调没有合理预期收益。

应保留 R3 作为 compiler evidence：AveLang 可以在一个完整 recurrence plan 中
express、lower 并验证 typed W/K pipeline；但要接近 Triton，下一步必须获得一种
不展开为 word-plane cross-lane shuffle 的 first-class dot-fragment mapping，或停止
该 native recurrence pipeline 路线。两者都应作为新的、单独预注册实验；本轮不继续
增加 schedule branch，也不改 RA。

## 复现

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:python:test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_r3_joint_v3.py \
  --T 64 128 512 2048 --out-dir <artifact>/correctness --dump-hsaco-dir <artifact>/hsaco

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_r3.py \
  --T 512 1024 2048 8192 --warmup 10 --repeat 50 --sessions 5 \
  --out-json <artifact>/benchmark/body_benchmark.json
```

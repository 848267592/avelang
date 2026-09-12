# Qwen Direct-K64 BV32 Full-Sequence B0

## 状态

**正确性通过；资源审计、正式多 session body benchmark 和 T=2048 PMC 已完成。**

B0 是一条 experimental-only 的 Avelang-native full recurrence。它把已通过
P1/P2 的单 BT64 chunk body 移入同一个 CTA 的 device-side chunk loop，使 FP32
state 在 CTA 中跨 chunk 反馈。它没有改 production selector、external HSACO、
allocator/RA、MFMA geometry、BV32 ownership、block-dot lowering、LDS layout 或
pred mapping。

B0 现在有完整的诊断性 body benchmark 结论：它是第一条正确的 Avelang-native
Direct-K64 BV32 full recurrence，但不是可晋级的 native 性能基线。它在完整序列内
出现高寄存器压力，且每 chunk 斜率显著高于 current Triton / current-vLLM bridge。

## 冻结的 B0 合约

- ABI：K/W/U/H/V-new 为 BF16；g、initial state、final state 为 FP32。
- `BT=64`，`BV=32`，`WG=128`，`32` CTA，two-wave cooperative ownership。
- pred 与 update 都使用 `v_mfma_f32_32x32x8_bf16`。
- update 为现有 C0 `persistent_typed_block` Direct-K64 lowering；K32 累加顺序不变。
- 每个 chunk 的严格边界为：

  ```text
  corrected = FP32(U_bf16) - pred_f32
  v_new_bf16 = BF16(corrected)
  v_decay_bf16 = BF16(FP32(v_new_bf16) * exp(g_last - g_t))
  ```

- FP32 `h_lo/h_hi` 是 feedback carrier；`H` 只是 pre-update BF16 snapshot。

新源码：

- `vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b0.py`
- `vllm_compare/bench_qwen_gdn_direct_k64_bv32_full_sequence_b0.py`

## 实现

每个 `(value_head, V32)` CTA 初始化两 wave 各自的 K64 半 state 后，在 GPU 内执行：

```text
for chunk in device:
  1. 从 persistent FP32 h_lo/h_hi 写 H BF16 snapshot 和 pred operand state_bf16
  2. 运行 P0 已验证的 MFMA32 pred mapping
  3. 写 BF16 V-new，并从该 BF16 值形成 BF16 V-decay
  4. C0 staged persistent typed Direct-K64 update 消费 CTA-local V-decay
  5. 将更新后的 FP32 h_lo/h_hi 作为下一 chunk feedback
  6. 写当前 chunk 的 H、V-new；最终写 FP32 final state
```

`emit_audit` 是编译期开关：correctness arm 写 pred/V-decay/state-after checkpoints；
body arm 完全删除这些非 ABI 调试写回，但保留相同的 pred、V-new BF16 boundary、update
和 FP32 feedback。T=64 已验证 body arm 与 audit arm 的 H、V-new、final state 逐字节一致。

## Correctness Gate

每个长度均使用同一套 nonzero-W BF16 K/W/U、FP32 g 与 FP32 initial state。B0 同时
比较 device-contract reference 与 P2 host feedback microscope；后者每 chunk 调用
已通过的 P1 body，仅用于 correctness，不用于 timing。

| T | chunks | B0 vs P2 | finite | 结论 |
|---:|---:|:---:|:---:|:---|
| 64 | 1 | H/pred/V-new/V-decay/state/final 全部 byte-exact | 是 | pass |
| 128 | 2 | 全部 byte-exact | 是 | pass |
| 512 | 8 | 全部 byte-exact | 是 | pass |
| 2048 | 32 | 全部 byte-exact | 是 | pass |

与独立 device-contract reference 的最大误差：

| T | H BF16 | pred FP32 | V-new BF16 | final state FP32 |
|---:|---:|---:|---:|---:|
| 64 | `0` | `9.31e-10` | `0` | `3.73e-09` |
| 128 | `0` | `9.31e-10` | `0` | `3.73e-09` |
| 512 | `3.05e-05` | `1.19e-06` | `7.63e-06` | `4.38e-07` |
| 2048 | `2.44e-04` | `2.24e-05` | `2.44e-04` | `4.84e-05` |

所有值低于既有 BF16 `1/128`、pred FP32 `5e-5` 与 state `0.02` 门槛。T=2048 每个
chunk 的 P2 comparison 也均逐字节一致，因此误差不是 B0 device-side feedback 引入的。

## BF16 Boundary Evidence

源码中 update 前必须经过：

```python
corrected_bf16 = al.convert(corrected, al.bf16)
corrected_from_bf16 = al.convert(corrected_bf16, al.f32)
vdecay_stage[...] = al.convert(corrected_from_bf16 * decay, al.bf16)
```

T=64 的 audit 与 body arm 对 H/V-new/final state 都是 byte-exact。这同时证明：

1. body mode 没有通过省略 BF16 round-trip 获得性能；
2. H BF16 snapshot 没有被误作 feedback；
3. FP32 persistent state 才是下一 chunk 的 pred 输入来源。

## Exact LTO / MIR Resource Audit

分别捕获了 audit body 和 compile-time-elided audit-store body 的真实 LTO 输入，
然后使用 `replay_qwen_v29_lto_mir.py` 重放 greedy、virtregrewriter 和
prolog/epilog。产物位于：

```text
rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b0_t2048/
  link_replay/                 # audit-store arm
  exact_lto_postra/
  body_link_replay/            # benchmark body arm
  body_exact_lto_postra/
  body_isa/
```

| T=2048 code object | audit checkpoints | benchmark body |
|:--|--:|--:|
| LDS group segment | 36,864 B | 36,864 B |
| private segment / scratch | 0 B | 0 B |
| SGPR | 40 | 38 |
| VGPR | 316 | **460** |
| VGPR spill count | 0 | 0 |
| SGPR spill count | 0 | 0 |
| `SI_SPILL_*` in post-greedy MIR | 0 | 0 |

因此 B0 的 first full-sequence native composition **没有 spilling**，但 body mode 已经
出现严重 register resource cliff：`VGPR=460`。这不是 rocprof 的
`Accum_VGPR_Count` 代理；它来自 final code-object metadata。该现象正是 B0 下一轮
只允许做 phase-lifetime scheduling 的触发条件，而不是重启 RA、LDS swizzle 或 tile sweep。

audit arm 的静态 ISA 统计为 48 `v_mfma_f32_32x32x8_bf16`、113 global loads、224
global stores、70 LDS reads、216 LDS writes、13 `s_barrier`。这些是单一 loop body 的
静态文本计数，不能误读为 T=2048 动态总指令数。body arm 的 exact LTO MIR 已保存，且
post-greedy / virtregrewriter 均无 spill。

## Formal Body Timing

正式协议已经实现：预分配、current HIP stream、无 Graph capture、warmup=5、repeat=20、
五个 fresh-process session、每 session 轮转 `[A,B,C,C,B,A]`。三臂为：

1. `avelang_b0_full_sequence`；
2. `current_triton_direct_preallocated`，直接调用 current Triton kernel 并提供预分配输出；
3. `current_vllm_hsaco_external_bridge`，复用既有 hash-guarded current-vLLM HSACO bridge。

正式矩阵使用五个 fresh-process session。每个 session 预分配全部输入、输出和
intermediate，使用 current HIP stream、无 Graph capture，`warmup=5`、`repeat=20`，
并以 `[A,B,C,C,B,A]` 轮转三臂位置。每个 T 的 JSON 还重新检查三臂 finite 输出和
device-contract 阈值；B0 保持前述 H/V-new/final-state 门槛，native/bridge 均在其
已冻结的 BF16/FP32 contract 阈值内。

| T | chunks | B0 ms | direct Triton preallocated ms | external HSACO bridge ms | B0 / Triton | B0 / bridge |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 8 | `0.158235` | `0.070084` | `0.039378` | `2.26x` | `4.02x` |
| 1024 | 16 | `0.358733` | `0.098306` | `0.066258` | `3.65x` | `5.41x` |
| 2048 | 32 | `0.678227` | `0.149081` | `0.115572` | `4.55x` | `5.87x` |
| 8192 | 128 | `2.778229` | `0.455137` | `0.416399` | `6.10x` | `6.67x` |

对四个点拟合 `latency = intercept + slope * chunks`：

| arm | intercept ms | slope us/chunk |
|:--|--:|--:|
| B0 | `-0.007405` | `21.755686` |
| direct Triton preallocated | `0.046067` | `3.197504` |
| external HSACO bridge | `0.015192` | `3.134995` |

因此 B0 的长序列斜率约为 direct Triton 的 `6.80x`，为 bridge 的 `6.94x`。这不是
Eager public API 的正式排名，而是相同预分配 ABI recurrence body 的诊断比较；direct
Triton 与 bridge 两臂的固定项不同，也不能将二者之间的差异单独归因到 kernel body。
原始 session/block 样本、逐臂 correctness 和拟合结果保存在：

```text
rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b0_benchmark/
  b0_body_benchmark.json
```

## T=2048 PMC / Resource Evidence

`rocprofv3` 对 B0 的 T=2048 body dispatch 记录了以下动态工作量：

| metric | value |
|:--|--:|
| MFMA | `65,536` |
| VMEM | `315,904` |
| VALU | `2,865,536` |
| SALU | `161,216` |
| LDS instructions | `495,616` |
| LDS block | `36,864 B` |
| scratch | `0 B` |
| occupancy percent | `0.6464` |

rocprof 同一 dispatch 的资源字段显示 `VGPR_Count=128`、`Accum_VGPR_Count=336`、
`SGPR_Count=112`。这些是 profiler 的 launch/resource metadata，**不能替代** code
object 的物理寄存器计数。关于 resource cliff 的正式依据仍是 exact LTO final code
object 的 `VGPR=460`、`SGPR=38`、`private=0`、零 VGPR/SGPR spill。

计数器 CSV、kernel trace、exact LTO replay 和 ISA 位于：

```text
rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b0_profile/
rocprof_outputs/qwen_direct_k64_bv32_full_sequence_b0_t2048/
```

## Decision

`native_full_recurrence_correctness_baseline=true`：B0 是第一条通过 P2 byte-exact
full-sequence feedback gate 的 Avelang-native Direct-K64 BV32 recurrence baseline。

`performance_baseline_promoted=false`：正式矩阵已确认 B0 从 T=512 的 `2.26x` Triton
差距扩大到 T=8192 的 `6.10x`，而且 B0 斜率为 `21.76 us/chunk`，显著高于 Triton 的
`3.20 us/chunk`。结合 exact-LTO `VGPR=460`，它不能作为 native performance baseline。

下一轮只允许一个 **same-work pred/update phase-lifetime scheduling A/B**：保持 B0 数学、
MFMA geometry、BV32 ownership、block-dot lowering 和 LDS layout不变，只改变 pred
accumulator、V-new fragment、update accumulator 的创建/释放时间，并用同一 full-sequence
correctness 与 body harness 复测。不得改 RA、allocator、MFMA geometry、BV32、LDS layout、
broad-K/compact-K、external HSACO 或 production dispatch。

## Reproduction

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:./test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

# Correctness first.
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b0.py \
  --T 64 128 512 2048 --seed 20260731 --json

# Only after the gate passes: diagnostic body timing, not public API ranking.
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_bv32_full_sequence_b0.py \
  --T 512 1024 2048 8192 --warmup 5 --repeat 20 --sessions 5 --json
```

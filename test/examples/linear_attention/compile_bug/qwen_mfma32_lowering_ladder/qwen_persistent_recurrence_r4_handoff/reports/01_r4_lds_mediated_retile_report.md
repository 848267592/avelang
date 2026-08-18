# Qwen Persistent Recurrence R4: LDS-Mediated Distributed Retile

## 结论

`gfx942_bt64_bv32_joint_v4` 已完成为一条正确的、完整的 native
full-recurrence experimental candidate。它保留 R2 的完整 BT64 联合 schedule，
将 R3 的 K-row fragment `ds_bpermute_b32` 路径替换成：

```text
contiguous BF16x8 global K packet
  -> token-major LDS vector store
  -> LDS address-mediated logical retile
  -> local LDS gather fragment
  -> Direct-K64 MFMA32 update
```

R4 的 nonzero-W full recurrence 在 `T=64/128/512/2048` 均通过；同 R1、R2、
R3、B0 和 P2 host microscope 的所有已导出张量 byte-exact。它没有 scratch 或
MIR spill，动态 MFMA、VMEM 和 R2 相同，并消除了 R3 的 `512`
`ds_bpermute_b32`。在统一的七臂 fresh-process body benchmark 中，R4 在所有
长度都优于 R2：T=2048 快 `19.749 us`（`3.79%`），T=8192 快 `83.003 us`
（`3.97%`）；拟合斜率从 `16.249` 降到 `15.591 us/chunk`（`-4.05%`）。

因此 R4 晋级为当前 **native full-recurrence experimental baseline**，取代 R2。
它不是 production promotion，也不表示已经复刻了 current Triton 的完整 typed
LDS-to-dot-fragment data path：R4 的 K consumer 仍有 `128` 条静态
`ds_read_u16`。这条 scalar gather 是当前公开 AveLang fragment API 无法表达
strided packed MFMA32 operand 时的明确、受控 fallback。R4 说明“大规模 cross-lane
shuffle”不是必需的；它也给出一个严格的 API 结论：要继续缩小 Triton 差距，需要
first-class swizzled LDS dot-fragment load/encoding，而不是回到 R3 微调。

## 范围与冻结条件

本轮只新增 experimental lowering mode：

```text
AVELANG_PERSISTENT_RECURRENCE_LOWERING=gfx942_bt64_bv32_joint_v4
```

保持不变的 contract：

| 项目 | 固定值 |
|---|---|
| target | gfx942 |
| tile / ownership | BT64, BV32, WG128, 32 CTA, two-wave cooperative |
| 输入和状态 dtype | K/W/U/H/V-new: BF16；g/state/final-state: FP32 |
| pred / update | MFMA32，`v_mfma_f32_32x32x8_bf16` |
| 数学 | P0 nonzero-W pred mapping、K32 accumulation order、BF16 V-new round-trip、FP32 feedback |
| full schedule | R2 one-chunk lookahead 与 same-bank tail commit |
| 禁止项 | 不改 production selector、external HSACO、RA、allocator、MFMA geometry、BV、old broad/compact-K、ping-pong 或 double buffer |

R4 没有创建独立 pred/update kernel；W、K、H、V-decay 和 bank lifetime 都继续由同一
persistent recurrence plan 管理。

## 为什么从 R3 转向 LDS-mediated retile

R3 的 producer-side typed BF16x8 packet 是正确的，但为了将 token-owned packet
转为 K-row-owned MFMA fragment，它在寄存器中做四个 i32 word plane、八个 token 的
cross-lane transpose。结果是正确的 `ds_bpermute_b32` machine graph，却有：

```text
512 ds_bpermute_b32
LDS 807,936
VALU 4,338,880
rocprof Accum_VGPR 384
```

R4 不再试图压缩或调度这条路径。K 的连续 global packet 直接写入
token-major physical LDS layout；最终 fragment 的 element ownership 由 LDS 地址
完成，而不是先在 VGPR 中拼 K-row packet 再跨 lane 交换。

现有 Stage-6R Triton mapping/S0-C verifier 证明了一个关键约束：已有 source
producer packet固定 `token x K[8 contiguous]`，而 native dot consumer packet 更接近
`K x token[8 contiguous]`。在不引入跨 lane transfer 的前提下，这两个 packet 不能由
同一 lane-local vector 直接变换。因此 R4 的 consumer 选择显式 LDS gather，而没有
伪称得到 Triton 的 packed wide fragment load。

## 实现

### 统一 planner

以下文件新增了 R4 schedule kind、shared encoding 和 dot operand encoding：

- `lib/Dialect/AveLang/Transforms/qwen_recurrence_schedule_plan.h`
- `lib/Dialect/AveLang/Transforms/qwen_persistent_recurrence_pass.cc`

`gfx942_bt64_bv32_joint_v4` 形成一个 `QwenRecurrenceSchedulePlan`，其关键属性为：

```text
shared_encoding = joint_v4_lds_mediated_retile_bank
dot_operand_encoding = lds_mediated_retile_dot
next-chunk stage = enabled
tail commit = current consumer release 后复用同一 bank
```

W/K producer、current/next stage、shared bank lifetime 和 pred/update operand tag
均由 planner 同时标记；没有拆成独立 K plan 或独立 update plan。

### Producer 与 consumer lowering

`lower_qwen_k64_pipeline_stage_pass.cc` 让 R4 的 W/K packet 使用 typed BF16x8
global load 与 vector LDS store。K 的 physical LDS address 为 token-major：

```text
[k-half, token-offset, K-feature]
```

`lower_qwen_block_dot_pass.cc` 中的 R4 consumer 从该 physical layout 读取所需
K-row fragment。它不会进入 R3 的 `gpu.shuffle`/`ds_bpermute` 逻辑；当前可表达的
正确版本是 `loadPreloadedTokenMajorKVector()`，以 8 个确定的 LDS element load
组成 BF16 fragment。该 fallback 保留 Direct-K64 MFMA32 数学和 K32 accumulation
order，不改变 global IO 或 loop feedback。

这也是 R4 的能力边界：producer 已是 packed/vectorized，consumer 仍不是 Triton 式
typed wide local-load。报告将它视为未完成的 source/IR expression feature，而不是把
scalar gather 混同为完整 packed operand lowering。

## 多层机器图证据

R4 编译日志明确输出：

```text
[qwen-persistent-recurrence] mode=gfx942_bt64_bv32_joint_v4
  lowered a complete joint recurrence region before block-dot lowering
[qwen-block-dot] mode=specialized operand=persistent_typed_block
```

证据链保存于：

`codex_qwen_persistent_recurrence_r4_lds_mediated_retile/machine/`

| 层次 | R4 证据 |
|---|---|
| planner/MLIR | `ir/post_recurrence_joint_planner.mlir`；完整 recurrence 先于 block-dot lowering 形成 |
| stage lowered MLIR | `ir/post_joint_v1_stage_lowering.mlir`；R4 stage packet 经过 typed LDS store lowering |
| LLVM | `ir/preopt_llvm.ll`、`ir/postopt_llvm.ll` |
| exact LTO MIR | `exact_lto/kernel_section_00.mir` pre-greedy，`01` post-greedy，`02` virtregrewriter |
| ISA | `machine/r4.isa` |
| code object | `hsaco/_qwen_gdn_persistent_recurrence_r4_joint_v4_kernel.hsaco` |

R4 HSACO SHA256：

```text
1f80bc215d7373c763d7761db083217af9d9cd9d5830d91f5d83d4bac21b70a4
```

这与 R2 的 `e131ab...`、R3 的 `9087df...` 不同；因此本轮不是重测旧的
machine graph。

### Static ISA

| static ISA occurrence | R2 | R3 | R4 |
|---|---:|---:|---:|
| `v_mfma_f32_32x32x8_bf16` | 48 | 40 | 40 |
| `ds_bpermute_b32` | 0 | 512 | **0** |
| `ds_read_b128` | 32 | 32 | 16 |
| `ds_read_b32` | 16 | 16 | 16 |
| `ds_read_u16` | not the R3 failure mode | not the R3 failure mode | **128** |
| `ds_write_b128` | 232 LDS writes total | 36 | 36 |
| `ds_write_b16` | -- | 72 | 72 |
| `s_barrier` | 13 | 11 | 11 |
| `s_waitcnt` | 109 | -- | 103 |

静态 MFMA occurrence 不能代替动态数学工作量：不同 unroll/reuse 会改变静态出现次数。
PMC 的动态 MFMA 在 R1/R2/R3/R4 都是 `65,536`，因此 performance 差异不能归因于
少算 MFMA。

### LTO 与 resource audit

R4 code object：

```text
.private_segment_fixed_size: 0
.vgpr_spill_count: 0
.sgpr_spill_count: 0
.vgpr_count: 276
.agpr_count: 64
.sgpr_count: 41
```

`vgpr_count/agpr_count` 是 code-object metadata，不应与 rocprof 的
`VGPR_Count/Accum_VGPR_Count` 当作同一单位直接相减。实际 PMC 资源字段为
`VGPR=128`、`Accum_VGPR=192`、`SGPR=112`、`LDS=53,248B`、`Scratch=0`。

exact-LTO pre/post-greedy MIR 与 virtregrewriter MIR 均显示：

```text
SI_SPILL_AV32_SAVE = 0
SI_SPILL_AV64_SAVE = 0
SI_SPILL_AV32_RESTORE = 0
SI_SPILL_AV64_RESTORE = 0
```

所以 R4 的收益或剩余差距都不能归因为以 private memory spill 换取速度。

## Nonzero-W correctness

R4 使用运行时 nonzero-W，而非 W=0 作为 correctness gate。检查包含 H snapshot、
raw FP32 pred、BF16 pred/V-new/V-decay、每 chunk state 和 final state。

| T | R4 vs R1/R2/R3/B0/P2 | reference H max abs | reference final-state max abs |
|---:|:---:|---:|---:|
| 64 | byte-exact | `0` | `3.7253e-09` |
| 128 | byte-exact | `5.9605e-08` | `3.7253e-09` |
| 512 | byte-exact | `2.4414e-04` | `1.9405e-05` |
| 2048 | byte-exact | `2.4414e-04` | `4.4488e-05` |

所有长度 finite 并通过既有 device-contract BF16/FP32 threshold。R4 仍执行
`corrected -> BF16 V-new -> FP32(v_new_bf16) -> BF16 V-decay`；FP32 persistent
state 是下一 chunk 的 feedback carrier，H BF16 snapshot 不是。

## T=2048 PMC

下表均为完整 recurrence 的同一 T=2048 机器工作量；trace 只用于 resource/
instruction accounting，不作为 latency。

| metric | R2 | R3 | R4 | R4 vs R2 |
|---|---:|---:|---:|---:|
| MFMA | 65,536 | 65,536 | 65,536 | 0 |
| VMEM | 202,240 | 205,696 | 202,240 | 0 |
| VALU | 2,227,392 | 4,338,880 | 2,294,144 | +66,752 (+3.00%) |
| SALU | 163,008 | 163,072 | 163,072 | +64 |
| LDS instructions | 385,536 | 807,936 | 381,952 | -3,584 (-0.93%) |
| LDS block | 53,248 B | 53,248 B | 53,248 B | 0 |
| rocprof VGPR / Accum_VGPR | 128 / 192 | 128 / 384 | 128 / 192 | unchanged |
| rocprof Scratch_Size | 0 B | 28 B collector field | 0 B | no cliff |

R4 相比 R3 的关键变化是：LDS `-425,984`（`-52.72%`）、VALU `-2,044,736`
（`-47.12%`）、VMEM `-3,456`，并把 rocprof Accum_VGPR 从 `384` 恢复到 `192`。
R4 相比 R2 的 VALU 略升，是 token-major LDS fragment gather 的直接代价；但它没有
抵消更少的 barrier/wait、较少 LDS 动态工作和更好的 full sequence scheduling。

current Triton 的同 contract T=2048 仍为 `VMEM=58,368`、`LDS=305,472`、
`LDS block=40,960B`。R4 因此没有消除 native data-path gap，只是用低成本的
LDS-mediated mapping 替换了 R3 的高成本寄存器 transpose。

## 正式 Fresh-Process Body Benchmark

方法固定为：预分配、current HIP stream、无 Graph capture、warmup=10、repeat=50、
每点 5 个 fresh-process sessions、七臂 rotating palindromic order、HIP event。
这是 recurrence-body diagnostic，而不是 Eager public API 或 production ranking。

| T | B0 ms | R1 ms | R2 ms | R3 ms | **R4 ms** | direct Triton ms | external bridge ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.156633 | 0.139187 | 0.137905 | 0.157273 | **0.133698** | 0.073830 | 0.040741 |
| 1024 | 0.357292 | 0.283502 | 0.278634 | 0.294538 | **0.269541** | 0.103394 | 0.068402 |
| 2048 | 0.677707 | 0.530889 | 0.521675 | 0.547474 | **0.501926** | 0.156192 | 0.118616 |
| 8192 | 2.778611 | 2.125920 | 2.090668 | 2.072541 | **2.007665** | 0.472783 | 0.421246 |

| implementation | fitted slope (us/chunk) |
|---|---:|
| B0 | 21.771093 |
| R1 | 16.526212 |
| R2 | 16.249126 |
| R3 | 15.918738 |
| **R4** | **15.590970** |
| direct Triton | 3.311115 |
| external bridge | 3.160771 |

R4 对 R2 的 paired endpoint improvement：

| T | R4 - R2 | improvement |
|---:|---:|---:|
| 512 | -4.207 us | 3.05% |
| 1024 | -9.093 us | 3.26% |
| 2048 | -19.749 us | 3.79% |
| 8192 | -83.003 us | 3.97% |

R4 仍慢于 direct Triton：`1.81x`（T=512）、`2.61x`（T=1024）、`3.21x`
（T=2048）和 `4.25x`（T=8192）。其 slope 是 Triton 的约 `4.71x`，说明主剩余差距
随 chunk 线性累积，不能由一次 launch/intercept 修复。

## 决策与停止条件

### R4 的决定

R4 通过预注册的 full recurrence correctness、无 spill/scratch、相同 MFMA 数量、
删除 R3 large-scale bpermute，且全部主要长度稳定优于 R2。因此：

```text
native full-recurrence experimental baseline = gfx942_bt64_bv32_joint_v4
historical controls = B0, R1, R2, R3
```

不修改 production selector；不以此 body benchmark 宣称 public API 超过 vLLM。

### 对纯 LDS retile 的精确判断

R4 不是“完全 typed-wide shared-to-dot operand”成功。它的 producer 已完成 BF16x8
vector load/store，但 consumer 还需要 128 条 `ds_read_u16`，这说明目前 AveLang
缺少一个可以表达下列语义的公开/中间 IR primitive：

```text
swizzled LDS physical layout
  -> non-contiguous BF16x8 MFMA32 fragment load
  -> typed dot operand
```

在现有 API 下，再枚举更多 source swizzle 会重复 D0-P/R3 已经否定的方向：

- 寄存器 transpose：正确但 512 `ds_bpermute_b32`，资源 cliff；
- token-major scalar gather：正确且比 R2 快，但不等价于 native packed consumer；
- 只压 producer store：C0.5S 已证明后端已经能合并连续 store，不能解决 consumer。

因此，**pure packed producer + pure packed consumer 的局部 LDS-layout 路线在当前
AveLang primitive 集合中正式 NO-GO**；R4 本身保留为有效的性能 baseline。下一步若
继续 native recurrence，应是一个独立、最小的 compiler/source feature：first-class
swizzled `qwen_mfma32_fragment_load` 或 dot-operand encoding，证明可以从固定 LDS
address map 直接产生合法 MFMA fragment，而不能把整个 block-dot schedule 隐藏在
lowering 中。该 feature 应先做 same-source generic/specialized A/B，再决定是否接入
R4；不应回到 R3 bpermute 微调。

## 复现

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention/vllm_compare
export PYTHONDONTWRITEBYTECODE=1

# Full nonzero-W correctness.
python3 test/examples/linear_attention/vllm_compare/repro_qwen_gdn_persistent_recurrence_r4.py \
  --T 64 128 512 2048 --seed 20260830 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_persistent_recurrence_r4_lds_mediated_retile/correctness \
  --json

# Seven-arm body diagnostic.
python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_persistent_recurrence_r4.py \
  --T 512 1024 2048 8192 --warmup 10 --repeat 50 --sessions 5 \
  --out-json test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_persistent_recurrence_r4_lds_mediated_retile/benchmark/body_benchmark.json
```

完整证据目录：

```text
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
  codex_qwen_persistent_recurrence_r4_lds_mediated_retile/
    correctness/
    machine/ir/
    machine/link/
    machine/exact_lto/
    machine/hsaco/
    machine/rocprof/r4_t2048/
    benchmark/body_benchmark.json
```

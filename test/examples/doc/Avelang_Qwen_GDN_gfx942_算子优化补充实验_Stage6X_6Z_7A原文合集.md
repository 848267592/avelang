# Qwen gfx942 BT64 Stage 6X-KS X0: KKT-Solve Handoff Ownership Audit

## Scope And Decision

This is the zero-source-change feasibility audit for **Stage 6X-KS: KKT-Solve
Handoff Elimination**.  It audits the current Stage 6W/W1 graph only:

```text
cumsum -> KKT FP32 a -> hierarchical FP32 solve / BF16 store -> fused W/U
```

No kernel, compiler, recurrence HSACO, default selector, or production path is
changed by X0.  Eager public API remains the formal ranking contract;
standalone body timings and rocprof are diagnostic gates only.

**X0 decision: proceed to X1 only.**  The current KKT and solve have compatible
per-`(chunk, value_head)` ownership and exactly compatible logical `a` layout.
X1 can therefore test a one-CTA KKT without changing the global handoff.  X2
is still conditional: source-level shared-memory aliasing/lifetime reuse must
be demonstrated rather than assumed.

## Frozen Contract

| item | value |
|:--|:--|
| target | gfx942 / MI300 |
| tensor shape | `B=1, Hk=4, Hv=8, K=V=128, BT=64` |
| KKT inputs | BF16 `k`; FP32 cumsum `g` and `beta` |
| KKT current output | FP32 `a`, contiguous `[1,T,8,64]` |
| solve current input | that same FP32 `a` layout |
| solve current output | BF16 `a_solved`, contiguous `[1,T,8,64]` |
| current experimental full path | Stage 6W / W1 |
| formal timing | complete Eager public API, no CUDA Graph replay |

At `T=2048`, `a` is 4 MiB; the producer write plus the solve read is 8 MiB.
At `T=8192`, it is 16 MiB and 32 MiB respectively.  It is the largest
source-native, one-use upstream boundary remaining after Stage 6W.

## 1. Current KKT Ownership

Source: `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`,
`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`.

The launch is:

```text
grid = num_chunks * 8 value_heads * 16 token16 tiles
workgroup = 64 threads = 1 wave
```

For one `(chunk, value_head)`, `tile_id in [0, 15]` maps as:

```text
row_tile = tile_id // 4
col_tile = tile_id % 4
row_base = 16 * row_tile
col_base = 16 * col_tile
```

The current kernel launches all 16 matrix tiles.  Only ten lower-or-diagonal
tiles satisfy `row_tile >= col_tile` and execute K staging plus MFMA.  The six
strict-upper CTA instances perform no MFMA; they only write their 16x16 output
region as zero.  This is intentional: it materializes the full `[64,64]`
layout expected by existing consumers while retaining strict-lower math.

### KKT Formula And Mask

For local token row `t` and source column `s` in the same BT64 chunk, current
writeback is:

```text
a[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t] - g[s])     when s < t
a[t,s] = 0                                               when s >= t
```

The MFMA produces a complete 16x16 dot tile.  The strict-lower/causal rule is
applied only at FP32 writeback by `source_offset < token_offset`.  Therefore
diagonal 16x16 tiles compute the full dot product but retain only their own
strict lower triangle; their diagonal and upper entries are written as zero.

### Current Per-Tile Work

Each active tile stages:

```text
row_k_bf16[16,128] = 4 KiB
col_k_bf16[16,128] = 4 KiB
```

It accumulates four `batch128` iterations, each containing two
`mfma_16x16x16_bf16_f32` operations.  Thus:

```text
8 dynamic BF16 MFMA / active 16x16 tile
10 active tiles / 64x64 matrix
80 dynamic BF16 MFMA / (chunk, value_head)
```

The Stage 4 T=2048 rocprof total of 20,480 MFMA instructions exactly matches
`32 chunks * 8 heads * 80`.  The current KKT resource tuple was `WG=64`,
`LDS=8192 B`, `VGPR=20`, `AccVGPR=4`, `scratch=0`.

The ten active current CTAs re-read the K inputs independently.  Ignoring
cache effects, their staged global K traffic is `10 * (4 KiB + 4 KiB) =
80 KiB` per matrix.  X1 can instead stage the chunk's whole
`K[64,128]` once, which is 16 KiB, while preserving the same 80 MFMA
operations and exact per-tile reduction order.

## 2. Current Solve Ownership And Inputs

Source: `vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`,
`_qwen_gdn_solve_hierarchical_bt64_fp32_compute_bf16_store_stage6u`.

The solve launch is already the desired coarse ownership:

```text
grid = num_chunks * 8 value_heads
workgroup = 256 threads = 4 waves
CTA = exactly one (chunk, value_head) 64x64 matrix
```

It consumes the exact KKT layout:

```text
a[0, chunk_start + row, head_idx, col]
```

No transpose, packing conversion, or dtype conversion occurs between the KKT
global store and solve load.  A fused handoff can therefore place the KKT
FP32 tile into an LDS `a_lds[row,col]` with the same row-major indices and
replace only the source of the solve's `a_frag` reads.

### Which A Blocks Solve Uses

Partitioning the 64x64 matrix into four 16x16 blocks, solve uses all strict
lower entries and no diagonal/upper `A` input values:

| input block | use count in the solve DAG |
|:--|--:|
| four diagonal strict-lower regions | initialize the four diagonal inverses |
| `A21`, `A32`, `A43` | level 1 |
| `A31`, `A32` | level 2a |
| `A42`, `A43` | level 2b |
| `A41`, `A42`, `A43` | level 3 |

The strict-lower matrix has `64 * 63 / 2 = 2016` FP32 semantic elements.
The current full output allocation has 4096 FP32 locations because KKT also
writes zeros for its diagonal and upper locations.  X2 should initially
retain all 4096 FP32 LDS locations: it preserves layout, avoids introducing
packed-lower address arithmetic, and keeps the experiment scoped to handoff
elimination.  Packed lower-triangle storage is an explicitly separate future
experiment, not an X2 addition.

### Current Solve LDS And DAG

The proven hierarchical solve uses:

```text
x[7,16,16] FP32       = 7 KiB
work[16,16] FP32      = 1 KiB
total                 = 8 KiB
```

`x` retains four diagonal inverse blocks and three level-1 lower blocks.
Slots are reused as the block DAG advances; `work` is first used for diagonal
row snapshots and later as one 16x16 product workspace.  This source schedule
has already passed correctness with `scratch=0`, `VGPR=44`, `AccVGPR=4`, and
`LDS=8192 B`.  At T=2048 it dynamically executes 64 FP32 MFMA16x16x4
instructions per matrix.

## 3. X1 Ownership: One CTA KKT, Still Global FP32 A

X1 uses the exact solve CTA mapping without changing the handoff:

```text
program_id -> chunk_idx = program_id // 8, head_idx = program_id % 8
WG256 -> wave_id = tid // 64
wave_id 0..3 owns row_tile 0..3
```

The proposed X1 staging is one BF16 shared tile:

```text
k_all_bf16[64,128] = 16 KiB
```

All four waves cooperatively stage it once.  Then each wave iterates the four
compile-time `col_tile` values.  For `row_tile >= col_tile`, it executes the
same eight-MFMA 16x16 dot sequence as current KKT and applies the same
writeback mask/decay.  For upper tiles it stores zero exactly as current KKT.
All writes retain the current global FP32 `a[1,T,8,64]` ABI.

| property | current KKT | X1 candidate |
|:--|:--|:--|
| CTA ownership | one 16x16 tile | one complete 64x64 chunk/head |
| grid per matrix | 16 WG64 CTAs | 1 WG256 CTA |
| active BF16 MFMA / matrix | 80 | 80 target |
| K LDS | 8 KiB per active CTA | 16 KiB per CTA |
| global `a` store | FP32 full 64x64 | unchanged |
| global `a` handoff | present | present |
| strict-upper handling | zero-store CTA | zero-store wave/tile |

X1 has a real risk: at `T=2048`, the KKT grid drops from 4096 total CTAs to
256 CTAs.  The fivefold reduction in theoretical K staging traffic can lose
to reduced inter-CTA parallelism, especially at T=64/128.  That is why X1 is
a hard gate rather than a presumed prerequisite for X2.

## 4. X2 LDS Lifetime And Capacity Audit

X2 would change only the KKT-to-solve boundary:

```text
KKT FP32 tile results -> a_lds[64,64] FP32 -> existing FP32 solve DAG
                                          -> BF16 a_solved global store
```

Capacity accounting is:

| region | bytes | KKT phase | solve phase |
|:--|--:|:--:|:--:|
| `a_lds[64,64]` FP32 | 16 KiB | live | live |
| KKT `k_all_bf16[64,128]` | 16 KiB | live | dead after final KKT tile |
| solve `x + work` FP32 | 8 KiB | unused | live |

There are two possible static-LDS outcomes:

1. **Correct reuse design:** reuse the 16 KiB K staging allocation after KKT
   as the solve's 8 KiB `x + work` area. Peak explicit LDS is 32 KiB:
   `a_lds 16 KiB + reusable scratch 16 KiB`.
2. **Naive separate allocations:** if the source declares independent K,
   `a_lds`, `x`, and `work` shared arrays, compiler allocation can retain all
   of them. Static LDS becomes 40 KiB even though K data is semantically dead
   before solve begins.

X0 does **not** claim that lexical phase order automatically aliases two
`al.make_shared` arrays.  Existing source uses `al.view` for byte-preserving
packed views, but X0 did not find a prior validated FP32-to-BF16 workspace
reinterpretation pattern for this exact use.  A future X2 implementation must
either demonstrate a single explicitly typed reusable backing store or accept
and measure the 40 KiB conservative case.  It must not call the memory reused
until HSACO resource metadata proves it.

Even the conservative 40 KiB is below a 64 KiB workgroup LDS budget, but it
can reduce resident workgroups and must be checked together with VGPR,
AccVGPR, scratch/private segment, spills, and occupancy.  The required
barriers are already substantial in solve.  X2 needs one additional
phase-publication barrier after the complete KKT `a_lds` write; it should not
add a barrier per output element or per KKT tile beyond the staging schedule.

## 5. Gate Definitions

### X1 Gate

Proceed from X1 to X2 only if all hold:

- KKT FP32 output matches current KKT at the frozen tolerance, with the first
  mismatch emitted on failure;
- solve-after-KKT matches current solve, proving layout compatibility;
- `scratch=0`, `private_segment=0`, and no VGPR/SGPR spills;
- dynamic MFMA is 80 per `(chunk,head)` matrix, not increased by predicate
  lowering or duplicated staged work;
- LDS/VGPR/AccVGPR show no resource cliff;
- isolated KKT is not materially slower than current KKT across T=64 through
  8192.  Any short-T launch-parallelism loss must be quantified rather than
  hidden by T=2048-only reporting.

### X2 Gate

Only after X1 passes:

- compare `current KKT -> current Stage 6U solve -> BF16 a_solved` against
  fused KKT+solve for T=64,128,512,2048,8192 with random, high-dynamic,
  cancellation, neutral-gate, non-default stream, output reuse, and NaN
  prefill cases;
- target BF16 bit exactness; if it fails, stop to identify numerical order
  changes rather than widening a tolerance silently;
- verify resource metadata, code-object spill fields, ISA MFMA mix, and
  barrier count;
- compare W1, X2, and native vLLM through the same complete Eager public API
  protocol at T=512,1024,2048,4096,8192,16384 with paired/Williams ordering,
  HIP event plus wall-clock, and cluster-aware confidence intervals.

## 6. X0 Answer To The Main Questions

| question | X0 answer |
|:--|:--|
| Does KKT calculate all 16 tiles? | It launches all 16; exactly 10 run MFMA, six strict-upper CTAs write zeros only. |
| How is diagonal strict-lower applied? | Full MFMA dot tile followed by `source_offset < token_offset` FP32 writeback mask. |
| Does solve need a layout conversion? | No. It reads the exact row-major KKT `[row,source]` layout. |
| Can `a` be CTA-local? | Yes, one 64x64 FP32 matrix is 16 KiB per chunk/head, not the whole T tensor. |
| Can KKT and solve LDS be reused? | Semantically yes; source-level static aliasing is not yet proven. |
| Is resource risk acceptable? | X1 has a 16 KiB K tile and is a justified test. X2 has a 32 KiB reuse target or 40 KiB conservative case and requires profiler proof. |
| Is a full fusion justified now? | No. X1 must first establish that the 16-CTA to one-CTA KKT ownership does not lose more parallelism than it saves. |

## Evidence

- `vllm_compare/qwen_gdn_bt64_nonrecurrence_mfma_v2.py`
- `vllm_compare/qwen_gdn_solve_bt64_hierarchical_bf16_stage6u.py`
- `qwen_gfx942_bt64_nonrecurrence_stage4_report.md`
- `qwen_gfx942_bt64_hierarchical_solve_stage5b_completion_report.md`
- `qwen_gfx942_bt64_stage6w_cluster_confirmation_and_intermediate_audit_report.md`
- `codex_qwen_bt64_stage6w_cluster_confirmation_and_intermediate_audit/stage6w_intermediate_accounting.csv`
# Qwen gfx942 BT64 Stage 6X-KS: KKT-Solve Handoff Elimination

## 状态

Stage 6X-KS 已完成 X0、X1 和 X2 的源码实现、直接正确性、预分配 body
benchmark、HSACO 检查、T=2048 rocprof，以及随后补齐的完整 Eager public-API
正式 sweep。X2 删除了 KKT 到 solve 的 FP32 全局矩阵边界，并保持相对于 Stage 6W
链的 bit-exact 语义。

**X2 已晋级为新的 Avelang BT64 experimental baseline。** 这不是从 body
benchmark 推断出来的结论：正式 Eager public API 在每个 T 独立进程中执行，5 个
session、50 个 paired Williams block、每实现 300 次调用，且 HIP event 与
wall-clock 均同向。T=2048 与 T=8192 的 event cluster CI 下界都大于零；long-text
slope 也从 W1 的 `6.341 us/chunk` 降至 X2 的 `5.694 us/chunk`。

人工正式测量的结构化汇总存于
`codex_qwen_bt64_kkt_solve_handoff_stage6x_manual_confirmation/`。旧的 Docker
产物目录由容器 `nobody` 所有，故没有覆盖其中的原始 JSON；本报告以这份明确标记为
`user_manual_formal_run` 的确认记录作为晋级证据。

本阶段没有修改 recurrence HSACO、W/U、chunk-o、vLLM 或既有 production
baseline。

## X0：源码、ownership 与 LDS 生命周期审计

### 现有 KKT

当前 BT64 KKT 是
`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0`：

- launch：`16 * num_chunks * 8` CTA，WG=64；
- 一个 CTA 对应一个 `(chunk, value-head, token16 row-tile, token16 col-tile)`；
- 每个 64x64 matrix 共有 16 个 tile；其中 10 个下三角或对角 tile 做 dot，6 个
  上三角 tile 只写零；
- active tile 以 8 次 `mfma_16x16x16_bf16_f32` 完成 K=128 reduction；故每个
  `(chunk, head)` 有 `10 * 8 = 80` 次动态 BF16 MFMA；
- 数学与 output layout 保持：

```text
a[t,s] = beta[t] * dot(k[t], k[s]) * exp(g[t]-g[s])  if s < t
         0                                          otherwise
```

`a` 的 global layout 是 FP32 `[1,T,8,64]`，每一个 `(chunk, head)` 对应其中
一行 64-wide matrix。Stage 6U solve 正好按同一 `(chunk, head)` 使用 WG256/4
waves 消费这一块；它只需要下三角，但现有 ABI 仍为完整 64x64 row-major matrix。

### 容量结论

每 CTA 的 A matrix 为 `64*64*4 = 16 KiB`。X1 只需一次性 stage
`K[64,128] BF16 = 16 KiB`。X2 的保守实现同时分配：

| LDS 对象 | 大小 | 生命周期 |
|---|---:|---|
| `k_all_bf16[64,128]` | 16 KiB | KKT phase |
| `a_lds[4,16,64] FP32` | 16 KiB | KKT 完成到 solve 完成 |
| Stage6U `x[7,16,16]` + `work[16,16]` | 8 KiB | solve phase |
| 合计 | 40 KiB | 保守、无 alias |

本轮没有假设 Avelang 可以安全地将 BF16 K staging 和 FP32 solve work 做类型
重解释 alias。40 KiB 在 gfx942 CTA LDS 容量内；是否值得做 32 KiB lifetime reuse
是后续独立 micro-experiment，而非本阶段的正确性前提。

## X1：1 CTA / chunk-head KKT

源码：
`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py`

`_qwen_gdn_kkt_bf16_kernel_bt64_one_cta_stage6x_x1` 采用 WG256。四个 waves 分别
拥有 4 个 token16 row tile，并循环四个 column tile。严格上三角不做 dot 但仍写零；
因此 a 的 layout、mask 和数值顺序保持不变。X1 仍写原 global FP32 `a`，仅验证
ownership、并行度与资源。

### X1 正确性

`test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` 已在 gfx942 执行：

- KKT-only：T=64/128/512/2048，random 与 high_dynamic；全部 FP32 bit-exact；
- X1 KKT 后接当前 Stage6U solve：T=64/512/2048，random 与 cancellation；全部
  BF16 bit-exact。

### X1 KKT body

预热 5、repeat 20、HIP-event body；仅供 ownership gate，不是 Eager full timing。

| T | current KKT ms | X1 KKT ms | current/X1 |
|---:|---:|---:|---:|
| 64 | 0.035473 | 0.031447 | 1.128x |
| 128 | 0.035333 | 0.032369 | 1.092x |
| 512 | 0.034832 | 0.029624 | 1.176x |
| 1024 | 0.035192 | 0.031106 | 1.131x |
| 2048 | 0.046910 | 0.032529 | 1.442x |
| 8192 | 0.170874 | 0.038117 | 4.483x |

T=2048 rocprof 显示 X1 保持 KKT 的 `20,480` MFMA，但把重复 tile staging/address
工作大幅降低。代价是 WG256 的资源和 occupancy；它没有 scratch。

| metric | current KKT | X1 KKT |
|---|---:|---:|
| grid work-items | 262,144 | 65,536 |
| workgroup | 64 | 256 |
| LDS | 8 KiB | 16 KiB |
| VGPR / AccVGPR / SGPR | 20 / 4 / 32 | 84 / 20 / 112 |
| scratch | 0 | 0 |
| MFMA | 20,480 | 20,480 |
| VALU | 1,565,696 | 639,488 |
| SALU | 187,904 | 93,184 |
| VMEM | 210,944 | 79,872 |
| LDS instructions | 184,320 | 47,104 |
| OccupancyPercent | 8.991% | 3.646% |
| median trace | 40.961 us | 9.815 us |

因此 X1 gate 通过：CTA 数从 16 降为 1 并未造成不可接受的并行度损失，且没有
scratch/spill cliff。

## X2：CTA-local KKT + hierarchical solve

### 实现

同一文件中的
`_qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2`：

```text
K/Beta/G
  -> KKT FP32 (CTA-local a_lds[4,16,64])
  -> Stage6U FP32 block-triangular solve in the same CTA
  -> BF16 a_solved global output
```

它不创建也不接收 global FP32 `a`。`a_lds` 的逻辑行布局是 `[row_block,
row_in_block,column]`；该布局与 solve 的 four-wave ownership 一致。

实现中发现了一个 Avelang fragment-layout 要点：`mfma_16x16x4_f32_f32` 的 A/B
operand 必须为 `vector<1xf32>`。初版错误地将 view 的最后一个 unit dimension
也索引掉，得到标量并触发 `MFMA operands must be vector types`。最终使用：

```python
a_lds_frag = al.view(a_lds, al.f32,
    al.make_layout((4, 16, 64, 1), (1024, 64, 1, 1)))
rhs = a_lds_frag[row_block, row_in_block, column]  # 保留最后的 1
```

这与现有 `x_frag` 的用法一致，直接将 LDS fragment 作为 FP32 MFMA operand。

### X2 直接正确性

已执行的 Stage6X KKT/solve matrix：

- current KKT -> current Stage6U solve vs X2：T=64/128/512/2048；
  random、high_dynamic、cancellation；全部 BF16 bit-exact；
- X1 KKT 与 current KKT：T=64/128/512/2048；全部 FP32 bit-exact；
- full Stage6W vs Stage6X：T=64/128/512/2048/8192，random、high_dynamic、
  cancellation、neutral_gate，且分别有/无 initial state；共 40 cases，public
  BF16 output 和 FP32 final-state 均 bit-exact。

补齐的接口回归：

- X2 caller-owned BF16 output 的 NaN prefill + 两次 reuse：通过；
- non-default stream T=64/2048：`2 passed`；
- 主 KKT/solve correctness matrix：`29 passed`。

这些测试证明 X2 相对 Stage6W 保持精确语义及接口行为；它们不把“与 Stage6W
bit-exact”偷换成“与 vLLM 完全 bit-exact”。完整输出/final-state 的跨实现接受范围仍
沿用 Stage 6S/6U 的冻结 contract。

### 预分配 KKT+solve body

直接 launch 到 caller-owned outputs，warmup=5、repeat=20。current 是原 KKT
kernel 加 Stage6U solve；X2 是一 kernel。全部 X2 BF16 bit-exact。

| T | current KKT+solve ms | X2 ms | speedup |
|---:|---:|---:|---:|
| 512 | 0.046850 | 0.034591 | 1.354x |
| 2048 | 0.056384 | 0.034691 | 1.625x |
| 8192 | 0.161080 | 0.076193 | 2.114x |

该趋势符合 handoff 消除的预期：收益随 chunk 数增长。但这仍是 standalone body，不能
替代 full Eager gate。

### X2 T=2048 rocprof / ISA

| metric | X2 |
|---|---:|
| grid work-items / WG | 65,536 / 256 |
| LDS block | 40 KiB |
| VGPR / AccVGPR / SGPR | 100 / 164 / 112 |
| scratch | 0 B |
| MFMA | 36,864 |
| VALU / SALU | 1,280,512 / 206,336 |
| VMEM / LDS inst | 90,112 / 285,696 |
| OccupancyPercent | 6.362% |
| median trace | 16.505 us |

MFMA 数为 `20,480` 次 KKT BF16 MFMA 加 `16,384` 次 solve FP32 MFMA，正好符合
融合的两阶段工作量；没有通过减少 solve 数学换取收益。HSACO 中可见：

- `v_mfma_f32_16x16x16_bf16` 静态 32 条；
- `v_mfma_f32_16x16x4_f32` 静态 56 条。

X2 的 VGPR/AccVGPR/LDS 均高于 X1，但 scratch 为零，且 trace 仍明显低于旧 KKT
单阶段 trace。资源 gate 因此为通过，但 40 KiB LDS / AccVGPR=164 意味着后续若尝试
buffer reuse，必须再次检查 occupancy，而不是假定 LDS 更少一定更快。

## Full Eager public API

全图入口为：

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py`

```text
cumsum -> X2 fused KKT+solve -> existing BF16 W/U
       -> immutable BF16 recurrence -> existing Stage6W BF16 chunk-o
```

权威计时脚本：

`vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py`

它固定输入、current stream、Eager public API（`cuda_graph_used=false`），对每个
block 打乱六种 Williams 三路顺序，记录 HIP event 和 wall-clock，并按 session/block
配对做 cluster bootstrap。

### 完整正式 Eager confirmation

每个 T 以独立进程运行。每个进程固定相同的 `q/k/v/g/beta/initial_state`、current
stream 和 Eager public API；compile、module load 与首次 allocation 在计时前 warmup。
随后执行 5 个 session，每个 session 运行 10 个 timed paired Williams block，block 的
起始顺序随机化。每实现共 300 个完整调用。收益为 `W1 - X2`，正值代表 X2 更快。

| T | W1 median ms | X2 median ms | paired mean gain us | event 95% CI us | X2 相对 W1 |
|---:|---:|---:|---:|:---|---:|
| 512 | 0.221349 | 0.199697 | 22.123 | [19.720, 24.647] | 快约 10.9% |
| 1024 | 0.262049 | 0.233467 | 26.070 | [23.592, 28.667] | 快约 10.9% |
| 2048 | 0.348378 | 0.319154 | 26.020 | [23.304, 28.563] | 快约 8.4% |
| 4096 | 0.518070 | 0.479532 | 40.465 | [37.439, 43.506] | 快约 7.4% |
| 8192 | 0.935690 | 0.852166 | 82.823 | [80.432, 85.154] | 快约 8.9% |
| 16384 | 1.773194 | 1.595190 | 177.540 | [176.004, 179.024] | 快约 10.0% |

T=1024/2048 的 HIP 与 wall-clock CI 都完全为正，且每个长度的多数 session 都为正；
T=4096、8192、16384 的长文本收益也稳定扩大。这满足预注册 gate：correctness、stream、
NaN/reuse、无 scratch/spill、T=2048 和 T=8192 paired CI 下界大于零、两类计时器方向
一致且 slope 不恶化。

对 T=1024--16384 的 event median 按 chunk 数拟合：W1 为 `6.341 us/chunk`，X2 为
`5.694 us/chunk`，因此回收 `0.646 us/chunk` 或约 `10.2%`。这说明收益不仅来自少一个
dispatch，也来自随 chunk 增长而消失的重复 K staging、地址计算和 FP32 `a` global
write/read。

### 同批 native vLLM 对比

| T | X2 ms | vLLM ms | X2/vLLM | 结论 |
|---:|---:|---:|---:|:---|
| 1024 | 0.233467 | 0.358813 | 0.651x | X2 快 |
| 2048 | 0.319154 | 0.401397 | 0.795x | X2 快 |
| 4096 | 0.479532 | 0.522356 | 0.918x | X2 快，gain CI [39.710, 45.394] us |
| 8192 | 0.852166 | 0.765016 | 1.114x | X2 慢 |
| 16384 | 1.595190 | 1.235115 | 1.292x | X2 慢 |

vLLM 的拟合 slope 为 `3.688 us/chunk`，仍低于 X2 的 `5.694 us/chunk`。因此当前准确
结论是：这批同口径 Eager 测试中 X2 在 `T <= 4096` 快于 vLLM，在 `T >= 8192` 稳定落后。
4096--8192 之间存在粗略 crossover，但没有把线性插值值宣称为 dispatch policy。

## 产物与复现

| 产物 | 作用 |
|---|---|
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | X1/X2 kernels 和 direct-out API |
| `vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | X2 full Eager graph |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | KKT/solve/reuse gates |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py` | 40-case full bit-exact matrix |
| `vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_stream.py` | non-default-stream gate |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | preallocated X1/X2 body benchmark and HSACO capture |
| `vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py` | rocprof driver |
| `vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py` | formal Williams Eager benchmark |
| `codex_qwen_bt64_kkt_solve_handoff_stage6x/` | JSON, HSACO, rocprof raw artifacts |

关键命令：

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py -s

PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --T 512 2048 8192 --warmup 5 --repeat 20

/opt/rocm/bin/rocprofv3 --kernel-trace --pmc \
  SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
  --kernel-include-regex _qwen_gdn_kkt_solve_bf16_kernel_bt64_one_cta_stage6x_x2 \
  -d test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_kkt_solve_handoff_stage6x/rocprof_x2 \
  -o stage6x_x2 -f csv -- python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_kkt_solve_handoff_stage6x.py \
  --implementation x2 --T 2048 --warmup 2 --repeat 5

# Run once per T in a fresh process after execution access is restored.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py \
  --T 2048 --sessions 5 --warmup-blocks 3 --blocks 10 \
  --out-dir /tmp/stage6x_eager_t2048
```

## 决策与下一步

版本状态：

| 版本 | 状态 |
|:--|:--|
| production/default | 不变 |
| U1 / Stage 6U | 历史 experimental baseline |
| W1 / Stage 6W | 上一 experimental baseline |
| X1 | 成功的 one-CTA KKT 组件/诊断 |
| X2 / Stage 6X | **当前 Avelang BT64 experimental baseline** |

冻结 X2；不立即压缩 `40 KiB` LDS，也不做 alias/reuse、packed-lower 或
recurrence--chunk-o fusion。唯一下一步是 **Stage 6Y updated full-gap audit**：以 X2
的五-dispatch 图重新比较 cumsum、fused KKT+solve、fused W/U、recurrence 和 chunk-o
对 native vLLM 的 standalone body slope、资源、真实 dispatch identity 及 global
intermediate accounting。只有该审计确认最大的可恢复缺口后，才选择一个新的 graph 或
kernel 实验。
# Qwen GDN Next Decision After Stage 6X-KS

X1 与 X2 的 source/body gates、NaN/reuse 和 non-default-stream gates 均通过。X2 在
同一 CTA 内保留 FP32 KKT matrix，直接执行现有 hierarchical FP32 solve，删除 global
FP32 `a` 的 allocation、store、read 和一个 dispatch；它对 Stage6W 为 full bit-exact。

随后补齐的正式 Eager public-API confirmation 使用每个 T 独立进程、5 sessions、50
paired Williams blocks、每实现 300 calls、HIP event 与 wall-clock。X2 对 W1 在
T=512/1024/2048/4096/8192/16384 都稳定更快；关键 gate 为：

| T | X2 相对 W1 paired gain | 95% event CI |
|---:|---:|:---|
| 2048 | +26.020 us | [23.304, 28.563] us |
| 8192 | +82.823 us | [80.432, 85.154] us |
| 16384 | +177.540 us | [176.004, 179.024] us |

W1 的长文本 slope 为 `6.341 us/chunk`，X2 降至 `5.694 us/chunk`，回收约 10.2%。
资源没有 scratch/spill cliff。因此：

```text
X2 / Stage 6X = new Avelang BT64 experimental baseline
W1 / Stage 6W = previous experimental baseline
production/default = unchanged
```

X2 不是所有长度都击败 native vLLM：同口径 Eager 下 X2 在 T<=4096 快于 vLLM，
在 T>=8192 因 vLLM `3.688 us/chunk` 的更低 slope 而落后。故不更改 production
selector，也不从两个点插值得出精确 crossover policy。

## 唯一下一步：Stage 6Y updated full-gap audit

冻结 X2 的五-dispatch 图：

```text
cumsum -> fused KKT+solve -> fused W/U -> recurrence -> chunk-o
```

Stage 6Y 是 measurement-only：重新审计 X2 与 vLLM 的实际 dispatch graph、每个
逻辑 body 的 T/chunk slope、资源/ISA 与所有剩余 global intermediate。它必须先产生
当前图的证据，才可以在以下候选中选择**一个**动作：`a_solved_bf16 -> W/U` 的 source
native handoff，或 immutable recurrence -> chunk-o 边界的后续设计。不得根据旧 Stage
6W 的六-dispatch账本，直接开始 LDS alias、packed lower、recurrence fusion 或其它
kernel 改动。
# Qwen gfx942 BT64 Stage 6Z Z0: Native Chunk-O Specialization Audit

## Scope And Result

Z0 is complete and passes its audit gate. No Avelang chunk-o implementation
was created before the native capture completed. The captured native vLLM
public API uses `chunk_fwd_kernel_o`, but it does not use one immutable
specialization across sequence lengths.

| T | BK | BV | workgroup | stages | grid | CTA | metadata LDS | scratch |
|---:|---:|---:|---:|---:|:---|---:|---:|---:|
| 2048 | 32 | 64 | 256 / 4 waves | 3 | `(2,32,8)` | 512 | 24576 B | 0 B |
| 8192 | 32 | 64 | 128 / 2 waves | 2 | `(2,128,8)` | 2048 | 12288 B | 0 B |

The tile/ownership is stable: one CTA computes `[BT=64, BV=64]` for one
chunk and one value head. There are two CTA per chunk-head because `V=128`.
The operational specialization changes at long sequence: native picks two
waves/two stages for the `T=8192` fresh capture. Per the Stage 6Z rule, the
one allowed Z1 candidate is based on this long-text ownership, not a sweep.

The raw machine-readable record is
[`z0_native_specializations.json`](codex_qwen_bt64_stage6z_native_chunko/z0_native_specializations.json).

## Exact Native Source And Mapping

The captured installed native source is
[`chunk_o.py`](codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/chunk_o.py).
Its launch grid is:

```python
grid = (triton.cdiv(V, BV), NT, B * H)
i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
```

Thus `i_v` selects a V64 block, `i_t` selects a BT64 chunk, and `i_bh`
selects a value head. This proves the CTA interpretation from source, rather
than inferring it from the T=2048 trace's 512 CTA count.

Each CTA keeps two FP32 register accumulator tiles:

```text
b_o [64,64] = q @ h^T
b_A [64,64] = q @ k^T
```

The `K=128` reduction uses four `BK=32` stages. It then applies decay and
the causal lower mask to `b_A`, converts that operand to BF16, performs the
`[64,64] @ [64,64]` score-times-V-new dot, and converts the final FP32 output
to BF16 at the global store. There is no global score tensor and no FP32
output staging.

## What Repeats, And What Does Not

| Item per chunk-head | Stage6W frozen | Native vLLM |
|:--|:--|:--|
| CTA | 8 x V16 | 2 x V64 |
| Q/K score evaluations | 8 V blocks x 4 source V16 tiles | 2 V64 blocks |
| Q/K score repeats | 8x | 2x |
| H elements | V16 partition, each H element once overall | V64 partition, each H element once overall |
| V-new elements | V16 partition, each V-new element once overall | V64 partition, each V-new element once overall |
| score global tensor | none | none |
| output boundary | BF16 direct | BF16 direct |

The key recoverable work is not H/V-new traffic. It is the Stage6W repeated
Q/K score preparation: it evaluates the same token-token score calculation
once for each of eight V16 blocks, while native evaluates it once for each
of two V64 blocks.

## Native LDS And Lifetime Evidence

The external-module rocprof trace reports `LDS_Block_Size=0`; that field is
not trustworthy here. The selected T=2048 TTGIR contains five local
allocations, three local deallocations, and the AMDGCN contains 72 `ds_read`,
40 `ds_write`, and 10 `s_barrier` instructions. Triton metadata records
24576 B of shared memory. T8192 similarly records 12288 B and 56/36
DS read/write instructions.

The important TTGIR ordering is:

1. Allocate and use Q/K/H source shared buffers for the K reduction.
2. Deallocate those source buffers.
3. Allocate BF16 score operand and V-new local buffers for the score-times-V
   dot.

That phase separation avoids O0's unfavorable simultaneous live region of
wide source staging, long-lived score storage, and multiple V16 accumulators.
It is the property Z1 must preserve, not merely its V64 CTA count.

## Captured Resources And ISA

| Item | T2048 | T8192 |
|:--|---:|---:|
| MFMA mnemonic | `v_mfma_f32_32x32x8_bf16` | same |
| static MFMA32 | 40 | 80 |
| static MFMA16 | 0 | 0 |
| buffer load/store | 14 / 4 | 28 / 8 |
| ds read/write | 72 / 40 | 56 / 36 |
| static barriers | 10 | 11 |
| profiler final WG, VGPR, AccVGPR | 256, 100, 36 | 128, 28, 196 |
| scratch | 0 B | 0 B |

The T2048 PMC attempt is retained only as a diagnostic: rocprof caused a new
autotune run and emitted several alternative specializations, so its dynamic
counters are not attributed to the exact Z0 final code object. It remains
under `native/T2048/rocprof_pmc/`, marked non-authoritative rather than
silently merged into this table.

## Z0 Decision

All Z0 prerequisites hold: final code objects and IR exist at both required
lengths; ownership and actual global/LDS lifetime are explicit; the native
schedule needs no compiler or recurrence ABI change; and Avelang has an
already-validated `mfma_32x32x8_bf16_f32` source primitive.

**Decision: GO to one Z1 prototype only.** It will fix `BT64/BV64/BK32`, use
the long-text V64 ownership, and retain a single fixed Avelang `WG=256`
four-wave implementation. It will not revive O0, perform a tile sweep, or
enter the full graph until isolated correctness, resource, and body gates all
pass.

## Evidence Paths

- T2048 selected source/IR/ISA/HSACO:
  `codex_qwen_bt64_stage6z_native_chunko/native/T2048/trace_capture/selected/`
- T8192 selected source/IR/ISA/HSACO:
  `codex_qwen_bt64_stage6z_native_chunko/native/T8192/trace_capture/selected/`
- Native trace captures:
  `codex_qwen_bt64_stage6z_native_chunko/native/T2048/rocprof_trace/` and
  `codex_qwen_bt64_stage6z_native_chunko/native/T8192/rocprof_trace/`
- Reproducible capture script:
  [`capture_qwen_gdn_bt64_stage6z_native_chunko.py`](../../vllm_compare/capture_qwen_gdn_bt64_stage6z_native_chunko.py)
# Qwen gfx942 BT64 Stage 6Z: Native-Style Chunk-O Result

## Final Decision

**No-Go. Stage 6Z stops at Z1.** The new kernel is correct and improves the
isolated body, but its captured ISA contains **41 static `s_barrier`**
instructions. The Stage 6Z hard bound is `barrier < 19`, so Z2 integration,
Eager full benchmarking, promotion, and selector changes are all forbidden.

## Z0 Native Evidence

Z0 captured final native vLLM `chunk_fwd_kernel_o` source, TTIR, TTGIR,
LLVM IR, AMDGCN, HSACO, metadata, and trace in fresh public-API processes.

| native final selection | T2048 | T8192 |
|:--|--:|--:|
| tile | BT64, BV64, BK32 | BT64, BV64, BK32 |
| workgroup / stages | 256 / 3 | 128 / 2 |
| CTA | 512 | 2048 |
| CTA per chunk-head | 2 | 2 |
| source metadata LDS | 24576 B | 12288 B |
| scratch | 0 B | 0 B |
| ISA MFMA | `v_mfma_f32_32x32x8_bf16` | same |

Native source maps `program_id(0/1/2)` to V block, token chunk, and value
head. One CTA owns a `[64,64]` output tile. It evaluates Q/K score twice per
chunk-head for two V64 blocks. Frozen Stage6W evaluates that score eight times
for eight V16 blocks. H and V-new remain partitioned across V, so this does
not claim to remove their traffic.

TTGIR deallocates Q/K/H source buffers before allocating score and V-new
operands. That phase separation was the important design constraint, not CTA
count alone. The detailed audit is
[`qwen_gfx942_bt64_stage6z_z0_native_chunko_audit.md`](qwen_gfx942_bt64_stage6z_z0_native_chunko_audit.md).

## Z1 Implementation

New source:
[`qwen_gdn_bt64_native_chunko_stage6z.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z.py)

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
q/k/v-new/h: BF16; g: FP32; accumulator: FP32; output: BF16
BT64, BV64, BK32, WG256, two CTA per chunk-head
```

The four waves own four `[row32,value32]` output quadrants. One 16 KiB phase
buffer serially holds Q/H staging, then the two score halves, then V-new
transpose. The two score halves are placed in distinct physical buffer halves
so Q/K staging for source-half 1 cannot overwrite source-half 0 score.

Two narrow correctness repairs were made and no schedule sweep occurred:

| issue | repair |
|:--|:--|
| score-owner-only barrier was undefined | every wave stages and joins barriers; owner waves alone update score accumulators |
| second Q/K stage overwrote first score half | second Q/K stage uses the phase-buffer upper half, later reused for V-new |

This avoids O0's long-lived V16 accumulator groups and adds neither a global
score tensor nor FP32 output staging.

## Isolated Correctness

Reference: frozen Stage6W chunk-o at the identical BF16 boundary.

| T | BF16 mismatch elements | max abs | mean abs |
|---:|---:|---:|---:|
| 64 | 1 | 9.313e-10 | 1.421e-14 |
| 128 | 2 | 7.451e-09 | 5.862e-14 |
| 512 | 34 | 1.526e-05 | 1.749e-10 |
| 2048 | 133 | 1.526e-05 | 1.344e-10 |
| 8192 | 306 | 1.526e-05 | 5.953e-11 |

All maxima are below frozen `1/128`. The small mismatch count is MFMA16 versus
MFMA32 reduction-order rounding, not a relaxed threshold. The suite also
passes zero V-new, caller-owned NaN-prefilled output reuse, and invalid FP32
V-new rejection: **7 passed in 10.29 s**.

## Isolated Body Signal

These are caller-owned diagnostics, not formal Eager public ranking. Each
implementation and T ran in its own process with warmup 10 and 100 samples.

| T | Stage6W CTA | Stage6W | Z1 CTA | Z1 | Z1 speedup |
|---:|---:|---:|---:|---:|---:|
| 2048 | 2048 | 0.092237 ms | 512 | 0.075592 ms | 1.220x |
| 8192 | 8192 | 0.252154 ms | 2048 | 0.196411 ms | 1.284x |

The two-point body slope is 1.666 us/chunk for Stage6W and 1.259 us/chunk for
Z1. Thus the ownership change recovers about 0.407 us/chunk. This positive
signal does not override the resource gate.

## Resource Gate

| item, T2048 | Stage6W | Z1 | O0 hard bound | result |
|:--|---:|---:|---:|:--|
| dynamic MFMA | 458752 | 81920 | N/A | reduced |
| dynamic VALU | 12918784 | 6626304 | N/A | reduced |
| dynamic VMEM | 851968 | 540672 | N/A | reduced |
| dynamic LDS instructions | 1343488 | 770048 | N/A | reduced |
| profiler AccVGPR | 64 | 172 | `<188` | pass |
| LDS block | 27136 B | 28672 B | `<33280 B` | pass |
| scratch | 0 B | 0 B | `0 B` | pass |
| code-object spill | 0/0 | 0/0 | `0/0` | pass |
| static MFMA32 | N/A | 32 | N/A | correct geometry |
| static MFMA16 | N/A | 0 | N/A | absent |
| **static barriers** | N/A | **41** | **`<19`** | **FAIL** |

Z1 HSACO has `VGPR=164`, `AGPR=32`, `SGPR=46`, `LDS=28672 B`, private segment
zero, and zero VGPR/SGPR spills. rocprof reports `VGPR_Count=4`, inconsistent
with HSACO, so code-object metadata is authoritative for static VGPR while
rocprof remains the source of collected AccVGPR and PMC values.

The 41 barriers come from safe CTA-wide K32 staging, two score halves, and
V-new transpose. This source schedule cannot reproduce native Triton's 10 to
11 barrier pipeline.

## Stop Record And Reproduction

The following are intentionally absent: Z2 full wrapper, X2 integration,
Eager full benchmark, paired bootstrap, promotion, selector change, and Z1b
tile/workgroup sweep. Stage6W/X2 are unchanged.

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z.py -s
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/vllm_compare/profile_qwen_gdn_bt64_native_chunko_stage6z.py \
  --implementation z1 --T 2048 --warmup 2 --repeat 5 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_stage6z_native_chunko/z1/profile
```

Raw selected native IR/ISA, Z1 HSACO/ISA, body JSON, and rocprof CSV/JSON live
under `codex_qwen_bt64_stage6z_native_chunko/`.
# Stage 6Z Chunk-O 实验复盘

## 结论

本轮没有把 Z1 接入 X2 full graph。不是因为它不正确，也不是因为它 body 不快；它正确且
body 有收益。但 HSACO 生成了 41 个静态 barrier，超过开始前冻结的 19 个上限，所以必须
停止，不能用 isolated 速度绕过资源风险。

## 为什么先做 Z0

Stage6Y 说明长文本 chunk-o 是最大的可恢复差距。旧 Stage6W 的一个 CTA 只处理 V16：

```text
一个 chunk-head 有 8 个 V16 CTA。
每个 CTA 都重新读取 Q，并重新计算 token-token score。
```

因此可以考虑做宽 V ownership，但不能只看到 native CTA 更少就复活 O0。O0 历史上有
AccVGPR 188、LDS 33280 B、19 barrier，收益有限。Z0 先抓真实 native source/IR/ISA。

## Z0 学到的真实结构

native `chunk_fwd_kernel_o` 的真实 ownership：

```text
CTA = [BT64 token, BV64 value, 一个 value head, 一个 chunk]
每个 chunk-head 有 2 个 CTA
BK32，BF16 MFMA32，FP32 accumulator，BF16 直接输出
```

T2048 选择 4-wave/3-stage；T8192 选择 2-wave/2-stage。tile 保持一致，只有 pipeline
参数不同。Z1 遵守规则，只采用长文本的 V64/BK32 ownership，不做两套实现。

关键不是 V64 本身，而是 native TTGIR 的 LDS 生命周期：先放 Q/K/H 做 K reduction，
结束后再放 score 和 V-new。它不会让 source tile、score tile、多个 V accumulator 长期
同时活着。

## Z1 如何实现

四个 wave 覆盖四个 `[row32,value32]` 输出象限。一个 16 KiB phase buffer 被分时复用：

1. Q/H K32 staging，累积 inter-state。
2. source-half 0 score 写入前半 LDS；source-half 1 的 Q/K 使用后半 LDS，避免覆盖。
3. score 完成后，后半 LDS 改装 V-new transpose；最后做 score times V-new。

出现过两次确定性错误，且只修了明确根因：

| 问题 | 原因 | 修复 |
|:--|:--|:--|
| 初始 NaN/大误差 | 只有两个 wave 进入了 workgroup barrier | 所有 wave 均参与 staging/barrier，owner wave 才累积 score |
| intra 大误差 | source-half 1 Q/K 覆盖了 source-half 0 score | second half 改用 phase buffer 上半区 |

没有换 tile、没有换 WG、没有改 compiler、没有加 fallback。

## 结果为什么仍然是 No-Go

数值正确：T64 到 T8192 最大误差最多 `1.526e-5`，小于 `1/128`；zero V-new、output
reuse、invalid dtype 都通过。

body 也正确变快：

| T | Stage6W | Z1 | Z1 加速 |
|---:|---:|---:|---:|
| 2048 | 0.092237 ms | 0.075592 ms | 1.220x |
| 8192 | 0.252154 ms | 0.196411 ms | 1.284x |

但是资源 gate 不是只看 latency：

```text
scratch = 0，spill = 0，AccVGPR = 172，LDS = 28672 B，均通过。
static s_barrier = 41，要求 < 19，失败。
```

所以不创建 Z2 full API，不跑 Eager public 排名，也不改 X2。这个反例说明 CTA 和 MFMA
数量下降不等于 source-level schedule 已经适合 promotion；同步形状同样是硬资源。
# Next Decision After Stage 6Z Native Chunk-O

## Decision

**Close Stage 6Z at Z1. Keep Stage6X X2 as the Avelang BT64 experimental
baseline and keep v24 as production/default.**

Z0 completed its native vLLM audit and Z1 passed isolated correctness plus
showed a positive body effect. However, Z1's captured ISA contains 41 static
`s_barrier` instructions, violating the Stage 6Z hard resource gate
`barrier < 19`. The experiment must not proceed to Z2 full integration.

| gate | result |
|:--|:--|
| Z0 final native specializations captured | pass |
| Z1 BF16 isolated correctness T64 to T8192 | pass, max abs <= 1.526e-5 |
| Z1 zero V-new/reuse/rejection checks | pass |
| Z1 scratch/spill | pass, zero |
| Z1 AccVGPR / LDS | pass, 172 / 28672 B |
| Z1 barrier | **fail, 41 >= 19** |
| Z1 T2048/T8192 isolated body | positive, 1.220x / 1.284x |
| Z2 full graph | not created |

## What The Evidence Means

The native-style V64 ownership is a real source of recoverable work: it
reduces Stage6W's Q/K score repetition from eight V16 blocks per chunk-head
to two V64 blocks and reduces caller-owned body slope by about 0.407
us/chunk in this prototype. But the Avelang Z1 expression implements that
schedule with an unsafe resource shape: source staging and score/V phases
need 41 static barriers. The correct action is to preserve this evidence, not
to accept the body win and create a full graph with an unbounded long-text
risk.

## Explicit Non-Actions

- No Stage6Z Z2 full wrapper or Eager public benchmark.
- No change to X2, Stage6U W/U, immutable recurrence, compiler, or selector.
- No Z1b tile/workgroup sweep and no revival of O0.
- No production/default promotion.

The next optimization must be chosen by a new, separately audited decision.
It cannot be a continuation of this failed barrier envelope under a different
name.
# Qwen gfx942 BT64 Stage 7A: Chunk-O Barrier Provenance And Phase Audit

## 结论

**Case C: 关闭当前 Stage 6Z source-native chunk-o 路线。**

Stage 7A 证明了两件同时成立的事：

1. Z1 的 `41` 个静态 `s_barrier` 不是 AMDGPU backend 额外保守插入的。
   它们在 source 显式 `al.syncthreads()` 的循环展开后已经存在于 pre-link
   LLVM，并且 pre-LTO assembly 与最终 HSACO ISA 都是同一个 `41`。
2. 某些局部 barrier 在最小 repro 中确实可删且 bit-exact；但按规则只做的
   一次完整 Z1 phase-compaction 在 `T=8192` 立刻失去 bit exactness。

因此，不能把最小 repro 的“same-lane fragment”结论直接推广到 full CTA MFMA
pipeline。对**当前** Avelang source schedule 来说，这些 phase boundary 是实际
需要的同步形状。不能继续删 barrier、不能改 tile/WG 扫描、不能接入 full graph。

Stage 6Z 保持 No-Go，Stage6W/X2/v24/default selector 均未改变。

## 冻结边界

本轮没有修改：

- Stage6Z Z1 kernel；
- Stage6W、Stage6X X2、v24、default selector；
- BT64/BV64/BK32、WG256、CTA mapping、MFMA32、dtype 或数学；
- recurrence ABI、full graph、compiler、LLVM/AMDGPU RA、assembly、vLLM source。

唯一新增 full-kernel source 是一个**未晋级的失败实验**：

[`qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage7a_phase_compact.py)

它只删除 lane-private `frag_words` 同步并把 score-half sync 延迟到已有的
V-new producer-to-consumer barrier。它未通过 correctness gate，不能使用。

## 1. 精确 Barrier 链

审计对象是冻结 Z1：

```text
_qwen_gdn_chunk_o_bf16_vnew_bf16_out_kernel_bt64_stage6z_z1
```

生成的四层 artifact：

| 层 | artifact | barrier 数 | 结论 |
|:--|:--|--:|:--|
| AveLang source | `qwen_gdn_bt64_native_chunko_stage6z.py` | 10 个语法 site | 全部是显式 `al.syncthreads()` |
| AveLang IR 语义 | `lib/IR/builtin_module.cc` | 1:1 | `syncthreads()` 直接构造 `gpu::BarrierOp` |
| pre-link LLVM | `z1_prelink.ll` | 41 | `fence release -> llvm.amdgcn.s.barrier -> fence acquire` |
| pre-LTO assembly | `z1_prelink.s` | 41 | 与 LLVM barrier ordinal 相同 |
| final HSACO ISA | `z1_final.isa` | 41 | 逐 ordinal 与 pre-LTO 链对齐 |

因此本轮能严谨地说：**没有看到 compiler/backend 额外创建 barrier。** 它做的是
保留 source barrier 并对 `al.range` 的部分循环展开。

初始 MLIR 也尝试在独立子进程导出，但当前 Docker binding 的 `get_mlir()` 触发
segmentation fault（return code `-11`）。这个调试 API 限制不会影响 LLVM/HSACO 的
确证：同一 AST source 的 LLVM、pre-LTO assembly 与 freshly captured final HSACO 都
完成且数目一致。它意味着 source-line DebugLoc 没有可用的 MLIR 打印 artifact，不能
声称拥有 MLIR source location 到 ISA PC 的逐条 debug metadata 映射。

完整 41 条 PC ledger 在：

- [`z1_barrier_ledger.md`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [`z1_barrier_ledger.json`](codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.json)
- [`audit_summary.json`](codex_qwen_bt64_chunko_barrier_stage7a/audit_summary.json)

ledger 使用已验证的同序 `source schedule -> LLVM call -> assembly -> final ISA`
ordinal 对齐，而不是伪造不存在的 DebugLoc。

## 2. 41 个 Barrier 的 source provenance

Z1 source 有 10 个同步 site。对 `T=2048, WG256` 的 exact specialization，静态
展开/保留结果如下：

| source line | site | static count | dynamic context | hazard | 初始分类 |
|--:|:--|--:|:--|:--|:--|
| 90 | `A.stage_qh` | 4 | 4 个 inter K32 stage | CTA Q/H producer -> MFMA consumer RAW | 必要 |
| 96 | `A.pack_frag` | 8 | 4 stage x 2 kt | `frag_words` pack -> load | 待验证 |
| 103 | `A.reuse_frag` | 8 | 4 stage x 2 kt | MFMA 后下一 fragment reuse | 待验证 |
| 125 | `B.stage_qk` | 2 | 2 个 source half 的 K-stage loop body，各动态 x4 | CTA Q/K producer -> owner MFMA RAW | 必要 |
| 131 | `B.pack_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | `frag_words` pack -> load | 待验证 |
| 139 | `B.reuse_frag` | 4 | 2 half x 2 kt，K-stage body 动态 x4 | MFMA 后 fragment reuse | 待验证 |
| 153 | `B.serialize_score` | 2 | 每个 score half 一次 | score write -> later score/V consume | 待验证 |
| 164 | `C.stage_v` | 1 | V-new transpose | score/V producer -> intra MFMA RAW | 必要 |
| 172 | `C.pack_frag` | 4 | 2 half x 2 kt | score/V fragment pack -> load | 待验证 |
| 179 | `C.reuse_frag` | 4 | 2 half x 2 kt | MFMA 后 fragment reuse | 待验证 |

总数为：`4 + 8 + 8 + 2 + 4 + 4 + 2 + 1 + 4 + 4 = 41`。

这也解释了表面矛盾：source 只有 10 行 barrier，但它不是 10 个静态 ISA barrier。
Phase A 的 `k_stage=4` 被展开；Phase B 保留了动态 K loop body；Phase C 的两个
score half/两个 fragment 被展开。

## 3. 与 Native vLLM 的阶段对齐

native selected `chunk_fwd_kernel_o` 在这次重新读取的 T2048 selected AMDGCN artifact
有 `11` 个 lexical `s_barrier`（此前 Z0 汇总的 `10` 是旧统计口径；Stage 7A 使用同一
selected file直接计数）。它的 Python source 没有 `tl.barrier()`；这些 barrier 是 Triton
local-memory/dot pipeline lowering 的结果，不能对 Python 行号做虚假的一对一归因。

| native phase | native source | Z1 对应 phase | 核心差异 |
|:--|:--|:--|:--|
| Q/K/H K32 load + two dots | `chunk_o.py:93-113` | A + B 的 source stage/fragment sequence | native 用 compiler-managed local operand pipeline；Z1 手动把 fragment 反复存入/读出 CTA LDS |
| decay/mask/score BF16 operand | `115-125` | B score serialize | native score 保持 local dot operand；Z1 将两个 score half 显式序列化到 `phase` |
| V load + score-times-V + BF16 store | `127-138` | C V transpose/intra | native TTGIR 释放 Q/K/H local buffers 后再分配 score/V local buffers；Z1 以 CTA-wide phase boundaries 保护共享复用 |

native T2048 TTGIR 有 5 个 local allocation、3 个 local deallocation；这就是它能以
11 个 barrier 完成多阶段局部 pipeline 的直接 evidence。Z1 的 41 个 barrier 不能仅用
“所有权变成 BV64”消掉。

## 4. 最小 Repro

新增 Qwen-free repro：

- [`repro_qwen_bt64_chunko_barrier_stage7a.py`](repro_qwen_bt64_chunko_barrier_stage7a.py)
- [`profile_qwen_bt64_chunko_barrier_stage7a.py`](profile_qwen_bt64_chunko_barrier_stage7a.py)

所有模式都是 WG256、shared BF16、MFMA32（A/C）且 zero scratch/spill。

| experiment | comparison | barrier | output | 解释 |
|:--|:--|--:|:--|:--|
| A | per-lane fragment pack without/with extra barrier | 1 / 2 | bit-exact | 单独的 lane-private write 后 barrier 可去掉 |
| B | score lower half write, then disjoint upper half write | 2 / 1 | bit-exact | 在没有中间 consumer 时，可合并为最终 consumer 前一条 barrier |
| C | all CTA stage, one owner wave MFMA | 1 | finite | owner wave 不意味着 source stage barrier 可以去掉 |

原始 JSON/ISA/readobj：

- [`minimal_repros.md`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)
- [`minimal_repros.json`](codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.json)

这些 repro 的价值是限定因果：它们证明 A/B 的 barrier 在**最小独立内存关系**中不是
必需的；它们没有证明完整 MFMA pipeline 在不同 wave 进度、反复 MFMA issue 与 LDS reuse
下也可安全删除。

## 5. 唯一允许的 Local Fix 与失败

依据 A/B，实施了唯一一个 phase-scheduling experiment：

```text
删除 A/B/C 的 frag_words pack/reuse barriers
删除两个 B.serialize_score barriers
保留 A.stage_qh、B.stage_qk、C.stage_v
```

理论上它会让静态 barrier 从 `41` 降至 `7`，且没有改动 tile、WG、MFMA、layout、dtype 或
math。该候选第一个完整 Z1 correctness case（T8192）即失败：

| comparison | result |
|:--|:--|
| phase-compact vs frozen Z1 | `bit_exact = false` |
| max abs | `0.00206613541` |
| gate | 失败，要求 bit-exact |

在该失配 kernel 后同一 pytest process 继续编译下一 specialization 时，HIP report 了
memory access fault 并 abort。该 abort 不用于归因；唯一可靠的 stop fact 是更早出现的
T8192 numerical mismatch。没有继续收集该无效候选的 body、rocprof 或 full graph 数据。

为什么最小 repro 不足以放行？最可能是完整 kernel 中 `frag_words` 的 reuse 不只是普通
“同一 lane store 后同一 lane load”：它夹在跨 wave 的 LDS source read、MFMA issue、下一
round LDS overwrite 和非锁步 wave progress 中。一个 barrier 可能同时充当 schedule-wide
phase boundary。当前一次修复同时移除了多类同步，Stage 7A 规则禁止再逐个恢复/扫组合，
所以不能把责任精确归给某一条 barrier。

## 决策

这不是 Case B：LLVM/pre-LTO/final ISA 都未显示额外 compiler-inserted barrier；问题不是
generic backend hazard analysis 平白增加了同步。

也不是可以继续的 Case A：唯一允许的 source compaction 未通过 full Z1 exactness。

所以是 **Case C for the current Avelang source schedule**。停止 Stage 6Z pure-source
native chunk-o 路线，不做 barrier subset sweep、tile/WG sweep、Z2 或 full integration。

后续如果追求端到端性能，下一条独立路线可以是已捕获 native `chunk_fwd_kernel_o` HSACO 的
external-kernel bridge，并明确标记 external integration；它不是 Avelang source kernel
优化。若坚持纯 Avelang source，按此前 Stage 6Z 排序转向 W/U runner-up gap，预期收益较小。

## 复现

```bash
cd /workspace/project/avelang

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/audit_qwen_bt64_chunko_barrier_stage7a.py \
  --T 2048 \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a

PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_bt64_chunko_barrier_stage7a.py \
  --out-dir test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a \
  --warmup 5 --repeat 20
```

The phase-compact test is intentionally skipped in ordinary collection because
its first full-kernel exactness gate already failed; the source is retained as
a documented failed experiment rather than a candidate baseline.
# Qwen GDN Next Decision After Stage 7A Chunk-O Barrier Audit

## Decision

**Close the Stage 6Z pure-Avelang native chunk-o source route.** Do not create
Z2, do not run full-graph timing, and do not promote the Stage 7A
phase-compact candidate.

The frozen Stage6Z Z1 kernel has 41 static barriers. Stage 7A established an
exact count-preserving chain:

```text
explicit al.syncthreads source schedule
  -> 41 pre-link LLVM barrier calls
  -> 41 pre-LTO assembly barriers
  -> 41 final HSACO s_barrier instructions
```

The backend did not create a hidden surplus of barriers. A one permitted
source scheduling change removed the candidate lane-private and early
score-half barriers, but failed the first full-Z1 exactness case at T8192:

```text
phase-compact vs Z1: max_abs=0.00206613541, bit_exact=false
```

The standalone barrier repro remains valuable evidence: a local per-lane
fragment barrier and a disjoint score/V store barrier can each be removed in
isolation. The full pipeline disproves treating those local facts as a global
license to erase the phase boundaries.

## What Remains Frozen

- Stage6W/X2 and v24/default selector;
- Stage6Z Z1 source and its No-Go status;
- BT64/BV64/BK32/WG256 mapping;
- recurrence ABI and all full paths;
- compiler, LLVM/AMDGPU RA, assembly, and vLLM source.

## Next Single Direction

Choose one independent objective, not both:

1. **End-to-end diagnostic:** integrate the captured native vLLM `chunk-o`
   HSACO behind a strict external-kernel bridge and measure the recoverable
   full-graph gap. It must be labeled external integration, not an Avelang
   source-kernel result.
2. **Pure Avelang source:** leave chunk-o closed and move to the previously
   ranked W/U runner-up gap. Its expected upside is smaller than native
   chunk-o replacement.

Do not reopen Stage6Z with barrier subset sweeps, alternate tile/WG shapes,
or a full-graph "just to see" integration.

## Evidence

- [Stage 7A report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_chunko_barrier_provenance_stage7a_report.md)
- [41-entry barrier ledger](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [minimal repro results](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)

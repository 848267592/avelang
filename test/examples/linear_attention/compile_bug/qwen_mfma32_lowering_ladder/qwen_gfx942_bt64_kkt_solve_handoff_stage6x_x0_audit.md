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

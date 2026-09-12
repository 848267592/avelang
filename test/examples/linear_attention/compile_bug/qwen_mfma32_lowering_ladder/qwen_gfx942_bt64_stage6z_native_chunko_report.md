# Qwen gfx942 BT64 Stage 6Z: Native-Style Chunk-O Result

## Later Addendum: C19-FPRO

本文件前面的 Z0/Z1/Z2 结论是当时的 Stage 6Z historical snapshot，其中的
“停止在 Z1/Z2、不要进入下一层”不覆盖后续已经完成的 C18/C19 compiler
ownership 路线。最新的 full-region 状态以
[`qwen_gfx942_c19_full_physical_region_ownership_completion.md`](qwen_gfx942_c19_full_physical_region_ownership_completion.md)
为准。

C19-FPRO 已完成一次性的 `FullPhysicalRegionPlan` ownership completion：Q/H/K/V
producer、shared placement、Q@H/Q@K/score@V consumer 和 actual shared allocation
均进入同一个 experimental compiler plan。C19 T=2048 的诊断 PMC 为
`MFMA/VMEM/LDS/VALU/SALU = 160/304/688/6418/610 per CTA`，actual LDS 为
`24576 B`，无 private/spill；T=64/128/512/1024/2048/4096/8192/16384 correctness
和 T=64/8192/16384 caller-owned zero-V/NaN-prefill 均通过。

因此当前路线状态是：

```text
Z5B = frozen historical isolated performance baseline
C18 = machine-distinct but incomplete ownership, No-Go
C19-FPRO = full ownership/allocation/correctness PASS,
            GO_FOR_PERFORMANCE，尚未运行正式性能或 production
```

本 addendum 不改写前面历史实验数据，也不把 C19 的 diagnostic trace 当作正式
body/Eager latency。C19 只开放下一轮正式 benchmark，不开放 selector 或 production。

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

## Stage 6Z fixed-source re-evaluation: current valid ranking

The historical Z2 timing/PMC above predates the Phase-B dead K overread fix and
must not be used as the current performance result. The repaired Z2 and Z3 were
recompiled and measured in
`codex_qwen_bt64_stage6z_fixed_rerun/`.

Correctness passed for both repaired arms at T64/512/1024/2048/4096/8192/16384;
Z3 was BF16 byte-exact with Z2, and both stayed below the frozen `1/128` Stage6W
contract. T64/8192/16384 zero-V NaN-prefilled caller-owned output checks passed.
Both code objects have private segment 0 and zero VGPR/SGPR spills. The new
static barrier counts are Z2=9 and Z3=21; Z3 therefore fails the pre-registered
`barrier < 19` gate.

| T | fixed Z2 ms | fixed Z3 ms | native selected direct-body ms | Z2/native | Z3/native |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.061431 | 0.074491 | 0.036194 | 1.697x | 2.058x |
| 1024 | 0.063634 | 0.078456 | 0.037756 | 1.685x | 2.078x |
| 2048 | 0.077475 | 0.094661 | 0.042763 | 1.812x | 2.214x |
| 4096 | 0.110724 | 0.126388 | 0.056224 | 1.969x | 2.248x |
| 8192 | 0.177203 | 0.220769 | 0.091917 | 1.928x | 2.402x |
| 16384 | 0.317692 | 0.393445 | 0.141591 | 2.244x | 2.779x |

The repaired Z2 is the current valid Avelang isolated baseline. Z3 is slower
than Z2 at every measured length and has both a barrier-gate failure and a
higher profiler AccVGPR (164 vs 32). Native selected chunk-o remains the
fastest diagnostic body, but no production selector or X2 integration was
changed. The final Stage 6Z decision is therefore: **fixed Z2 is the current
Avelang research baseline; fixed Z3 is a correctness-passing but performance
No-Go candidate; old pre-fix Z2 numbers are history only.**

## Stage 6Z fixed Z2 vs native WG256 exact same-shape machine-gap audit

The detailed Chinese audit is
`qwen_gfx942_bt64_stage6z_fixed_z2_vs_native_machine_gap_audit.md`.
This is a read-only audit: no kernel source, X2 graph, selector, allocator,
RA, recurrence HSACO, or Z4 candidate changed.

The audit deliberately re-captured native public selection at T=2048 and
pinned the selected `BK=32, BV=64, num_warps=4, num_stages=2` configuration
before its direct body measurement. It therefore supersedes the historical
native WG128/direct-wrapper and autotune-contaminated PMC numbers.

| T=2048 fresh body | median | note |
|:--|--:|:--|
| repaired fixed Z2 | `0.0783565 ms` | 5 fresh processes, HIP event |
| native selected WG256 | `0.0426830 ms` | same shape, same stream |
| fixed Z2 / native | `1.8358x` | paired session median |
| paired difference | `35.6735 us` | Z2 minus native |

The clean WG256 PMC capture has 512 CTAs. Per CTA, MFMA is identical at 160,
while Z2/native is VMEM `928/140`, LDS `928/480`, VALU `11400/3376`, and SALU
`1072/660`. Both code objects have zero private segment and zero spill. The
machine-gap report therefore attributes the remaining gap primarily to Z2's
source-visible Q/K/V phase materialization, shared-memory round trips, and
address/layout feeding, rather than MFMA count, scratch, or RA spill.

The only next candidate registered by this audit is a matched typed
Phase-B/C producer-consumer materialization experiment with WG256/BV64/BK32
and MFMA32 frozen. It is not implemented here and is not a Z4 promotion.

## Shape guard follow-up

The Z2 source now contains an explicit `Z2_WORKGROUP_CONTRACT=256` guard and
its host launch remains fixed at `(256, 1, 1)`. The unsafe mixed-benchmark
native direct call that could silently let Triton choose WG128 was removed;
native comparisons now use the fresh-public WG256 config pin helper. The Z2
report records the root cause, the invalid historical path, the exact command
rule for later agents, and the post-fix T=64 smoke evidence. WG128 remains only
the separate Z3 source candidate or a native long-text selector result.

## Stage 6Z fixed Z2 VMEM provenance audit

The read-only per-operand audit is recorded in
`qwen_gfx942_bt64_stage6z_fixed_z2_vmem_provenance_audit.md`. It uses the
fixed T=2048 same-shape WG256 artifacts and does not modify kernel source,
implement Z4, or connect X2.

The audit separates three quantities that must not be conflated:

```text
static lexical ISA instruction count
dynamic PMC instruction count
logical operand bytes / memory transaction bytes
```

The fixed Z2/native dynamic PMC remains `VMEM=928/140 per CTA`, with equal
`MFMA=160/CTA`. The final ISA shows fixed Z2 BF16 input sites as scalar
`global_load_ushort` and native typed BF16 sites as `buffer_load_dwordx4`.
The LLVM/source mapping proves that fixed Z2 reloads the logical Q tile in
Phase A and again in Phase B, with Phase B repeated for both `source_half`
values. Native TTGIR keeps one typed Q operand available to both the `Q*H` and
`Q*K` dot consumers in the same loop body. This is the strongest presently
proven excess-VMEM dataflow.

K scalarization and V_new narrow loads are also real, but the current artifacts
do not provide a per-operand dynamic transaction-byte counter. Therefore the
audit does not claim that K or V_new contributes more than Q, and it does not
convert `928-140` into bytes.

The single next candidate is consequently **Q Phase-A/Phase-B tile residency**:
keep the scaled Q producer available for both score halves and the inter-state
consumer, while freezing WG256/BV64/BK32/MFMA32/math/ABI. This is registered
only; no implementation was made in Stage 6Z.

## Stage 6Z Z4 Q-producer residency ladder

The registered Q-residency candidate was tested as three independent arms from
the repaired fixed-Z2 source. The complete experiment is recorded in
[`qwen_gfx942_bt64_stage6z_z4_q_residency_ladder.md`](qwen_gfx942_bt64_stage6z_z4_q_residency_ladder.md).
This section supersedes the earlier statement that Q residency was only a
future candidate, while preserving the Stage 6Z hard stop: no arm was connected
to X2, a selector, production dispatch, recurrence HSACO, allocator, or RA.

### Frozen scope

All four arms use `BT64/BV64/BK32/WG256`, two CTAs per chunk-head, MFMA32,
the fixed BF16 ABI, the same K32 reduction order, output contract and
mathematics. K/H/V-new/g/output paths were not changed. The only intended
source differences are Q load width or Q producer lifetime:

| arm | intended change | Q producer passes |
|:--|:--|--:|
| fixed Z2 | repaired baseline | 3 |
| Z4A | legal vector-Q packet only | 3 |
| Z4B | Phase-A Q slice also feeds score half 0; half 1 reloads | 2 |
| Z4C | one K32 loop feeds inter, score half 0 and score half 1 | 1 |

Z4B and Z4C retain the scalar source Q width. Z4C does not add a second full
Q LDS buffer, private Q array or double buffer. It intentionally exposes the
three-accumulator live-range tradeoff.

### Correctness and machine identity

The fresh Docker correctness suite passed **31 tests**. Z4A, Z4B and Z4C are
BF16 byte-exact with fixed Z2 for `T=64/512/1024/2048/4096/8192/16384`, pass
finite checks, and pass the caller-owned NaN-prefilled zero-V checks at
`T=64/8192/16384`. No arm was stopped by correctness.

Each arm produced a distinct code object and all had zero private segment and
zero VGPR/SGPR spills:

| arm | HSACO SHA256 (prefix) | VGPR | AGPR | SGPR | LDS |
|:--|:--|--:|--:|--:|--:|
| fixed Z2 | `2afaa8867da65642...` | 168 | 32 | 44 | 16384 B |
| Z4A | `44c958d03d169da3...` | 92 | 16 | 32 | 16384 B |
| Z4B | `6998fc3e16d37e1...` | 188 | 48 | 42 | 16384 B |
| Z4C | `f9d12fbce1d82be7...` | 220 | 64 | 36 | 16384 B |

The Z4A typed packet survived to ISA (`global_load_dwordx4` and
`ds_write_b128` are present), so it was not silently folded back to fixed Z2.
The Z4B/Z4C scalar Q arms also remain distinct machine graphs. The capture
driver could not emit the initial high-level MLIR because the existing Docker
MLIR printer segfaulted; no missing MLIR artifact is presented as evidence.
Lowered LLVM, pre-LTO AMDGCN, exact-LTO MIR, final ISA, HSACO and readobj
metadata are under:

```text
codex_qwen_bt64_stage6z_z4_q_machine/{z2,z4a,z4b,z4c}/
```

### Static and dynamic work

Static ISA counts and dynamic PMC counts are separate measurements. Dynamic
counts below are fresh T=2048 rocprof data divided by 512 CTAs, not inferred
from lexical ISA counts:

| arm | MFMA/CTA | VMEM/CTA | LDS/CTA | VALU/CTA | SALU/CTA | static barrier |
|:--|--:|--:|--:|--:|--:|--:|
| fixed Z2 | 160 | 928 | 928 | 11400 | 1072 | 9 |
| Z4A | 160 | 3504 | 480 | 17684 | 7464 | 27 |
| Z4B | 160 | 800 | 800 | 9746 | 964 | 9 |
| Z4C | 160 | 672 | 672 | 7514 | 778 | 8 |

Z4A's packet load reduced scalar Q load sites in the static ISA but produced
more dynamic VMEM and much more address/layout VALU/SALU in this AveLang
lowering. Z4B removed one Q producer pass and Z4C removed two; their dynamic
VMEM/LDS/VALU/SALU reductions confirm that repeated Q production is real work.
The unchanged MFMA count proves these changes did not win by deleting math.
Z4C nevertheless raises code-object VGPR/AGPR relative to Z2, which is the
machine signature expected from keeping `inter`, `score_half0` and
`score_half1` live in one K32 loop.

### Fresh body timing

Caller-owned isolated bodies used HIP events, the current stream, no Graph,
warmup 10, repeat 50 and five independent processes per arm. These are body
diagnostics, not Eager public-API promotion data:

| arm | T=2048 median | paired delta vs Z2 | T=8192 median | paired delta vs Z2 |
|:--|--:|--:|--:|--:|
| fixed Z2 | 0.080500 ms | baseline | 0.177483 ms | baseline |
| Z4A | 0.111245 ms | -30.785 us (-38.2%) | not run | T=2048 failed |
| Z4B | 0.079438 ms | +0.781 us (+1.3%) | 0.192466 ms | -14.802 us (-8.4%) |
| Z4C | 0.078196 ms | +3.105 us (+2.9%) | 0.192726 ms | -15.322 us (-8.6%) |

Z4C's five T=2048 paired samples were positive, but that short-length gain
does not survive T=8192. The two-point diagnostic slopes from T=2048 to
T=8192 are approximately `1.010 us/chunk` for fixed Z2, `1.177 us/chunk`
for Z4B and `1.193 us/chunk` for Z4C.

### Stage 6Z decision after Z4

The ladder answers the three registered questions:

1. **Q scalar load width:** Z4A did not provide a benefit. A legal vector
   packet reached ISA, but the surrounding generic lowering caused a large
   dynamic work increase. This is not evidence that the hardware rejects
   vector loads; it is evidence that this source/lowering form is not a
   profitable isolated control.
2. **Q pass 3 to 2:** Z4B lowers dynamic VMEM from `928` to `800` per CTA and
   reduces other dynamic work, but the latency result is only marginal at
   T=2048 and regresses at T=8192.
3. **Q pass 3 to 1:** Z4C lowers dynamic VMEM to `672` per CTA and gives a
   small T=2048 gain, but the longer live accumulator region and increased
   resource footprint erase the benefit at long text.

No Z4 arm simultaneously reduced machine work, improved T=2048 robustly and
avoided the T=8192 regression. Therefore the current Pareto choice remains
**repaired fixed Z2**. Z4C is diagnostic-only and is not a new baseline. The
single registered next action is a read-only live-range/provenance audit of
Z4C's three accumulators; no Z4-derived optimization, X2 integration, selector
or production change is authorized by this result.

Raw Z4 artifacts are located at:

```text
codex_qwen_bt64_stage6z_z4_q_machine/
codex_qwen_bt64_stage6z_z4_q_pmc_T2048/
codex_qwen_bt64_stage6z_z4_q_benchmark_T2048/
codex_qwen_bt64_stage6z_z4_q_benchmark_T8192/
```

Reproduction entry points are:

```text
vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py
vllm_compare/test_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py
vllm_compare/bench_qwen_gdn_bt64_native_chunko_stage6z_z4_q_ladder.py
vllm_compare/dump_qwen_gdn_bt64_stage6z_z4_q_machine_artifacts.py
vllm_compare/profile_qwen_gdn_bt64_native_chunko_stage6z_z4_q.py
```

## Stage 6Z Z4C accumulator live-range audit closure

The read-only exact-LTO audit is now complete. It is recorded in
[`qwen_gfx942_bt64_stage6z_z4c_accumulator_live_range_audit.md`](qwen_gfx942_bt64_stage6z_z4c_accumulator_live_range_audit.md).

The conclusion is **Case A**: Z4C's resource growth is mainly explained by
`inter_acc`, `score_acc0` and `score_acc1` being initialized before the shared
K32 loop and kept as three distinct `areg_512_align2` accumulator families.
The exact post-RA MFMA destinations are:

```text
inter_acc   -> $agpr32..47
score_acc0  -> $agpr16..31
score_acc1  -> $agpr0..15
```

fixed Z2 reuses `$agpr0..31` across phase-separated inter/score/intra regions.
Z4C's additional `$agpr48..63` range is a fragment/copy bridge accompanying the
overlap, not a fourth mathematical accumulator. Both arms remain at 20 static
MFMA32 instructions, 16 KiB LDS, zero private segment and zero spills. The
profiler `Accum_VGPR=32 -> 52` is recorded separately from code-object
`AGPR=32 -> 64`; these fields are correlated resource observations, not a
one-to-one vreg mapping.

The registered action was implemented and is recorded in
[`qwen_gfx942_bt64_stage6z_z5a_dedicated_q_lds.md`](qwen_gfx942_bt64_stage6z_z5a_dedicated_q_lds.md).
Z5A reduces the source Q producer pass from 3 to 1, keeps dynamic MFMA at
160/CTA, reduces dynamic VMEM from 928 to 672/CTA, and passes the full
correctness matrix. Its fresh-process body result is positive at both T=2048
and T=8192, so it is now the **isolated Stage 6Z research baseline**. It is
not a production baseline: it has 32 KiB LDS, higher Accum_VGPR, a larger
lexical ISA graph, and has not been connected to X2 or the Eager public API.
Z4C remains diagnostic-only; fixed Z2 remains the production-style Stage 6Z
reference.

## Stage 6Z Z5A dedicated full-Q LDS cache

The complete implementation, correctness evidence, exact-LTO artifacts, dynamic
PMC, paired T=2048/T=8192 benchmark and decision are recorded in
[`qwen_gfx942_bt64_stage6z_z5a_dedicated_q_lds.md`](qwen_gfx942_bt64_stage6z_z5a_dedicated_q_lds.md).

The final Z5A comparison is:

| metric | fixed Z2 | Z5A |
|:--|--:|--:|
| Q source producer pass | 3 | 1 |
| LDS block | 16,384 B | 32,768 B |
| dynamic MFMA/CTA | 160 | 160 |
| dynamic VMEM/CTA | 928 | 672 |
| dynamic LDS/CTA | 928 | 1,440 |
| dynamic VALU/CTA | 11,400 | 7,136 |
| dynamic SALU/CTA | 1,072 | 768 |
| T=2048 body | 0.078056 ms | 0.067720 ms |
| T=8192 body | 0.176743 ms | 0.161119 ms |

Z5A is a research-only candidate. No selector, X2 integration, recurrence
HSACO replacement, allocator/RA change or production dispatch change was made.

## Stage 6Z Z5B direct Q-cache consumer

Z5B is the direct-consumer continuation of Z5A. The full implementation and
evidence are recorded in
[`qwen_gfx942_bt64_stage6z_z5b_direct_q_cache_consumer.md`](qwen_gfx942_bt64_stage6z_z5b_direct_q_cache_consumer.md).

The only source change was to remove the per-K32 Q cache -> old phase-Q
republish. Phase A/B reads now use the persistent Q shared cache directly;
H/K/V-new still use the old phase area, and the Z5A phase-separated
accumulator order remains unchanged.

| metric | fixed Z2 | Z5A | Z5B | native WG256 diagnostic |
|:--|--:|--:|--:|--:|
| Q global producer pass | 3 | 1 | 1 | native implementation |
| Q republish | baseline | yes | no | no comparable AveLang phase |
| LDS allocation | 16,384 B | 32,768 B | 32,768 B | collector metadata 0; dynamic LDS present |
| dynamic MFMA/CTA | 160 | 160 | 160 | 160 |
| dynamic VMEM/CTA | 928 | 672 | 672 | 140 |
| dynamic LDS/CTA | 928 | 1,440 | 672 | 480 |
| dynamic VALU/CTA | 11,400 | 7,136 | 7,072 | 3,376 |
| dynamic SALU/CTA | 1,072 | 768 | 768 | 660 |
| T=2048 body | 0.077555 ms | 0.066138 ms | 0.065618 ms | 0.042643 ms |
| T=8192 body | 0.177143 ms | 0.160178 ms | 0.156473 ms | 0.090755 ms |

Z5B passed the full T=64/512/1024/2048/4096/8192/16384 BF16 byte-exact
correctness matrix, including caller-owned zero-V/NaN-prefill checks. Its
T=2048 paired result was faster than Z5A in 4/5 sessions, and its T=8192
result was faster in 5/5 sessions. The endpoint slope improved from about
`0.980` to `0.946 us/chunk`.

T=2048 的五个 session 差值均值约 `-0.625 us`，小样本 cluster bootstrap
95% CI 为约 `[-1.887, +0.441] us`；T=8192 的五个差值均值约 `-4.362 us`，
CI 为约 `[-5.620, -3.461] us`。因此 T=2048 的收益是方向性小收益，T=8192
的收益更稳定，报告不把 T=2048 的小差值写成高置信度结论。

The Z5B machine object has a different SHA256 from Z5A, zero private segment,
zero VGPR/SGPR spills, and unchanged code-object AGPR=32. Static ISA
`ds_read/ds_write` drops from `152/240` to `56/144`; dynamic MFMA remains
160/CTA. This is a genuine machine-graph change, not a timing-only alias.

**Current Stage 6Z isolated baseline: Z5B.** It is still research-only and
must not be connected to X2 or production. It remains about `1.54x` native at
T=2048 and `1.72x` native at T=8192. The remaining evidence-backed gap is
mostly VMEM and VALU (`672/7072` per CTA versus native `140/3376`), so the
next investigation, if authorized, must audit the remaining operand dataflow
rather than revisit Q republish.

## Stage 6Z Z5B remaining VMEM operand ledger

本轮不是重新审计 Z5A/Z5B 的资源差异，也没有修改 kernel。它只补完
此前 provenance audit 中仍 unresolved 的 `672 VMEM/CTA` operand attribution，
并与同形状 native WG256 的 `140 VMEM/CTA` 对齐。Z5B 的 baseline、性能、
correctness 和资源结论全部冻结。

新报告为：

`qwen_gfx942_bt64_stage6z_z5b_remaining_vmem_operand_ledger.md`

关键冻结数据：

| 指标 | Z5B | native WG256 |
|:--|--:|--:|
| dynamic MFMA/CTA | 160 | 160 |
| dynamic VMEM/CTA | 672 | 140 |
| dynamic LDS/CTA | 672 | 480 |
| dynamic VALU/CTA | 7,072 | 3,376 |
| dynamic SALU/CTA | 768 | 660 |

source、lowered LLVM、ISA 和 loop ownership 可以确认 Z5B 的 Q duplicate 已由
Z5B 消除：Q pointer 只有一个 global producer region，Phase A/B 从 dedicated
Q cache 读取。剩余 Q 成本是一次 scalar/narrow Q-fill，而不是三次 global reload。
K、H、V-new 仍存在 typed packet/layout 差距，但现有工件不能证明它们有比 g
更大的重复 producer。

本轮唯一证据支持的最大剩余 offender 是 **g target/source/final residency**：
score target/source 和 final scaling 具有多个 logical consumer role，source/LLVM
可建模约 `192` 个 FP32 issuing-load opportunities，并伴随 address/index VALU；
native 则以 block g tile 给多个 consumer 复用。下一候选只登记为设计，不实现：
`Z6G typed FP32 g-tile residency`。本轮不接入 X2、selector 或 production。

## Stage 6Z Z6G g residency stable/ideal

Z6G 已按上一轮 ledger 登记的唯一候选完成两个独立 arm：

- `Z6G-S stable`：64-token FP32 g tile 写入 256 B CTA-local shared cache；
- `Z6G-I ideal`：每 token 用合法 `raw_buffer_load_x4` 形成连续四-head packet，
  选出当前 head 后写入同一 g cache。

完整中文报告为
[`qwen_gfx942_bt64_stage6z_z6g_g_residency_stable_vs_ideal.md`](qwen_gfx942_bt64_stage6z_z6g_g_residency_stable_vs_ideal.md)。
源码为
`vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z6g_g_residency.py`。

本轮没有设置资源 No-Go gate。两条 arm 都先完成 correctness，再进入实际
性能测试；没有修改 X2、selector、production、RA 或 recurrence HSACO。

### Correctness

Z6G-S/Z6G-I 相对 Z5B 的 T=64/512/1024/2048/4096/8192/16384 BF16 byte-exact、
finite 测试全部通过；T=64/8192/16384 caller-owned output、zero-V-new、
NaN-prefill 测试全部通过。

### 同口径 body latency

| T | Z5B | Z6G-S | Z6G-I | native WG256 |
|--:|--:|--:|--:|--:|
| 2048 | 0.065618 ms | 0.068422 ms | 0.072968 ms | 0.042464 ms |
| 8192 | 0.157354 ms | 0.168230 ms | 0.183172 ms | 0.090835 ms |
| 16384 | 0.274929 ms | 0.306896 ms | 0.337021 ms | 0.141610 ms |

T=2048/8192 使用 7 个 fresh-process paired sessions，T=16384 使用 5 个。
Z6G-S 相对 Z5B 的 paired difference 分别为 `+3.165/+11.045/+30.950 us`；
Z6G-I 分别为 `+7.253/+26.222/+61.295 us`。三个长度上两个 arm 都是稳定
回退，故 Z5B 保持 isolated baseline。

### T=2048 dynamic PMC（每 CTA）

| arm | MFMA | VMEM | LDS | VALU | SALU | profiler Accum_VGPR | Occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Z5B | 160 | 672 | 672 | 7072 | 768 | 100 | 14.540% |
| Z6G-S | 160 | 469 | 725 | 6003 | 757 | 184 | 8.222% |
| Z6G-I | 160 | 532 | 725 | 6106 | 911 | 188 | 8.326% |
| native WG256 | 160 | 140 | 480 | 3376 | 660 | 36 | 9.511% |

Z6G 确实减少了动态 VMEM，但 shared g cache 增加一条 barrier、约 256 B
shared allocation、LDS read/write 和更长的跨 phase residency；profiler
Accum_VGPR/occupancy 也明显恶化。Z6G-I 还要为当前 head 实际不需要的三个
相邻 head 携带 x4 packet，因此比 stable 更慢。动态 MFMA 数学工作保持
`160/CTA` 不变。

### Machine identity

两条 Z6G arm 都成功导出了 lowered LLVM、exact-LTO MIR、final ISA 和 HSACO，
并且 HSACO hash 与 Z5B 不同。Z6G-S/I 的 final ISA 都是 56 条静态 MFMA32、
33 条 `s_barrier`、89 条 `ds_read`、145 条 `ds_write`；Z6G-S 有 113 条
`global_load_*`，Z6G-I 有 112 条 `global_load_*` 加一条
`buffer_load_dwordx4`。对应 Z5B 为 56/32/56/144/192。

Z6G-S HSACO：
`851043fbafc58a29a61d7b7b5d860e0db0aaa8569baad4126bf289bef8237ad4`。

Z6G-I HSACO：
`6914616828369f0881127d47900372d80519050ae4e20ef4b11e52a3cc1a0ca8`。

Z6G machine artifacts 位于：

```text
codex_qwen_bt64_stage6z_z6g/machine/z6g_s/
codex_qwen_bt64_stage6z_z6g/machine/z6g_i/
```

initial MLIR 的直接 `get_mlir()` probe 触发后端 native segfault；跳过该调试
打印后，LLVM、LTO、MIR、ISA、HSACO 和运行测试均成功。报告明确保留这个工具
边界，没有把 initial MLIR 缺失伪装成已捕获。

### Stage 6Z 决策更新

Z6G 的 source/LLVM 证据证明 g global producer 已收敛到单一 cache fill，
score target/source/final consumer 均从 addrspace(3) g cache 取得；但是减少
VMEM 没有转化成 latency/slope 收益。最终选择：

```text
Z5B 继续作为当前 Stage 6Z isolated research baseline
Z6G-S 不晋级
Z6G-I 不晋级
不接入 X2、selector 或 production
```

下一轮不应继续叠加 g cache 变体。剩余差距仍由完整 W/K/H/V-new operand
feeding、LDS layout、ownership 和 phase schedule 的组合构成，而不是单独
一个尚未缓存的 g scalar。

## Stage 6Z BF16 operand-feeding/layout feasibility audit

Z6G 之后又完成了一个只读的 BF16 operand feeding 审计，正式报告为
[`qwen_gfx942_bt64_stage6z_bf16_operand_feeding_layout_feasibility_audit.md`](qwen_gfx942_bt64_stage6z_bf16_operand_feeding_layout_feasibility_audit.md)。
本轮没有修改 source、compiler、lowering、allocator/RA、X2、selector 或
production，也没有重跑旧 benchmark。

审计固定比较 Z5B 与 native selected chunk-o 的 Q/K/H/V-new 路径。两边的
dynamic MFMA 都是 `160/CTA`，所以差距不是 MFMA 数量；已有 T=2048 PMC 为：

| 指标 | Z5B | native WG256 diagnostic |
|:--|--:|--:|
| VMEM/CTA | 672 | 140 |
| LDS/CTA | 672 | 480 |
| VALU/CTA | 7,072 | 3,376 |
| SALU/CTA | 768 | 660 |
| MFMA/CTA | 160 | 160 |

Z5B 的 Q full-cache 已经消除了 Q global duplicate；但 Q/K/H/V-new 仍以
scalar BF16 producer、generic phase/shared view 和 fragment reconstruction
进入 MFMA。native TTGIR 实际保留 `blocked`、`swizzled_shared`、
`amd_rotating_shared` 和 `dot_op`：

```text
Q      64x32 BF16 -> shared -> dot_op0
K      32x64 BF16 -> shared -> dot_op1
H      64x32 BF16 -> transpose -> dot_op1
V-new  64x64 BF16 -> rotating shared -> dot_op1
```

C0.5S 已证明连续 source store 可以被后端自动合并为 `ds_write_b128`；D0-P
已证明 source 能发出部分 raw load/shuffle/packed LDS primitive，但完整
non-contiguous producer-to-MFMA fragment bridge 仍不能稳定表达且 full arm
correctness 失败。因此当前缺口不是普通 store vectorization，而是
**producer ownership + physical shared layout + typed dot operand 的组合边界**。

本轮没有把这个现象直接定性为已经完成的“纯 compiler bug”证据，因为现有
Z5B 高层表示本身没有等价的 first-class dot operand encoding。下一步只登记
一个、不叠加其它优化的候选：

```text
Stage 6Z-BF16-DOT
same-source generic/specialized typed BF16 operand late-lowering A/B
```

它必须保持 pre-branch IR、ownership、shared shape、barrier、MFMA、数学和
ABI 完全相同，只在 `block_dot_bf16_f32` 的 generic 与 gfx942 specialized
lowering 处分叉；B 必须把 typed packet/shared/dot operand 证据保留到
LLVM/MIR/ISA 后再测 correctness 和 dynamic PMC。本候选目前只登记，尚未实现。

## Stage 6Z Z7AB joint Q/H/K superphase

Z7AB 是在最终 Z5B source 上实现的一个单一 correctness-locked 候选，正式报告为
[`qwen_gfx942_stage6z_z7ab_joint_qhk_superphase.md`](qwen_gfx942_stage6z_z7ab_joint_qhk_superphase.md)。
它不改变 BT64/BV64/BK32、WG256、two-CTA ownership、MFMA32、BF16 ABI、
causal math 或 K32 reduction order：完整 Q LDS cache 仍只由 global 产生一次，
而 Phase A `Q@H` 与 Phase B source-half 0 `Q@K` 在一个 K32 joint loop 中
消费同一 `q_words/q_frag` source SSA 值。score half 1 仍在 inter/score0
accumulator 结束后执行，故不回到 Z4C 的三 accumulator overlap。

Z7AB 的完整 byte-exact matrix 已通过：GPU 恢复后复跑为 `11 passed in 11.85s`，
覆盖 T=64/512/1024/2048/4096/8192/16384、finite、caller-owned output 与
zero-V-new/NaN-prefill 边界。首次 initial `get_mlir()` probe 曾触发 ROCm
container OOM kill；恢复后使用 `--skip-initial-mlir` 成功捕获 lowered LLVM、
pre-LTO AMDGCN、llc pre/post-RA stop-point MIR、final ISA 和 HSACO。

Z7AB 的 SHA256 为 `ff4762fd9635efbf3163daa5871f94125da9cf635d43d3a74e1962683e965778`，
不同于 Z5B 的 `979889c1…ed67`；静态 `ds_write` 从 144 降至 92、barrier 从 32
降至 24，MFMA 保持 56，code-object 仍为 VGPR/AGPR/LDS=`104/32/32768 B` 且
scratch/spill=0。当前 runtime 没有由 link-debug hook 导出 argv，因此 exact
linker-LTO MIR 仍不可得；报告没有把 llc MIR 称为 exact LTO MIR。

新鲜 T=2048 PMC 显示动态 VMEM/LDS/VALU/SALU 为
`2464/448/11114/5000 per CTA`，Z5B 为 `672/672/7072/768`；MFMA 都是 160。
7 个 clean fresh-process sessions 的 caller-owned body timing 也稳定回退：

| T | Z5B ms | Z7AB ms | Z7AB - Z5B |
|--:|--:|--:|--:|
| 2048 | `0.067841` | `0.100409` | `+32.454 us`, CI `[31.773, 33.040]` |
| 8192 | `0.157915` | `0.287987` | `+129.744 us`, CI `[129.011, 130.228]` |

所以 Z7AB 是 correctness/machine-identity 通过但 performance **No-Go** 的反例。
静态 LDS/barrier 缩减无法抵消 typed H/K packet producer 与 address/control 路径导致的
每-chunk VMEM/SALU 扩张。Z5B 继续是唯一 Stage 6Z isolated performance baseline；
Z7AB 不接入 X2、selector 或 production，也不运行条件性的 T=16384/Eager。

可复现 source-level A/B consumer map、native TTGIR/ISA schedule 和完整 machine
delta 分别在：

- [`stage6z_z7ab_ab_consumer_map.json`](stage6z_z7ab_ab_consumer_map.json)
- [`stage6z_z7ab_native_ab_schedule.json`](stage6z_z7ab_native_ab_schedule.json)
- [`stage6z_z7ab_machine_delta.json`](stage6z_z7ab_machine_delta.json)

## Stage 6Z Z7AB dynamic-multiplicity closure and fusion-only control

The follow-up closure is recorded in
[`qwen_gfx942_stage6z_z7ab_dynamic_multiplicity_and_fusion_only.md`](qwen_gfx942_stage6z_z7ab_dynamic_multiplicity_and_fusion_only.md)
with the machine-readable ledger in
[`stage6z_z7ab_dynamic_multiplicity.json`](stage6z_z7ab_dynamic_multiplicity.json).

It establishes that Z7AB's `+1792 VMEM/CTA`, `+4232 SALU/CTA` and
`+4042 VALU/CTA` do not come from a duplicated K0 logical load, MFMA count,
spill or occupancy.  The four lexical BF16x8 raw load sites expand in final
ISA into per-distinct-address `v_readfirstlane -> saveexec -> buffer_load ->
cbranch_execnz` convergence loops.  H/K account for 2,048 unique packets per
CTA, which covers 83.1% of the measured Z7AB VMEM count.  K0 lookahead covers
stages 1--3 once each rather than reloading the same logical stage, but has no
machine-level evidence of useful overlap.

The requested `Z7AB-F` scalar-producer fusion-only source was implemented but
failed the first byte-exact gate at T=64/512/2048; a zero-V-new diagnostic
also fails, and an extra H-to-K scalar producer barrier did not repair it.
It therefore has no PMC or timing result by rule.  Stage 6Z remains closed
with Z5B as the only isolated performance baseline.  Do not create Z7AC or
integrate this line into X2, a selector, or production.

## Stage 6Z Z8W waterfall-free packet lowering

Z8W closes a compiler-lowering defect in the Z7AB H/K BF16x8 packet producer.
The high-level Z7AB source and schedule are identical in both arms; only
`AVELANG_STAGE6Z_PACKET_LOAD_LOWERING=current_raw|waterfall_free` changes the
raw-buffer address lowering.  The corrected lowering moves a divergent byte
offset from scalar `soffset` to VGPR-compatible `vindex` and uses immediate
zero `soffset`, preserving the raw-buffer effective address.  The final ISA
removes all 12 H/K packet waterfall sites, while MFMA, global/LDS static work,
barriers, LDS allocation and no-spill state are preserved.

At T=2048, dynamic per-CTA work changes from Z7AB current
`MFMA/VMEM/LDS/VALU/SALU=160/2464/448/11114/5000` to Z8W
`160/448/448/6990/840`.  Seven fresh-process body sessions confirm a
`31.988 us` Z8W gain over current Z7AB, but Z8W remains `0.727 us` slower
than Z5B at T=2048 and `6.347 us` slower at T=8192.  Therefore Z8W is a
compiler-correctness and machine-work success, but not a performance
promotion; Z5B remains the isolated baseline.  T=16384 conditional timing,
X2 integration and public Eager timing were not run.

The complete Chinese report and machine-readable provenance are:

- [`qwen_gfx942_stage6z_z8w_waterfall_free_packet_lowering.md`](qwen_gfx942_stage6z_z8w_waterfall_free_packet_lowering.md)
- [`stage6z_z8w_waterfall_provenance.json`](stage6z_z8w_waterfall_provenance.json)
- [`stage6z_z8w_machine_delta.json`](stage6z_z8w_machine_delta.json)

## Stage 6Z Z9S final-machine critical-path closure

Z9S is the one permitted same-source compiler scheduling experiment after the
Z8W final-machine audit. It keeps the waterfall-free typed packet lowering,
then uses `AVELANG_STAGE6Z_PACKET_SCHEDULING=bounded_consumer_point` to issue
one closed K0 `v4i32` raw packet before an existing independent MFMA region
and delay only its required VMEM wait to the existing LDS commit point.

The final ISA proves that this is a real machine-graph difference: Z8W issues
the audited packet at `0x2D68` then waits at `0x2D70`; Z9S issues it at
`0x2BEC`, runs the existing K0 MFMA region, and waits at `0x2D8C`. The LDS
commit and cross-wave barrier remain at the consumer point. MLIR carries the
schedule marker, the HSACO hash changes, and both arms preserve the same
MFMA/VMEM/LDS work, 32 KiB LDS, `104/32/33` code-object registers and zero
private/spill.

The 13-test byte-exact matrix passes, but seven fresh-process body sessions
close the performance case: at T=2048 Z9S versus Z5B is `+2.515 us` with a
CI crossing zero, and at T=8192 it is stably `+5.305 us`. Z9S is slightly
faster than Z8W at T=8192, but it does not beat Z5B. Thus this is **Case B**:
packet scheduling is formally closed and Z5B remains the only isolated
performance baseline. The next registered direction is a read-only Z8W vs
native remaining-VMEM producer-ownership audit, not another schedule or
pipeline variant.

Full evidence: [`qwen_gfx942_stage6z_z9s_critical_path_scheduler.md`](qwen_gfx942_stage6z_z9s_critical_path_scheduler.md),
[`stage6z_z9s_dependency_schedule.json`](stage6z_z9s_dependency_schedule.json), and
[`stage6z_z9s_machine_delta.json`](stage6z_z9s_machine_delta.json).

## Stage 6Z Z10V：Remaining-VMEM Producer-Ownership Closure

Z10V 对 Z8W 与 selected native WG256 做了逐 logical tensor 的 producer ownership
审计。完整中文报告是
[`qwen_gfx942_stage6z_z10v_remaining_vmem_ownership.md`](qwen_gfx942_stage6z_z10v_remaining_vmem_ownership.md)，
机器可检查 ledger 是
[`stage6z_z10v_producer_ownership.json`](stage6z_z10v_producer_ownership.json)。

本轮没有修改 kernel、compiler、allocator、selector 或 production，也没有重新跑
性能。冻结的动态 T=2048 per-CTA 数据为：

| arm | MFMA | VMEM | LDS | VALU | SALU |
|:--|--:|--:|--:|--:|--:|
| Z8W | 160 | 448 | 448 | 6990 | 840 |
| selected native WG256 | 160 | 140 | 480 | 3376 | 660 |

审计把四种概念分开：unique logical bytes、logical packet count、lexical load
sites 和 dynamic PMC。Q/H/K0/K1/V-new 的逐张量结果如下：

- Q：Z8W 已由一个 Q-cache producer 生成 4 个 distinct K-stage block；Phase A
  和 Phase B 都从 shared cache 读。Q global duplicate 已关闭。
- H：Z8W 和 native 都是 4 个 distinct K-stage producer group；Z8W 是 raw
  BF16x8/phase 路径，native 是 typed block/memdesc 路径，没有证明 Z8W 把同一
  H tile 重复 global load。
- K0：prologue stage 0 与 lookahead stages 1--3 覆盖不同 logical stage，不是
  同一 K 字节重复 producer。
- K1：第二 source half 的 4 个 distinct stage，同样没有证明重复 producer。
- V-new：Z8W 一次生产 `[64,64]` tile 后由两个 source-half intra consumer 从
  phase 复用；native 也只有一个 V block global producer。
- g：存在 score/final 多角色，但 Z6G-S/I 已经是独立且关闭的支线，本轮不重复。
- output：冻结的 public BF16 ABI store，不是可删除 producer。

因此 Z10V gate **未通过**：没有一个非 Q、非 g、非 output tensor 同时满足“Z8W
logical global producer multiplicity 明确高于 native”和“可只删除真实 producer”
这两个条件。`448 -> 140` 的剩余差距只能暂时归为 typed packet granularity、
producer-to-consumer physical layout、fragment feeding、地址/ownership 辅助和
aggregate PMC 不可唯一分摊的组合，不能硬选 H、K 或 V-new 做 single-producer
rewrite。

决策：`STOP_Z10V_NO_CODE`。Z5B 继续是 Stage 6Z isolated performance baseline；
Z8W/Z9S 保留为 correctness/machine-work evidence，不接 X2、selector 或 production。

## Stage 6Z Z11D：Static V-new Producer-to-Dot Physical Contract

Z11D 没有实现 compiler candidate，结论为 **Case C /
`STOP_Z11D_MAPPING_NOT_DIRECT`**。完整中文报告是
[`qwen_gfx942_stage6z_z11d_static_vnew_physical_contract.md`](qwen_gfx942_stage6z_z11d_static_vnew_physical_contract.md)，
机器可检查 mapping 是
[`stage6z_z11d_vnew_physical_contract.json`](stage6z_z11d_vnew_physical_contract.json)。

该审计只冻结并恢复 D phase `score @ V-new -> intra` 的完整物理映射。Z5B/Z8W 的
global V-new packet 是固定 token 的连续 value BF16x8，而当前 phase/local MFMA
consumer 读取的是固定 value 的连续 token BF16x8。精确地址式证明一个 global
packet 的相邻 member 在 consumer-compatible LDS 中相隔 `128 B`；它不能用一个或少量
连续 `ds_write_b128/b64` 直接放置。selected native WG256 也没有提供 direct identity
反例：其 TTGIR 明确为
`buffer_load -> amdg.in_thread_transpose -> shared4 -> dot_op`。

因此，若继续实现所谓 static-direct lowering，必然重引入 runtime transpose/scatter、
fragment rebuild 或 dynamic mapping，违反 Z11D 的硬 gate，也重复 C0.5/D0-P/BDV2
已经关闭的路线。没有新增 selector、compiler pass、HSACO、PMC、correctness 或性能
结果；这正是预注册 stop rule 的要求。Z5B 继续保持唯一 isolated performance baseline，
且 Z11D 不接入 X2、production 或 full recurrence。

## Stage 6Z C12：Full Chunk-O Native-Shaped Physical Lowering

C12 已完成 Step 0 external native HSACO upper-bound control，以及 selected native
WG256 的完整 Q/H/K/V physical-plan 恢复。external launcher 使用 selected native
`chunk_fwd_kernel_o` 的真实 symbol、ABI、grid、dynamic LDS 和 SHA256：T=2048
为 `9cc107ec...f53bfb95c`，T=8192 为 `e201dd58...17066ee5`。在 caller-owned
output、current stream、no Graph、fresh-process 7-session body 口径下，external
与 selected native output exact；Z5B/external 的 session-median 中位数为：

| T | Z5B ms | external native HSACO ms | Z5B/external |
|--:|--:|--:|--:|
| 2048 | `0.0671000` | `0.0300245` | `2.235x` |
| 8192 | `0.1570135` | `0.0771345` | `2.037x` |

这证明 caller/ABI/stream/launch 不是 chunk-o machine gap 的障碍，并给出 native
code-object upper bound；它不是 Avelang C12 kernel，也没有接入 production。

physical audit 恢复了 native 的：

- Q：`#blocked2 -> #shared -> dot_op(opIdx=0)`，同一 Q physical source 服务 Q@H 和 Q@K；
- H：`#blocked2 -> #shared1 -> #linear -> fixed tt.trans -> dot_op(opIdx=1)`；
- K：`#blocked1 -> #shared2 -> dot_op(opIdx=1)`；
- V：`#blocked -> amdg.in_thread_transpose -> #linear1 -> #shared4 rotating -> dot_op`；
- Q/H/K source shared buffers 在 score/V phase 前 dealloc，峰值 LDS 为 24576 B。

当前 AveLang full-scope block-dot 仍只接受 K/H source role，且
`makeLogicalBlockLayoutPlan` 通过 runtime `DivUI`/`RemUI` 物化 wave/lane/packet
ownership；没有能保存 native blocked/shared/dot encoding、Q 多 consumer 物理复用、
V in-thread transpose 和完整 phase lifetime 的统一 static region plan。
因此 C12 按 hard gate 关闭：

```text
STOP_C12_PHYSICAL_PLAN_INCOMPLETE
```

没有创建 C12 machine delta、没有运行 C12 correctness/PMC/performance，也不应继续
创建 C12-Q/H/K/V 或 transpose/packet-width 局部变体。详细报告、机器可读 physical
plan 和 external control：

- [`qwen_gfx942_c12_full_chunk_o_native_shaped_lowering.md`](qwen_gfx942_c12_full_chunk_o_native_shaped_lowering.md)
- [`stage6z_c12_native_chunk_o_physical_plan.json`](stage6z_c12_native_chunk_o_physical_plan.json)
- [`stage6z_c12_external_native_control.json`](stage6z_c12_external_native_control.json)

Z5B 仍是唯一 AveLang isolated performance baseline；native-shaped distributed
layout 应登记为独立的长期 compiler infrastructure 课题，而不是继续堆 Qwen
局部 lowering patch。

## Stage 6Z C13-SPR：Static Physical Representation & Plan MVP

C13-SPR 修正了 C12 结论中的表述边界。准确结论不是“AveLang 完全无法表达
static full-region encoding”，而是：`al.make_layout` 已能表达普通 shape/stride，
block-dot 已能携带字符串 metadata，但缺少 typed distributed/shared/MFMA/dot/
transform encoding，以及统一保存 Q/H/K/V、dual-consumer 和 phase lifetime 的
compiler-owned full-region plan。

C13 新增了五类 typed MLIR attrs 和纯整数的 `ChunkOPhysicalPlan`。它完整承载了
C12 T2048 WG256 需要的：

- Q `#blocked2 -> #shared -> dot_op0`，同一物理 Q source 同时服务 Q@H/Q@K；
- H `#blocked2 -> #shared1 -> fixed transpose -> dot_op1`；
- K `#blocked1 -> #shared2 -> dot_op1`；
- V `#blocked -> #linear1/fixed transform -> #shared4 rotating -> dot_op1`；
- gfx942 MFMA v3 32x32x8、warpsPerCTA `[2,2]`；
- source Q/H/K lifetime 到 `source_release`，再进入 score/V phase。

静态 mapping algebra 不接收 `mlir::Value`，不会生成 `DivUI/RemUI/Add/Mul/select`
ownership SSA。Q/H/K/V ownership、shared offset、dot slot、fixed transform 和负例
均通过单测验证；typed attrs 能经过 canonicalizer 与真实
`lower_qwen_block_dot` pass boundary，并能 MLIR textual parse/print。最终为
`7/7` tests passed。

本轮没有实现 C13 codegen，没有修改 Qwen source、production selector、recurrence、
RA/allocator，也没有运行 GPU benchmark。V native `amdg.in_thread_transpose` 的
精确 lane/register table，以及 native bank-conflict formula 仍明确标为未知，没有
猜测补齐。C13 的完整证据在：

[`qwen_gfx942_c13_static_physical_representation_mvp.md`](qwen_gfx942_c13_static_physical_representation_mvp.md)

以及四份 `stage6z_c13_*.json` 工件。状态是：

```text
C13_SPR_MVP_GO_FOR_CODEGEN
```

这只是允许下一阶段进入 codegen 设计，不改变当前 Z5B isolated performance
baseline，也不改变 Stage 6Z 对 X2/full graph 的既有停止决策。

## Latest Addendum: C21-NSM Selected-Native Pipeline Reconstruction

C21 使用 fresh selector capture 固定了真正的 T=2048 native 对照：gfx942、
BT64/BV64/BK32、WG256、4 waves/CTA、`num_stages=2`、selected shared metadata
`12288 B`。该 stage-2 code object 的 SHA256 为
`cc74acf84104a0900ed327fc469826c228c94fdf07fd55d8d302c3e88c04e72d`；当前运行时
cache 后续若选择其它 specialization，不能替代此对照。

C21 的 source 与 `ChunkOPipelinePlan` 将 Q@H、Q@K half-0、Q@K half-1 放进同一
K32 superloop。它通过 T=2048 random、zero-V-new、caller-owned NaN prefill、
structured one-hot 和 token/value pattern 五项检查，均相对 Z5B BF16 byte-exact。
但是 exact final ISA 仍保留 `global load -> immediate wait -> LDS publish ->
barrier -> consumer` 子阶段；没有证明 next-stage producer 在当前 MFMA window
结束前发射。因此 C21 的 plan 没有 materialize 成 native-style pipeline。

7 个 fresh-process、no-Graph、HIP-event T=2048 body session 的中位数为：Z5B
`66.499 us`、C19 `94.140 us`、C21 `91.355 us`、fresh selected native `60.390 us`。
C21 虽比 C19 快 `2.785 us`，但仍比 Z5B 慢 `24.857 us`，所以不是新的 isolated
baseline。

状态明确封档为：

```text
STOP_C21_PIPELINE_NOT_MATERIALIZED
```

不得据此自动开始 C22、barrier/packet/VALU 微调或接入 X2。完整 machine timeline、
PMC、正式样本和 JSON 证据见
[`qwen_gfx942_c21_selected_native_pipeline_reconstruction.md`](qwen_gfx942_c21_selected_native_pipeline_reconstruction.md)。

## Latest Addendum: C26-WDO Work Decomposition And Ownership Audit

C26 是 C25 停止后的只读归因审计。它修正了一个重要的历史测量错误：C25 的
native `320 MFMA/CTA` 来自宽匹配 `chunk_fwd_kernel_o`（混入 public selector/
autotune dispatch）以及硬编码 CTA 归一化。C26 对 T=2048/8192/16384 的 final
direct-tail 使用 exact `Kernel_Name` 和 `Grid_Size / Workgroup_Size` 重算后，Z5B
与 selected native 都是 `160 MFMA/CTA`。

两边完成同一个 `[64 token,128 value]` logical chunk-head 都需要两个 V64 CTA，
理论/实测均为 `320 MFMA/logical unit`；因此 CTA 数量或 output partition 不是当前
大 gap 的解释。same-work 下，Z5B 仍有明显更高的 VMEM/LDS/VALU/SALU per useful
MFMA，后续只能把它作为 per-CTA operand-feeding/materialization 的研究问题，不能
再以“native 一个 CTA 做两倍数学”作为假设。

主分类为：

```text
CASE C: C26_PMC_NORMALIZATION_ERROR_FOUND
```

本轮没有实现 C27、CTA redesign、packet/barrier/layout/RA 优化，也没有接入 X2。
formal timing 继续显示 native 更快；Triton selector 在独立 fresh cache/process 中会
出现 W/BK specialization 漂移，故 report 明确将 W2 final-tail PMC 与 session-level
current-selector timing identity 分开保存。完整证据见
[`qwen_gfx942_c26_work_decomposition_ownership_audit.md`](qwen_gfx942_c26_work_decomposition_ownership_audit.md)
及 `stage6z_c26_*.json`。

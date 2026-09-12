# Qwen gfx942 Stage 6Z: Z7AB Dynamic-Multiplicity Closure and Fusion-Only Control

## Result

This task closes the Z7AB regression rather than adding another schedule
variant.  The key finding is that Z7AB's apparently smaller static load graph
does not mean it performs less dynamic global-memory work.  Four source
`raw_buffer_load_x4` sites lower to four LLVM
`llvm.amdgcn.raw.buffer.load.v4i32` sites, but their final ISA has a per-unique-
address exec-mask convergence loop.  Across H, K0 and K1, one CTA has `2048`
distinct BF16x8 packets.  That model alone covers `83.1%` of Z7AB's observed
`2464 VMEM/CTA` at T=2048.

The requested scalar-producer `Z7AB-F` control was implemented from Z5B.  It
keeps only the adjacent physical Q fragment dual use, removes K0 lookahead,
alternate banks, ping-pong and source pipeline state, and restores scalar
current-stage H/K producers.  It is finite but fails BF16 byte exactness at
T=64, T=512 and T=2048.  A zero-V-new diagnostic also diverges, placing the
first observable failure in the H/inter path rather than the score/V-new path.
One extra producer-to-producer barrier did not repair it.  By the frozen
correctness rule, no Z7AB-F machine capture or performance test was run.

Therefore the final outcome is a **source-level A+B fusion closure**:

```text
Z5B remains the sole Stage 6Z isolated performance baseline.
Z7AB remains correct but performance No-Go.
Z7AB-F is correctness No-Go before performance.
No Z7AC, X2, selector or production follow-up is permitted here.
```

## Scope

All comparisons keep gfx942, BT64/BV64/BK32, WG256, two CTA per chunk-head,
MFMA32 math, K32 reduction order, BF16 Q/K/H/V-new/output, FP32 g and the
caller-owned output ABI.  No allocator, RA, recurrence HSACO or production
path changed.

## Aggregate Evidence

T=2048 rocprof counters are normalized by `131072 / 256 = 512` CTA.

| metric / CTA | Z5B | Z7AB | delta |
|:--|--:|--:|--:|
| MFMA | 160 | 160 | 0 |
| VMEM | 672 | 2,464 | +1,792 |
| LDS | 672 | 448 | -224 |
| VALU | 7,072 | 11,114 | +4,042 |
| SALU | 768 | 5,000 | +4,232 |
| profiler VGPR | 76 | 88 | +12 |
| profiler AccumVGPR | 100 | 88 | -12 |
| occupancy | 14.48% | 16.71% | +2.23 points |

The change is thus not an MFMA-count, spill, AGPR-cliff or occupancy failure.
Z7AB keeps code-object VGPR/AGPR at `104/32`, LDS at `32768 B`, and
private/spills at zero.  Its clean fresh-process body still regresses:

| T | Z5B | Z7AB | Z7AB minus Z5B |
|--:|--:|--:|--:|
| 2048 | `0.067841 ms` | `0.100409 ms` | `+32.454 us`, CI `[31.773, 33.040]` |
| 8192 | `0.157915 ms` | `0.287987 ms` | `+129.744 us`, CI `[129.011, 130.228]` |

The two-point body slope is `0.9383 us/chunk` for Z5B and `1.9539 us/chunk`
for Z7AB.  Native diagnostic slope is `0.4984 us/chunk`.

## Dynamic Multiplicity Closure

### Source to ISA chain

Z7AB has four lexical raw packet sites:

| logical bucket | source | LLVM | final ISA region | unique BF16x8 packets / CTA |
|:--|:--|:--|:--|--:|
| C: K0 current stage 0 | `k0_initial_words` | `lowered_llvm.ll:280` | `0x2B3C..0x2B60` | 128 |
| B: H current | `h_current_words` | `lowered_llvm.ll:335` | `0x2C18..0x2C3C` | 1,024 |
| D: K0 lookahead stages 1-3 | `k0_prefetch_words` | `lowered_llvm.ll:577` | `0x2DD4`, `0x2F28`, `0x307C` regions | 384 |
| E: K1 current | `k1_words` | `lowered_llvm.ll:739` | `0x2E2C`, `0x2F80`, `0x30D4` regions | 512 |

Every region follows the same final ISA template:

```text
v_readfirstlane_b32   scalar_offset, vector_offset
v_cmp_eq_*            matching-lane mask
s_and_saveexec_b64    select matching lanes
buffer_load_dwordx4   one BF16x8 packet
s_cbranch_execnz      elect another distinct offset until EXEC is empty
```

This is a CFG/ISA multiplicity model, not a static mnemonic estimate.  The
logical packet math is exact: H has `64 * 128 / 8 = 1024` packets; K0 has
`32 * 128 / 8 = 512`; K1 has `512`; total `2048`.  The model intentionally
does not claim that each packet maps one-for-one to a hardware memory
transaction or to an exact PMC PC count.  The available rocprof PC sampling
mode is beta and did not offer reliable per-PC execution attribution, so it
was not used to manufacture a finer split.

`2048 / 2464 = 83.1%`.  The remaining 416 VMEM/CTA covers Q fill, g/causal,
V-new, output, helper traffic and any attribution that cannot be separated by
aggregate counters.

### No K0 double fetch

The K0 lookahead is structurally unhealthy, but it is not a duplicate logical
global producer.  Stage 0 is issued by `k0_initial_words` (128 packets), then
the lookahead sites issue stages 1, 2 and 3 (384 packets).  Each K0 logical
packet is loaded once and then consumed from its selected LDS bank on the next
current stage.  Ping-pong changes storage location and creates extra issue
sites/control, but it does not explain the full `+1792 VMEM/CTA` as a second
read of the same K0 data.

There is also no hard evidence of useful MFMA overlap: the relevant issue
blocks appear before the release barriers and aggregate PMC has no per-PC
latency timeline.  It is therefore correct to call it *unproven overlap*, not
to claim it hides memory latency.

### SALU and VALU attribution

The largest SALU family is the packet convergence controller.  With 2,048
unique H/K packets, just `s_and_saveexec_b64` plus `s_cbranch_execnz` gives a
minimum 4,096 scalar control instructions.  This closely matches the measured
`+4232 SALU/CTA`; exec restores and packet-address setup account for the
remainder.  The source `k_stage` conditions are compile-time unrolled, so
bank parity and last-stage checks do not form a 4,232-iteration dynamic scalar
loop.

The same packet regions add vector offset formation, first-lane extraction,
comparison/masking and typed packet placement.  Roughly two vector-side
address/election instructions per packet explain the `+4042 VALU/CTA` scale.
No evidence attributes the increase to MFMA, C/D/E math or an occupancy cliff.

| rank | root cause | evidence | fusion? | lookahead/ping-pong? |
|--:|:--|:--|:--:|:--:|
| 1 | Raw BF16x8 H/K producer lowers to per-address EXEC convergence | 2,048 packet model; 83.1% of Z7AB VMEM; >=4,096 saveexec/backedge SALU | no | no |
| 2 | K0 lookahead adds distinct early producer sites with no measured overlap proof | 384 packets, stages 1-3 only; no duplicate logical read | no | yes |
| 3 | Current source-level scalar producer plus dual-accumulator consumer does not preserve established physical mapping | Z7AB-F fails T64/512/2048 including zero-V-new | yes | no |

The Q fragment dual use has no independently measurable dynamic cost in Z7AB:
Z7AB changed producer representation and scheduling simultaneously.  It is
not sound to label any fraction of `+1792/+4232/+4042` as Q reuse cost.  The
only clean control intended to measure it, Z7AB-F, did not pass correctness.

## Z7AB-F Control

The new experimental-only source is
[`qwen_gdn_bt64_native_chunko_stage6z_z7abf_fusion_only.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z7abf_fusion_only.py).
It intentionally contains only this fusion:

```text
q_frag = persistent-Q-LDS current K32 fragment
inter_acc  = Q@H(q_frag)
score0_acc = Q@K0(q_frag)
q_frag dies
```

It restores scalar current-stage H, K0 and K1 producers and deletes all raw
packet loads, next-stage K0 issue, alternate K0 banks, parity, valid state and
pipeline state.  B1 and C/D/E preserve Z5B order.  H and K0 use disjoint rows
inside Z5B's existing 32 KiB allocation; no new LDS or private allocation is
introduced.

### Correctness gate

The first gate ran the exact required byte comparison:

```text
T=64   failed byte exact, finite
T=512  failed byte exact, finite
T=2048 failed byte exact, finite
```

`V-new=0` still diverges, so score/V-new cannot explain the first difference.
Adding one producer-phase barrier between scalar H and scalar K0 stores did not
change the failure.  This is not evidence that scalar producers are generally
wrong: Z5B's phase-separated scalar path is correct.  It proves only that this
specific source-level joint consumer representation fails to preserve the
physical mapping that Z7AB established as correct.

The frozen rule prohibits PMCs, body timing, long-length tests, Eager tests or
X2 integration for a non-byte-exact arm.  The complete negative result is
also stored in [`stage6z_z7abf_machine_delta.json`](stage6z_z7abf_machine_delta.json).

## Decision

This is a pre-gate form of Case C.  The A+B source fusion cannot currently be
tested in isolation from producer representation while keeping the frozen
BF16 output contract.  Do not add a Z7AC source variant, revive Z6G, expand a
V-new/g superphase, or touch production.  Any future revisit must be a generic
compiler-internal consumer-group/scheduler representation that can preserve
the known-correct physical operand mapping without source-level packet
controllers.  That is outside this task.

The machine-readable closure is
[`stage6z_z7ab_dynamic_multiplicity.json`](stage6z_z7ab_dynamic_multiplicity.json).


# v29 Block-Dot Integration Feasibility and Eager Comparison

## Summary

This audit separates two things that must not be conflated:

1. the fastest **complete, correctness-validated Avelang graph** is the
   Stage 6X BT64 graph;
2. the best **pure-Avelang direct-K64 update lowering candidate** is the C0.5
   `persistent_typed_lds_layout` block-dot suffix.

They cannot be directly combined by substituting one function call in old
v29. The C0.5 kernel consumes an already materialized BF16 `V-new` tensor and
owns the entire recurrence state. Old v29 instead computes the nonzero-W
prediction, derives corrected/decayed values from its evolving state, and
performs the update in the same kernel. Producing all `V-new` values before
the C0.5 suffix would be mathematically circular: later chunks need the state
created by earlier updates.

Therefore this pass intentionally did **not** create a fast-but-invalid
"v29 plus C0.5" full operator. It measured the two valid boundaries:

- Stage 6X versus native vLLM as a complete uncaptured Eager public operator;
- C0.5 versus the current Triton W=0 recurrence-update control as an isolated
  direct-K64 update suffix.

No production file, old v29 source, compiler lowering, allocator, or external
HSACO was modified in this pass.

## What Is Currently Best

### Complete Avelang graph: Stage 6X

The complete experimental graph is:

```text
cumsum
  -> Stage 6X CTA-local fused KKT + solve
  -> Stage 6U BF16 solved-boundary fused W/U
  -> current-vLLM BF16 recurrence HSACO bridge
  -> Stage 6W BF16 V-new / BF16 output chunk-o
```

Entry point:

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_full.py`

The recurrence bridge is intentionally external HSACO integration. It is not
the C0.5 Avelang block-dot suffix.

### Direct-K64 compiler-lowering candidate: C0.5

The C0.5 kernel is:

`vllm_compare/repro_qwen_gdn_direct_k64_block_dot_bv32_coop.py`

It has the ABI:

```text
BF16 K + BF16 already-corrected V-new + FP32 g + FP32 initial state
  -> BF16 H snapshots + FP32 final state
```

Its source invokes `al.amdgpu.block_dot_bf16_f32`; its `persistent_typed_lds_layout`
option is the best direct-K64 lowering arm measured in this audit.

## Why C0.5 Cannot Be a Direct v29 Replacement

The old full-v29 kernel has this loop-carried dependency:

```text
state(chunk i)
  -> pred = W(chunk i) @ state(chunk i)
  -> corrected = U(chunk i) - pred
  -> V-decay(chunk i)
  -> state(chunk i + 1)
```

The C0.5 suffix begins only at the third line, but requires its `V-new` input
to reside in global BF16 memory before its recurrence loop begins. A separate
producer kernel cannot prepare `V-new` for all chunks because it would need
the same evolving state that C0.5 is supposed to calculate.

The only mathematically legitimate integration would be a new fused op with
all of the following semantics in one loop:

```text
pred MFMA32 + corrected/V-decay construction + direct-K64 block dot update
```

It would require the block-dot lowering to accept the current kernel's shared
corrected-value tile rather than a global `V-new` pointer. That is a new
compiler/source experiment, not a mechanical substitution. It also reopens
the old v29 nonzero-W correctness gate. This audit does not claim that such a
new fused op is infeasible; it documents that no correct drop-in replacement
exists today.

## Fresh Complete Eager Public Measurement

The complete measurement reused:

`vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py`

Contract:

- same random BF16 `q/k/v`, FP32 `g/beta/initial_state`;
- same current stream;
- uncaptured Eager public APIs (`cuda_graph_used=false`);
- compile, module load and first allocations warmed before timing;
- three sessions, one warmup Williams block, five randomized six-order
  Williams blocks per T;
- HIP-event and wall-clock collected by the harness.

Before timing, a fresh T=512 Stage 6X to native-vLLM check produced:

| quantity | value |
|:--|--:|
| output max abs | `6.103515625e-04` |
| output mean abs | `4.5858127e-05` |
| final-state max abs | `4.78097796e-03` |
| final-state mean abs | `4.3256947e-04` |
| finite | yes |

The values remain inside the previously frozen Stage 6X full-contract limits.

### Full-Operator Results

| T | Stage 6X full ms | native vLLM public ms | Stage 6X / vLLM | Interpretation |
|--:|--:|--:|--:|
| 512 | `0.199176` | `0.352684` | `0.565x` | Stage 6X faster, `1.77x` vLLM-normalized speedup |
| 2048 | `0.318854` | `0.404061` | `0.789x` | Stage 6X faster, `1.27x` speedup |
| 8192 | `0.856193` | `0.752759` | `1.137x` | Stage 6X slower, vLLM is `1.14x` faster |

The paired event-gain confidence intervals for Stage 6X minus vLLM were:

| T | mean Stage 6X gain | clustered 95% CI | conclusion |
|--:|--:|:--|:--|
| 512 | `+152.999 us` | `[+148.948, +156.836] us` | Stage 6X faster |
| 2048 | `+82.588 us` | `[+78.234, +87.097] us` | Stage 6X faster |
| 8192 | `-103.521 us` | `[-107.495, -99.360] us` | Stage 6X slower |

Measured endpoint-fit slopes over these three points were `5.510 us/chunk`
for Stage 6X and `3.419 us/chunk` for vLLM. Thus the short/mid sequence
advantage does not imply long-sequence parity.

## Fresh C0.5 Direct-K64 Kernel Measurement

This is a **kernel body/suffix** comparison, not a full public operator. It
uses the existing current-Triton W=0 control: supplied BF16 `V-new` is fed as
native `v`, BF16 `W=0` is supplied, and the native recurrence still executes
its fused pred pipeline. It is the closest available ABI/mathematics control,
but it is not a raw Triton update-only kernel.

All rows are medians of three fresh-process session medians, warmup=5,
repeat=20. C0 is the generic scalar-transpose lowering and C0.5 is the best
typed/persistent LDS-layout lowering.

| T | C0 generic ms | C0.5 specialized ms | C0.5/C0 | Triton W=0 control ms | C0.5/Triton |
|--:|--:|--:|--:|--:|--:|
| 512 | `0.079297` | `0.076134` | `0.960x` | `0.037857` | `2.011x` |
| 2048 | `0.217023` | `0.210733` | `0.971x` | `0.119698` | `1.761x` |
| 8192 | `0.714842` | `0.685679` | `0.959x` | `0.412412` | `1.663x` |

The specialized lowering is a real but modest `2.9%` to `4.3%` kernel-body
improvement over its generic direct-K64 counterpart. It does not make the
Avelang suffix close to native Triton, and it cannot be quoted as a full
operator speedup.

## What The Numbers Do and Do Not Show

They show:

1. The best complete Avelang graph is already measurable against native vLLM
   under a valid Eager public-API contract.
2. The C0.5 lowering helps its isolated direct-K64 suffix, but the suffix
   remains `1.66x` to `2.01x` slower than its native control.
3. A direct substitution of C0.5 into v29 would produce a false full-operator
   comparison because it lacks v29's in-loop pred-to-corrected-V dependency.

They do not show:

1. that the Stage 6X full graph contains C0.5 lowering;
2. that C0.5 has passed full-v29 nonzero-W correctness;
3. that the C0.5 suffix timing can be added to or subtracted from the Stage
   6X Eager timing;
4. that vLLM's W=0 suffix control equals its complete public operator.

## Next Valid Integration Gate

Do not modify old v29 by a pointer-level or ABI reinterpretation substitution.
The next valid experiment, if this line is resumed, is a new experimental
fused pred-update op with an explicit contract:

```text
inputs: BF16 K, FP32 W/U/g/state
internal: MFMA32 pred + shared corrected V tile + direct-K64 block_dot update
outputs: BF16 H, FP32 final state
```

It must first pass T=64/512/2048 nonzero-W recurrence reference correctness,
then be evaluated as a complete Eager operator. Until then, Stage 6X remains
the only valid full-operator baseline and C0.5 remains an isolated lowering
result.

## Reproduction

```bash
cd /workspace/project/avelang
export PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python:.

# Full Eager graph: Stage 6X versus W1 and native vLLM.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_eager.py \
  --T 512 2048 8192 --sessions 3 --warmup-blocks 1 --blocks 5 \
  --out-dir /tmp/stage6x_current_eager

# Direct-K64 suffix: generic and specialized Avelang versus native W=0 control.
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_direct_k64_block_dot_bv32_typed_lds_c05.py \
  --T 2048 --warmup 5 --repeat 20 --sessions 3 --json
```

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

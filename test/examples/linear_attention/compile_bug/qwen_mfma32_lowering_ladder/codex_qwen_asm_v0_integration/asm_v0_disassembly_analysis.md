# ASM v0 Static Disassembly Analysis

## Origin and Core-Schedule Check

The source is normalized by replacing only
`qwen_gdn_bt64_gfx942_asm_v0` with
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`; the normalized source is
byte-identical to the frozen Triton compiler-stage AMDGCN source.  The source
patch therefore changes only ELF/metadata/debug symbol spellings.

The rebuilt v0 object uses the new symbol:

```text
qwen_gdn_bt64_gfx942_asm_v0
```

It retains the original fixed launch ABI: 88-byte kernarg segment, 256-thread
workgroup, and 57,344-byte dynamic LDS launch argument.

## Static ISA Counts

Counts from `assembly/disassembly.txt`:

| instruction family | count |
|:--|--:|
| `v_mfma_f32_32x32x4_xf32` | 96 |
| `v_mfma_f32_32x32x8_bf16` | 48 |
| scratch load/store | 0 |
| `v_accvgpr_write_b32` | 123 |
| `s_barrier` | 44 |
| `ds_read` / `ds_write` | 483 |

The maximum referenced accumulator register in a direct
`v_accvgpr_read/write_b32` is `a63`; no index `>= a100` appears.  This is
fundamentally different from the failed full-v29 generic-lowering path, whose
post-RA code parked ordinary values in the `a100..a254` range.

## Code-Object Metadata

`assembly/metadata.txt` reports:

```text
.agpr_count: 64
.vgpr_count: 320
.sgpr_count: 79
.vgpr_spill_count: 0
.sgpr_spill_count: 0
.private_segment_fixed_size: 0
```

Those are static AMDGPU metadata fields.  ROCprof's resource accounting uses
different presentation units and is recorded separately once the v0 profile
is run.  The static evidence already proves that the code object has no
private scratch allocation or reported register spills.

## Consequence

The old full-v29 cliff is absent from this opaque code-object route: it never
passes the recurrence through generic AveLang memref/vector/MFMA lowering or
through LLVM register allocation for the core kernel.  This is an execution
path result, not a claim that generic Avelang lowering has been repaired.

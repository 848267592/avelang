# L6 MIR / Regalloc Audit

## Summary

This audit adds late LLVM/MIR/register-allocation evidence for the two existing L6 variants:

- `L6_baseline_current_update`
- `L6_subtile16_stage_full_update_like`

No new source variant was added, and no full Qwen kernel was modified.

The main result is:

- The real Avelang-produced hsaco for `L6_baseline_current_update` contains the problematic high AGPR copy region, including `v_accvgpr_write_b32 a100..a131` and later matching reads.
- The `L6_subtile16_stage_full_update_like` hsaco does not contain high AGPR writes/reads; its explicit AGPR copies stay at `a0..a15`.
- Both variants have the same dynamic MFMA shape count in the relevant kernel: 8 `mfma_32x32x8_bf16` and 32 `mfma_16x16x16_bf16`.
- MIR after `virtregrewriter` reports no VGPR spills and no scratch reservation, so this is not a memory-spill issue. It is register allocation using AGPR copies under pressure.
- The high `a100..a131` values in baseline are ordinary address/data temporaries around global `k` loads and LDS staging, not required MFMA accumulator state.

The strongest classification is:

```text
primary cause:
  broad k_all_t / kall_vec shared-view address lowering,
  plus update-side lifetime pressure,
  causing LLVM/AMDGPU RA to park ordinary VGPR temporaries in AGPRs

not primary:
  update MFMA intrinsic alone
  pred/v_decay value alone
  scratch spill
```

## Artifact Paths

New dump script:

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_mir_regalloc_artifacts.py`

Generated artifacts:

- `test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/`

Important files per variant:

- `lowered_optimized.ll`
- `llc_structure.s`
- `stop_after_amdgpu-isel.mir`
- `stop_after_finalize-isel.mir`
- `stop_after_greedy.mir`
- `stop_after_virtregrewriter.mir`
- `stop_after_post-RA-sched.mir`
- `print_after_greedy.txt`
- `print_after_virtregrewriter.txt`

Prior hsaco/ISA evidence reused for final binary behavior:

- `test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/hsaco/`

## Dump Status

Available:

- Optimized LLVM IR from Avelang codegen.
- AMDGPU MIR at `amdgpu-isel`, `finalize-isel`, `greedy`, `virtregrewriter`, `prologepilog`, `postrapseudos`, and `post-RA-sched`.
- Final structured `llc` assembly from the dumped LLVM IR.
- Real hsaco objdump ISA from the Avelang JIT path.

Not available:

- Initial Avelang/ROCDL MLIR text. The current Docker build hits an MLIR printer assertion while dumping initial MLIR:

```text
llvm::dyn_cast ... Assertion `dyn_cast on a non-existent value' failed
```

- Dedicated LLVM live-interval pressure logs. The optimized LLVM build accepted machine dumps, but did not produce a useful live interval/register pressure stream from the attempted flags. We therefore rely on MIR metadata, physical register assignment, and final ISA.

## Counter Context

From the previous L6 profiling pass:

| variant | trace_us | VGPR | AccVGPR | Scratch | LDS block | MFMA | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L6_baseline_current_update` | `34.371` | 128 | 264 | 0 | 45056 | 5120 | 182144 | 22528 | 28672 |
| `L6_subtile16_stage_full_update_like` | `19.189` | 96 | 168 | 0 | 32768 | 5120 | 120128 | 16384 | 21504 |
| `L6_update_mfma_no_pred_dependency` | `10.175` | 96 | 80 | 0 | 20480 | 2048 | 69504 | 8192 | 7168 |
| `L6_update_mfma_minimal_frag` | `3.164` | 12 | 4 | 0 | 1024 | 256 | 4096 | 2432 | 512 |

The baseline-to-subtile improvement keeps the same MFMA count but reduces trace, LDS block, VMEM, LDS instructions, and AccVGPR.

## ISA AGPR Summary

Measured from real hsaco objdump ISA:

| artifact | max AGPR in ISA | `v_accvgpr_write/read` | high AGPR writes | high AGPR reads | MFMA32 | MFMA16 |
|:---|---:|---:|---:|---:|---:|---:|
| baseline hsaco | 131 | 155 / 155 | 51 | 51 | 8 | 32 |
| subtile hsaco | 15 | 32 / 32 | 0 | 0 | 8 | 32 |
| baseline `llc` from dumped LLVM | 15 | 32 / 32 | 0 | 0 | 8 | 32 |
| subtile `llc` from dumped LLVM | 15 | 32 / 32 | 0 | 0 | 8 | 32 |

The `llc` route is still useful for MIR structure and source comments, but it does not reproduce the exact high physical AGPR numbering from the Avelang JIT hsaco. The final hsaco remains the authoritative source for the `a100..a131` symptom.

## Baseline High AGPR Region

The problematic baseline hsaco region writes high AGPRs while computing pointer/address values. Example:

```asm
v_lshl_add_u64 v[16:17], v[16:17], 0, v[2:3]
v_accvgpr_write_b32 a101, v17
...
v_accvgpr_write_b32 a100, v16
...
v_accvgpr_write_b32 a103, v15
v_accvgpr_write_b32 a102, v14
...
v_lshlrev_b64 v[12:13], 10, v[12:13]
v_lshl_add_u64 v[12:13], s[12:13], 0, v[12:13]
v_lshl_add_u64 v[12:13], v[12:13], 0, v[2:3]
v_accvgpr_write_b32 a107, v13
v_accvgpr_write_b32 a106, v12
...
v_accvgpr_write_b32 a131, v13
v_accvgpr_write_b32 a130, v12
```

These instructions are surrounded by address-generation operations such as:

- `v_lshrrev_b32`
- `v_lshlrev_b32`
- `v_lshlrev_b64`
- `v_lshl_add_u32`
- `v_lshl_add_u64`
- `v_sub_u32`
- `v_and_b32`

The later readback region uses the high AGPR values as pointers for global loads and LDS stores:

```asm
v_accvgpr_read_b32 v12, a100
v_accvgpr_read_b32 v13, a101
global_load_ushort v2, v[12:13], off
v_accvgpr_read_b32 v12, a98
s_waitcnt vmcnt(0)
ds_write_b16 v12, v2
...
v_accvgpr_read_b32 v12, a130
v_accvgpr_read_b32 v13, a131
global_load_ushort v2, v[12:13], off
v_accvgpr_read_b32 v12, a128
s_waitcnt vmcnt(0)
ds_write_b16 v12, v2
```

This strongly indicates the high AGPR values are address/data temporaries feeding global `k` loads and LDS staging, not MFMA accumulator outputs.

## MIR Evidence

After `virtregrewriter`, both variants report no scratch/spill fallback:

| variant | LDS size | occupancy metadata | hasSpilledVGPRs | vgprForAGPRCopy | scratchReservedForDynamicVGPRs |
|:---|---:|---:|:---|:---|---:|
| baseline | 45056 | 1 | false | empty | 0 |
| subtile | 32768 | 1 | false | empty | 0 |

The update MFMA region in MIR uses normal low accumulator registers:

```mir
%1259:vreg_64_align2 = DS_READ_B64_gfx9 %85, 0 ...
%1260:vreg_64_align2 = DS_READ_B64_gfx9 %86, 0 ...
%1890:areg_128_align2 =
  V_MFMA_F32_16X16X16BF16_1K_e64 %1259, %1260, 0 ...
...
GLOBAL_STORE_DWORD ... %1890.sub0 ...
```

The source comments on those loads point to:

```text
%ir.669
%ir.invariant.gep63
%ir.gep64.*
%ir.gep68.*
%ir.gep72.*
%ir.gep76.*
```

These are exactly the generic shared-memory addresses produced by the staged update inputs. In the baseline they come from the broad `k_all_t[128,BT]` / `kall_vec` path; in the subtile variant they come from narrower fixed subtile staging.

## LLVM IR Evidence

The optimized LLVM IR shows large addrspace(3) workgroup globals in baseline:

```llvm
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_0 =
  internal unnamed_addr addrspace(3) global [16384 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_1 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_2 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_3 =
  internal unnamed_addr addrspace(3) global [8192 x i8] undef
@__wg__qwen_mfma32_l6_kstage_update_variants_kernel_4 =
  internal unnamed_addr addrspace(3) global [4096 x i8] undef
```

The IR also contains long runs of generic `getelementptr` chains into addrspace(3), for example:

```llvm
%247 = getelementptr i8, ptr addrspace(3) %222, i32 %.idx17
%248 = getelementptr float, ptr addrspace(3) %247, i32 %221
%249 = getelementptr float, ptr addrspace(3) %248, i32 %213
...
%298 = getelementptr i8, ptr addrspace(3) %261, i32 3168
%299 = getelementptr float, ptr addrspace(3) %298, i32 %221
%300 = getelementptr float, ptr addrspace(3) %299, i32 %213
```

This matches the broad shared-view lowering pattern rather than a compact fixed-layout fragment load.

## Classification Of `a100..a131`

### kall_vec shared-view address lowering

Supported as the primary culprit.

Evidence:

- The high AGPR write region is dominated by address generation.
- The readback region feeds `global_load_ushort` and `ds_write_b16`.
- Baseline uses broad shared K staging and generic `kall_vec` view lowering.
- Subtile reduces the broad shared view and removes high AGPR copies entirely in hsaco.
- MFMA counts are unchanged between baseline and subtile.

### update B operand fragment construction

Contributing, but not the direct high-AGPR instruction source.

Evidence:

- The later update MFMA reads from shared K fragments.
- MIR comments tie the update B operand loads to generic shared GEPs.
- However, the explicit `a100..a131` hsaco writes are mostly pointer/address temporaries before the actual MFMA16 update region.

So the issue is best phrased as:

```text
the update B operand path inherits expensive generic shared K view lowering
```

not:

```text
mfma_16x16 update intrinsically needs high AGPR
```

### pred/v_decay temporary

Secondary pressure source, not the direct high-AGPR source.

Evidence:

- `L6_update_mfma_no_pred_dependency` reduces AccVGPR from 264 to 80, so pred/v_decay lifetime pressure matters.
- But the high `a100..a131` writes are address-generation values, not pred/v_decay arithmetic values.
- The lifetime marker experiment did not change counters, likely because the marker is erased too early.

### generic RA spill/use-AGPR fallback

Supported, with a specific caveat.

It is not scratch spilling:

- `hasSpilledVGPRs: false`
- `Scratch_Size: 0`
- `scratchReservedForDynamicVGPRs: 0`

It is register allocation using AGPR as an overflow/parking class for ordinary VGPR temporaries under pressure. The values being parked are address/data temporaries around global load and LDS staging.

## Should Fixed-Layout Qwen K Fragment Lowering Live In AveLang/MLIR Or LLVM/AMDGPU?

It should live in the AveLang / MLIR / Avelang-AMDGPU lowering layer, before LLVM register allocation.

Reason:

- The problematic structure is visible before LLVM as broad addrspace(3) shared buffers and generic GEP chains.
- LLVM/AMDGPU can only allocate the temporaries it receives; by the time RA runs, the high-level meaning of `k_all_t[128,BT] -> kall_vec -> update B operand` is mostly lost.
- The subtile experiment shows that changing the source/lowering shape removes high AGPR copies without changing the MFMA math.

LLVM/AMDGPU may still need a late lifetime or scheduling primitive later, but fixed-layout K fragment lowering is more naturally and safely handled before LLVM.

## Should Ordinary Address Temporaries Be Restricted From AGPR?

As a diagnostic guard: yes, this is worth exploring.

As the main fix: no, not yet.

A blanket restriction could force more values into VGPRs, increase VGPR pressure, or create real scratch spills. The better primary fix is to avoid generating the large number of ordinary address temporaries in the first place.

Recommended policy:

- First implement fixed-layout K fragment lowering.
- Then optionally add an assertion/debug mode that flags address/pointer temporaries copied through AGPR, so future regressions are visible.
- Only consider a real RA restriction after measuring its effect on VGPR pressure and scratch.

## Next Minimal Compiler Patch

Add a narrow AveLang/MLIR lowering pattern or helper for the Qwen update K fragment.

Target pattern:

```text
k_all_t[128, BT] staged in shared
kall_vec = view(k_all_t, i32, ...)
kall_vec[base_k + lane_col, token_pack]
feeding mfma_16x16x16_bf16_f32 update B operand
```

Lower it to a fixed-layout packed fragment load:

```text
for each update tile:
  compute constant-ish LDS offsets for base_k + lane_col and token pack
  emit compact DS_READ_B64 / packed load sequence
  feed vreg_64_align2 directly to MFMA16 B operand
```

Constraints:

- Preserve current source semantics and output ABI.
- Do not materialize full transposed `k_all_t[128,BT]` when only one 16-column K subtile is consumed.
- Avoid generic shared-view GEP ladders in the update B path.
- Keep this as a specific AMDGPU/Qwen fragment helper first, not a broad lifetime system.

If this patch lands and reduces the L6 baseline toward `L6_subtile16_stage_full_update_like` without scratch, then the same helper should be applied to the production Qwen chunk_gdr lowering path.

## Exact Commands

Artifact dump:

```bash
cd /workspace/avelang
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_mir_regalloc_artifacts.py
```

Key inspection commands:

```bash
rg -n "v_accvgpr_write_b32|v_accvgpr_read_b32|v_mfma|a1[0-9][0-9]|a[89][0-9]" \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_lowering_audit/hsaco/*.isa

rg -n "ldsSize|occupancy|hasSpilledVGPRs|scratchReservedForDynamicVGPRs|vgprForAGPRCopy" \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/*/stop_after_virtregrewriter.mir

rg -n "DS_READ|DS_WRITE|GLOBAL_LOAD|V_MFMA|V_ACCVGPR|%ir\\." \
  test/examples/linear_attention/rocprof_outputs/qwen_l6_mir_regalloc_audit/L6_baseline_current_update/stop_after_virtregrewriter.mir
```


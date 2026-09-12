# L6 Compiler Culprit Classification

## Classification

The culprit is closest to **Case C plus Case E**:

- **Case C: view/cast/subview address lowering** is the strongest concrete culprit.
- **Case E: no single isolated source operation** also applies, because pred/v_decay dataflow and update staging interact.

It is not mainly Case B, and it is not fixed by the current lifetime-marker mechanism.

## Case Review

### Case A: MFMA32 accumulator unpack lowering

Partially implicated, but not the whole problem.

Evidence:

- Previous `pred32_only_sink` already had high AccVGPR around `184`.
- Baseline and subtile both have 8 static MFMA32 instructions.
- Subtile still reduces AccVGPR from `264` to `168`.

Conclusion: pred MFMA32 contributes a high base cost, but the extra baseline peak is not explained by accumulator unpack alone.

### Case B: update MFMA fragment construction

Not primary.

Evidence:

- `L6_update_mfma_minimal_frag`: AccVGPR `4`.
- `L6_update_mfma_no_pred_dependency`: AccVGPR `80`.
- Baseline: AccVGPR `264`.

Conclusion: MFMA16 update fragments are cheap when isolated and moderate in the no-pred full-update path.  They become expensive only in the Qwen-shaped dataflow/staging context.

### Case C: view/cast/subview address lowering

Primary concrete culprit.

Evidence from baseline to subtile, with the same dynamic MFMA count:

| metric | baseline | subtile | delta |
|:---|---:|---:|---:|
| trace_us | 34.331 | 19.109 | -15.222 |
| AccVGPR | 264 | 168 | -96 |
| VGPR | 128 | 96 | -32 |
| LDS block | 45056 | 32768 | -12288 |
| VALU | 182144 | 120128 | -62016 |
| VMEM | 22528 | 16384 | -6144 |
| LDS inst | 28672 | 21504 | -7168 |
| `v_lshl` | 722 | 410 | -312 |
| `v_lshl_add` | 401 | 242 | -159 |
| explicit acc write/read | 155 / 155 | 32 / 32 | -123 / -123 |

The static high ACC indices in baseline occur in address/staging regions, not just in MFMA-local windows.  This points at generic lowering of the broad shared K view and its index arithmetic.

### Case D: shared allocation / LDS packing

Secondary.

Baseline has a larger LDS block (`45056`) than subtile (`32768`) and more `ds_write` (`152` vs `104`).  However, shared allocation size alone does not explain the high explicit AGPR traffic or the no-pred drop to AccVGPR `80`.

### Case E: no single culprit

Also true at the full-kernel level.

The source-only K-subtile pattern helped isolated L6 but failed in full Qwen:

| kernel | trace_us | VGPR | AccVGPR | scratch | LDS block |
|:---|---:|---:|---:|---:|---:|
| original full v29 | 817.775 | 128 | 264 | 0 | 61440 |
| k_subtile full exp | 2239.567 | 128 | 384 | 84 | 49152 |

This means the fix should not be another direct full-Qwen source rewrite.  The isolated data is still useful because it identifies what the compiler should lower better: the Qwen update K fragment/view path.

## Minimal Compiler Fix Proposal

Do not implement a broad lifetime system or a full Triton-like block-dot rewrite from this evidence.

Proposed minimal compiler/backend fix:

```text
Add a fixed-layout lowering path for the Qwen update K fragment.
```

Concrete target:

- Detect or expose the shared K update pattern currently represented as broad `k_all_t[128,BT]` plus packed `kall_vec` view.
- Lower it to the compact per-update-tile DS read pattern needed by `mfma_16x16x16_bf16_f32`.
- Avoid generic address-generation expansion for the full transposed shared view.
- Keep non-MFMA address/staging temporaries out of AGPR allocation when possible.

This is narrower than a new source K-subtile variant: it preserves the production source schedule but gives the backend a fixed update-fragment layout to lower.

## One More Artifact Before Coding

Before implementing the fix, collect one late backend artifact for baseline vs subtile:

```text
post-ROCDL or LLVM/MIR register-allocation dump around the region that emits
v_accvgpr_write_b32 a100..a131 in baseline.
```

Reason:

- ISA already proves the symptom.
- The missing detail is exactly which lowering/register-allocation pass turns address/staging temporaries into high-index AGPRs.
- A MIR/liveness dump would tell whether the fix belongs in MLIR shared-view lowering, LLVM address arithmetic simplification, or AMDGPU register-allocation constraints.

If that dump is hard to add, the next best implementation is still the fixed-layout Qwen update K fragment lowering helper.

# Qwen Persistent Recurrence R0: Existing Compiler Inventory

## Purpose

This inventory is the R0 entry gate. It separates existing experimental
evidence from reusable compiler components, so the persistent recurrence is
not another name for a pred/update schedule experiment.

The R0 target is a first-class semantic boundary for the complete device-side
recurrence: chunk loop, FP32 feedback state, pred, BF16 V-new boundary,
V-decay, Direct-K64 update, ABI-visible H/V-new, and final FP32 state.

## Component Inventory

| Component | Location | Responsibility | Status | R0 disposition |
|:--|:--|:--|:--|:--|
| B0 full sequence | `vllm_compare/repro_qwen_gdn_direct_k64_bv32_full_sequence_b0.py`; `qwen_direct_k64_bv32_full_sequence_b0_report.md` | Correct device-side chunk loop, P0 pred, BF16 boundary, Direct-K64 update, FP32 feedback | Experimental and correctness-proven | Reused verbatim as the `legacy_b0` region body and machine-work oracle |
| P0/P1/P2/P3 ladder | `repro_qwen_gdn_direct_k64_*`; corresponding reports | Pred mapping, one-chunk composition, feedback microscope, feedback-source decomposition | Correctness infrastructure | Reused as R0 oracle and failure-localization ladder |
| B1 recurrence step | `al.amdgpu.qwen_gdn_recurrence_step_bf16_f32`; `lower_qwen_gdn_recurrence_step_pass.{h,cc}`; B1 stream32 repro | A single BT64 recurrence step with stream32-local scheduling | Correctness-positive, performance-negative | Preserved as a step-level migration adapter; not selected as the complete R0 op |
| Typed block dot | `al.amdgpu.block_dot_bf16_f32*`; `lower_qwen_block_dot_pass.cc` | Typed Direct-K64 BF16 dot and MFMA32 update lowering | Specialized `persistent_typed_block` is the correct native update baseline | Kept inside the R0 recurrence region until recurrence lowering |
| S0 stage token | `amdgpu_qwen_k64_pipeline_stage_load/commit`; `lower_qwen_k64_pipeline_stage_pass.cc` | Opaque K64 source-load and shared-bank commit token | S0-C packet commit is No-Go | Reserved in the future schedule interface; not invoked in R0 |
| B2 lookahead | B2 source and report | One-chunk-ahead source schedule | Historical, not selected | Excluded from R0 |
| D0-P/C0.5 layouts | Direct-K64 layout reports | Candidate LDS producer/consumer layouts | Not an R0 dependency | Excluded from R0 |
| Generic region conversion | `lower_ave_lang_to_memref_pass.cc` | Preserve structured operations across type conversion | Compiler correctness infrastructure | R0 relies on its region-preserving repair |
| Exact LTO replay | `replay_qwen_v29_lto_mir.py`; linker debug capture | Final ROCm LTO pre/post-RA MIR | Proven audit tool | Reused unchanged for R0 |
| Same-source lowering A/B | K-fragment/block-dot reports | Separate source schedule from lowering effect | Audit infrastructure | Retained for R1 validation, not a R0 branch |
| Current Triton oracle | Stage 6R and S0-C artifacts | Phase/layout evidence | External diagnostic oracle | Read-only R1 input; no TTIR/ISA clone or HSACO call |

## Required Conclusions

### B1 is not the final persistent-recurrence op

`qwen_gdn_recurrence_step_bf16_f32` owns one BT64 step and was created for
stream32. Its lowering is allowed to make step-local decisions. It cannot
represent the outer device loop and its loop-carried FP32 state. Using B1 as
the final parent op would reproduce the split-planning problem R0 is intended
to solve.

R0 does not delete B1. B1 remains a valid historical experiment and a future
planner may lower a legal subregion through its implementation. `legacy_b0`
does not select B1.

### Block dot stays inside the recurrence region

`block_dot_bf16_f32` remains a typed region-internal update operation. It is
not copied into a Qwen mega-pass and is not lowered before the recurrence
planning boundary. This preserves future generic/specialized operand choices.

### Reusable S0 contribution

S0 contributes an opaque load/commit stage-token capability. S0-C proved the
current token packet layout cannot be committed as a Triton-like operand
packet without forbidden cross-lane/layout work. R0 therefore reserves the
interface but uses no S0 packet mapping.

### Historical controls

| Item | R0 treatment |
|:--|:--|
| `AVELANG_QWEN_BLOCK_DOT_LOWERING=specialized` | Reused for B0-compatible update lowering |
| `AVELANG_QWEN_BLOCK_DOT_OPERAND_LOWERING=persistent_typed_block` | Reused for B0/C0 update contract |
| `AVELANG_PERSISTENT_RECURRENCE_LOWERING=legacy_b0` | New R0-only switch; the only accepted R0 mode |
| B1 stream32 and B2 lookahead | Historical only; never selected |
| S0 load/commit lowering | Available but not selected |
| external HSACO bridge | Forbidden |

### Earlier semantic loss

Before R0, B0 lowered as ordinary operations and the compiler saw pred/update
pieces but not their complete loop-carried recurrence. B1 preserved a step and
block-dot preserved an update dot, but neither preserved the complete unit.
The generic AveLang-to-memref conversion also reconstructed unknown operations
without transferring regions. R0 fixes this generic region-loss path, rather
than adding a Qwen-only workaround.

The machine-readable matrix is
`qwen_persistent_recurrence_reuse_matrix.json` beside this document.

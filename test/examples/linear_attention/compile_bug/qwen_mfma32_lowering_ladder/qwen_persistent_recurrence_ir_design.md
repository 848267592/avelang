# Qwen Persistent Recurrence IR Design

## Goal

R0 adds a complete recurrence semantic boundary plus a small target schedule
interface. The semantic operation does not embed a lane map, LDS offsets,
Triton instruction sequence, Qwen symbol, or external HSACO call.

The initial lowering is intentionally non-optimizing: it uses the existing B0
body and restores that body structurally before block-dot lowering. This gives
future planning a stable whole-recurrence control point without changing B0
machine work.

## Layer 1: Persistent Recurrence Semantics

The concrete operation is printed as a generic region operation:

```mlir
"ave.gpu.amdgpu_qwen_persistent_recurrence"(
  %k, %w, %u, %g, %initial_state,
  %h, %pred_f32, %pred_bf16, %v_new, %v_decay, %state_after,
  %final_state, %num_chunks, %emit_audit) ({
  // B0 device-side chunk loop.
  "ave.gpu.amdgpu_qwen_persistent_recurrence_yield"() : () -> ()
}) {avelang.amdgpu.recurrence_plan = {...}} : (...) -> ()
```

R0 source emits a marked empty region around the B0 body and an end delimiter.
`FormQwenPersistentRecurrencePass` moves the exact body into that region,
validates it, and removes the delimiter. The front-end marker is rejected if
it contains anything other than its yield, preventing an accidental partial
recurrence.

| Semantic value | R0 representation | Required ordering |
|:--|:--|:--|
| Initial and feedback state | FP32 input plus region-local carried state | FP32 `state_i` is the only next-chunk feedback carrier |
| H snapshot | BF16 output | Snapshot is not feedback storage |
| Pred | MFMA32 body plus FP32/BF16 audit outputs | P0 fixed nonzero-W mapping |
| Corrected | Region-local FP32 expression | `FP32(U_bf16) - pred_f32` |
| V-new boundary | BF16 ABI output | `BF16(corrected)` must precede update consumption |
| V-decay | BF16 transient/audit output | `BF16(FP32(v_new_bf16) * decay)` |
| Update | Region-local typed block-dot | Direct-K64, B0 K32 order |
| Final state | FP32 output | State after final chunk |
| Loop extent | `num_chunks` operand, R0 plan `bt=64` | Device-side loop extent and fixed BT contract |

R0 has one parent region because B0 is a fully ordered body. A future form may
use pred/update subregions only while they remain inside one parent loop and
share one schedule plan. They may not become independent kernels or unrelated
lowering passes.

### Verification, effects, and canonicalization

* The operation has exactly one region/block and an explicit yield terminator.
* Its intrinsic verifier checks all 14 operands, integer/index chunk count,
  i1 audit flag, and region/terminator shape.
* It is not Pure and is not CSE-able: explicit inputs/outputs plus FP32
  feedback establish observable ordering.
* Safe canonicalization may simplify region-local expressions, but cannot move
  the BF16 V-new conversion after update, outline pred/update independently,
  or hoist a state-dependent value across chunks.
* Generic AveLang-to-memref conversion transfers operation regions when it
  rebuilds an operation. This is a generic compiler correctness repair and
  prevents the semantic body from being lost at type conversion.

## Layer 2: Target Schedule Interface

`qwen_recurrence_schedule_plan.h` contains a deliberately small plan:

```c++
struct QwenRecurrenceSchedulePlan {
  target = "gfx942";
  bt = 64; bv = 32; workgroupSize = 128; waves = 2;
  stateType = "f32"; boundaryType = "bf16";
  distributedTile = Deferred;
  shared = Deferred;
  dotOperand = Deferred;
  supportsPipelineStageToken = true;
};
```

The interface is attached as MLIR attributes in R0. It is not an implicit
scheduler. R1 may choose legal distributed/shared/dot encodings and a stage
token lifetime only after seeing the whole recurrence.

## Pass Ordering

### R0 implemented order

```text
safe inlining / canonicalization / CSE
  -> AveLang-to-memref with region preservation
  -> persistent recurrence formation around exact B0 body
  -> future unified schedule-planning boundary
  -> legacy_b0 recurrence lowering (structural inline)
  -> block-dot / typed operand lowering
  -> intrinsic implementation linking and GPU outlining
  -> GPU-to-ROCDL -> LLVM -> ROCm LTO
```

The key invariant is that block-dot lowering occurs after the persistent
recurrence planning boundary. R0 has no planner, so `legacy_b0` chooses the
known B0 body and then allows the existing specialized block-dot lowering.

### R1 required order

```text
canonicalize-safe
  -> persistent recurrence formation
  -> unified recurrence schedule planning
  -> distributed-layout lowering
  -> recurrence lowering
  -> block-dot / operand lowering
  -> GPU-to-ROCDL -> LLVM/LTO
```

## Migration

* B0 is the immediate source and compatibility oracle.
* B1 remains an optional step-level adapter, not the parent semantic op.
* Block-dot remains typed inside the region until the plan selects a lowering.
* S0 contributes an opaque stage-token capability only; S0-C forbids its
  present packet mapping as a native dot operand.
* Triton evidence is an external legality/performance oracle only.

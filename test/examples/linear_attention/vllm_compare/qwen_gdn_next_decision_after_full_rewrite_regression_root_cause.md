# Next Decision After Full Rewrite Regression Root Cause

## Decision

Full v29 remains stopped. v24 remains the production baseline.

Isolated L6 succeeds because it has the same four persistent fragment
consumers but only a short outer loop and a smaller live set: AccVGPR 180 in
the previous L6 production-shaped result, no scratch, and trace 18.628 us.

Full v29 fails for a different reason than repeated K loads:

- it removes the broad producer correctly;
- it does not duplicate static K global loads;
- it clones only three scalar reloads, which R1 also has without scratch;
- but its generic rewritten B-fragment loads coexist with the real MFMA32
  pred accumulator, pred-partial/state/v-decay, update, and 32-chunk live
  structure.

That combination reaches AccVGPR 384 and introduces 736 B scratch, raising
trace from 830.974 us to 1302.635 us.

## Minimal Fix Result

Hoisting the compact replacement alloca outside the long loop was exact but
ineffective:

| metric | rewrite | hoisted candidate |
|:---|---:|---:|
| trace us | 1302.635 | 1301.974 |
| AccVGPR | 384 | 384 |
| scratch bytes | 736 | 736 |

The candidate was reverted.

## Next Single Action

Implement a late GPU-lowering representation for the persistent update
B-fragment that emits a dedicated fixed LDS packed read into the MFMA16
operand, instead of generic dynamic vector.load. Gate it first in a reduced
repro with the real MFMA32 pred schedule. Do not create another full Qwen
source rewrite until that gate removes scratch.

Production v23/v24/v26/v27/v28 were not modified.

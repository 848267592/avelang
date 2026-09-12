# Qwen GDN Next Decision After Real Phase Boundary

## Decision

The real pred/update phase boundary did not reduce reduced-repro latency, so
do not pursue a compiler phase-splitting pass and do not revisit full v29.
Keep v24 as the production baseline.

| reduced variant | normal median ms | rocprof trace us | AccVGPR | Scratch |
|:--|--:|--:|--:|--:|
| current fused | `0.532050` | `513.423` | 144 | 0 B |
| hard shared phase boundary | `0.546571` | `539.181` | 144 | 0 B |
| no real pred accumulator control | `0.334236` | `314.126` | 132 | 0 B |

The hard boundary is `2.73%` slower in normal timing and `5.02%` slower in
rocprof trace. Its dynamic MFMA/VALU/SALU/VMEM/LDS counts and its register
allocation are identical to current fused. Removing the real pred accumulator
is `37.18%` faster, proving the pred phase matters, but materializing/reloading
through workgroup memory did not solve the combined live-region problem.

The problem is therefore still the isolated-K-to-full composition, but deeper
than an uncrossed single SSA edge: MFMA32 pred, pred-partial, state, v-decay,
and update coexist in a form that the shared-memory boundary does not make
cheaper.

The next single action is to stop this v29 compiler experiment line and
preserve the evidence for a future design that changes the full composition,
rather than adding another local K-load or phase-boundary variation.

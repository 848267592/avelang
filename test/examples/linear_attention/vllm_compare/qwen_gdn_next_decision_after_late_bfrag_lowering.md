# Qwen GDN Next Decision After Late B-Fragment Lowering

## Decision

Do not revisit the full v29 rewrite from this result. Keep v24 as the
production baseline.

## What Worked

The experimental dedicated B-fragment op survived producer-consumer rewriting
and GPU outlining, then late-lowered to a direct 8-byte LDS vector load. The
path did not use the old generic `vector.load` / memref-view chain, and no
temporary alloca was added for the fragment.

## What Did Not Work

On the reduced loop with the real MFMA32 pred accumulator, the direct path was
not a register-pressure win:

| path | median ms | VGPR | AccVGPR | Scratch |
|:--|--:|--:|--:|--:|
| generic B load | 0.5292 | 128 | 144 | 0 B |
| late direct LDS B load | 0.5634 | 124 | 148 | 0 B |

The direct load reduces VGPR by four but increases AccVGPR by four. Because
the generic control does not spill, the test cannot support the claim that
this local B-load replacement will remove full v29's 736-B scratch spill.

## Next Single Action

Stop the v29 late-B-fragment lowering line. The exact blocker is that a
standalone replacement of the B-fragment access does not shrink the combined
MFMA32 pred / pred_partial / state / v-decay live region enough to reduce
AGPR pressure. Any further attempt needs a different, evidence-backed way to
split or schedule that full live region; it should not be another source
rewrite or another local B-load lowering variation.

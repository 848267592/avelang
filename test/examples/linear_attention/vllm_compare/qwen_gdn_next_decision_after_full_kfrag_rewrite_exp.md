# Next Decision After Full K-Fragment Rewrite Experiment

## Decision

Stop the full v29 K-fragment rewrite branch here. It is exact relative to
original v29, but misses every performance gate:

| T=2048 metric | original v29 | rewrite exp |
|:---|---:|---:|
| normal chunk_gdr ms | `0.837143` | `1.338669` |
| rocprof trace us | `830.974` | `1302.635` |
| AccVGPR | `264` | `384` |
| scratch bytes | `0` | `736` |

MFMA work is unchanged. The rewrite lowers VALU/SALU but introduces scratch,
raises VMEM, and increases AccVGPR. It must not be migrated to full forward
or any production baseline.

## What Transferred

The persistent helper and pass fired and preserve full v29 chunk_gdr
semantics exactly at T=512/1024/2048.

## What Did Not Transfer

The intended L6 lowering benefit did not compose with the full recurrence:

- AccVGPR did not fall;
- scratch was introduced;
- LDS allocation and MFMA count stayed unchanged;
- VMEM increased; and
- trace regressed about 57%.

## Recommended Next Action

Do not make another Qwen source rewrite. Return to a compiler-side,
loop-shaped reduced repro that explains the `AccVGPR=384`,
`Scratch=736 B` full-loop behavior. First reproduce and remove that
regression in isolation; only then reconsider full Qwen migration.

Production files v23/v24/v26/v27/v28 were not modified.

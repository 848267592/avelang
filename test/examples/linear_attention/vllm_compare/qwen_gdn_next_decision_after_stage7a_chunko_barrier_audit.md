# Qwen GDN Next Decision After Stage 7A Chunk-O Barrier Audit

## Decision

**Close the Stage 6Z pure-Avelang native chunk-o source route.** Do not create
Z2, do not run full-graph timing, and do not promote the Stage 7A
phase-compact candidate.

The frozen Stage6Z Z1 kernel has 41 static barriers. Stage 7A established an
exact count-preserving chain:

```text
explicit al.syncthreads source schedule
  -> 41 pre-link LLVM barrier calls
  -> 41 pre-LTO assembly barriers
  -> 41 final HSACO s_barrier instructions
```

The backend did not create a hidden surplus of barriers. A one permitted
source scheduling change removed the candidate lane-private and early
score-half barriers, but failed the first full-Z1 exactness case at T8192:

```text
phase-compact vs Z1: max_abs=0.00206613541, bit_exact=false
```

The standalone barrier repro remains valuable evidence: a local per-lane
fragment barrier and a disjoint score/V store barrier can each be removed in
isolation. The full pipeline disproves treating those local facts as a global
license to erase the phase boundaries.

## What Remains Frozen

- Stage6W/X2 and v24/default selector;
- Stage6Z Z1 source and its No-Go status;
- BT64/BV64/BK32/WG256 mapping;
- recurrence ABI and all full paths;
- compiler, LLVM/AMDGPU RA, assembly, and vLLM source.

## Next Single Direction

Choose one independent objective, not both:

1. **End-to-end diagnostic:** integrate the captured native vLLM `chunk-o`
   HSACO behind a strict external-kernel bridge and measure the recoverable
   full-graph gap. It must be labeled external integration, not an Avelang
   source-kernel result.
2. **Pure Avelang source:** leave chunk-o closed and move to the previously
   ranked W/U runner-up gap. Its expected upside is smaller than native
   chunk-o replacement.

Do not reopen Stage6Z with barrier subset sweeps, alternate tile/WG shapes,
or a full-graph "just to see" integration.

## Evidence

- [Stage 7A report](../compile_bug/qwen_mfma32_lowering_ladder/qwen_gfx942_bt64_chunko_barrier_provenance_stage7a_report.md)
- [41-entry barrier ledger](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/z1_barrier_ledger.md)
- [minimal repro results](../compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_chunko_barrier_stage7a/minimal_repros.md)

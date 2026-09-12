# Qwen gfx942 C24 Region Pending Packet Infrastructure

## Decision

**`STOP_C24_PENDING_PACKET_NOT_SURVIVING_BACKEND`**.

C24 completed the requested compiler-infrastructure work, not a new chunk-o
candidate.  The new internal Issue/Commit SSA representation is valid, the
synthetic gfx942 test produces a real delayed VMEM wait, and the same
representation compiles through the real C21 H-to-K edge without C23's
segmentation fault.  But the final C21 ISA waits for the pending K load before
the H MFMA window.  It therefore fails the pre-registered real-overlap gate.

No full Q/H/K expansion, correctness test, PMC collection, or body/public
performance benchmark was run.  Continuing would violate the C24 stop rule.

## C23 Root Cause

C23's `C23LatePipelineMaterializer` kept a mutable
`DenseMap<Operation *, C23PendingKPacket>` inside a greedy
`OpRewritePattern`.  Its raw operation keys and vector SSA payload were
created in one H/K rewrite and later consumed from another rewrite.  Greedy
rewriting is allowed to replace or erase those operations, but this map has no
region-owned SSA graph, no dominance contract, and no atomic lifetime for its
issue/commit pair.  The old T64 path terminated in `get_llvm_ir` with exit
139 before producing LLVM or MIR.

That is the precise compiler **design** defect.  There is no captured ASan
stack for the individual faulting instruction: the isolated Debug JIT attempt
did not complete.  C24 does not hide that gap.  It replaces the invalid model
with verifier-checked IR and retires the old environment mode with a controlled
compiler diagnostic.  The controlled C23 repro now exits 1 with the explicit
message that cross-op ownership inside greedy rewriting is invalid.

## C24 Design

C24 adds two compiler-internal operations:

```text
ave.gpu.amdgpu_region_pending_packet_issue
  source, predicate, indices -> vector<8xbf16>

ave.gpu.amdgpu_region_pending_packet_commit
  vector<8xbf16>, predicate, shared destination, indices
```

They are not source-facing Qwen operations.  Their verifier requires BF16
memory, an `i1` predicate, `vector<8xbf16>`, matching index arity, and a direct
same-block Issue producer for Commit.  `C24RegionPendingPacketPlanner` forms
the pair over the frozen C21 function region before greedy block-dot lowering.
The C21 proof contains exactly one edge: issue K0 before H, then commit it to
the existing K shared region before K0.  It does not speculate across K0 to K1
or across a loop iteration.

The implementation is in
[AveLangOps.td](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/IR/AveLangOps.td),
[AveLangOps.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/IR/AveLangOps.cc),
[lower_qwen_block_dot_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_block_dot_pass.cc),
and
[lower_qwen_k64_pipeline_stage_pass.cc](/home/jiandongliu/project/avelang/lib/Dialect/AveLang/Transforms/lower_qwen_k64_pipeline_stage_pass.cc).

## Synthetic Gate

The new compiler regression
`C24RegionPendingPacketCodegenTest.SyntheticIssueMfmaCommitSurvives` passed.
Its final ISA has:

```text
line 13  global_load_dwordx4       ; pending next packet
line 19  v_mfma ...                ; current work
...      10 more current MFMAs
line 53  s_waitcnt vmcnt(0)
line 54  ds_write_b128             ; commit
```

Thus the minimum representation is capable of exactly
`Issue -> multiple MFMA -> delayed wait -> LDS store`.  Artifacts are under
`codex_qwen_gfx942_c24_region_pending_packet_infrastructure/synthetic/`.

## Real H-to-K Result

The real C21 T64 compile is valid and generated a new HSACO:

```text
HSACO SHA256: 4436f8b6b506fc39200cb9de908ed511d3fbcd920a479716b9858c384cadd1d4
VGPR/AGPR/SGPR: 192 / 80 / 38
LDS: 24576 B
private segment and spills: 0
```

The pending pair survives through the compiler graph:

| level | evidence |
|:--|:--|
| post-block MLIR | Issue and Commit are direct SSA operations |
| post-kfrag MLIR | predicated K `vector.load`, then four H MFMA calls, then K `vector.store` |
| LLVM | K `%153` load -> `%155` phi -> later LDS store; four H MFMA calls lie between source-level def/use |
| MIR | selected as `%517:vreg_128_align2`; greedy allocation uses `%2015:av_128_align2`; zero spills |
| final ISA | K store is delayed, but `vmcnt(0)` is not |

The decisive ISA sequence is:

```text
364  global_load_dwordx4           ; C24 K Issue
367  global_load_dwordx4           ; current H packet
389  s_waitcnt vmcnt(0)
390  ds_write_b128                 ; H publication
396  v_mfma                         ; H MFMA 0
401  v_mfma                         ; H MFMA 1
405  v_mfma                         ; H MFMA 2
409  v_mfma                         ; H MFMA 3
411  ds_write_b128                 ; C24 K Commit
414/415 s_barrier
438  v_mfma                         ; first K consumer
```

So C24 did delay **publication** of K, but not the VMEM dependency.  H must
publish its own globally loaded packet before its MFMA, and AMDGPU emits one
`vmcnt(0)` that resolves both H and the outstanding K load.  There are zero H
MFMAs between K Issue and that wait.  This is exactly the Case-C stop
condition, not a resource or correctness failure.

## Liveness and Resources

The packet is only `vector<8xbf16>`.  In LLVM it is `%153`, flows through
`%155`, and is consumed by the K Commit store after the H MFMA region.  MIR
shows a 128-bit virtual packet; greedy assigns the observed load to an
`av_128_align2` value.  C24 changes code-object resources from historical C21
`188/80/36` VGPR/AGPR/SGPR to `192/80/38`; it introduces no scratch or spill
and does not enlarge the H/K accumulator lifetime.  The modest packet live
range is therefore not the reason for the stop.

Static C21 to C24 instruction counts are `94/18/40/100/20/94/16` to
`94/18/40/100/20/93/17` for global-load/global-store/ds-read/ds-write/MFMA/
wait/barrier.  They are code-shape evidence only, not dynamic PMC results.

## Validation

- Avelang binding rebuilt and copied to `python/`; build and installed binding
  SHA256 matched.
- C24 synthetic codegen regression: pass.
- Real C21 T64 compile, final ISA, HSACO, LTO replay and exact MIR: pass.
- Exact LTO reports zero AV32/AV64 spill saves.
- C23 selected mode: deterministic diagnostic, no segmentation fault.
- `git diff --check`: pass.

Correctness, PMC and performance were intentionally not run because the real
ISA overlap gate failed first.

## Final Answers

1. C23 failed because a per-pattern mutable map attempted to own a
   cross-rewrite pending SSA lifetime; the exact machine fault instruction was
   not captured.
2. Greedy block-dot rewrites cannot safely own that lifetime because rewrites
   may replace/erase the raw `Operation *` keys and no region SSA verifies the
   edge.
3. C24 moves ownership to a function-region planner and direct Issue/Commit
   SSA pair.
4. Synthetic ISA proves delayed wait works for the representation.
5. Real MLIR/LLVM/MIR preserve the packet and delayed store, but final ISA
   waits before the H MFMA window.
6. The pending packet adds a small packet/address lifetime, not a new
   accumulator region or spill.
7. T64/T2048/T8192 correctness, full Q/H/K materialization and all formal
   timings are `not_run` by the hard gate.
8. The result is `STOP_C24_PENDING_PACKET_NOT_SURVIVING_BACKEND`; C25, packet
   sweeps, barrier sweeps, RA work and new layouts were not started.

Machine artifacts and every machine-readable result are in
`codex_qwen_gfx942_c24_region_pending_packet_infrastructure/` and the adjacent
`stage6z_c24_*.json` files.

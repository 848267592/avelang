# Qwen GDN Next Decision After Persistent Recurrence R0

## Decision

R0 is accepted as the experimental semantic architecture baseline.  Its
`legacy_b0` lowering preserves full nonzero-W recurrence correctness through
T=2048 and produces a byte-identical B0 HSACO.

Do not promote R0 as a performance version.  It intentionally made no
schedule change and must remain the compatibility baseline for R1.

## Evidence

* R0 versus B0 and P2 is byte-exact for H, pred, BF16 V-new, BF16 V-decay,
  per-chunk state, and final state at T=64/128/512/2048.
* R0 and B0 HSACO SHA256 are identical:
  `c565372cc0eaf1ea83dcba77b43bba5f1a2a2155c0e5444f35e897759ec88053`.
* Exact ROCm LTO MIR has zero AV32/AV64 spill saves; code-object scratch and
  VGPR/SGPR spill counts are zero.
* Formed MLIR contains the complete recurrence and typed block-dot; dot
  lowering occurs only after the legacy recurrence boundary has been removed.

## Next Single Action: R1 Planner Design Gate

Implement no schedule yet.  First define the input/output contract of a joint
planner over the whole R0 region: persistent state, pred operands, BF16
boundary, K0/K1 update, H/V-decay transients, current/next chunk candidates,
and all shared-bank lifetimes.  Its output must be one recurrence schedule
plan, not a pred plan plus an update plan.

Do not create B3, S1, stream32, local-array lookahead, an LDS swizzle, or an
external-HSACO path in that action.

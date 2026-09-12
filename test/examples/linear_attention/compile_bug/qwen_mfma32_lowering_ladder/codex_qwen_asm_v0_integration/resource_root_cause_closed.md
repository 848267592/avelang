# Resource Root-Cause Closure for ASM v0

## What Is Closed

The old full-v29 compiler path combined generic address formation, fragment
packing, MFMA32 pred values, and update state in one LLVM allocation region.
Its documented failure mode was `Accum_VGPR=384`, `736 B` private scratch,
and 190 spilled VGPR words.

ASM v0 does not lower this body through that path.  It launches a fixed,
independently assembled gfx942 code object whose static metadata reports:

```text
private segment: 0 B
VGPR spills:     0
SGPR spills:     0
high a100+ direct read/write instructions: 0
```

The frozen Triton profile established the corresponding runtime target as
`VGPR=128`, `AccVGPR=192`, `SGPR=80`, `Scratch=0`.  A targeted v0 rocprof run
is still required to record the independently measured v0 row, even though
the normalized compiler-stage source is identical apart from symbol spelling.

## Scope Limit

This closes the resource cliff for the raw opaque FLA/Triton contract.  It
does not close the semantic CASE-C gap to Avelang BT64 v29 or v24, and it does
not make generic Avelang lowering perform like this code object.

# Intended CTA, Wave, and Lane Ownership

The intended workgroup is 256 threads, or four 64-lane wavefronts.

- Wave 0 owns diagonal inverse `X11` and first-level `X21` work.
- Wave 1 owns `X22` and `X32` work.
- Wave 2 owns `X33` and `X43` work.
- Wave 3 owns `X44` and the later dependency levels.

Each wave maps lane `0..63` to the standard FP32 MFMA16 accumulator fragment:
four FP32 accumulator elements per lane for a 16x16 output tile.  Each
`16x16x4` product needs four MFMA instructions for its K=16 reduction.  The
exact fragment load/store mapping must be demonstrated by the availability
repro before this ownership plan can be encoded in a correctness candidate.

Cross-wave consumers read a published 16x16 FP32 block only at DAG-level
boundaries.  No ownership or lane mapping is claimed as implemented while the
FP32 MFMA source feature gate is unavailable.

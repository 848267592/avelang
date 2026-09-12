# v24 Versus Stage 2 W/U

v24 uses two specialized 64-lane MFMA kernels: one for W and one for U.  Each
stages a 16x16 BF16 A/B tile in LDS, exposes packed `(16,2,4)` i32 views to
the MFMA, and writes four FP32 outputs per lane.  Its coefficient rounding
correction keeps the public FP32 result close to scalar accumulation.

Stage 2 deliberately used v6 generic W/U for BT64 correctness bring-up.  It
has a one-thread workgroup and evaluates the complete local matrix with scalar
loops.  The BT64 port therefore reuses the 16x16 primitive, not the scalar
algorithm: every 64-token source dimension becomes four MFMA accumulation
steps and every 64-token output dimension becomes four 16-token row tiles.

The result remains an FP32 W/U tensor because the frozen asm recurrence ABI
requires it, even though the MFMA operands are BF16 staged.


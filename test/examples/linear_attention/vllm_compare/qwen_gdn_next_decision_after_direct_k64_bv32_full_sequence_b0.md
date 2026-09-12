# Next Decision After Direct-K64 BV32 Full-Sequence B0

## Current State

B0 is correctness-locked but not performance-promoted.

- T=64/128/512/2048 passed device-contract checks.
- B0 was byte-exact with the P2 host feedback microscope at every checked chunk.
- B0 audit/body H, V-new and final state are byte-exact at T=64.
- Body code object has zero private scratch and zero MIR/code-object spills.
- Body VGPR count is `460`, a resource cliff.
- The formal five-fresh-process-session matrix completed without Graph capture or allocation in the timed region.
- B0 is `0.158235/0.358733/0.678227/2.778229 ms` at T=512/1024/2048/8192.
- Direct current Triton is `0.070084/0.098306/0.149081/0.455137 ms`; B0 is `2.26x/3.65x/4.55x/6.10x` slower.
- B0's fitted slope is `21.755686 us/chunk`, versus `3.197504 us/chunk` for direct Triton and `3.134995 us/chunk` for the external current-vLLM HSACO bridge.
- T=2048 PMC confirms zero scratch, 65,536 MFMA, 315,904 VMEM, 2,865,536 VALU, 161,216 SALU and 495,616 LDS instructions. Exact LTO still shows `VGPR=460` with zero spills.

## Next Single Action

Do exactly one same-work **pred/update phase-lifetime scheduling A/B** on B0. The two arms must retain B0's full-sequence device loop, BF16 ABI boundary, MFMA32 geometry, BT64/BV32/WG128/32-CTA ownership, C0 block-dot lowering, LDS layout, K32 accumulation order and output contract. Only the live ranges of pred accumulator, V-new fragment and update accumulator may differ.

The acceptance gate is unchanged full-sequence correctness at T=64/128/512/2048 plus the same five-session diagnostic body matrix. The objective is to reduce the `VGPR=460` resource cliff and the `21.76 us/chunk` slope without a scratch/spill regression.

- Do not modify RA, allocator, MFMA geometry, BV32 ownership, block-dot lowering, LDS swizzle, broad-K/compact-K, external HSACO or production dispatch.
- Do not promote B0 as a native performance baseline: the formal matrix already rejects that promotion.

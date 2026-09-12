# Stage 4 Source Audit

## KKT-S0

`_qwen_gdn_kkt_bf16_kernel_bt64_mfma_v2_s0` launches one 64-thread CTA for
each `(chunk, value-head, row16, col16)` tile. It stages two `[16,128]` K
tiles, performs eight MFMA16 reductions, then applies the strict-lower mask,
FP32 decay, and beta during layout-compatible writeback. Upper tiles only
zero their output. This directly ports v24's proven dot primitive without a
single-thread BT64 program.

## WU-S0 and WU-S1

One 256-thread CTA owns a full BT64 row extent and one 16-column output tile;
its four waves own the four token16 row tiles. Shared A and B subtiles are
reused by all waves, cutting CTA count by four relative to Stage 3. Exact S0
retained a scalar FP32 residual loop and missed the performance gate.

S1 keeps the BF16 main product and stages `A_fp32 - bf16(A)` as a second BF16
MFMA product. It removes the 64-step scalar correction while retaining the
accuracy needed by the frozen full contract. The correction-free diagnostic
was faster but failed correctness and remains explicitly named
`qwen_gdn_w_u_bt64_mfma_v2_no_correction_failed`.

## chunk-o-S0

One 256-thread CTA owns `(chunk, value-head, V16)` and four waves own the four
output token16 tiles. Q, H, K, and V-new are staged for shared reuse. Inter
and causal intra products stay in separate FP32 MFMA accumulators and are
summed at final writeback. An attempted single combined accumulator changed
rounding enough to fail the numerical gate, so it was not retained.

## Boundaries

The module accepts only gfx942 experiment-contract shapes: contiguous
`B=1,Hk=4,Hv=8,K=V=128,BT=64`, with `T % 64 == 0`. Unsupported inputs raise
`ValueError`; no production dispatch was changed. Cumsum, v18 solve, and the
external asm-v0 recurrence are reused unchanged.

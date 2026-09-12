# Stage 6B Design

O0: one 256-thread CTA owns `[token64,V64]`; its four waves own token16 rows.
It stages Q `[64,128]` once, constructs the lower ten QK score tiles once per
V64 CTA, retains them in `score_decay_bf16[4,4,16,16]`, and processes four
V16 subtiles sequentially. Each subtile stages the matching H16 and V-new16,
uses short-lived FP32 inter/intra accumulators, and writes the final FP32
output exactly once.

O1 is the only allowed follow-up: identical schedule except V32 ownership and
two sequential V16 subtiles. It tests whether reducing the unrolled V region
reduces resource pressure. It did not.

No partial output tensor, atomics, direct-BF16 output, cast fusion, or
cross-stage fusion is introduced.

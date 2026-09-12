# Stage 5B: One Recommended Implementation

## Decision

Implement exactly one new **audit-gated, standalone hierarchical FP32 BT64
block-inverse solve**, then compare it with v18 before any full-pipeline
integration. Do not alter v18, Stage 4 KKT/W-U/chunk-o, the asm recurrence,
or compiler/assembly.

## Fixed design

- Input/output: contiguous FP32 `[1,T,8,64]`, strict-lower A, output
  `X=(I+A)^-1`, same layout and W/U contract as v18.
- CTA: one `(chunk, head)` matrix; grid `(T/64 * 8,1,1)`.
- Workgroup: 256 threads (four wavefronts), matching the profiled vLLM
  launch shape as a measured starting point, not an assembly dependency.
- Algebra: partition each 64x64 matrix into a 4x4 lower grid of 16x16 blocks;
  form four `D_i^-1`; calculate `X21`, `X32`, `X43`, then `X31`, `X42`, then
  `X41` with the formulas in `solve_math_contract.md`.
- Matrix primitive: FP32 16x16x4 MFMA where AveLang source support and
  correctness permit it. The real vLLM ISA proves this instruction class is
  viable on this exact gfx942 shape; it does not guarantee identical lowering.
- Storage: keep block operands/fragments in registers/LDS only as needed for
  the block DAG. Initial LDS budget gate is <= 8 KiB explicit shared memory;
  private scratch must remain zero.

## Parallel/dependency schedule

The four 16x16 diagonal local inverse calculations have 14 active row updates
each (source range 2..15). They are independent mathematically. The
off-diagonal DAG then has three levels: `{21,32,43}`, `{31,42}`, `{41}`.
Do not reintroduce a 63-row whole-matrix barrier loop.

## Risks and correctness gates

The reduction order differs from v18, so compare independently against the
FP32 authority and v18 for normal, high-dynamic, near-singular, sparse,
single-subdiagonal, cancellation, and real Stage 4 KKT inputs. Start with
the same 54-case matrix. Require no new tolerance beyond the existing
`atol=rtol=1e-5` authority gate, zero scratch, and a non-regressing W/U
consumer check.

## Performance gates

The measured vLLM body is a lower reference of 0.035613 ms at T=2048. A
realistic first Avelang gate is **<= 0.060 ms** body-only, with strong target
**<= 0.050 ms**. Against Stage 4's 0.125687 ms solve stage this would save
about 0.0657-0.0757 ms. Applying that arithmetic to the historical
0.475867 ms Stage 4 full median estimates a 0.400-0.410 ms T=2048 full
result, subject to remeasurement because full latency is not perfectly
additive.

The fallback is not a second competing solve design: if this one fails its
standalone correctness/resource gates, retain v18 and stop. The audit does
not support a compiler or assembly escalation first because neither path
spills and the primary gap is structural.

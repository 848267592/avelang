# Stage 6U Source-Level Contract Audit

## Hierarchical solve

The source is `vllm_compare/qwen_gdn_solve_bt64_hierarchical_fp32_v1.py`.

- Input and current output pointers are FP32 at lines 28-43.
- The retained block matrix `x[7,16,16]`, product workspace `work[16,16]`,
  diagonal recurrence, and every MFMA accumulator remain FP32 (lines 62-70,
  81-112, 125-315).
- The output is cleared at lines 72-79. This defines strict-upper blocks as
  zero. There is no tail/padding contract because the wrapper rejects T not
  divisible by 64 at lines 318-330.
- Diagonal identity is added only after strict-lower recurrence at lines
  107-112. Diagonal blocks are written at lines 114-123.
- Strict-lower cross blocks are written at lines 167-177, 216-219, 258-261,
  and 313-315.
- The public wrapper allocates FP32 output and launches WG=256 at lines
  333-344.

Therefore P0 can preserve every internal FP32 value, DAG edge, barrier, LDS
allocation, MFMA and launch dimension while changing only `out_ptr`, the
global output tensor element type, and final output conversion to BF16.

## Stage 6T F1 W/U

The source is `vllm_compare/qwen_gdn_bt64_fused_wu_eager_stage6t.py`.

- FP32 solved input is declared and viewed at lines 300-314.
- W main coefficient is formed at lines 339-348 and consumed by the four
  lane-group MFMA branches at lines 355-362.
- W residual coefficient is formed at lines 364-375 and consumed at lines
  376-383.
- W accumulators are stored at lines 385-390 before U begins.
- U main coefficient and MFMA are lines 398-421.
- U residual coefficient and MFMA are lines 422-441.
- U is stored at lines 442-447.

The residual is `coeff - f32(bf16(coeff))`. It compensates for information
lost when an FP32 coefficient is represented by a BF16 MFMA operand. Once
solved A is itself BF16, retaining this residual no longer preserves a valid
upstream precision contract and is removable.

## Ownership and duplicate work

Each CTA owns one `(chunk,value_head)` and contains four waves. Four
`col_pair` iterations cover 128 output columns as two 16-column accumulators.
Four `source_tile` iterations cover BT=64 in 16-token reductions. The four
lane-group conditions are source-level fragment selection, but current
lowering executes all four predicated calls, producing the measured 4x
factor. MFMA16 versus native MFMA32 contributes the separate 2x geometry
factor.

Measured F1 accounting is 512 MFMA/CTA for each of W-main, W-residual,
U-main and U-residual: 2048 total. Removing both residual phases predicts
1024 MFMA/CTA and 262144 MFMA at T=2048 (256 CTAs).

W and U do not retain large accumulators simultaneously: W is fully stored
before U accumulators are created. Coefficients are recomputed independently
for W and U because W includes `exp(g)` and U does not. Within each phase,
main and residual repeat the A/beta coefficient load and arithmetic. This is
visible high-level duplication, not hidden compiler duplication.

## Classification

- Mathematically required: hierarchical inverse DAG, identity placement,
  causal zero/strict-lower layout, W and U products, four source tiles, and
  coverage of 128 output columns.
- Current schedule only: FP32-to-BF16 residual correction, four predicated
  lane-group call sites, and MFMA16 geometry.
- Stage 6U changes only the first schedule item. Geometry/predicate work is
  audited later and is not mixed into P0/C0.

# Stage 6U-Solved Final Summary

Stage 6U completed as CASE A. U1 is the selected opt-in experimental Eager
public candidate; Stage 6S production/default remains unchanged.

- P0 is bit-exact to FP32-solve-then-BF16-cast across 42 producer cases.
- C0 removes both W/U residual phases and halves F1 MFMA from 2048 to 1024
  per CTA, with zero scratch/spills.
- Full correctness passes with max output abs `0.001953125` and max final-state
  abs `0.0152155161`.
- T=2048 U1 gain versus Stage 6S is `122.551 us`, paired 95% CI
  `[108.741, 136.236] us`.
- Gap slope improves by `0.851649 us/chunk` versus Stage 6S.
- U2 is N/A; the next action is an isolated C0 lane-predicate collapse.
- All authoritative timing uses complete Eager public API calls; Graph was not
  used.

See `../qwen_gfx942_bt64_bf16_solved_boundary_stage6u_report.md` and
`../qwen_gfx942_bt64_bf16_solved_boundary_stage6u_replay_cn.md`.

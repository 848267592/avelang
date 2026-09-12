# First Divergence

All 37 full-forward cases satisfy the frozen public thresholds: output absolute error <= `1/128` and final-state absolute error <= `2e-2`. The first non-bitwise stage is `g_cumsum`, where FP32 implementation order differs slightly from the vLLM capture. The first downstream differences therefore appear in KKT/solve/W/U. The immutable asm recurrence is bit-exact only when fed its own Stage 1-compatible inputs; in this full path the candidate's upstream FP32 W/U differs from vLLM's BF16 intermediate contract. The final BF16 output and FP32 final state remain within the predeclared acceptance thresholds.

Maximum observed absolute errors are recorded in `final_decision.json` under `notes.stage_max_abs` and the per-case first mismatch coordinates are in `stage_correctness_results.csv`.

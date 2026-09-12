# First Divergence

The diagnostic comparisons intentionally use different solve-to-W/U precision contracts: F1 consumes FP32 `a_solved`, while native vLLM `wy_fast` consumes BF16 `A` from `solve_tril`. Their first nonzero divergence is therefore expected and is diagnostic-only, not a public correctness failure.

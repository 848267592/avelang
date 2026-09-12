# Stage 6U Decision

CASE A. U1 passes full and expanded correctness, removes FP32 solved/W/U
materialization and all solved/W/U casts, halves F1 MFMA, retains zero scratch,
and improves T=2048 versus Stage 6S by 122.551 us with paired 95% CI
[108.741, 136.236] us. Keep U1 as the opt-in experimental best candidate;
production/default Stage 6S remains unchanged.

U2 is N/A in this pass. The next single action is an isolated C0
predicate-collapse experiment targeting the measured 4x lane-group factor.
Do not combine it with the remaining 2x MFMA16/MFMA32 geometry change.

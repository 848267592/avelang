# F1 Predicate Source Map

The four branch groups are:

- W main: `qwen_gdn_bt64_fused_wu_eager_stage6t.py:355-362`.
- W residual: `:376-383`.
- U main: `:413-420`.
- U residual: `:433-440`.

Each group selects one packed BF16 fragment according to `lane_group` (defined
at lines 315-319). It is not a head or tile-boundary predicate. Existing
counter normalization shows all four source call sites contribute dynamic
MFMA work, creating the 4x predicated-fragment factor.

Stage 6U C0 deliberately preserves this mapping. Removing it would mix a
second schedule variable into solved-boundary propagation and is reserved for
the post-correctness Phase E/U2 decision.

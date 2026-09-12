# First Divergence

There is no Stage 4 versus O0/O1 divergence in any executed FP32-staging
test: every reported maximum and mean absolute error is zero. Therefore a
first bad token, V column, token16 block, and V tile are all `N/A`.

The direct Stage4-vLLM body comparison has a small, expected implementation
ordering difference. Its worst executed point is random T=2048:
`max_abs=1.1281809e-05`, `mean_abs=1.1724628e-06`; it is far below the
standalone `4e-3` tolerance and public BF16 contract threshold.

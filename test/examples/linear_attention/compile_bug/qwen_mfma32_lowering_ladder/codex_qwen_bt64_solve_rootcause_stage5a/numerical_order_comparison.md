# Numerical Equivalence and Order

The harness passes the exact same contiguous FP32 tensor object to v18 and
vLLM in each case.  `input_contract_validation.json` records a representative
hash and confirms `same_input_object_for_both=true`; every case records its
own SHA-256 in `correctness.csv`.

The authority is a FP32 `torch.linalg.solve_triangular` evaluation of
`(I+A)X=I` for each local 64x64 chunk/head.  All 54 cases passed
`atol=rtol=1e-5`:

| quantity over all cases | v18 | vLLM |
|:--|--:|--:|
| maximum absolute error vs authority | `4.5776367e-05` | `3.0517578e-05` |
| maximum mean absolute error | `8.4649315e-08` | `8.4892939e-08` |
| maximum infinity residual | `4.8995018e-05` | `1.9311905e-05` |
| maximum cross-implementation absolute difference | `6.1035156e-05` | same comparison |

The maxima arise in deliberately high-dynamic synthetic inputs, not the
ordinary Stage 4 KKT set.  Differences follow from legal FP32 association
changes: row/group partial reductions in v18 versus local 16x16 inverse and
block-dot reductions in vLLM.  No BF16 downcast, altered diagonal rule, or
approximate solve was used in the audited paths.

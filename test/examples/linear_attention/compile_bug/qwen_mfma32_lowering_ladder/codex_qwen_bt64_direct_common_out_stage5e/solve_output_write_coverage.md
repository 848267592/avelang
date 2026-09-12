# Solve Output Write Coverage Audit

## v18

`_qwen_gdn_solve_kernel_v18_parallel` accepts `a_ptr` and `out_ptr` explicitly.
For BT64, 128 threads execute 32 load/store repetitions, covering exactly
`128 * 32 = 4096` matrix elements per chunk/head. The kernel loads all input
elements into LDS before solving, writes every row/column element, and adds one
to each diagonal during final store.

- full 64x64 output coverage: yes;
- upper triangle: copied from `-a`; valid Stage 4 KKT input has zero upper triangle;
- diagonal: explicitly adds one;
- output pre-zero dependency: none;
- output read/atomic/RMW: none;
- safe input/output alias: no; audit wrapper rejects shared storage.

## hierarchical_fp32_v1

`_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1` also accepts `a_ptr` and
`out_ptr`. Its first loop has 256 threads and 16 repetitions, explicitly
zeroing exactly `4096` output elements. Later phases overwrite all diagonal and
strict-lower blocks produced by the block DAG.

- full 64x64 output coverage: yes;
- upper triangle: explicitly zeroed;
- diagonal: explicitly written from four solved diagonal blocks;
- output pre-zero dependency: none;
- output read/atomic/RMW: none;
- safe input/output alias: no; early output clear can destroy input, so rejected.

Both kernels require T divisible by 64. No tail is accepted. Both use FP32,
contiguous `[1,T,8,64]` input/output and current-stream JIT launch semantics.


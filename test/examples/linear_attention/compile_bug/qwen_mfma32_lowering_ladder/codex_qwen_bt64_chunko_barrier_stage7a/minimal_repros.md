# Stage 7A Minimal Barrier Repros

| mode | barriers | MFMA32 | ds read/write | private | spills | median ms | finite |
|:---|---:|---:|:---|---:|:---|---:|:---:|
| A_direct_shared_to_mfma | 1 | 1 | 3/10 | 0 | 0/0 | 0.023234 | True |
| A_fragment_shared_to_mfma | 2 | 1 | 3/10 | 0 | 0/0 | 0.022653 | True |
| B_score_reuse_split_barriers | 2 | 0 | 2/16 | 0 | 0/0 | 0.023335 | True |
| B_score_reuse_merged_barrier | 1 | 0 | 2/16 | 0 | 0/0 | 0.023274 | True |
| C_all_cta_stage_owner_wave | 1 | 1 | 3/10 | 0 | 0/0 | 0.023455 | True |

## Paired Semantics

- `A_direct_shared_to_mfma` vs `A_fragment_shared_to_mfma`: bit_exact=True, max_abs=0.
- `B_score_reuse_split_barriers` vs `B_score_reuse_merged_barrier`: bit_exact=True, max_abs=0.

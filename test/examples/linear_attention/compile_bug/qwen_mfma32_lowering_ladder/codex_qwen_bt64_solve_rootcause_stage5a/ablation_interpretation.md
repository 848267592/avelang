# Audit-Only Ablations

`ablation_results.csv` is copied at the audit root and in `ablations/`.  None
of these kernels is imported by a production path.

| T/input | actual v18 | launch floor | load/store only | no-store checksum |
|:--|--:|--:|--:|--:|
| 64 random | 0.116053 ms | 0.018027 ms | 0.018908 ms | 0.105997 ms |
| 2048 random | 0.122302 ms | 0.017866 ms | 0.024397 ms | 0.106959 ms |
| 2048 diagonal-boundary | 0.122061 ms | 0.017266 ms | 0.022473 ms | 0.106419 ms |

Interpretation boundaries:

- `launch_floor` only writes one element.  It is a lower-bound estimate, not
  a valid solve.
- `load_store_only` keeps the original `[1,T,8,64]` mapping and full output
  traffic, but replaces the recurrence with a copy.  Its small cost rules out
  final global I/O as the main v18 body cost.
- `no_store_checksum` retains LDS staging, the 63-row recurrence and a
  checksum of the last row; it is diagnostic only because dead-code analysis
  could still remove work not feeding that checksum.  Its 0.106959 ms value
  nevertheless bounds final full-output stores plus some output-side work to
  roughly 15.3 us for this input, not the 86.7 us total gap to vLLM body.
- The diagonal-boundary input does not speed v18 materially.  This is
  consistent with structurally fixed barriers and loops dominating over data
  values; it is not a proof of exact branch execution counts.

No safe isolated off-diagonal vLLM diagnostic was added: separating its block
products would require a new solve-like candidate and violate Stage 5A scope.

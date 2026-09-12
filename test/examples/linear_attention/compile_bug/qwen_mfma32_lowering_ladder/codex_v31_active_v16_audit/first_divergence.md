# v31 First Divergence

The first material D1 failure is not partial state feedback. The isolated
P32 padded MFMA32 predicate differs from the BF16 reference at logical output
`[0, 0]`; its largest deterministic mismatch is `[11, 8]`:

| source | value |
|:--|--:|
| BF16 torch reference | `0.0090225022` |
| D1/P32 padded path | `-0.0126709975` |
| P16 direct path | `0.0090225022` |

P32 max/mean absolute errors are `0.02169350` / `0.00614372`. P16 matches the
same reference. D2 T=64 consequently returns to a `3.81e-6` final-state
error. At T=128 the second chunk starts with only `5.72e-6` h error but its
post-update state differs by `0.0400558`, identifying a separate recurrent
numerical-order issue after the pred mapping is fixed.

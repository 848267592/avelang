# F1 MFMA Source Accounting

At T=2048 there are `(2048/64)*8 = 256` CTAs.

| Region | MFMA/CTA | Dispatch MFMA |
|:--|--:|--:|
| W main | 512 | 131072 |
| W residual | 512 | 131072 |
| U main | 512 | 131072 |
| U residual | 512 | 131072 |
| F1 total | 2048 | 524288 |
| Predicted C0 total | 1024 | 262144 |

The source loops are four column pairs times four source tiles times two
simultaneous 16-column accumulators. MFMA16 geometry supplies 2x relative to
MFMA32. Four lane-group branches lower as predicated MFMA calls, supplying the
observed 4x fragment factor. Main plus residual supplies the removable 2x.

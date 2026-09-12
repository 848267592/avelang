# Pred/V-Decay Lifetime Boundary Opportunity

## Why This Note Exists

The isolated K-subtile helper reproduced the favorable `L6_subtile16_stage_full_update_like` counters, but the full Qwen k-subtile experiment did not transfer:

| kernel | trace_us | VGPR | AccVGPR | Scratch | LDS block | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| original full v29 | `817.775` | 128 | 264 | 0 | 61440 | 4977280 | 399360 | 1242304 |
| k_subtile full exp | `2239.567` | 128 | 384 | 84 | 49152 | 3370304 | 1347008 | 2159808 |

The full k-subtile version is correctness-equivalent to original v29, but it is much slower and introduces scratch.

## Evidence From Isolated L6

The strongest lifetime signal is:

| variant | trace_us | AccVGPR |
|:---|---:|---:|
| `L6_baseline_current_update` | `34.371` | 264 |
| `L6_update_mfma_no_pred_dependency` | `10.175` | 80 |
| `L6_update_mfma_minimal_frag` | `3.164` | 4 |

This suggests the update MFMA intrinsic alone is not the problem.  The high pressure appears when pred/v_decay state remains live into the update path.

## Minimal Helper Idea

A possible future helper is an explicit lifetime/staging-boundary marker, conceptually:

```python
al.end_lifetime(pred_acc)
al.end_lifetime(pred_partial)
```

or:

```python
pred_partial = al.discard(pred_partial)
```

The goal would be to tell lowering/register allocation that pred-side temporaries no longer need to remain live once `v_decay_t_bf16` is fully staged.

## Why This Is Still Only A Proposal

- No existing `al.end_lifetime` or `al.discard` mechanism was identified.
- Adding one would require compiler/language work and validation beyond this pass.
- The full k-subtile experiment shows source-level K staging alone is insufficient in the full recurrence kernel.

## Recommendation

Do not implement a broad lifetime system yet.  If v29 continues, the next compiler-facing experiment should be a minimal lifetime-boundary intrinsic or lowering marker placed immediately after `v_decay_t_bf16` is staged and before update MFMA begins.

This is not Triton-like block-dot lowering; it is a narrow lifetime/allocation hint for the Qwen-shaped pred-to-update boundary.

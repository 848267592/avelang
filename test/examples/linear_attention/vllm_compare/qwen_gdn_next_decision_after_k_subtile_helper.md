# Qwen GDN Next Decision After K-Subtile Helper

## Summary

The K-subtile source helper works in the isolated L6 repro but does not transfer to the full v29 fused chunk_gdr kernel as a performance fix.

Decision: do not productionize K-subtile staging for full Qwen.  Do not continue v29 by only tweaking K staging.  The next credible v29 direction would be a narrow pred/v_decay lifetime-boundary experiment; otherwise stop v29 and return to v24-local work.

## What Exactly Is The Compiler/Lowering Problem?

The broad source pattern:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (8 * 4, 4, 1)))
```

is expensive in isolated Qwen-shaped L6.  It combines broad transposed shared K staging with pred/v_decay state feeding update MFMA.

However, after full-kernel testing, the problem is not solved by K staging alone.  Full v29 also suffers from pred/v_decay lifetime pressure and update scheduling/re-staging pressure.

## What Minimal Helper/Source Pattern Fixes It?

In isolation, this source-level pattern fixes the broad K staging cost:

```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (2 * 4, 4, 1)))
```

Isolated result:

| variant | trace_us | VGPR | AccVGPR | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|
| L6 baseline | `34.331` | 128 | 264 | 182144 | 22528 | 28672 |
| L6 subtile | `19.308` | 96 | 168 | 120128 | 16384 | 21504 |
| L6 helper | `19.029` | 96 | 168 | 120128 | 16384 | 21504 |

This supports Option A: source-level helper/pattern, not a compiler rewrite.

## Did The Fix Transfer To Full Qwen?

No.

Full v29 result at `T=2048`:

| kernel | latency_ms | trace_us | VGPR | AccVGPR | Scratch | VALU | VMEM | LDS inst |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| original full v29 | `0.834399` | `817.775` | 128 | 264 | 0 | 4977280 | 399360 | 1242304 |
| k_subtile exp | `2.281168` | `2239.567` | 128 | 384 | 84 | 3370304 | 1347008 | 2159808 |

Correctness transfer:

- k_subtile exp matches original v29 exactly for `h` and `final_state` on tested inputs.
- Both original and k_subtile remain wrong against torch reference for nonzero `w`.
- w=0 update sanity remains good.

So the rewrite is functionally equivalent to original v29, but much slower.

## What Remains?

The remaining issue is not just broad K staging.  The full-kernel negative result points to:

- pred/v_decay live dependency into update;
- repeated K-subtile restaging inside the full update loop;
- higher AccVGPR and scratch after the full rewrite.

The earlier isolated result still matters:

| variant | trace_us | AccVGPR |
|:---|---:|---:|
| L6 baseline | `34.371` | 264 |
| L6 no-pred-dependency update | `10.175` | 80 |
| L6 minimal update frag | `3.164` | 4 |

This suggests a lifetime-boundary experiment is more relevant than another K-staging-only tweak.

## Next Action

Choose one:

1. If continuing v29: implement a minimal compiler/language lifetime-boundary experiment around the pred/v_decay to update transition.
2. If avoiding compiler work: stop v29 full path and return to v24-local optimizations.

Do not:

- productionize full-kernel K-subtile staging;
- keep tuning K staging only;
- claim a generic MFMA lifetime bug;
- modify v23/v24/v26/v27/v28 baselines.

## Evidence Files

- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/k_all_t_bad_pattern_note.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/k_subtile_helper_design.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/k_subtile_helper_report.md`
- `test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/pred_vdecay_lifetime_boundary_opportunity.md`
- `test/examples/linear_attention/vllm_compare/qwen_gdn_v29_k_subtile_exp_report.md`

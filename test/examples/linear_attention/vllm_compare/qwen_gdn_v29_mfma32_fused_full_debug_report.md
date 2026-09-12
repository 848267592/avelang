# Qwen GDN v29 MFMA32 Fused Full Debug Report

## Summary

The v29 fused full failure is not the old FullOp/MFMA16 pollution bug, and it is not a state writeback/readback layout bug.  The first chunk already has BF16-level pred error, and that error is amplified by recurrence when chunk1 uses the updated state.

Key findings:

- chunk0 pred error: `max_abs ~= 2.78e-02`
- chunk0 state_after error: `max_abs ~= 2.62e-01`
- chunk1 normal pred error: `max_abs ~= 1.06e+01`
- state writeback/readback through pred layout: exact after BF16 cast, `0`
- decay_off still fails, so decay/g_last indexing is not the primary issue
- feedback_disabled keeps errors at chunk0 scale, so feedback amplification is the trigger

No debugfix was applied because Track A did not identify a small source-level layout/indexing bug.

## Files

- `qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py`
- `bench_qwen_gdn_v29_mfma32_fused_full_debug.py`

## Modes

`normal`:

Runs BT64 chunks with normal state feedback.

`decay_off`:

Uses `g=0`, so `decay=1` and `g_last_exp=1`.

`feedback_disabled`:

Resets pred state to original `initial_state` each chunk.  This is diagnostic only.

## Results

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 64 --mode normal
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode normal
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode decay_off
PYTHONDONTWRITEBYTECODE=1 python3 qwen_gdn_chunked_avelang_v29_mfma32_fused_full_debug.py --T 128 --mode feedback_disabled
```

### Normal

| T | tensor | max_abs | mean_abs |
|---:|:---|---:|---:|
| 64 | pred | `2.78058201e-02` | `6.17189426e-03` |
| 64 | ucorr | `2.78058052e-02` | `6.17189333e-03` |
| 64 | vdecay | `3.04311514e-02` | `6.31419849e-03` |
| 64 | delta | `2.62058258e-01` | `5.27455807e-02` |
| 64 | state_after | `2.62058258e-01` | `5.27455807e-02` |
| 64 | state_readback_vs_written_bf16 | `0` | `0` |
| 128 | pred | `1.06056023e+01` | `1.10271597e+00` |
| 128 | ucorr | `1.06056023e+01` | `1.10271597e+00` |
| 128 | vdecay | `1.00293064e+01` | `1.08891606e+00` |
| 128 | delta | `8.16010284e+01` | `8.46710968e+00` |
| 128 | state_after | `8.15782013e+01` | `8.46663666e+00` |
| 128 | state_readback_vs_written_bf16 | `0` | `0` |

### Decay Off

| T | tensor | max_abs | mean_abs |
|---:|:---|---:|---:|
| 128 | pred | `1.09993839e+01` | `1.15943897e+00` |
| 128 | ucorr | `1.09993839e+01` | `1.15943885e+00` |
| 128 | vdecay | `1.09993839e+01` | `1.15943885e+00` |
| 128 | delta | `8.80725403e+01` | `9.01883698e+00` |
| 128 | state_after | `8.81770630e+01` | `9.01819420e+00` |
| 128 | state_readback_vs_written_bf16 | `0` | `0` |

### Feedback Disabled

| T | tensor | max_abs | mean_abs |
|---:|:---|---:|---:|
| 128 | pred | `3.00049763e-02` | `6.21640682e-03` |
| 128 | ucorr | `3.00049782e-02` | `6.21640682e-03` |
| 128 | vdecay | `2.97214985e-02` | `6.03089388e-03` |
| 128 | delta | `2.42783546e-01` | `4.91721332e-02` |
| 128 | state_after | `2.42783546e-01` | `4.91721332e-02` |
| 128 | state_readback_vs_written_bf16 | `0` | `0` |

## Interpretation

Chunk0 does not pass at strict recurrence tolerance.  The error is initially BF16-sized in pred, but the state update multiplies and accumulates it into state.  Chunk1 then sees that perturbed state and pred error jumps by roughly three orders of magnitude.

The state writeback/readback probe passed:

```text
state_readback_vs_written_bf16 max_abs=0
```

This rules out the suspected bug where updated state was written to one layout but read by pred through another.

Decay is not the root cause.  With `decay=1` and `g_last_exp=1`, chunk1 still fails at similar scale.

Feedback is the trigger.  When pred is forced to reuse the original `initial_state`, chunk1 remains at chunk0 error scale.

## Decision

No source-level debugfix was made.  The evidence points to recurrence amplification of MFMA32 pred/update numerical differences, not a simple K-half, decay, or state-layout bug.

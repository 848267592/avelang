# Qwen GDN v29 Pred-Only MFMA32 Report

## 1. Summary

v29 `pred_only` compiles, runs, and passes correctness against the torch reference, and ISA confirms real `32x32` BF16 MFMA:

- generated MFMA mnemonic: `v_mfma_f32_32x32x8_bf16`
- no `v_mfma_f32_16x16x16_bf16` found in the v29 kernel ISA

However, performance is a negative result versus the closest available v28 `pred_only` baseline:

- v29 `T=2048`: `0.482717 ms`
- v28 `pred_only T=2048`: `0.241719 ms`

rocprof is even clearer:

- v29 trace median: `519.852 us`
- v28 `pred_only` trace median: `215.420 us`

So this is currently a **no-go** for implementing `v29_update_only` next. The kernel is source-level feasible and mathematically correct, but the current lowering/resource behavior is too expensive.

## 2. What Was Fixed From The Seed Code

No MFMA32 fragment-shape rewrite was needed.

The seed kernel already:

- compiled,
- executed,
- and matched the torch reference closely enough at BF16-level error.

The only source change to the v29 file was auxiliary:

- added optional `--dump-hsaco-dir` support for ISA inspection,
- added a small guard so repeated launches do not fail once the hsaco is already dumped.

I did **not** need to change:

- the candidate accumulator mapping,
- the candidate `a_row = lane_mod32` / `b_row = lane_mod32` operand mapping,
- the MFMA32 operand order.

## 3. MFMA32 Accumulator Layout Mapping

The seed kernel uses:

```python
out_row = ((acc_i & 3) * 8) + (lane_col & 7)
out_col = ((acc_i >> 2) * 8) + ((lane_col >> 3) * 4) + lane_group
```

This mapping was validated empirically by end-to-end correctness against the torch reference at:

- `T=64`
- `T=512`
- `T=2048`

Because correctness already held at those sizes with nonzero `initial_state`, a separate one-hot layout derivation was not required in this pass.

## 4. Operand Mapping

The seed kernel uses:

```python
a_row = lane_mod32
b_row = lane_mod32
```

and issues:

```python
pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[0], a_frag[0], pred_acc)
pred_acc = al.amdgpu.mfma_32x32x8_bf16_f32(b_frag[1], a_frag[1], pred_acc)
```

This matches the working source-level MFMA32 probe pattern and also passed the torch-reference correctness check, so no operand remap was needed here.

## 5. Correctness Table

Reference:

- `qwen_gdn_pred_only_torch_reference`
- nonzero `initial_state`

Results:

| T | max_abs | mean_abs |
|---:|---:|---:|
| 64 | `3.36031318e-02` | `6.10408187e-03` |
| 512 | `3.85222882e-02` | `6.07094867e-03` |
| 2048 | `4.11690474e-02` | `6.08642027e-03` |

Notes:

- The absolute error is stable as `T` grows.
- `max_rel` is not a useful gating metric here because the reference contains many near-zero elements.
- The pytest threshold was therefore set on `max_abs` and `mean_abs`.

Pytest result:

```text
2 passed in 6.17s
```

## 6. Benchmark Table

Measured with:

```bash
python3 bench_qwen_gdn_v29_pred_only_mfma32.py --T 512 1024 2048 --warmup 5 --repeat 20
```

| T | v29 pred_only ms | v28 pred_only ms | v29 speedup vs v28 | max_abs vs ref | mean_abs vs ref |
|---:|---:|---:|---:|---:|---:|
| 512 | `0.102513` | `0.084446` | `0.8238x` | `4.12282348e-02` | `6.06351718e-03` |
| 1024 | `0.237573` | `0.145977` | `0.6145x` | `4.42132801e-02` | `6.06358750e-03` |
| 2048 | `0.482717` | `0.241719` | `0.5007x` | `4.24863696e-02` | `6.09207246e-03` |

Extra comparison:

- `max_abs(v29, v28)` at `T=2048` is `3.69272232e-02`

So v29 and v28 are in the same numerical neighborhood, but v29 is materially slower.

## 7. rocprof Counter Table

v29 `pred_only`, `T=2048`:

| metric | v29 |
|---|---:|
| trace median | `519.852 us` |
| Workgroup_Size | `128` |
| Grid_Size | `4096` |
| VGPR | `60` |
| AccVGPR | `204` |
| SGPR | `112` |
| LDS block | `24576` |
| Scratch | `0` |
| SQ_INSTS_MFMA | `32768` |
| SQ_INSTS_VALU | `3944448` |
| SQ_INSTS_SALU | `279744` |
| SQ_INSTS_VMEM | `198656` |
| SQ_INSTS_LDS | `264192` |
| OccupancyPercent | `0.637270114` |

Closest available v28 `pred_only` baseline from the earlier profiling report:

| metric | v28 pred_only |
|---|---:|
| trace median | `215.420 us` |
| Workgroup_Size | `256` |
| Grid_Size | `8192` |
| VGPR | `64` |
| AccVGPR | `40` |
| SGPR | `112` |
| LDS block | `16384` |
| Scratch | `0` |
| SQ_INSTS_MFMA | `65536` |
| SQ_INSTS_VALU | `1033472` |
| SQ_INSTS_SALU | `79744` |
| SQ_INSTS_VMEM | `196608` |
| SQ_INSTS_LDS | `327680` |
| OccupancyPercent | `1.2433` |

Important comparison:

- v29 MFMA count is lower: `32768` vs `65536`
- but v29 trace is much worse: `519.852 us` vs `215.420 us`
- v29 `AccVGPR` explodes: `204` vs `40`
- v29 occupancy drops: `0.6373` vs `1.2433`
- v29 VALU rises sharply: `3944448` vs `1033472`
- v29 SALU rises sharply: `279744` vs `79744`
- VMEM is similar: `198656` vs `196608`

This strongly suggests the loss is not due to missing MFMA32. The kernel uses fewer MFMA instructions, yet the surrounding lowering and accumulator/control overhead is much worse.

## 8. ISA Evidence

HSACO:

- [v29 hsaco](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/v29_pred_only_mfma32/hsaco/_qwen_gdn_pred_only_bf16_kernel_v29_mfma32.hsaco)

Disassembly:

- [v29_pred_only_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/v29_pred_only_mfma32/v29_pred_only_isa.s)

Confirmed present:

- `v_mfma_f32_32x32x8_bf16`

Representative lines:

- [v29_pred_only_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/v29_pred_only_mfma32/v29_pred_only_isa.s:1313)
- [v29_pred_only_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/v29_pred_only_mfma32/v29_pred_only_isa.s:1318)

Confirmed absent:

- no `v_mfma_f32_16x16x16_bf16` found by grep

So the go/no-go outcome is **not** blocked by MFMA32 lowering failure.

## 9. Comparison To v28 Pred_Only

This is the key result of the experiment:

1. v29 does achieve the intended source-level `32x32` BF16 MFMA lowering.
2. v29 is still slower than v28 `pred_only` at every measured `T`.
3. The slowdown grows with sequence length.

The most plausible explanation from current evidence:

- lower MFMA count alone is not enough,
- accumulator lifetime/allocation is much worse in v29 (`AccVGPR=204`),
- occupancy is cut roughly in half,
- VALU and SALU overhead are much higher,
- VMEM is not the main differentiator here.

So the current implementation is paying a large price for the `32x32` schedule even though it uses the intended hardware instruction.

## 10. Recommendation

Do **not** implement `v29_update_only` yet.

Current recommendation is to stop here and analyze the pred-only resource problem first. The data says:

- correctness passes,
- MFMA32 lowering works,
- but trace and benchmark regress badly versus v28 pred-only.

Most likely next debugging direction:

1. inspect why `AccVGPR` reaches `204` in this pred-only kernel,
2. understand whether LDS staging and partial-reduction layout are forcing bad accumulator lifetime,
3. reduce VALU/SALU overhead before adding update.

So the decision for this pass is:

- **No-go** for `v29_update_only` right now.
- Keep `v29_pred_only` as evidence that source-level MFMA32 works.
- Treat the current kernel as a lowering/resource-pressure negative result, not a production direction yet.

## 11. Files

- [v29 kernel](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_pred_only.py)
- [v29 pytest](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/test_qwen_gdn_v29_pred_only_mfma32.py)
- [v29 bench](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_pred_only_mfma32.py)
- [v29 report](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_v29_pred_only_mfma32_report.md)

# Triton vs Avelang v28 Lowering and AccVGPR Report

## 1. Summary Conclusion

Two profiling tasks were run for the fixed Qwen GDN chunk_delta_h/chunk_gdr shape:

- `B=1`, `T=2048`, `Hk=4`, `Hv=8`, `K=128`, `V=128`
- BF16 `k`, FP32 `w/u/g`
- chunk size `64`

Main findings:

1. Avelang v28 high `Accum_VGPR_Count` is not caused by `h` store, `vn` store, or decay. Those ablations keep AccVGPR high, and even increase it from `212` to `220`.
2. `pred_only` is low AccVGPR (`40`) and `update_only` is moderate (`136`). The high AccVGPR appears only when pred and update are both present in one kernel.
3. Evidence suggests the v28 AccVGPR jump is caused by combined pred/update accumulator lifetime and reuse pressure around the persistent `h1/h2` state fragments, not by output stores.
4. Triton selected config is `BV=32`, `num_warps=2`, `num_stages=2`.
5. Triton is much faster, but not because it has clearly lower VGPR/AccVGPR. Forced selected-config Triton has `VGPR=128` and `AccVGPR=200` without initial state, or `AccVGPR=224` with initial state/final state. Avelang v28 full has `VGPR=52`, `AccVGPR=212`.
6. The stronger Triton-vs-Avelang evidence is instruction/traffic shape: v28 has much higher MFMA, SALU, VMEM, and LDS instructions, plus larger workgroup/grid work-items and a nonzero LDS block allocation.

So the Triton comparison weakens the narrow claim that v28 is slow simply because Avelang has worse VGPR/AccVGPR allocation than Triton. It supports a broader lowering/geometry-cost hypothesis: Avelang v28 expresses a Triton-like `h1/h2` recurrence, but lowers it into substantially more MFMA and memory/LDS work.

## 2. v28 Ablation Counter Table

Normal benchmark command:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  python profile_v28_accvgpr_ablation.py --T 2048 --variant all --warmup 10 --repeat 30
'
```

Normal benchmark latency:

| variant | latency ms |
|:---|---:|
| `full_v28` | 0.604137 |
| `no_h_store` | 0.578940 |
| `no_vn_store` | 0.577078 |
| `no_decay` | 0.572050 |
| `pred_only` | 0.243642 |
| `update_only` | 0.388157 |

rocprof counters:

| variant | trace us | WG | grid work-items | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full_v28` | 577.338 | 256 | 8192 | 36864 | 0 | 52 | 212 | 112 | 327680 | 2758784 | 614912 | 466944 | 1015808 | 1.2819 |
| `no_h_store` | 558.570 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2751744 | 614656 | 401408 | 1015808 | 1.2851 |
| `no_vn_store` | 526.662 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2471808 | 614912 | 401408 | 1015808 | 1.2797 |
| `no_decay` | 544.909 | 256 | 8192 | 36864 | 0 | 44 | 220 | 112 | 327680 | 2176896 | 605952 | 397312 | 1015808 | 1.2749 |
| `pred_only` | 215.420 | 256 | 8192 | 16384 | 0 | 64 | 40 | 112 | 65536 | 1033472 | 79744 | 196608 | 327680 | 1.2433 |
| `update_only` | 341.588 | 256 | 8192 | 20480 | 0 | 16 | 136 | 112 | 262144 | 751232 | 539136 | 165888 | 688128 | 1.2732 |

## 3. v28 AccVGPR Interpretation

Answers to the requested questions:

1. Does `pred_only` already have high AccVGPR?

No. `pred_only` has `AccVGPR=40`, far below `full_v28=212`.

2. Does `update_only` already have high AccVGPR?

Not at the v28 full level. `update_only` has `AccVGPR=136`, close to the older v23/v24 range and far below `212`.

3. Does `full_v28` become high only when pred and update are both present?

Yes. The high value appears in `full_v28` and in variants that keep both pred and update (`no_h_store`, `no_vn_store`, `no_decay`).

4. Do `h`/`vn` stores extend accumulator lifetime?

The evidence says no. Removing `h` store or `vn` store does not reduce AccVGPR. Both variants report `AccVGPR=220`, slightly higher than `full_v28=212`.

5. Does decay extend accumulator lifetime?

The evidence says no. `no_decay` also reports `AccVGPR=220`.

6. Is `AccVGPR=212` caused by unavoidable `h1/h2` state fragments, or by poor accumulator reuse between `pred_acc` and `update_acc`?

Evidence suggests the jump is caused by the combined pred/update region and accumulator lifetime/reuse pressure. The persistent state fragments alone are not enough to reach `212`: `update_only` still has persistent state/update and lands at `136`; `pred_only` lands at `40`. The jump appears when v28 stages persistent state for pred, computes pred partials, creates `v_decay`, and then performs update in the same kernel.

This is still Avelang-side evidence, so the careful conclusion is: evidence suggests the high AccVGPR comes from combined pred+update lowering/lifetime, not from stores or decay.

## 4. vLLM Triton Config and Kernel Identification

Direct vLLM workload:

- Python wrapper: `vllm.model_executor.layers.fla.ops.chunk_delta_h.chunk_gated_delta_rule_fwd_h`
- Triton kernel: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- Regex used for final profiling: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`

Autotune discovery before forced profiling:

```text
patched_vllm_rocm_autotune_configs=12->8,disabled_num_stages=4
selected cache entry:
BV=32,num_warps=2,num_stages=2
```

For final counters, the profiling script forced only this selected config:

```text
--force-bv 32 --force-num-warps 2 --force-num-stages 2
```

This avoids rocprof mixing counters from autotune candidate kernels.

Normal direct chunk_delta_h latency:

| Triton case | initial_state | output_final_state | latency ms |
|:---|:---:|:---:|---:|
| selected forced | false | false | 0.174199 |
| selected forced | true | true | 0.181049 |

## 5. vLLM Triton vs Avelang v28 Counter Table

Closest comparison uses vLLM with `initial_state=True`, `output_final_state=True`, because v28 full loads/stores final state in this benchmark. The no-initial-state vLLM result is also included because it was requested first.

| kernel | case | trace us | WG | grid work-items | LDS block | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occ% |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Avelang v28 | full_v28 | 577.338 | 256 | 8192 | 36864 | 0 | 52 | 212 | 112 | 327680 | 2758784 | 614912 | 466944 | 1015808 | 1.2819 |
| Triton | no init, no final | 138.767 | 128 | 4096 | 0 | 0 | 128 | 200 | 96 | 97280 | 1600896 | 137600 | 77760 | 293824 | 0.6046 |
| Triton | init + final | 142.111 | 128 | 4096 | 0 | 0 | 128 | 224 | 112 | 98304 | 1637824 | 141120 | 78848 | 298432 | 0.6078 |

Derived ratios, Avelang v28 full vs Triton init+final:

| metric | Avelang / Triton |
|:---|---:|
| trace median | 4.06x |
| MFMA | 3.33x |
| VALU | 1.68x |
| SALU | 4.36x |
| VMEM | 5.92x |
| LDS inst | 3.40x |

## 6. Interpretation

### a. Does Triton have much lower AccVGPR than Avelang v28?

No, not under the selected-config counters.

- Triton no-init: `AccVGPR=200`, slightly lower than Avelang v28 `212`.
- Triton init+final: `AccVGPR=224`, higher than Avelang v28 `212`.

This weakens the narrow hypothesis that v28 is slow because Avelang simply uses much more accumulator register file than Triton.

### b. If Triton also has high AccVGPR but is faster, what explains the speed difference?

The strongest counter evidence is instruction and memory/LDS footprint:

- v28 uses `327680` MFMA vs Triton init+final `98304`.
- v28 uses `466944` VMEM vs Triton init+final `78848`.
- v28 uses `1015808` LDS instructions vs Triton init+final `298432`.
- v28 uses `614912` SALU vs Triton init+final `141120`.
- v28 has `LDS_Block_Size=36864`, while Triton reports `0`.
- v28 launches with workgroup `256` and grid work-items `8192`; selected Triton uses workgroup `128` and grid work-items `4096`.

So Triton is faster despite similar/high AccVGPR because its generated kernel does substantially less work in the profiled counters.

### c. If Triton counters cannot be extracted reliably

They were extracted, but an important methodology issue was found: a normal rocprof run with the autotuner enabled mixes candidate configs into the trace/counter CSV. The report therefore uses forced selected config counters (`BV=32,num_warps=2,num_stages=2`) for the final comparison.

## 7. Exact Commands

Syntax check:

```bash
env PYTHONPYCACHEPREFIX=/tmp/profile_v28_vllm_pycache \
  python3 -m py_compile \
  avelang/test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py \
  avelang/test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py
```

v28 normal benchmark:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  python profile_v28_accvgpr_ablation.py --T 2048 --variant all --warmup 10 --repeat 30
'
```

v28 rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  for variant in full_v28 no_h_store no_vn_store no_decay pred_only update_only; do
    outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_${variant}
    /opt/rocm/bin/rocprofv3 \
      --kernel-trace \
      --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
      --kernel-include-regex chunk_gdr \
      -d ${outdir} \
      -o v28_accvgpr_${variant} \
      -f csv \
      -- python test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py \
           --T 2048 --variant ${variant} --warmup 2 --repeat 5
  done
'
```

vLLM selected-config normal benchmark:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
    --T 2048 --warmup 10 --repeat 30 \
    --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

vLLM selected-config rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_forced
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    -d ${outdir} \
    -o vllm_triton_chunk_delta_h_forced \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
         --T 2048 --warmup 2 --repeat 5 \
         --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

vLLM selected-config init+final benchmark and rocprof:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
    --T 2048 --warmup 10 --repeat 30 \
    --with-initial-state --output-final-state \
    --force-bv 32 --force-num-warps 2 --force-num-stages 2 &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_init_state_forced
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    -d ${outdir} \
    -o vllm_triton_chunk_delta_h_init_state_forced \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
         --T 2048 --warmup 2 --repeat 5 \
         --with-initial-state --output-final-state \
         --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

## 8. File Paths

Scripts:

- `test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py`
- `test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py`

Report:

- `test/examples/linear_attention/vllm_compare/triton_vs_avelang_v28_lowering_and_accvgpr_report.md`

rocprof outputs:

- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_full_v28`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_h_store`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_vn_store`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_no_decay`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_pred_only`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_accvgpr_update_only`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_forced`
- `test/examples/linear_attention/rocprof_outputs/qwen_profile_vllm_triton_chunk_delta_h_init_state_forced`

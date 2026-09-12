# Triton vs Avelang v28 ISA and Memory Gap Report

## 1. Executive Summary

This pass did not change any v23/v24/v26/v27/v28 kernel implementation.  It only added a profiling/analysis helper and collected ISA plus memory-counter evidence for the existing Avelang v28 full kernel and the actual vLLM Triton `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` selected config.

The 4.06x trace gap is not explained by AccVGPR alone.  The strongest evidence is dynamic instruction/traffic shape:

- Avelang v28 dynamic MFMA count is `3.33x` Triton.
- Avelang v28 dynamic VMEM count is `5.92x` Triton, split as `5.92x` reads and `5.94x` writes.
- Avelang v28 dynamic LDS instruction count is `3.40x` Triton.
- Avelang v28 dynamic SALU count is `4.36x` Triton.

Static ISA confirms a likely mechanism: Avelang v28 uses only `v_mfma_f32_16x16x16_bf16`, while Triton uses larger `v_mfma_f32_32x32x8_bf16` plus `v_mfma_f32_32x32x4_xf32`.  Avelang also emits many narrow `global_store_dword` stores, while Triton uses vectorized `buffer_store_dwordx4`.  However, store ablations from the earlier report only save tens of microseconds, so writeback is secondary rather than the main 4x cause.

Diagnosis ranking:

1. **Strong evidence:** MFMA shape/decomposition plus loop geometry causes more dynamic MFMA work.
2. **Strong evidence:** VMEM/LDS traffic is much higher in Avelang v28.
3. **Strong evidence:** SALU/address/control overhead is much higher dynamically.
4. **Moderate evidence:** Avelang writeback is narrower/scalarized, but not enough to explain the whole gap.
5. **Not supported as primary:** AccVGPR alone, because Triton selected config has similar or higher AccVGPR.

Recommended next step: create a minimal lowering repro focused on MFMA tile shape plus LDS/VMEM staging, especially the `16x16x16` decomposition versus Triton `32x32` forms.  A store-only repro is useful but lower priority.

## 2. Existing Counter Recap

From `triton_vs_avelang_v28_lowering_and_accvgpr_report.md`, fixed shape:

- `B=1`, `T=2048`, `Hk=4`, `Hv=8`, `K=128`, `V=128`
- BF16 `k`, FP32 `w/u/g`
- chunk size `64`
- Triton selected config: `BV=32`, `num_warps=2`, `num_stages=2`

| metric | Avelang v28 full | Triton init+final | Avelang/Triton |
|:---|---:|---:|---:|
| trace median us | 577.338 | 142.111 | 4.06x |
| VGPR | 52 | 128 | 0.41x |
| AccVGPR | 212 | 224 | 0.95x |
| MFMA | 327680 | 98304 | 3.33x |
| VALU | 2758784 | 1637824 | 1.68x |
| SALU | 614912 | 141120 | 4.36x |
| VMEM | 466944 | 78848 | 5.92x |
| LDS inst | 1015808 | 298432 | 3.40x |
| LDS block | 36864 | 0 | N/A |
| workgroup | 256 | 128 | 2.00x |
| grid work-items | 8192 | 4096 | 2.00x |

## 3. ISA Extraction Method and Paths

Added helper:

- `test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py`

Avelang hsaco was dumped without changing kernels by wrapping `AmdgpuCompiler.compile` and writing `CompiledKernel.kernel` bytes during the existing v28 full call.

Triton hsaco was generated in a dedicated cache directory by forcing the known selected config:

- `--force-bv 32`
- `--force-num-warps 2`
- `--force-num-stages 2`
- `--with-initial-state`
- `--output-final-state`

Dumped files:

| item | path |
|:---|:---|
| Avelang hsaco | `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/hsaco/_qwen_gdn_chunk_gdr_bf16_kernel_v28_triton64_geometry.0.hsaco` |
| Triton hsaco | `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/triton_cache/52F75CUYDCKXZWREEMMJIHXGC37GWPACH6ZGZA7K7A7PJEMLESJQ/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.hsaco` |
| Avelang ISA | `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/avelang_v28_full.s` |
| Triton ISA | `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/triton_chunk_delta_h_selected_init_final.s` |
| ISA summary JSON | `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/isa_category_summary.json` |

Tool:

- `/opt/rocm/llvm/bin/llvm-objdump -d --no-show-raw-insn`

## 4. Whole-Kernel ISA Category Table

These are static instruction counts from disassembly.  They should not be treated as dynamic work; the dynamic rocprof counters above are more representative because loop trip counts and branch paths differ.

| category | Triton static | Avelang static | Avelang/Triton |
|:---|---:|---:|---:|
| total instructions | 3724 | 1986 | 0.53x |
| v_mfma | 96 | 96 | 1.00x |
| global/buffer/flat load | 68 | 107 | 1.57x |
| global/buffer/flat store | 24 | 42 | 1.75x |
| ds_read | 172 | 168 | 0.98x |
| ds_write | 171 | 96 | 0.56x |
| other VALU | 2314 | 896 | 0.39x |
| SALU/control | 816 | 501 | 0.61x |
| branch | 62 | 79 | 1.27x |

Static Triton has more total lines, but dynamic profiling shows Triton executes far fewer MFMA/VMEM/LDS/SALU instructions.  This means the main issue is not raw static code size; it is the executed loop geometry and instruction forms.

## 5. MFMA Mnemonic Comparison

| kernel | static MFMA mnemonics |
|:---|:---|
| Avelang v28 | `v_mfma_f32_16x16x16_bf16`: 96 |
| Triton selected | `v_mfma_f32_32x32x8_bf16`: 32; `v_mfma_f32_32x32x4_xf32`: 64 |

The static MFMA count is equal, but the instruction shapes differ.  This matches the dynamic counter gap: Avelang lowers the work through smaller 16x16x16 BF16 MFMA forms, while Triton uses 32x32 MFMA forms for this K64/BV32 geometry.

Dynamic MFMA is the decisive number:

- Avelang v28: `327680`
- Triton init+final: `98304`
- Ratio: `3.33x`

Conclusion: MFMA shape/decomposition is a primary suspect.

## 6. Epilogue and Writeback Comparison

Static store mnemonics:

| kernel | store mnemonics |
|:---|:---|
| Avelang v28 | `global_store_dword`: 40; plus two helper `buffer_store_format_*` stubs in the object |
| Triton selected | `buffer_store_dwordx4`: 24 |

Avelang v28 has visibly narrower scalarized global stores (`global_store_dword`).  Triton uses vectorized 4-dword buffer stores.

Tail inspection caveat:

- The last 250 static instructions of Triton are mostly trailing `s_nop`, so the report uses direct `store` grep across the whole kernel rather than relying only on the file tail.
- Avelang's tail window includes `16` `global_store_dword` plus two helper store stubs.

Prior ablation evidence:

- `no_h_store` normal latency: `0.578940 ms` vs `full_v28=0.604137 ms`, about `25 us` saved.
- `no_vn_store` normal latency: `0.577078 ms`, about `27 us` saved.

Conclusion: writeback is inefficient and narrow in Avelang, but store removal only saves tens of microseconds.  It is a secondary contributor, not the main 4.06x trace gap.

## 7. Memory and VMEM Throughput Evidence

`rocprofv3 --list-metrics` is not supported in this container.  The available command is:

```bash
/opt/rocm/bin/rocprofv3 --list-avail
```

The attempted EA/DRAM derived counter group failed:

```text
Could not construct profile cfg failed with error code 38:
Request exceeds the capabilities of the hardware to collect
```

So exact DRAM bytes and effective GB/s are unavailable from this run.  The usable memory evidence is VMEM read/write instruction and cycle counters.

Successful VMEM counter group:

```text
SQ_INSTS_VMEM_RD
SQ_INSTS_VMEM_WR
SQ_INST_CYCLES_VMEM_RD
SQ_INST_CYCLES_VMEM_WR
```

| metric | Avelang v28 full | Triton init+final | Avelang/Triton |
|:---|---:|---:|---:|
| trace median us in VMEM run | 578.620 | 141.931 | 4.08x |
| SQ_INSTS_VMEM_RD | 366592 | 61952 | 5.92x |
| SQ_INSTS_VMEM_WR | 100352 | 16896 | 5.94x |
| SQ_INSTS_VMEM total | 466944 | 78848 | 5.92x |
| SQ_INST_CYCLES_VMEM_RD | 366592 | 61952 | 5.92x |
| SQ_INST_CYCLES_VMEM_WR | 100352 | 16896 | 5.94x |

This supports inefficient extra VMEM work in Avelang v28.  Because byte counters were not available, the data cannot prove bandwidth saturation or compute exact GB/s.  It does show Avelang issues about six times as many VMEM read/write instructions under the same shape.

## 8. Diagnosis

| rank | possible cause | evidence | classification |
|---:|:---|:---|:---|
| 1 | MFMA instruction shape/decomposition | Dynamic MFMA `3.33x`; Avelang uses only `16x16x16_bf16`, Triton uses `32x32x8_bf16`/`32x32x4_xf32` | strong |
| 2 | Extra VMEM/global traffic | Dynamic VMEM total `5.92x`; RD `5.92x`; WR `5.94x`; Avelang static stores are narrow `global_store_dword` | strong |
| 3 | Extra LDS staging/reduction | Dynamic LDS inst `3.40x`; Avelang has nonzero `LDS_Block_Size=36864` while Triton reports `0` | strong |
| 4 | SALU/address/control overhead | Dynamic SALU `4.36x`; static Avelang has many `s_waitcnt`, exec-mask branches, and address ops | strong |
| 5 | Epilogue/global stores only | Avelang stores are narrower, but `no_h_store`/`no_vn_store` save only ~25-27 us | weak as primary, moderate as secondary |
| 6 | AccVGPR alone | Triton init+final AccVGPR is `224`, Avelang is `212` | not supported as primary |

Interpretation:

- The trace gap is primarily a lowering/geometry issue, not a pure register allocation issue.
- Avelang appears to execute substantially more dynamic work for the same high-level recurrence.
- The strongest next target is matching Triton's larger MFMA geometry and reducing staging/reduction traffic, not tuning stores first.

## 9. Recommended Next Step

1. Create a minimal backend repro around the MFMA lowering difference:
   - Avelang `16x16x16_bf16` decomposition for BT64/BV32/K64-style pred/update.
   - Triton-like `32x32` MFMA shape as the reference target.
   - Include persistent state as later MFMA operand only if needed after the geometry repro.

2. Create a second LDS/VMEM staging repro:
   - same dynamic loop count,
   - one path using Avelang-style LDS staging/partial reduction,
   - one path minimizing LDS traffic.

3. Store/prologue/epilogue repro is lower priority:
   - Avelang scalarized `global_store_dword` vs vectorized `buffer_store_dwordx4` is real,
   - but current ablations show it is not the main 4x leak.

## 10. Exact Commands

Syntax check:

```bash
env PYTHONPYCACHEPREFIX=/tmp/v28_isa_gap_pycache \
  python3 -m py_compile \
  avelang/test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py
```

Dump Avelang v28 hsaco:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    dump-avelang-v28 \
    --out-dir test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/hsaco \
    --T 2048 --with-initial-state
'
```

Dump Triton selected-config hsaco:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  rm -rf test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/triton_cache &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    dump-triton-selected \
    --out-dir test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap \
    --T 2048 --with-initial-state --warmup 1 --repeat 1
'
```

Disassemble:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    disassemble \
    --hsaco test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/hsaco/_qwen_gdn_chunk_gdr_bf16_kernel_v28_triton64_geometry.0.hsaco \
    --out test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/avelang_v28_full.s &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    disassemble \
    --hsaco test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/triton_cache/52F75CUYDCKXZWREEMMJIHXGC37GWPACH6ZGZA7K7A7PJEMLESJQ/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.hsaco \
    --out test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/triton_chunk_delta_h_selected_init_final.s
'
```

Analyze ISA:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    analyze-isa \
    --isa \
      test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/avelang_v28_full.s \
      test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/triton_chunk_delta_h_selected_init_final.s \
    --out test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/isa/isa_category_summary.json \
    --tail 250
'
```

List available rocprof metrics:

```bash
docker exec ac739c57a0bf sh -lc '
  /opt/rocm/bin/rocprofv3 --list-avail
'
```

Avelang VMEM counters:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_v28_full_vmem &&
  rm -rf ${outdir} &&
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INST_CYCLES_VMEM_RD SQ_INST_CYCLES_VMEM_WR \
    --kernel-include-regex chunk_gdr \
    -d ${outdir} \
    -o v28_full_vmem \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_v28_accvgpr_ablation.py \
      --T 2048 --variant full_v28 --warmup 2 --repeat 5
'
```

Triton VMEM counters:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  outdir=test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_triton_vmem &&
  rm -rf ${outdir} &&
  /opt/rocm/bin/rocprofv3 \
    --kernel-trace \
    --pmc SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INST_CYCLES_VMEM_RD SQ_INST_CYCLES_VMEM_WR \
    --kernel-include-regex chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    -d ${outdir} \
    -o triton_vmem \
    -f csv \
    -- python test/examples/linear_attention/vllm_compare/profile_vllm_triton_chunk_delta_h.py \
      --T 2048 --warmup 2 --repeat 5 \
      --with-initial-state --output-final-state \
      --force-bv 32 --force-num-warps 2 --force-num-stages 2
'
```

Parse VMEM summaries:

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    parse-rocprof \
    --counter-csv test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_v28_full_vmem/v28_full_vmem_counter_collection.csv \
    --trace-csv test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_v28_full_vmem/v28_full_vmem_kernel_trace.csv \
    --kernel-substr v28_triton64_geometry \
    --out test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_v28_full_vmem/summary.json &&
  python test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py \
    parse-rocprof \
    --counter-csv test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_triton_vmem/triton_vmem_counter_collection.csv \
    --trace-csv test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_triton_vmem/triton_vmem_kernel_trace.csv \
    --kernel-substr chunk_gated_delta_rule_fwd_kernel_h_blockdim64 \
    --out test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/memory_triton_vmem/summary.json
'
```

Failed EA/DRAM derived counter attempt:

```bash
/opt/rocm/bin/rocprofv3 \
  --kernel-trace \
  --pmc FETCH_SIZE WRITE_SIZE MemWrites32B BANDWIDTH_EA \
  --kernel-include-regex chunk_gdr \
  ...
```

Result:

```text
Could not construct profile cfg failed with error code 38:
Request exceeds the capabilities of the hardware to collect
```

## 11. File Paths

- Helper script: `test/examples/linear_attention/vllm_compare/profile_v28_isa_memory_gap.py`
- This report: `test/examples/linear_attention/vllm_compare/triton_vs_avelang_v28_isa_memory_gap_report.md`
- Prior AccVGPR report: `test/examples/linear_attention/vllm_compare/triton_vs_avelang_v28_lowering_and_accvgpr_report.md`
- Generated evidence root: `test/examples/linear_attention/rocprof_outputs/qwen_profile_v28_isa_memory_gap/`

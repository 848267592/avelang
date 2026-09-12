# Source-Level MFMA32 And StoreX4 Probe Report

## 1. Summary

Both source-level probes are feasible in current Avelang AMDGPU source code:

- BF16 `32x32` MFMA compiles and lowers to `v_mfma_f32_32x32x8_bf16`.
- `raw_buffer_store_x4` compiles and lowers to `buffer_store_dwordx4`.

This means the two source primitives needed for a v29-style experiment are available from Avelang source without compiler changes.

## 2. Whether MFMA32 Probe Compiles

Yes.

Command:

```bash
cd test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python3 repro_mfma32_and_storex4_probe.py --mode mfma32 --warmup 5 --repeat 20
```

Observed runtime result:

- `latency_ms = 0.0203705`
- `finite = 1`
- `checksum = 8823.265625`
- `max_abs = 31.5097008`

The probe is intentionally a fragment-layout probe, not a reconstructed row-major `C[32,32]` correctness test. Its purpose here is source-level lowering evidence.

## 3. Exact MFMA ISA Mnemonic Generated

Dumped HSACO:

- [mfma hsaco](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/mfma32/_mfma32_bf16_fragment_probe_kernel.hsaco)

Disassembly:

- [mfma32_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/mfma32_isa.s)

ISA grep confirms:

```text
v_mfma_f32_32x32x8_bf16
```

Representative lines:

- [mfma32_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/mfma32_isa.s:21)
- [mfma32_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/mfma32_isa.s:22)

So the source-level call

```python
al.amdgpu.mfma_32x32x8_bf16_f32(...)
```

does reach the expected `32x32` BF16 MFMA ISA.

## 4. MFMA32 Probe Latency And Counters

rocprof files:

- [mfma32 trace](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/mfma32/mfma32_probe_kernel_trace.csv)
- [mfma32 counters](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/mfma32/mfma32_probe_counter_collection.csv)

Median kernel data for `_mfma32_bf16_fragment_probe_kernel`:

| metric | value |
|---|---:|
| trace median | `2.484 us` |
| Workgroup_Size | `64` |
| Grid_Size | `64` |
| VGPR | `24` |
| AccVGPR | `16` |
| SGPR | `16` |
| LDS block | `0` |
| Scratch | `0` |
| SQ_INSTS_MFMA | `16` |
| SQ_INSTS_VALU | `22` |
| SQ_INSTS_SALU | `0` |
| SQ_INSTS_VMEM | `20` |
| SQ_INSTS_LDS | `0` |
| OccupancyPercent | `0.00201065454` |

Interpretation:

- The probe is tiny and clean.
- `AccVGPR=16` is low.
- No LDS and no scratch appear.
- The generated code is not silently falling back to 16x16 MFMA.

## 5. Whether Store_X4 Probe Compiles

Yes.

Command:

```bash
cd test/examples/linear_attention/vllm_compare
PYTHONDONTWRITEBYTECODE=1 python3 repro_mfma32_and_storex4_probe.py --mode store_x4 --warmup 5 --repeat 20 --n-i32 1048576
```

Observed runtime result:

- `latency_ms = 0.0322075`
- `correct = 1`
- `checksum = 549755289600`

So the vectorized raw-buffer copy works functionally from source.

## 6. Exact Store ISA Mnemonic Generated

Dumped HSACO:

- [store hsaco](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/store_x4/_raw_buffer_store_x4_probe_kernel.hsaco)

Disassembly:

- [store_x4_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/store_x4_isa.s)

ISA grep confirms:

- `buffer_load_dwordx4`
- `buffer_store_dwordx4`

Representative lines:

- [store_x4_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/store_x4_isa.s:43)
- [store_x4_isa.s](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/store_x4_isa.s:53)

This is the key source-level proof we wanted for the store side.

## 7. Store_X4 Probe Latency And Counters

rocprof files:

- [store trace](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/store_x4/store_x4_probe_kernel_trace.csv)
- [store counters](/home/jiandongliu/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/store_x4/store_x4_probe_counter_collection.csv)

Median kernel data for `_raw_buffer_store_x4_probe_kernel`:

| metric | value |
|---|---:|
| trace median | `19.228 us` |
| Workgroup_Size | `256` |
| Grid_Size | `262144` |
| VGPR | `8` |
| AccVGPR | `0` |
| SGPR | `32` |
| LDS block | `0` |
| Scratch | `0` |
| SQ_INSTS_MFMA | `0` |
| SQ_INSTS_VALU | `1073152` |
| SQ_INSTS_SALU | `1126400` |
| SQ_INSTS_VMEM | `528384` |
| SQ_INSTS_LDS | `0` |
| OccupancyPercent | `18.154847` |

Interpretation:

- The kernel remains fully scratch-free.
- The path is a real VMEM vectorized copy path.
- The exact store lowering is `buffer_store_dwordx4`, not scalarized `global_store_dword`.

## 8. Any Source Changes Needed To Fix The Provided Code

Only one small helper change was needed in the file:

- Added optional `--dump-hsaco-dir` support so the probe can dump compiled HSACO directly for ISA inspection.

No intrinsic-shape fix was required for the actual kernels:

- `mfma_32x32x8_bf16_f32` call shape compiled as written.
- `make_rsrc + raw_buffer_load_x4 + raw_buffer_store_x4` compiled as written.

Command-side adjustments:

- Host syntax check needed `python3` instead of `python`.
- Host `py_compile` needed `PYTHONPYCACHEPREFIX=/tmp/...` because the local `__pycache__` path was not writable.

## 9. Conclusion

The two source-level capabilities we cared about are both present and working:

- MFMA32 works from source and generates `v_mfma_f32_32x32x8_bf16`.
- vectorized raw-buffer store works from source and generates `buffer_store_dwordx4`.

That means:

- implementing `v29_pred_only` with `2-wave`, `128-thread`, `32x32` MFMA is source-level feasible;
- vectorized store should stay a secondary optimization candidate for v29 writeback/epilogue work;
- there is no immediate evidence here of a source API limitation blocking either primitive.

This report does not yet prove that a full v29 Qwen GDN kernel will be fast. It proves the two critical source-side building blocks are available.

## 10. Short v29 Plan

Recommended next implementation target:

- `qwen_gdn_chunked_avelang_v29_gdr_bt64_bv32_mfma32_layout_fixed.py`

Minimal plan:

1. Start with a `pred-only` chunk_gdr microkernel, not full GDN.
2. Use `BT=64`, `BV=32`, `Kq=32`, split state as `h1[BV,64] + h2[BV,64]`.
3. Map pred GEMM to `32x32x8` MFMA over K-quarter tiles, with `2` waves / `128` threads first.
4. Keep update/writeback simple at first; correctness and ISA shape come before fusion.
5. If pred-only lowers cleanly and counters stay reasonable, add update path next.

## 11. Exact Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_probe python3 -m py_compile \
  /home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/repro_mfma32_and_storex4_probe.py
```

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  PYTHONDONTWRITEBYTECODE=1 python3 repro_mfma32_and_storex4_probe.py \
    --mode mfma32 --warmup 5 --repeat 20 \
    --dump-hsaco-dir /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/mfma32
'
```

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare &&
  PYTHONDONTWRITEBYTECODE=1 python3 repro_mfma32_and_storex4_probe.py \
    --mode store_x4 --warmup 5 --repeat 20 --n-i32 1048576 \
    --dump-hsaco-dir /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/store_x4
'
```

```bash
docker exec ac739c57a0bf sh -lc '
  /opt/rocm/llvm/bin/llvm-objdump -d --no-show-raw-insn \
    /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/mfma32/_mfma32_bf16_fragment_probe_kernel.hsaco \
    > /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/mfma32_isa.s
'
```

```bash
docker exec ac739c57a0bf sh -lc '
  /opt/rocm/llvm/bin/llvm-objdump -d --no-show-raw-insn \
    /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/hsaco/store_x4/_raw_buffer_store_x4_probe_kernel.hsaco \
    > /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/store_x4_isa.s
'
```

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex _mfma32_bf16_fragment_probe_kernel \
    -d test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/mfma32 \
    -o mfma32_probe -f csv \
    -- python3 test/examples/linear_attention/vllm_compare/repro_mfma32_and_storex4_probe.py \
      --mode mfma32 --warmup 5 --repeat 20
'
```

```bash
docker exec ac739c57a0bf sh -lc '
  cd /workspace/project/avelang &&
  /opt/rocm/bin/rocprofv3 --kernel-trace \
    --pmc SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_VMEM SQ_INSTS_LDS OccupancyPercent \
    --kernel-include-regex _raw_buffer_store_x4_probe_kernel \
    -d test/examples/linear_attention/rocprof_outputs/mfma32_storex4_probe/rocprof/store_x4 \
    -o store_x4_probe -f csv \
    -- python3 test/examples/linear_attention/vllm_compare/repro_mfma32_and_storex4_probe.py \
      --mode store_x4 --warmup 5 --repeat 20 --n-i32 1048576
'
```

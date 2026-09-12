# Qwen Update K-Fragment Helper Lowering Report

## Summary

The narrow helper is mathematically correct, but it does not improve L6 lowering.

`al.amdgpu.qwen_update_kfrag_load_bf16x4(shared_k, k_col, token_base)` replaces the source-level `kall_vec` access and returns the exact `vector<4xbf16>` consumed by one MFMA16 call. Baseline/helper output is bit-identical (`max_abs=0`). However, the helper currently lowers immediately to `ave.memref.load_vec` and then to a standard vector load. Canonicalization reconstructs the same addrspace(3) GEP and broad shared-staging structure as baseline.

Consequently:

- `Accum_VGPR_Count` remains `264`;
- trace changes from `34.251 us` to `34.572 us`;
- the real hsaco still writes ordinary temporaries through AGPR indices up to `a131`;
- MFMA, VMEM, LDS, VGPR, LDS allocation, and scratch are unchanged.

This experimental helper should not be migrated to original full v29 in its current form.

## Implementation

The helper was added to the AMDGPU named intrinsic module:

```python
b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
    k_all_t_fixed,
    tile * 16 + lane_col,
    token_tile * 32 + lane_group * 4,
)
acc = al.amdgpu.mfma_16x16x16_bf16_f32(a_frag, b_frag, acc)
```

Its contract is intentionally narrow:

- input shared tensor: BF16 `[128,64]` in workgroup memory;
- dynamic `k_col` and `token_base` integer/index values;
- result: `vector<4xbf16>`, exactly one 64-bit MFMA16 B fragment.

The frontend implementation emits one `AveLangMemRefLoadVecOp` directly from the original shared K tensor. It does not create the rank-3 i32 `kall_vec` view.

## L6 Source Paths

Baseline:

```python
k_all_t = al.make_shared((128, BT), al.bf16)
kall_vec = al.view(k_all_t, al.i32, al.make_layout((128, 8, 4), (32, 4, 1)))
b_words = kall_vec[tile * 16 + lane_col, token_pack]
b_frag = al.view(b_words, al.Tensor((2, 4, 1), al.bf16))
```

Helper:

```python
k_all_t_fixed = al.make_shared((128, BT), al.bf16)
b_frag = al.amdgpu.qwen_update_kfrag_load_bf16x4(
    k_all_t_fixed, tile * 16 + lane_col, token_base
)
```

Subtile control:

```python
k_sub_t = al.make_shared((128, 16), al.bf16)
ksub_vec = al.view(k_sub_t, al.i32, al.make_layout((128, 2, 4), (8, 4, 1)))
```

## Correctness And Smoke

GPU: AMD Instinct MI300X, ROCm 7.2.2, workgroup 128, grid work-items 4096.

| variant | normal median ms | finite | checksum abs |
|:---|---:|:---:|---:|
| `L6_baseline_current_update` | `0.045567` | yes | `129497.109375` |
| `L6_fixed_kfrag_helper_update` | `0.046069` | yes | `129497.109375` |
| `L6_subtile16_stage_full_update_like` | `0.033790` | yes | `129497.109375` |

Direct baseline/helper comparison:

| metric | value |
|:---|---:|
| baseline finite | true |
| helper finite | true |
| max abs | `0` |
| mean abs | `0` |

## Rocprof Counters

| metric | baseline | fixed helper | subtile control |
|:---|---:|---:|---:|
| trace median us | `34.251` | `34.572` | `19.188` |
| Workgroup_Size | `128` | `128` | `128` |
| Grid_Size | `4096` | `4096` | `4096` |
| VGPR_Count | `128` | `128` | `96` |
| Accum_VGPR_Count | `264` | `264` | `168` |
| SGPR_Count | `112` | `112` | `112` |
| LDS_Block_Size | `45056` | `45056` | `32768` |
| Scratch_Size | `0` | `0` | `0` |
| SQ_INSTS_MFMA | `5120` | `5120` | `5120` |
| SQ_INSTS_VALU | `182144` | `182208` | `120128` |
| SQ_INSTS_SALU | `11200` | `11200` | `11264` |
| SQ_INSTS_VMEM | `22528` | `22528` | `16384` |
| SQ_INSTS_LDS | `28672` | `28672` | `21504` |
| OccupancyPercent | `0.5035` | `0.5030` | `0.4197` |

The helper misses every success gate except correctness, equal MFMA count, and zero scratch. It does not reduce AccVGPR or trace.

## ISA And MIR Audit

### MFMA and memory instructions

| static ISA metric | baseline | fixed helper | subtile control |
|:---|---:|---:|---:|
| `v_mfma_f32_32x32x8_bf16` | `8` | `8` | `8` |
| `v_mfma_f32_16x16x16_bf16` | `32` | `32` | `32` |
| `ds_read` | `88` | `88` | `80` |
| `ds_write` | `152` | `152` | `104` |
| `global_load` | `144` | `144` | `96` |
| total instructions | `3144` | `3145` | `2132` |

Baseline and helper both contain 64 static `ds_read_b64` instructions. Their ISA differs materially only by one additional shift in the helper variant.

### High AGPR copies

| metric | baseline | fixed helper | subtile control |
|:---|---:|---:|---:|
| maximum explicit AGPR write index | `131` | `131` | `3` |
| `v_accvgpr_write_b32` count | `155` | `155` | `32` |
| writes with AGPR index >= 100 | `31` | `31` | `0` |
| maximum explicit AGPR read index | `131` | `131` | `3` |

The baseline and helper have the same high-AGPR index set in the relevant region (including `a100..a128`, `a130`, and `a131`; `a129` is not present in this build). The values written there are address/data temporaries around broad K staging, not MFMA accumulators.

### Late MIR metadata

| metric | baseline | fixed helper | subtile control |
|:---|---:|---:|---:|
| `ldsSize` | `45056` | `45056` | `32768` |
| `hasSpilledVGPRs` | false | false | false |
| `vgprForAGPRCopy` | empty | empty | empty |
| `scratchReservedForDynamicVGPRs` | `0` | `0` | `0` |

This remains register allocation under pressure, not scratch spilling.

## Lowering Diagnosis

The helper does remove the explicit source-level `kall_vec` view. It does not preserve a distinct fixed-fragment operation through the compiler pipeline. The frontend implementation creates `AveLangMemRefLoadVecOp`; AveLang-to-memref converts it to `vector.load`; LLVM optimization then exposes the same broad addrspace(3) GEP/staging graph as baseline.

The optimized LLVM IR still contains:

- the full `[128,64]` workgroup allocation;
- the full global-K to LDS staging address chain;
- repeated addrspace(3) GEPs for the update fragment loads;
- the same MFMA16 fragment load count and broad live ranges.

Therefore the measured failure is primarily:

```text
helper lowering did not survive as a fixed-offset primitive and did not bypass
the generic GEP/staging graph before register allocation
```

The unchanged AGPR placement is a consequence. This experiment does not establish that AMDGPU RA would still choose high AGPRs after a genuinely distinct late fixed-offset lowering, because such a lowering was not reached.

There is a second important constraint: read-side replacement alone leaves the full broad K staging work intact. The subtile control reduces VMEM by `6144`, LDS instructions by `7168`, and LDS allocation by `12288` bytes. A read-only helper cannot reproduce those reductions while the producer still stages the full `[128,64]` tile.

## Required Answers

### Did the helper replace generic kall_vec lowering?

At source/AveLang construction level, yes: helper code has no `al.view(k_all_t, i32, ...)` and directly requests a four-BF16 fragment.

At optimized LLVM/ISA level, no: it canonicalizes to the same generic GEP/vector-load structure and nearly identical ISA.

### Did high AGPR copies disappear?

No. Both baseline and helper reach explicit AGPR index `131`, with 31 writes at index 100 or above.

### Did AccVGPR or trace fall?

No. AccVGPR remains `264`; trace regresses slightly from `34.251 us` to `34.572 us` (about `0.9%`).

### Is it worth migrating to original full v29?

No. The helper is correct but has no lowering or performance benefit. Migrating it would add a Qwen-specific API without solving the measured issue.

### Where did the attempt fail?

It failed before RA: the helper lowered too early to `vector.load`, so it did not bypass the GEP/staging graph. RA then made the same high-AGPR placement decision as baseline.

## Next Compiler Action

Do not hard-ban ordinary temporaries from AGPR and do not migrate this helper to full v29.

The next compiler experiment must use a real dialect op that survives beyond AveLang-to-memref lowering, together with a targeted lowering/rewrite that owns both sides of the pattern:

1. preserve a `qwen_update_kfrag` op through memref type conversion and GPU outlining;
2. recognize the broad K staging producer and the exact fragment consumer footprint;
3. lower to a compact physical LDS fragment/staging region with direct 64-bit loads;
4. remove dead broad-tile stores and the full transposed view before LLVM RA.

Merely changing the read expression is insufficient; the producer staging graph must also be transformed at the compiler level.

## Reproduction

Build:

```bash
cmake -S . -B build-kfrag -G Ninja \
  -DAVE_LANG_BACKEND=rocm -DWITH_PYTHON=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/local/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++
cmake --build build-kfrag --target _avelang_bindings -j 16
```

Profile:

```bash
PYTHONPATH=build-kfrag/python:python PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/profile_qwen_kfrag_helper_lowering.py \
  --warmup 10 --repeat 30 --rocprof-warmup 2 --rocprof-repeat 5
```

Late LLVM/MIR dump:

```bash
PYTHONPATH=build-kfrag/python:python PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/dump_l6_mir_regalloc_artifacts.py \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering/mir
```

Artifacts:

- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering/`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering/hsaco/`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_helper_lowering/mir/`


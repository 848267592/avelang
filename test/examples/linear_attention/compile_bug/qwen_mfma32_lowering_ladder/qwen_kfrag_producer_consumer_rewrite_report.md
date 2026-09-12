# Qwen K-Fragment Producer-Consumer Rewrite Report

## Summary

The guarded persistent K-fragment rewrite passes the isolated L6 gate.
It is bit-exact against the broad `[128,64]` baseline and removes the high
AGPR-copy region associated with that staging path.

At workgroup `128` on MI300X, the rewrite changes the rocprof trace median
from `34.412 us` to `18.628 us` (`1.847x`, 45.9% lower). It keeps the dynamic
MFMA count at `5120`, uses no scratch, and is slightly faster than the source
subtile control (`19.188 us`).

## Compiler Changes

- `lib/Dialect/AveLang/IR/AveLangOps.td`
- `lib/Dialect/AveLang/IR/AveLangOps.h`
- `lib/Dialect/AveLang/IR/AveLangOps.cc`
- `lib/IR/Intrinsics/amdgpu_module.cc`
- `lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.h`
- `lib/Dialect/AveLang/Transforms/qwen_kfrag_producer_consumer_rewrite_pass.cc`
- `lib/Dialect/AveLang/Transforms/CMakeLists.txt`
- `lib/Target/GPU/lower_to_llvm.cc`

The opt-in source helper is:

```python
al.amdgpu.qwen_update_kfrag_load_bf16x4(
    shared_k, source_k, thread_id, key_head,
    token_window_base, k_column, token_fragment_base,
)
```

It creates `ave.gpu.amdgpu_qwen_update_kfrag_load`, rather than lowering
immediately to a generic vector load. The dedicated pass runs after
AveLang-to-memref conversion and before intrinsic linking/LLVM lowering.

## Rewrite And Guards

The pass only accepts the isolated Qwen L6 pattern:

- BF16 shared K tile `[128,64]` in workgroup memory;
- BF16 source K `[1,64,4,128]`;
- one `scf.for 0..64` producer containing one matching shared store and one
  rank-4 BF16 K load;
- exactly four persistent `vector<4xbf16>` consumers from the same producer;
- each consumer used exactly once as operand B of
  `mfma_f32_16x16x16bf16_1k`.

It replaces the broad producer and generic `kall_vec` consumer chain with a
workgroup `[128,16]` compact tile for the active token window, then emits a
direct `vector.load vector<4xbf16>` for each MFMA B fragment. The original
`[128,64]` producer loop and dead shared allocation chain are erased before
LLVM register allocation.

The first implementation had two integration issues, both fixed before the
measurement:

1. Function-argument lowering introduced equivalent memref views/casts, so
   producer matching must use the fixed source shape instead of only SSA
   identity.
2. Scalar lexical values are private-memref reloads in consumer branches.
   The pass clones those scalar reloads at the compact-stage insertion point
   so all new operands dominate their use.

Any unrewritten persistent op is a pass failure, so successful compilation is
evidence that the persistent op survived through type conversion and was
consumed by this pass. MLIR remarks are not forwarded by the Python JIT in
this build; the `rewrite_remark_seen=false` profiler field is not a rewrite
failure.

## Correctness

Command:

```bash
PYTHONPATH=/tmp/avelang-build-kfrag-qwen-rocm722/python:/workspace/project/avelang/python \
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/repro_qwen_mfma32_l6_kstage_update_variants.py \
  --check-rewrite-equivalence --seed 20260621
```

| comparison | finite | max_abs | mean_abs |
|:---|:---:|---:|---:|
| baseline vs producer-consumer rewrite | yes / yes | `0` | `0` |

All three smoke variants have the same sink checksum:
`129497.109375`.

## Benchmark

| variant | normal median ms | trace median us | speedup vs baseline trace |
|:---|---:|---:|---:|
| `L6_baseline_current_update` | `0.046389` | `34.412` | `1.000x` |
| `L6_fixed_kfrag_producer_consumer_rewrite` | `0.033590` | `18.628` | `1.847x` |
| `L6_subtile16_stage_full_update_like` | `0.033930` | `19.188` | `1.793x` |

## Rocprof Counters

| metric | baseline | rewrite | subtile control |
|:---|---:|---:|---:|
| Workgroup_Size | `128` | `128` | `128` |
| Grid_Size | `4096` | `4096` | `4096` |
| VGPR_Count | `128` | `84` | `96` |
| Accum_VGPR_Count | `264` | `180` | `168` |
| SGPR_Count | `112` | `112` | `112` |
| LDS_Block_Size | `45056` | `32768` | `32768` |
| Scratch_Size | `0` | `0` | `0` |
| SQ_INSTS_MFMA | `5120` | `5120` | `5120` |
| SQ_INSTS_VALU | `182144` | `118656` | `120128` |
| SQ_INSTS_SALU | `11200` | `11200` | `11264` |
| SQ_INSTS_VMEM | `22528` | `16384` | `16384` |
| SQ_INSTS_LDS | `28672` | `21504` | `21504` |
| OccupancyPercent | `0.5045` | `0.4199` | `0.4251` |

Relative to baseline, the rewrite reduces VGPR by 34.4%, AccVGPR by 31.8%,
LDS allocation and VMEM by 27.3%, LDS instructions by 25.0%, and VALU by
34.9%. OccupancyPercent is the profiler's reported normalized value; it is
not the explanation for the speedup because the rewrite lowers the dominant
register and memory work while MFMA work is unchanged.

## ISA And MIR Evidence

| static ISA / MIR metric | baseline | rewrite | subtile control |
|:---|---:|---:|---:|
| MFMA32 instructions | `8` | `8` | `8` |
| MFMA16 instructions | `32` | `32` | `32` |
| global_load | `144` | `96` | `96` |
| ds_read | `88` | `80` | `80` |
| ds_write | `152` | `104` | `104` |
| total ISA instructions | `3144` | `2082` | `2132` |
| `v_accvgpr_write_b32` count | `155` | `32` | `32` |
| high AGPR writes, index >= 100 | `31` | `0` | `0` |
| max AGPR write index | `131` | `3` | `3` |
| MIR ldsSize | `45056` | `32768` | `32768` |
| MIR hasSpilledVGPRs | false | false | false |
| MIR scratchReservedForDynamicVGPRs | `0` | `0` | `0` |

The fixed variant therefore removes the observed `a100..a131` high-AGPR-copy
class rather than merely moving it. It also reaches the same K staging traffic
class as the subtile control while retaining a compiler-owned, opt-in source
interface.

## Decision

The helper successfully replaces the generic broad `kall_vec` lowering for
this guarded pattern. It is worth testing in an experimental original full
v29 copy, but it must not replace any current production baseline yet.

The remaining limitation is scope: the pass intentionally recognizes only the
L6 fixed shape and has not been validated against the full recurrent Qwen
chunk_gdr control flow. The next action is a chunk_gdr-only experimental full
v29 copy that uses the persistent helper at the exact broad K producer and
MFMA16 B-consumer site, then compares state-update correctness before any full
forward benchmark.

## Artifacts

- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_producer_consumer_rewrite/`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_producer_consumer_rewrite/mir/`
- `test/examples/linear_attention/rocprof_outputs/qwen_kfrag_producer_consumer_rewrite/hsaco/`


# Full v29 LTO MIR Audit and Pred Streaming Experiment

## Scope

This report answers two narrow questions without modifying a production
baseline:

1. What exactly creates the full-v29 K-fragment rewrite's `736 B` scratch and
   `Accum_VGPR_Count=384` report?
2. Can one source-level MFMA32 accumulator serialization change reduce the
   pred epilogue's register/LDS pressure while preserving original-v29
   semantics?

The audited kernel is the full v29 K-fragment rewrite experiment, not v24:

`_qwen_gdn_fused_chunk_gdr_full_kfrag_rewrite_exp_bf16_kernel_v29_mfma32`.

## Exact LTO Debug Chain

The initial JIT LLVM IR is not sufficient here: final AMDGPU register
allocation occurs inside ROCm `ld.lld` full LTO. The Avelang backend now
records a replayable linker command when this variable is set:

```bash
AVELANG_AMDGPU_LINK_DEBUG_DIR=<directory>
```

The backend change is in `lib/Target/AMDGPU/amdgpu_backend.cc`. It preserves
the JIT pre-link bitcode, final code object, and replayable argv. The replay
script appends the exact LTO-plugin flags:

```text
-plugin-opt=save-temps
-plugin-opt=-print-before=greedy
-plugin-opt=-print-after=greedy
-plugin-opt=-print-after=virtregrewriter
-plugin-opt=-print-after=prologepilog
```

Artifacts:

- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/linked.hsaco.0.5.precodegen.bc`
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/post_lto_precodegen.ll`
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_07.mir`:
  spill-producing post-greedy MIR
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_08.mir`:
  post-virtregrewriter MIR
- `rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra/kernel_section_18.mir`:
  later physical-register MIR containing the high-AGPR copies

The reproducible driver is
`replay_qwen_v29_lto_mir.py` in this directory.

## Exact Scratch Location

The fresh T=2048 profile of the full K-fragment rewrite is:

| metric | value |
|:--|--:|
| trace median | `1302.317 us` |
| scratch | `736 B` |
| VGPR | `128` |
| AccVGPR | `384` |
| SGPR | `112` |
| LDS block | `61440 B` |
| MFMA | `294912` |
| VMEM | `601984` |

The final code object independently reports:

```text
.private_segment_fixed_size: 736
.vgpr_spill_count: 190
.sgpr_spill_count: 25
.agpr_count: 256
```

The exact post-greedy machine section has:

| MIR operation | count | virtual-register classes |
|:--|--:|:--|
| `SI_SPILL_AV32_SAVE` | `70` | `vgpr_32` |
| `SI_SPILL_AV64_SAVE` | `60` | `vreg_64_align2`, `av_64_align2` |

Thus the VGPR spill-word count is exactly:

```text
70 * 1 + 60 * 2 = 190
```

This matches `.vgpr_spill_count: 190`; it is the direct MIR explanation for
the `736 B` private segment. Representative spilled virtual registers are
`%6842:vreg_64_align2`, `%6848:vreg_64_align2`, `%6854:vreg_64_align2`, and
the scalar sequence `%7554` through `%7863:vgpr_32`. The full list is in
`exact_lto_postra/summary.json` and `kernel_section_07.mir`.

`Accum_VGPR_Count` is a profiler resource metric, not a one-to-one virtual
register ID. The direct mapping available from MIR is physical allocation:
the final physical section contains high AGPR copies beginning at
`$agpr100` and reaching `$agpr254` (`166` high-AGPR copy lines). The sequence
is preceded by broad `V_LSHL_ADD_U64` address formation and followed by
fragment copies feeding MFMA paths. The post-LTO LLVM IR contains repeated
`<2 x i64>`/`<2 x i32>` extraction chains in the same generic vector-style
lowering region.

This is strong evidence that the rewrite's generic dynamic vector/address
path crosses the AGPR/VGPR allocation threshold. It does not prove that every
high AGPR is a K fragment alone: the exact machine output shows the combined
address/fragment/live-region composition, including pred and update state.

## One Pred Accumulator Serialization Experiment

New experimental kernel:

`vllm_compare/qwen_gdn_chunked_avelang_v29_mfma32_pred_epilogue_streaming_exp.py`

Only the pred MFMA32 epilogue changes. The original stores `pred_acc[16]`
directly to a permuted shared `pred_partial[2,32,32]` tile. The experiment
stores the same values in a lane-major `pred_acc_serial[2,64,16]` tile, with
the same 8 KiB LDS footprint. The correction epilogue uses the inverse
MFMA32 layout to reload the two K-half partials and generate `v_decay`.

No pred schedule, update recurrence, K staging, chunk size, or public
`h/final_state` interface changes.

### Semantic Check Against Original v29

| T | h max abs | final-state max abs |
|--:|--:|--:|
| 64 | `0` | `0` |
| 512 | `0` | `0` |
| 1024 | `0` | `0` |
| 2048 | `0` | `0` |

The experiment is bit-exact relative to original v29. This does not resolve
the separate known nonzero-W v29-vs-reference recurrence issue.

### Normal Timing

| T | original v29 ms | serialized epilogue ms | speedup |
|--:|--:|--:|--:|
| 512 | `0.235409` | `0.222250` | `1.0592x` |
| 1024 | `0.437590` | `0.430279` | `1.0170x` |
| 2048 | `0.835021` | `0.819657` | `1.0187x` |

### T=2048 rocprof

| metric | original | serialized epilogue | change |
|:--|--:|--:|--:|
| trace median us | `827.169` | `785.347` | `-5.06%` |
| VGPR | `128` | `128` | `0` |
| AccVGPR | `264` | `256` | `-8` |
| scratch | `0 B` | `0 B` | `0` |
| LDS block | `61440 B` | `61440 B` | `0` |
| MFMA | `294912` | `294912` | `0` |
| VALU | `4977280` | `4961792` | `-15488` |
| SALU | `810496` | `810688` | `+192` |
| VMEM | `399360` | `399360` | `0` |
| LDS instructions | `1242304` | `1164480` | `-77824` |
| occupancy percent | `0.64499` | `0.64514` | effectively unchanged |

## Conclusion

The LTO debug chain now proves the `736 B` scratch is real register spilling:
`190` VGPR spill words are visible in exact post-greedy MIR. It also exposes
the high `$agpr100..$agpr254` copy region that accompanies the generic
vector/address lowering under the full pred+update live region.

The one allowed source experiment is a positive but small result. Compact
lane-major serialization eliminates the permuted accumulator-unpack path,
keeps original-v29 semantics bit-exact, reduces AccVGPR by `8`, and improves
T=2048 trace by `5.06%`. It does not eliminate the deeper full-composition
pressure or make v29 a production candidate. Keep v24 as the production
baseline; use this as evidence that a future design should keep MFMA32 output
serialization compact rather than expanding it through generic permuted
epilogue lowering.

## Commands

```bash
# Capture an exact Avelang LTO input while compiling the full rewrite.
export AVELANG_AMDGPU_LINK_DEBUG_DIR=\
  /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_link_replay

# Replay the captured ROCm LTO command with pre/post RA dumps.
python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/replay_qwen_v29_lto_mir.py \
  --argv-file test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_link_replay/amdgpu-link-0.argv.txt \
  --out-dir test/examples/linear_attention/rocprof_outputs/qwen_v29_full_mir_regalloc/exact_lto_postra

python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_v29_pred_epilogue_streaming.py -s

python3 test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v29_pred_epilogue_streaming.py \
  --T 512 1024 2048 --warmup 5 --repeat 20
```

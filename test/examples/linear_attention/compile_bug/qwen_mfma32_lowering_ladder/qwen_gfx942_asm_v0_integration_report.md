# Qwen gfx942 ASM v0 Integration Report

## Summary

The experimental raw BT64 recurrence route is implemented and passes its
target stage gate. The contract is **CASE C**: the frozen Triton operator is
correct for the raw FLA/vLLM recurrence, but its pred cast/order and public
`h` dtype differ from the historical Avelang BT64/v24 pipeline.

Final symbol:

```text
qwen_gdn_bt64_gfx942_asm_v0
```

The core assembly body was not algorithmically changed. The compiler-stage
Triton AMDGCN source was copied and mechanically renamed to the new symbol;
after normalizing the symbol name, the source is byte-identical to the frozen
golden source.

The v0 path is opt-in and opaque. It loads the dedicated HSACO through the
existing HIP external-kernel bridge and does not lower the recurrence into
generic AveLang memref/vector/MFMA operations. v24 and all production
defaults remain unchanged.

## Contract Difference

| Item | asm v0 / Triton | historical Avelang BT64 or v24 |
|:--|:--|:--|
| pred | FP32 resident `w/state`, `v_mfma_f32_32x32x4_xf32` | BF16-staged pred path |
| `h` | BF16 `[1,T/64,8,128,128]` | public Avelang containers are FP32; v24 is BT16 |
| gate input | raw `g`, exponentiation in kernel | v29 commonly receives precomputed decay/scale |
| chunk | BT64, BV32 | v24 production chain is BT16 |
| initial state | strict non-null FP32 in v0 | Avelang wrappers commonly allow `None` |

The common recurrence pieces are `key_head = value_head // 2`, `v_new =
u - pred`, BF16 corrected-decay/key update, and FP32 final state. Because the
differences are observable, the adapter does not convert v0 output into a
false v24-equivalent full-forward result. Detailed records are in
`codex_qwen_asm_v0_integration/contract_diff.md` and `.json`.

## Runtime Integration

The experimental high-level entry point is:

```python
qwen_gdn_bt64_gfx942_asm_v0(k, w, u, g, initial_state)
```

The explicit container adapter is
`qwen_gdn_chunk_gdr_avelang_bt64_gfx942_asm_v0(...)`. Runtime checks cover
gfx942, fixed shapes/dtypes/layouts, contiguity, `T % 64`, HSACO SHA256,
current HIP stream, and the exact 88-byte kernarg ABI. Guard failures use an
explicit caller-provided fallback; launch failures are raised.

| Field | Value |
|:--|:--|
| grid | `[4,8,1]` |
| workgroup | `[256,1,1]` |
| dynamic LDS argument | `57344` bytes |
| kernarg segment | 88 bytes |
| stream | current PyTorch HIP stream |

## Correctness

The GPU correctness runner completed **46 cases**: 40 random nonzero-W cases
plus six special modes: `zero_w`, `zero_state`, `unit_decay`, `high_dynamic`,
`cancellation`, and `small_scale`. It covered T=64, 128, 512, and 2048.

Against the real vLLM golden kernel, original HSACO, rebuilt HSACO, and asm v0
all had zero max absolute error, zero mean absolute error, and no first
mismatch for `h`, `v_new`, and `final_state` in every case.

| Test | Result |
|:--|:--|
| asm v0 pytest | `5 passed` |
| external HSACO bridge tests | `8 passed` |
| v31 P16 existing tests | `55 passed` |
| 46-case raw correctness runner | exit 0; all three artifacts exact vs vLLM |
| nonzero-W | passed against vLLM authority |
| T=64/128/512/2048 | passed against vLLM authority |

The project/v29 diagnostic reference is intentionally separate. It shows the
expected CASE-C numerical difference from BF16 staging and recurrence order;
it is not used as the v0 authority. In ordinary random cases, the largest
observed project-reference errors were approximately `0.001953` for `h`,
`0.000608` for `v_new`, and `0.000931` for final state. `high_dynamic`
amplifies this cast/order difference. This prevents claiming v0 is a drop-in
v24/v29 numerical replacement, but does not weaken the bit-exact vLLM gate.

## Same-Harness Benchmark

The strict comparison used the same C++ HIP harness, same inputs, same dynamic
LDS launch, module loading before timing, warmup 10, repeat 50, and three
independent sessions. T=512 is shown as session medians because its
sub-0.1-ms workload had a visible cold/cache outlier.

| T | golden original ms | golden rebuilt ms | asm v0 ms | old v29 context ms | v31 context ms |
|---:|---:|---:|---:|---:|---:|
| 512 | `1.586597/0.081921/0.086609` | `1.615040/0.082843/0.089292` | `1.615480/0.082963/0.154350` | `0.433924` | `0.647702` |
| 2048 | `0.195331` | `0.197573` | `0.196371` | `0.833899` | `1.582211` |
| 8192 | `0.567604` | `0.567364` | `0.567084` | `3.227254` | `6.230597` |
| 16384 | `1.164608` | `1.164450` | `1.164971` | `6.649940` | `12.539170` |

At T=2048, asm v0 is `+0.53%` relative to original golden and `-0.61%`
relative to rebuilt golden. At T=8192 and 16384 it remains within about
`0.1%` of golden. The T=512 row is not treated as a stable optimization
claim. Old-v29 and v31 are context only: their visible outputs and contracts
are different and they are not semantic speedup baselines.

## rocprof Resources

All rows below were collected with the same harness and rocprof command at
T=2048. `LDS_Block_Size=0` is the profiler's static LDS field; the launch
still supplies the required 57,344-byte dynamic LDS argument.

| implementation | trace us | grid work-items | WG | LDS field | scratch | VGPR | AccVGPR | SGPR | MFMA | VALU | SALU | VMEM | LDS inst | Occupancy |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| golden original | 145.755 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.201728 |
| golden rebuilt | 145.776 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.210285 |
| asm v0 | 145.656 | 8192 | 256 | 0 | 0 | 128 | 192 | 80 | 196608 | 2535040 | 214016 | 91136 | 588928 | 1.204427 |
| old v29 context | 798.866 | 4096 | 128 | 61440 | 0 | 128 | 264 | 112 | 294912 | 4977280 | 810496 | 399360 | 1242304 | 0.644988 |
| v31 context | 1550.784 | 4096 | 128 | 27648 | 0 | 36 | 228 | 112 | 327680 | 4671424 | 1447616 | 661504 | 1690816 | 0.644323 |

Static v0 metadata reports private segment 0, VGPR spill count 0, and SGPR
spill count 0. Its disassembly has no scratch loads/stores and no direct
`v_accvgpr_read/write_b32` reference at or above `a100`; the maximum direct
accumulator index is `a63`. The old failed generic full-v29 rewrite remains
documented separately as `AccVGPR=384`, `736 B` scratch, and 190 spilled VGPR
words. The v0 path avoids that compiler region; it does not repair generic
Avelang lowering.

## ISA Evidence

The v0 disassembly contains:

- `v_mfma_f32_32x32x4_xf32`: 96 static instructions;
- `v_mfma_f32_32x32x8_bf16`: 48 static instructions;
- `s_barrier`: 44;
- LDS read/write instructions: 483;
- scratch load/store instructions: 0;
- direct high-AGPR references at `a100` or above: 0.

This confirms the Triton compiler-stage MFMA schedule was preserved,
including the XF32 pred path and BF16 update path. No padded MFMA32 schedule
was introduced.

## Full-Forward Boundary

The asm v0 recurrence was not wired into v24 full forward. This is a
correctness boundary:

1. v24 is a BT16 cumsum/KKT/solve/w_u/chunk_gdr/chunk_o pipeline;
2. asm v0 is BT64 and emits BF16 chunk-start `h`;
3. the pred cast/order differs;
4. v24 chunk_o expects BT16 chunk layout.

Widening BF16 `h` to FP32 alone does not repair chunk boundaries or numerical
semantics. `full_forward_benchmark.csv` therefore records the comparison as
not applicable, and no v24/asm-v0 full-forward latency is claimed.

The next full-forward bottleneck is a separately validated BT64 upstream and
downstream contract: KKT/solve/w_u/chunk_o plus the XF32/BF16 cast policy.

## Reproduction and Decision

The consolidated runner is:

```bash
cd /workspace/project/avelang
bash test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_asm_v0_integration/commands.sh
```

The raw recurrence gate passes: exact vLLM correctness, golden-matched
resources, and same-stage trace within 0.1% at long-text sizes. The old
generic-lowering resource cliff is avoided for this opaque route.

It is not production-ready as a v24 full-forward replacement because the
contract is CASE C. Keep it opt-in. `ready_for_step_2_profile` is false until
a correct BT64 upstream/downstream full-forward boundary is defined and
validated; no full-forward performance claim is made here.

# gfx942 BT64 Single-Step Assembly Report

## Status

This report records the completed ABI and baseline gates for the dedicated raw
assembly route. It does **not** claim a full Qwen BT64 assembly kernel exists:
the full pred/epilogue/update/state kernel has not yet been implemented, so
there is no valid assembly correctness or performance result and no AveLang
external-HSACO integration.

That distinction is deliberate. The task requires strict reference correctness
and a performance gate before integration; reporting smoke numbers as Qwen
numbers would be misleading.

## Why Raw Assembly Is Being Audited

The v29/v30/v31 sequence showed that generic lowering can create an unfavorable
combined pred/update live region, including high AGPR pressure and spills. A
standalone gfx942 assembly step is intended to test a fixed P16 schedule whose
register, LDS, waitcnt, and barrier behavior is explicit. Production v24 and
all existing production paths remain untouched.

## Assembly Smoke ABI Gate: Passed

Files are under
`compile_bug/qwen_mfma32_lowering_ladder/codex_gfx942_asm_bt64_step_audit/smoke/`.

- Symbol: `qwen_gfx942_asm_smoke`
- Build target: `amdgcn-amd-amdhsa`, `gfx942`
- Code object: ELF AMDGPU HSA ABI version 4, assembly code object directive v6
- Launch: grid `4`, block `128`
- Kernarg: two pointers plus one `uint32_t`, `24` bytes / align `8`
- Runtime: `hipModuleLoad`, `hipModuleGetFunction`, `hipModuleLaunchKernel`
- Correctness: all `512` `output[i] = input[i] + addend` values pass
- ISA: real `global_load_dword` and `global_store_dword` are present
- Static/private scratch: `0 B`
- rocprof: `Scratch_Size=0`, `LDS_Block_Size=0`, `VGPR_Count=4`,
  `Accum_VGPR_Count=4`, `SGPR_Count=16`

The first descriptor-only object assembled but failed
`hipModuleLoad(...): no kernel image is available for execution on the device`.
Adding the `.amdgpu_metadata` AMDGPU note was the necessary fix.

## Single-Step Math And Baselines

The common BT64 reference is the existing v31 authoritative recurrence:

```text
pred[t, v] = bf16(W[t, :]) @ bf16(old_state[v, :])
v_decay[t, v] = bf16((U[t, v] - pred[t, v]) * decay[t])
update[v, k] = sum_t(v_decay[t, v] * bf16(K[t, k]))
new_state[v, k] = scale * old_state[v, k] + update[v, k]
```

Shape: `B=1,T=64,Hk=4,Hv=8,K=V=128`; one v31 launch contains the normal
`32` independent head/value blocks, grid `32`, workgroup `128` / two waves.

The correct direct P16 pred mapping is retained as the only permitted pred
mapping. It splits K into two K64 wave partials and each P16 tile uses:

```text
lane_col = lane & 15
lane_group = lane >> 4
vec_idx = lane_group + seg32 * 4, for seg32 in {0, 1}
mfma_16x16x16_bf16_f32(fragment[0])
mfma_16x16x16_bf16_f32(fragment[1])
```

The existing P16 primitive regression suite passed `55` cases on the MI300
container. It includes random and extreme primitive inputs. The full assembly
debug-intermediate suite is not applicable until that kernel exists.

## Measured Gate Baseline

Warmup `10`, repeat `50`, same seeded nonzero inputs (`20260712`):

| implementation | median ms | p10 ms | p90 ms | reference state max abs |
|:--|--:|--:|--:|--:|
| v31 P16 BT64 step | `0.081842` | `0.079959` | `0.085367` | `2.99215e-05` |
| actual vLLM Triton BT64 chunk-delta-h | `0.056984` | `0.056243` | `0.062052` | `1.68294e-04` |
| hand-written Qwen assembly step | not implemented | - | - | not run |

The vLLM comparison is the actual `chunk_gated_delta_rule_fwd_h` wrapper with
`chunk_size=64`, `head_first=False`, and an initial state. Its BF16 numerical
ordering differs slightly from the v31 reference; that is recorded in
`reference_and_triton_single_step.json` rather than hidden.

Triton is currently `1.436x` faster than the v31 P16 baseline. A candidate
assembly kernel must be at most `1.5x` Triton's latency (currently
`<=0.085476 ms`) and at least 30% faster than v31 (`<=0.057289 ms`) while
remaining correct. The two gates are nearly the same numerical target.

## Full Assembly Gate

The full raw `.s` kernel must still implement all of the following before any
claim of performance success:

- direct P16 two-K64 pred, with no MFMA32;
- cross-wave partial reduction and immediate U/decay epilogue;
- compact BF16 `v_decay[64,32]` LDS representation;
- K16-at-a-time update without a broad `[128,64]` transposed LDS view;
- FP32 old-state scaling and exactly one final global state store;
- debug output mode for pred, v_decay, update, and new state;
- fifty-plus randomized/extreme reference tests.

`assembly/register_plan.md` and `assembly/waitcnt_barrier_plan.md` hold the
constrained plan, not an implemented kernel. No generic lowering, LLVM RA
change, or inline-assembly shortcut is part of that plan.

## Decision

| Gate | Status |
|:--|:--|
| gfx942 raw assembly build/load/launch | pass |
| smoke correctness and scratch-zero | pass |
| authoritative BT64 reference | available |
| v31 P16 mapping verification | pass, 55 tests |
| actual vLLM Triton baseline | available |
| full assembly correctness | not run |
| 30% v31 speedup | not evaluable |
| <=1.5x Triton | not evaluable |
| minimal AveLang external HSACO dispatch | not started |
| multi-chunk recurrence | prohibited / not started |

The current only remaining issue is implementing the full explicit P16
pred-to-update assembly schedule without reintroducing the generic lowering
live-region pressure. It is not valid to enter multi-chunk recurrence or
Avelang integration until that single-step kernel clears all gates.

## Reproduction

Use `codex_gfx942_asm_bt64_step_audit/commands.sh` inside the MI300 container.
The complete smoke source, HSACO, disassembly, metadata, rocprof CSV files,
reference manifest, and commands live in the audit directory.

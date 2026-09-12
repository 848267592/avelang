# Qwen Triton Full-Sequence Assembly Audit

## Summary

The verified vLLM/Triton `chunk_delta_h` BT64 kernel extends to real long
sequences without a new specialization: `T` is runtime and the same gfx942
HSACO executes a persistent `ceil(T/64)` loop inside each program. Original,
reassembled AMDGCN, and the audit-only full-sequence external bridge are
bit-exact against the real vLLM wrapper on their tested contracts.

The LDS alias gate did **not** pass. The assembly shows that Triton already
reuses its high LDS fragment band across barrier-separated phases, while the
remaining low/K ranges are dynamically addressed and loop-carried. More
importantly, the fixed `(4,8)` grid contains only 32 workgroups at every T;
reducing LDS would not expose additional independent grid work. No assembly
variant was created.

## Actual Long-Sequence Path

| T | HSACO | config | grid / WG | dynamic LDS | vLLM wrapper event median |
|--:|:--|:--|:--|--:|--:|
| 64 | same `cb1811...f03f9` | BV32, 4 warps, 2 stages | `(4,8)` / 256 | 57,344 B | 0.058588 ms |
| 128 | same | same | same | same | 0.060390 ms |
| 512 | same | same | same | same | 0.088191 ms |
| 2048 | same | same | same | same | 0.193968 ms |
| 8192 | same | same | same | same | 0.616917 ms |
| 16384 | same | same | same | same | 1.214906 ms |

`@triton.jit(do_not_specialize=["T"])` and cache hashes both confirm that no
new T-specific code object is created. A wrapper call launches one
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` dispatch. The long-sequence
slope is therefore the persistent per-workgroup recurrence, not dispatch
count growth.

## Golden Reproduction

The compiler-stage `.amdgcn` was copied from the selected Triton artifact and
reassembled unchanged for gfx942. Original and rebuilt HSACOs match vLLM
bit-exactly for all visible outputs (`h`, `v_new`, and final state):

- 36 cases: 30 random cases spread across T=64/128/512/2048 plus six extreme
  modes; all original/rebuilt-vLLM max errors are zero.
- T=8192 and T=16384: original/rebuilt smoke checks are likewise zero for all
  three outputs.
- Full external bridge pytest at T=512/2048: `3 passed`, including cache,
  guard/fallback, hash rejection, and non-default stream coverage.

The direct torch recurrence is recorded as a numerical reference diagnostic.
Its ordinary-case errors are small, but high-dynamic inputs diverge due to
Triton MFMA/exp reduction order (observed maxima: h `0.5625`, v_new `1.1608`,
state `0.7231`). It was not used to weaken the stronger bit-exact vLLM gate.

## Resource and Trace Evidence

Original and rebuilt have identical static ISA and per-dispatch counters. At
T=2048 rocprof reports `WG=256`, grid work-items `8192` (=32 WGs),
`VGPR=128`, `AccVGPR=192`, `SGPR=80`, `Scratch=0`, and dynamic counts
`MFMA=196608`, `VALU=2535040`, `SALU=214016`, `VMEM=91136`, `LDS=588928`.

The raw rocprof trace medians are retained in `fullseq_device_trace.csv`.
They show substantial shared-system noise, particularly at the shortest and
longest measurements; rebuilt has no changed instruction sequence, resource
tuple, or mathematical code. Neither the small trace deltas nor the noisy
HIP-event external timings are claimed as an optimization.

v24 is a full-forward pipeline with cumsum/KKT/solve/w_u/chunk_o and is not a
same-operator latency row. Its existing production timing is contextual only;
this audit compares the `chunk_delta_h` recurrence stage directly.

## LDS and Occupancy Gate

The 57,344 B dynamic LDS map is documented in `lds_region_map.md` and
`lds_regions.json`:

| band | observed purpose | result |
|:--|:--|:--|
| 0--32767 | dynamic layout/transpose/dot scratch | no strict free lifetime proven |
| 32768--49151 | K/update repacking | consumed in every recurrence update |
| 49152--57343 | W/V/MFMA fragment conversion | already reused across barriers by Triton |

By LDS capacity alone, 57,344 B and 48 KiB allow one 256-thread WG per CU;
32 KiB could allow two. That does not cross a useful residency threshold for
this launch: there are only 32 WGs in the entire grid, regardless of T, and
the 192 AccVGPR resource is also high. No safe alias and no measurable
occupancy benefit were established. Therefore:

```text
lds_alias_v1: not implemented
old dynamic LDS: 57,344 B
new dynamic LDS: n/a
performance gate: n/a
```

## External Full Bridge

`qwen_gdn_bt64_gfx942_external_full(...)` is an audit-only opt-in dispatcher.
It accepts only gfx942, B=1, exact dtypes/layouts, T divisible by 64, and a
non-null FP32 initial state. It checks a local HSACO SHA256, passes the current
HIP stream, exact `kernelParams`, and 57,344 B dynamic LDS, then caches the
module/function. It remains separate from all production dispatch.

## Decision

`final_decision.json` records the result. The next bottleneck is the fixed
32-workgroup geometry with each program serially walking all BT64 chunks. Do
not continue LDS aliasing: it lacks both a proven safe region and a residency
benefit. The appropriate future work is an algorithmic/work-distribution or
fusion design that creates more independent long-sequence work, not another
local assembly offset edit.

All commands and generated artifacts live under
`compile_bug/qwen_mfma32_lowering_ladder/codex_triton_fullseq_asm_opt_audit/`.

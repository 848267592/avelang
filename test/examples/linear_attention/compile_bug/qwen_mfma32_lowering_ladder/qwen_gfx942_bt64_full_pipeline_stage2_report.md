# Qwen gfx942 BT64 Full-Pipeline Stage 2

## Result

Stage 2 is complete as an opt-in experimental pipeline. The new entry point is
`qwen_gdn_full_bt64_gfx942_asm_v0(...)` in
`vllm_compare/qwen_gdn_full_bt64_gfx942_asm_v0_experimental.py`. Its formal
execution path contains no `vllm` import and does not call the vLLM full
wrapper. vLLM is used only by the golden and benchmark harnesses.

The observed next bottleneck is the **generic-v6 BT64 `chunk_o` fallback**.
This is not the v24 production MFMA output path. Stage 2 did not implement a
native BT64 output kernel, fuse stages, or alter the immutable assembly
recurrence.

## Real Operator Boundary

The authoritative entry is
`vllm.model_executor.layers.fla.ops.chunk.chunk_gated_delta_rule`.
Its source-level graph is:

```text
chunk_local_cumsum
  -> chunk_scaled_dot_kkt_fwd
  -> solve_tril
  -> recompute_w_u_fwd
  -> chunk_gated_delta_rule_fwd_h
  -> chunk_fwd_o
  -> BF16 public output + optional FP32 final state
```

The candidate graph and exact source mapping are recorded in
`codex_qwen_bt64_full_pipeline_stage2/full_operator_execution_graph.md` and
`stage_source_map.md`.

## Frozen Contract

- Fixed target: gfx942, `B=1,Hk=4,Hv=8,K=V=128`, `[B,T,H,D]`, contiguous.
- `BT=64`; reject `T % 64 != 0`; no tail support is claimed.
- Inputs: BF16 `q/k/v`; FP32 `g/beta`; optional FP32 initial state.
- `initial_state=None` becomes a zero FP32 state outside asm.
- Reused asm is unchanged CASE C: FP32 `w/u/g/h0`, XF32 prediction, BF16
  `h`, FP32 `v_new` and final state.
- Candidate cumsum/KKT/solve/W-U reuse v6/v18 BT64-capable stages. Candidate
  W/U is FP32 for the asm ABI; the vLLM captured W/U is BF16.
- Experimental chunk-o uses the generic v6 primitive with `chunk_size=64` and
  a FP32 view of BF16 `h`; it is not the v24 BT16 MFMA chunk-o wrapper.
  Likewise, the experimental W/U is generic v6 rather than v24's v14 MFMA
  W/U implementation.
- Public output is BF16 and final state is FP32. Acceptance thresholds were
  frozen before the matrix: output abs <= `1/128`, final-state abs <= `0.02`.

Details are in `bt64_numerical_contract.md`, `bt64_numerical_contract.json`,
`cast_policy.md`, and `shape_layout_contract.md`.

## Correctness

`stage2_runner.py --random-cases 30 --capture` executed 37 cases: 30 random
nonzero cases covering T=64/128/512/2048, plus neutral-gate, high-dynamic,
cancellation, small-value, and two T=8192 smoke cases. Zero and nonzero
initial states are both included. Every full result was accepted.

| quantity | maximum abs error | threshold |
|:--|--:|--:|
| public BF16 output | `0.001953125` | `0.0078125` |
| FP32 final state | `0.014624655` | `0.020000000` |
| KKT | `5.78135e-04` | diagnostic |
| solve | `1.28478e-03` | diagnostic |
| W | `1.83117e-03` | diagnostic |
| U | `6.01139e-02` | diagnostic |
| recurrence H | `0.03125` | diagnostic |
| recurrence V_new | `0.0869646` | diagnostic |

The first non-bitwise difference is the FP32 cumsum (`9.53674e-06` max).
Downstream stage values differ further because the candidate's FP32
intermediates intentionally bridge to the immutable asm ABI while vLLM's
captured solve/W/U are BF16. This is documented rather than hidden; the public
outputs remain under the predeclared thresholds. Per-case data and first
coordinates are in `stage_correctness_results.csv` and
`first_divergence_report.md`.

## Regression Tests

- asm-v0 recurrence: `5 passed`.
- external HSACO/full bridge, gfx942 smoke, and P16: `63 passed, 1 skipped`.
- new full BT64 pipeline suite: `4 passed`.

The assembly body, symbol, grid/workgroup ABI, LDS, MFMA order, and dispatch
were not modified. The recurrence's resource tuple remains `VGPR=128`,
`AccVGPR=192`, `SGPR=80`, and zero scratch/spills.

## End-to-End Timing

Each row is the median of three independent HIP-event session medians, with
warmup=10 and repeat=50. Candidate/v24 and vLLM run in separate Python
processes. Candidate full timing is explicitly marked cached-allocator because
the reused generic wrappers retain internal allocation behavior.

| T | candidate BT64 asm-v0 ms | v24 BT16 ms | vLLM ms |
|--:|--:|--:|--:|
| 512 | `4.925` | `0.303` | `0.308` |
| 2048 | `16.041` | `0.590` | `0.364` |
| 8192 | `80.907` | `2.584` | `0.735` |
| 16384 | `201.451` | `5.206` | `1.273` |

At T=2048 the experimental candidate is `27.19x` v24 and `44.12x` vLLM.
These comparisons are diagnostic: v24 uses a different BT16 numerical path,
while the candidate has the correct BT64 full API but deliberately generic
upstream/downstream stages. The raw three-session samples and p10/p90 context
are retained in `full_pipeline_benchmark.csv`.

## T=2048 Targeted Profile

The table uses the median of the last five matching rocprof dispatches. These
are device trace measurements, not the HIP-event full latency above.

| stage | trace ms | workgroup | grid work-items | scratch | MFMA | VALU | SALU | VMEM |
|:--|--:|:--|:--|--:|--:|--:|--:|--:|
| cumsum | `0.012` | `1` | `256` | 0 | 0 | 17,152 | 22,272 | 32,768 |
| KKT | `0.325` | `1` | `16,384` | 0 | 0 | 207,568,896 | 12,812,288 | 11,116,544 |
| solve | `0.109` | `128` | `32,768` | 0 | 0 | 2,695,424 | 2,535,680 | 32,768 |
| W/U | `3.612` | `1` | `16,384` | 0 | 0 | 800,227,328 | 17,006,592 | 34,603,008 |
| asm recurrence | `0.472` | `256` | `1024 x 8` | 0 | 196,608 | 2,535,040 | 214,016 | 91,136 |
| chunk-o | `11.761` | `1` | `16,384` | 260 B | 0 | 1,079,058,432 | 317,063,168 | 215,351,296 |

For asm, rocprof reports global work-items `1024 x 8`; this normalizes to the
frozen launch grid `(4,8,1)` with workgroup 256. Its uninstrumented preallocated
HIP-event time at T=2048 was `0.205 ms`, or about `1.28%` of the candidate's
full median. Profiling perturbation explains why its trace value is higher.

This generic-v6 `chunk_o` fallback dominates the experimental profile: it has
no MFMA or LDS work, one-thread workgroups, 260 B scratch, and 1.079B VALU /
317M SALU / 215M VMEM instructions. Its `11.761 ms` trace is about 3.26x the
generic-v6 W/U trace and roughly 72% of the profiled stage sum. This identifies
the first missing native BT64 downstream stage; it is not evidence that the
existing v24 MFMA `chunk_o` is slow.

## Decision

`ready_for_stage3=true` because the candidate is opt-in, does not call the
vLLM full wrapper, passes the ordinary/multi-chunk/T8192 correctness gates,
and has a complete targeted profile. Stage 3 should port the **v24-style MFMA
output design to a native BT64 chunk-o** against the frozen BF16-H/V-new
interface. It cannot directly call v24's kernel: both v24 `w_u` and `chunk_o`
hard-reject `chunk_size != 16`, and splitting a BT64 recurrence snapshot into
four BT16 output calls changes the causal intra-chunk math. The work should not
touch asm v0, alter v24, or modify compiler/register allocation. Native BT64
W/U is the next missing companion stage, but BT64 chunk-o is the first target
because its generic fallback has the largest measured trace.

## Evidence

All generated artifacts and exact commands live under
`codex_qwen_bt64_full_pipeline_stage2/`; the structured decision is
`final_decision.json`. No Stage 3 kernel was implemented in this pass.

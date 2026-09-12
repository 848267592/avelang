# Qwen gfx942 BT64 Stage 6S: BF16 Recurrence Full-Graph Contract

## Conclusion

Stage 6S completes as Case A. Graph B connects the Stage 6R current-vLLM BF16 recurrence HSACO through explicit, opt-in boundaries. It does not replace the default path. At T=2048, the complete graph improves from 0.335679 ms to 0.309019 ms: a 26.647 us gain with paired bootstrap 95% interval [26.595, 26.700] us. This exceeds the primary 20 us gate.

The full result is not the isolated recurrence gain copied directly into the graph. The recurrence saves 40.099 us at T=2048, while the three materialized boundary casts cost 20.750 us together. At long sequence lengths Graph B remains faster and reduces the native-vLLM gap slope from 4.396726 to 3.284716 us/chunk.

No production selector, asm-v0, current-vLLM HSACO, compiler, KKT, solve, W/U, chunk-o, FP32 output staging, or final BF16 cast changed.

## New Wrapper and Contract

New experimental-only entry:

- qwen_gdn_full_bt64_stage6s_bf16_recurrence_bridge(...)
- [qwen_gdn_bt64_bf16_recurrence_full_stage6s.py](/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_bt64_bf16_recurrence_full_stage6s.py)

The contract is gfx942, B=1, Hk=4, Hv=8, K=V=128, BT=64 and T divisible by 64. The wrapper checks device, dtype, shape, contiguity, HSACO hash and current stream. Every mismatch raises. There is no pointer reinterpretation, no silent fallback and no default-selector change.

| boundary | before | recurrence side | after |
|:--|:--|:--|:--|
| W | FP32 [1,T,8,128] | BF16 | numeric FP32-to-BF16 cast |
| U | FP32 [1,T,8,128] | BF16 | numeric FP32-to-BF16 cast |
| V-new | bridge output BF16 | unchanged chunk-o needs FP32 | numeric BF16-to-FP32 cast |

The recurrence bridge code object is SHA256 632026a536877921e7c057ecb1b1527983d947339608bbf71f3d1bd8c788077e, symbol chunk_gated_delta_rule_fwd_kernel_h_blockdim64, grid (4,8,1), workgroup 128, dynamic LDS 40960 B. Graph A keeps the distinct historical asm-v0 SHA256 eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226.

## Actual Dispatch Graphs

| graph | count | logical dispatches |
|:--|--:|:--|
| A, current companion | 8 | cumsum, KKT, hierarchical solve, W, U, asm recurrence, chunk-o, final cast |
| B, Stage 6S | 11 | shared stages plus W cast, U cast, current-vLLM recurrence and V-new cast |
| C, native vLLM | 7 | cumsum, KKT, BF16 fill, inverse solve merge, combined W/U, recurrence, chunk-o |

Graph C is from a new direct T=2048 rocprof capture rather than inferred from Graph B. Its final four complete replay sequences each show these seven dispatches. Native chunk-o writes public BF16 output and has no standalone final cast.

## Correctness

The full matrix covers T=64, 128, 512, 1024, 2048 and 8192 with random nonzero state, zero state, high dynamic, small values, cancellation, neutral gate and multichunk feedback. It contains 12 accepted full cases.

| check | result |
|:--|:--|
| bridge vs native recurrence at identical BF16 boundary | h, V-new and final-state bit-exact |
| Graph-B output | max abs 0.001953125, below 0.0078125 |
| Graph-B final state | max abs 0.0172200203, below 0.0200000000 |
| non-default stream | pass |
| graph replay | pass |
| invalid dtype and T guard, no fallback | pass |
| captured hierarchical solve vs v18 contract | pass |

W rounding widened back to FP32 has maximum absolute error 0.0009517372 and U has 0.0145950317. Widened BF16 V-new is exact. No executed case showed threshold violation, NaN, Inf, saturation report, subnormal-specific failure or layout change.

## Boundary Body Timing

All bodies use graph replay, warmup=20, repeat=100, five sessions and the same stream. Individual cast timings are not additive because each isolated graph has its own launch/intercept cost. The all-boundary measurement is the authoritative cast total.

| T=2048 body | ms | us |
|:--|--:|--:|
| historical asm-v0 recurrence | 0.154669 | 154.669 |
| current-vLLM bridge recurrence | 0.114570 | 114.570 |
| recurrence gain | - | 40.099 |
| W FP32-to-BF16 alone | 0.014061 | 14.061 |
| U FP32-to-BF16 alone | 0.014061 | 14.061 |
| W plus U together | 0.017026 | 17.026 |
| V-new BF16-to-FP32 alone | 0.014581 | 14.581 |
| all three boundaries together | 0.020750 | 20.750 |
| diagnostic net before full | - | 19.349 |

The recurrence-body slope is 4.513539 us/chunk for asm-v0 and 3.108600 us/chunk for the bridge. The combined boundary-cast slope is 0.252201 us/chunk.

## Same-Harness Full Timing

All values are medians of five session medians in one process, with identical inputs, current stream, capture-before-timing and balanced ABBA/BCCB/ACCA replay. Profiler time is not used as latency.

| T | Graph A current ms | Graph B Stage6S ms | Graph C vLLM ms | B gain vs A us | B-vLLM gap us |
|--:|--:|--:|--:|--:|--:|
| 512 | 0.121721 | 0.120579 | 0.100450 | 1.370 | 20.049 |
| 1024 | 0.191685 | 0.181350 | 0.129493 | 10.319 | 51.865 |
| 2048 | 0.335679 | 0.309019 | 0.188800 | 26.647 | 120.206 |
| 4096 | 0.608985 | 0.550138 | 0.324282 | 58.960 | 225.767 |
| 8192 | 1.159743 | 1.033295 | 0.604138 | 126.288 | 429.022 |
| 16384 | 2.302501 | 2.024268 | 1.182217 | 278.926 | 841.739 |

Graph-A minus vLLM slope is 4.396726 us/chunk. Graph-B minus vLLM is 3.284716 us/chunk, so Stage 6S recovers 1.112009 us/chunk. No long-text point regresses. Graph B is still slower than vLLM, so it remains an experimental candidate rather than a production promotion.

### Eager-versus-Graph Reconciliation

The 0.188800 ms native-vLLM number is a CUDA Graph replay device-schedule result. It must not be compared directly with the older approximately 0.364 ms eager public-call reports. To verify the distinction, the old Stage-2 eager HIP-event harness was rerun in the current container with the same public vLLM API, T=2048, current input contract and the same autotune patch. It produced 0.360936 ms, p10 0.353244 ms and p90 0.373635 ms.

Thus both series are real but answer different questions: eager timing includes host launch/queue gaps between the seven vLLM dispatches after the start event is recorded; graph replay enqueues the already-captured graph as one replay and removes those host-side gaps. The Stage-6S A/B/C comparison remains internally fair because all three use the same graph-replay protocol. It is not a claim that ordinary eager vLLM invocation is 0.188800 ms.

### Supplemental Eager Public-API Leaderboard

After the graph-replay audit, the primary leaderboard was rerun with direct
eager public calls for v24, Stage 6S, and native vLLM. CUDA/HIP Graph replay
was not used. Every row uses the same seeded BF16/FP32 input contract, current
stream, warmup=20, repeat=100, five sessions, and pair-balanced
ABBA/BCCB/ACCA order. Public wrappers retain their warmed cached-allocator
behaviour; this is a public-API result, not an allocation-free claim.

| T | v24 eager ms | Stage 6S eager ms | vLLM eager ms | v24 / vLLM | Stage 6S / vLLM |
|--:|--:|--:|--:|--:|--:|
| 512 | `0.346034` | `0.293796` | `0.375658` | `0.921x` | `0.782x` |
| 1024 | `0.457640` | `0.324823` | `0.389739` | `1.174x` | `0.833x` |
| 2048 | `0.617018` | `0.391782` | `0.412713` | `1.495x` | `0.949x` |
| 4096 | `1.060757` | `0.601955` | `0.527264` | `2.012x` | `1.142x` |
| 8192 | `2.163678` | `1.093706` | `0.769584` | `2.811x` | `1.421x` |
| 16384 | `4.276980` | `2.088145` | `1.243489` | `3.439x` | `1.679x` |

All 12 Avelang-vLLM correctness checks passed. The largest Stage-6S error was
output max abs `0.0009765625` and final-state max abs `0.005528152`, within
the frozen `1/128` and `0.02` thresholds.

This is the first eager measurement that makes Stage 6S directly comparable
with v24 on the primary leaderboard. Stage 6S is faster than native vLLM at
T=512, 1024, and 2048, with its best relative result `0.782x` at T=512. It
crosses behind vLLM between T=2048 and T=4096, but remains substantially
better than v24 at every measured length. A least-squares fit over the sweep
gives Stage 6S `7.340 us/chunk`, native vLLM `3.569 us/chunk`, and a
Stage-6S-minus-vLLM gap slope of `3.771 us/chunk`; v24's corresponding gap
slope is `12.425 us/chunk`.

The raw per-length sessions and exact contract are in
`eager_public_leaderboard/by_t/`. The executable harness is
`stage6s_eager_public_leaderboard.py`.

## Resource and Identity Audit

The direct Graph-B trace contains chunk_gated_delta_rule_fwd_kernel_h_blockdim64, not qwen_gdn_bt64_gfx942_asm_v0:

| recurrence | WG | VGPR | AccVGPR | SGPR | scratch | MFMA | VMEM | LDS instructions |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Graph B current-vLLM bridge | 128 | 104 | 160 | 96 | 0 | 65,536 | 58,368 | 305,472 |
| Graph A historical asm-v0 | 256 | 128 | 192 | 80 | 0 | 196,608 | 91,136 | 588,928 |

The external-module rocprof metadata reports LDS_Block_Size=0 for Graph B while the guarded Stage-6R ABI/code-object reports dynamic LDS 40960 B. The fixed bridge ABI is authoritative; this is a collector metadata limitation, not an alternative launch.

Graph B adds three cast dispatches. The mixed direct trace recognizes the expected PyTorch BF16 copy classes, but contains warmup/replay and the unchanged final output cast. Individual cast VMEM, LDS-instruction and barrier counts are N/A, recorded as such in conversion_instruction_counts.csv rather than estimated.

## Decision

Case A: retain Graph B as an opt-in experimental full path. Do not promote it to the default path.

The only next action is BF16 storage-boundary propagation: have W/U producers natively write the recurrence BF16 contract and let chunk-o consume BF16 V-new, removing the three explicit casts one at a time under this same full-graph correctness/benchmark contract. Do not change the recurrence HSACO, compiler, asm-v0, KKT, solve, output staging or final public cast in that next action.

## Validation and Artifacts

- Stage 6S, Stage 6R bridge and asm-v0 regression: 8 passed in 21.95s.
- Shared Stage-4 nonrecurrence KKT/W/U/chunk-o regression: 29 passed.
- Direct source-JIT hierarchical-solve regression: 7 failed because the active runtime binding does not export al.amdgpu.mfma_16x16x4_f32_f32. This is the pre-existing feature-export limitation that requires the immutable, hash-guarded Stage-5B solve HSACO bridge in this experiment. The Stage-6S captured-solve versus v18 contract test is included in the 8 passing tests.
- Syntax checks passed with isolated /tmp/pycache_stage6s.
- git diff --check passed.
- [Stage 6S artifacts](/home/jiandongliu/project/avelang/test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/codex_qwen_bt64_bf16_recurrence_full_contract_stage6s) contain raw samples, correctness matrices, bridge source, commands and traces.

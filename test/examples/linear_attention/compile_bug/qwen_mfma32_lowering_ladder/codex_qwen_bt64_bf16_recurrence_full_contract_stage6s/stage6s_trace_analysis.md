# Stage 6S Structural Trace Analysis

ROCprof is used only for resource and dispatch identity. Full latency in this experiment is HIP-event CUDA/HIP graph replay because prior Stage-5F work showed that kernel tracing materially perturbs short graph timing.

## Candidate Identity

The direct Graph-B trace contains `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, not `qwen_gdn_bt64_gfx942_asm_v0`. Its recorded resource tuple is `VGPR=104`, `AccVGPR=160`, `SGPR=96`, `Scratch=0`, `MFMA=65536`, `VMEM=58368`, and `LDS instructions=305472`, matching the Stage-6R current-vLLM recurrence specialization. The ABI guard fixes workgroup `128`, grid `(4,8,1)`, and dynamic LDS `40960 B`.

The direct rocprof CSV reports `LDS_Block_Size=0` for this externally loaded module even though the fixed module ABI and Stage-6R code-object audit report `40960 B` dynamic LDS. This is a trace metadata limitation for this external module, not evidence that the launch changed; `40960 B` is the authoritative ABI value in `graph_b_dispatch_map.json` and `bf16_boundary_contract.json`.

## Casts

Graph B has exactly three new cast dispatches per replay: W FP32-to-BF16, U FP32-to-BF16, and V-new BF16-to-FP32. Its existing final FP32-to-BF16 output cast remains unchanged. The mixed direct trace identifies the expected PyTorch BF16 copy kernel classes, but includes warmup and final-output casts; therefore individual cast VMEM/MFMA/LDS/barrier counts are deliberately marked `N/A` in `conversion_instruction_counts.csv` rather than estimated.

No scratch/spill is reported for Graph-B recurrence or the shared unchanged Avelang stages. The direct trace contains no recurrence fallback or hidden copy between recurrence and chunk-o beyond the explicitly declared V-new numeric cast. Full trace files are retained under `rocprof/full_t2048/`; the direct isolated Graph-B trace is under `rocprof/direct_b_t2048/`.

## Native Graph C Topology

A direct native-vLLM trace at T=2048 records exactly seven logical dispatches in each final replay: cumsum, KKT, one BF16 fill, inverse solve merge, one combined W/U kernel, the same named current-vLLM recurrence specialization, and chunk-o. It has no independent final output cast. The last four complete sequences in `rocprof/direct_c_t2048/stage6s_kernel_trace.csv` establish this count; earlier rows are Triton/JIT/autotune activity and are excluded.

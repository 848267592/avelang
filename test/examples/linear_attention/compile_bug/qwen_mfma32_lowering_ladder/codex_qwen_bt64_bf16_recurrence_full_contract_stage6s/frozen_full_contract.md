# Stage 6S Frozen Full-Graph Contract

## Scope

This is an opt-in, audit-only BT64 experiment on gfx942. It does not modify the production selector, compiler, KKT, hierarchical solve source, W/U source, chunk-o source, output staging, final cast, historical asm-v0 HSACO, or the captured current-vLLM recurrence HSACO.

The measured contract is `B=1, Hk=4, Hv=8, K=V=128, BT=64`, contiguous `[B,T,H,D]`, BF16 `q/k/v`, FP32 `g/beta/initial_state`, with `T in {512,1024,2048,4096,8192,16384}`. All full timings use one process, the current stream, CUDA/HIP graph replay after warmup/capture, `warmup=20`, `repeat=100`, five sessions, and balanced `ABBA`, `BCCB`, `ACCA` ordering. Compile, module load, allocation, and profiler time are outside timing.

## Frozen Graphs

| graph | dispatch count | recurrence | explicitly changed boundary |
|:--|--:|:--|:--|
| A, current experimental companion | 8 | immutable historical asm-v0, FP32 W/U/v-new | none |
| B, Stage 6S candidate | 11 | immutable current-vLLM BF16 HSACO bridge | W FP32->BF16, U FP32->BF16, v-new BF16->FP32 |
| C, native vLLM public full | 7 | native vLLM runtime | direct Stage-6S trace at T=2048 |

Graph A and B share cumsum, Stage-4 KKT, the captured immutable Stage-5B hierarchical FP32 solve code object, Stage-4 W, Stage-4 U, Stage-4 FP32 chunk-o, FP32 output staging, the final BF16 cast, input tensors, stream and all static shape guards. The sole Graph-B changes are its three numeric conversions and the recurrence code object.

## Reproducibility

The executable measurement contract is `frozen_measurement_contract.json`. Raw samples, session summaries, slopes, paired bootstrap results, correctness matrices, and trace CSVs live in this directory. ROCprof is structural-only; HIP-event graph replay is the latency authority.

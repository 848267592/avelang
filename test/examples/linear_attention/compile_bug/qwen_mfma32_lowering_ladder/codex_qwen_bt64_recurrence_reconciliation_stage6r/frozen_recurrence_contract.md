# Stage 6R Frozen Recurrence Contract

This audit compares the current Stage 6A vLLM recurrence selected after its
actual cumsum/KKT/solve/WU producers with the immutable asm-v0 recurrence.
It is not a performance patch.

- Target: gfx942 / MI300, `B=1,Hk=4,Hv=8,K=V=128,BT=64`.
- Lengths: `T=512,2048,8192,16384`; all are divisible by 64.
- Inputs: identical deterministic Stage 6A seed per `T`, contiguous tensors,
  nonzero FP32 initial state, and the same current PyTorch HIP stream.
- Timing: compile, autotune, module load, and allocation are excluded.  Each
  body is prewarmed, CUDA/HIP graph captured, then timed with HIP events using
  warmup 20, repeat 100, five ABBA sessions.
- ROCprof is used only for dispatch/resource structure.  It is not a latency
  authority because Stage 5F established measurable instrumentation effects.

The raw ABI of each implementation is recorded from the fresh T=2048 capture
in `current_kernels/*/abi.json`; no historical cache is used as the claimed
current vLLM specialization.

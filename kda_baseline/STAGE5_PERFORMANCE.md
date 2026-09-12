# Stage 5 — KDA performance benchmark

Run date: 2026-09-10 UTC.  This stage benchmarks only forward inference KDA
operators.  It does not modify a KDA implementation, implement Avelang, load
model weights, or start an SGLang/vLLM server.

## Test environment

- Host: `nimrodmi300`; physical GPU: host `card0` / MI300X, `gfx942`.
- Host driver reported by `rocm-smi`: `6.16.13`; the host has eight GPUs.
- Both runs used the same physical GPU, sequentially, with
  `HIP_VISIBLE_DEVICES=0` and `ROCR_VISIBLE_DEVICES=0`.  GPU clocks, ROCm,
  the driver, and Docker were not changed.
- SGLang container: `ljd_sglang_kda`, image
  `lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909`, digest
  `sha256:16e1ae7e199418fe4df58f6dffb7b2d6b8382cc998aae0d525f8d8b2f7959c98`.
  Source commit: `908226fea2df861769e2720161a75649ae4c6f92`.
  Runtime: PyTorch `2.11.0+rocm7.2`, HIP `7.2.26015`, Triton `3.7.0`.
- vLLM container: `ljd_vllm_kda`, image `vllm/vllm-openai-rocm:kimi-k3`,
  digest
  `sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b`.
  Source commit: `40e6042ec83eb8f2971f21043a5da40496bd188a`.
  Runtime: PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`, Triton `3.7.0`.
- Both containers remained persistent (`running`, `restart=unless-stopped`)
  after the benchmark.

## Common benchmark contract

- Input: BF16; recurrent state: FP32; `H=12`, `K=V=128`,
  `lower_bound=-5.0`.
- Warmup: `50`; timed repetitions: `300`.  Triton JIT compilation and
  autotuning were completed during warmup, before timed CUDA events.
- Each shape was run three times.  The reported latency is the median of the
  three per-run medians.
- Input construction, tensor cloning, state reset/copy, and result handling
  were outside the timed event.  Decode state reset was performed by the
  `prepare()` callback before each launch.
- Speedup is `slower median / faster median`; therefore the winner has a
  speedup greater than 1.0x.

### Operator boundaries and paths

| Phase | SGLang reference | vLLM AMD reference |
|---|---|---|
| Prefill | `sglang.kernels.ops.attention.fla.kda.chunk_kda`, ordinary Triton | `vllm.models.kimi_k3.amd.ops.kda_prefill.chunk_kda_prefill(use_fused_chunk=False)`, AMD-vendored Triton |
| Decode | Direct `sglang.kernels.ops.attention.fla.fused_recurrent.fused_recurrent_kda_packed_decode_kernel`, ordinary Triton | `vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent_kda_packed_decode`, AMD-vendored Triton |

Prefill is the raw `q/k/v + gate + beta + initial_state -> output + final_state`
boundary.  Decode is the post-convolution packed recurrent KDA boundary and
excludes convolution, RMSNorm, and projection.  Neither side uses FlashKDA,
CuTeDSL, Helion, the gfx950 fused-chunk path, or the optional SGLang CUDA-JIT
decode path.

## Prefill results

Tokens/s is `T / median_us * 1e6`.

| T | SGLang median (us) | vLLM median (us) | speedup | winner | SGLang tokens/s | vLLM tokens/s |
|---:|---:|---:|---:|---|---:|---:|
| 64 | 310.662 | 356.950 | 1.149x | SGLang | 206,012 | 179,297 |
| 128 | 311.544 | 358.713 | 1.151x | SGLang | 410,858 | 356,831 |
| 256 | 311.103 | 363.962 | 1.170x | SGLang | 822,879 | 703,371 |
| 512 | 309.601 | 361.457 | 1.167x | SGLang | 1,653,744 | 1,416,489 |
| 1024 | 336.420 | 393.986 | 1.171x | SGLang | 3,043,814 | 2,599,077 |
| 2048 | 496.318 | 466.233 | 1.065x | vLLM | 4,126,387 | 4,392,653 |
| 4096 | 637.948 | 605.841 | 1.053x | vLLM | 6,420,581 | 6,760,855 |
| 8192 | 930.503 | 936.993 | 1.007x | SGLang | 8,803,840 | 8,742,861 |

The corresponding three-round aggregate min/p90 values (us) are retained
here for completeness:

| T | SGLang min | SGLang p90 | vLLM min | vLLM p90 |
|---:|---:|---:|---:|---:|
| 64 | 299.445 | 329.489 | 342.229 | 371.592 |
| 128 | 299.966 | 324.964 | 346.716 | 370.350 |
| 256 | 298.524 | 324.042 | 349.720 | 376.039 |
| 512 | 297.322 | 321.077 | 349.880 | 374.196 |
| 1024 | 327.888 | 346.475 | 382.889 | 404.441 |
| 2048 | 483.639 | 511.360 | 452.432 | 475.627 |
| 4096 | 622.165 | 651.969 | 594.083 | 616.997 |
| 8192 | 917.885 | 940.478 | 923.413 | 945.205 |

Prefill winner: SGLang wins 6/8 shapes.  vLLM is faster at `T=2048` and
`T=4096`; the `T=8192` result is effectively a tie (0.7%).  The strongest
overall Prefill reference for this sweep is **SGLang**, with that mid-length
vLLM exception noted.

## Decode results

Tokens/s is `B / median_us * 1e6`.  Effective recurrent-state bandwidth counts
one FP32 state read and one FP32 state write:
`2 * B * H * K * V * sizeof(fp32) / latency`.

| B | SGLang median (us) | vLLM median (us) | speedup | winner | SGLang tokens/s | vLLM tokens/s | SGLang state TB/s | vLLM state TB/s |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 39.299 | 41.843 | 1.065x | SGLang | 25,446 | 23,899 | 0.0400 | 0.0376 |
| 8 | 42.363 | 44.707 | 1.055x | SGLang | 188,844 | 178,943 | 0.2970 | 0.2815 |
| 16 | 47.310 | 49.994 | 1.057x | SGLang | 338,195 | 320,038 | 0.5319 | 0.5034 |
| 32 | 56.164 | 59.088 | 1.052x | SGLang | 569,765 | 541,565 | 0.8962 | 0.8518 |
| 64 | 58.067 | 60.830 | 1.048x | SGLang | 1,102,185 | 1,052,112 | 1.7336 | 1.6548 |
| 128 | 63.174 | 68.762 | 1.088x | SGLang | 2,026,150 | 1,861,493 | 3.1869 | 2.9279 |
| 256 | 97.705 | 102.312 | 1.047x | SGLang | 2,620,132 | 2,502,150 | 4.1211 | 3.9355 |

The corresponding three-round aggregate min/p90 values (us) are retained
here for completeness:

| B | SGLang min | SGLang p90 | vLLM min | vLLM p90 |
|---:|---:|---:|---:|---:|
| 1 | 36.695 | 48.352 | 37.055 | 52.558 |
| 8 | 37.255 | 44.586 | 38.818 | 47.631 |
| 16 | 44.266 | 49.513 | 43.946 | 52.678 |
| 32 | 51.156 | 58.527 | 52.878 | 62.132 |
| 64 | 53.079 | 61.171 | 55.923 | 64.817 |
| 128 | 55.442 | 68.502 | 61.090 | 73.469 |
| 256 | 94.741 | 99.948 | 98.466 | 105.276 |

Decode winner: SGLang wins all 7 tested batch sizes.  The strongest Decode
reference is therefore **SGLang**.

## Fairness and conclusions

All requested fairness checks are **PASS**:

1. Input shapes match exactly for every paired shape.
2. Input dtype is BF16 on both sides; recurrent state dtype is FP32.
3. Initial-state layout is the same cache-pool semantic (`[state_row,H,V,K]`)
   and the vLLM wrapper's input reshaping is outside the timed region.
4. Prefill and Decode use the same operator boundaries described above.
5. Both Prefill paths are ordinary Triton paths; vLLM uses
   `use_fused_chunk=False`.
6. Both Decode paths measure only the packed recurrent KDA core.
7. JIT/autotune compilation is completed during the 50 warmup launches.
8. State reset/copy and input clone operations are outside CUDA event timing;
   every result row records `state_reset_copy_excluded=true`.
9. All rows report capability `[9, 4]`, `HIP_VISIBLE_DEVICES=0`, and
   `ROCR_VISIBLE_DEVICES=0`.

The largest Prefill relative gap is **1.171x** (17.1%, `T=1024`; absolute
difference 57.566 us).  The largest Decode relative gap is **1.088x** (8.8%,
`B=128`; absolute difference 5.588 us).  These are close same-order baselines,
not an order-of-magnitude difference; Prefill is especially close at `T=8192`.

## Artifacts

Raw three-round results are on the host under
`/home/jiandongliu/project/avelang/kda_baseline/`:

- `results/perf_prefill_sglang.jsonl`
- `results/perf_prefill_vllm.jsonl`
- `results/perf_decode_sglang.jsonl`
- `results/perf_decode_vllm.jsonl`
- `results/perf_*_round*.log` (including the separate `*_b256_round*.log`)

The shared benchmark adapters used for this stage are:

- `bench_common.py`
- `bench_kda_prefill.py`
- `bench_kda_decode.py`

No KDA implementation or Avelang implementation was changed for Stage 5.
Stage 5 is complete and execution stops here; no Stage 6/Avelang work was
started.

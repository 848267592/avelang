# Qwen GDN v9 Resweep After chunk_o Optimization

Date: 2026-06-13

## Primary Target

Single-GPU TP4 per-rank operator benchmark for vLLM Qwen3Next / Qwen3.5 GDN linear attention.

- GPU: AMD Instinct MI300X
- dtype: BF16 q/k/v, FP32 state and accumulators
- layout: q/k/v `[B,T,H,D]`
- initial_state: `[B,Hv,V,K]`
- fixed model shape: `B=1,Hk=4,Hv=8,K=128,V=128`
- legacy/dev `K=64,V=64` was not used for this report
- timing: CUDA/HIP events with synchronize, `warmup=3,repeat=10` for sweeps, `warmup=5,repeat=20` for final T validation

This is not an end-to-end TP4 multi-GPU serving benchmark. It is a single-GPU per-rank operator benchmark.

## Phase A: chunk_size Resweep

Fixed blocks:

- chunk_gdr: `block_v=4,block_k=64`
- chunk_o: `block_v=4,block_k=16`

| chunk_size | full_ms | vLLM_ms | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o_v9 | output_err | final_state_err | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1.261073 | 0.319995 | 0.034251 | 0.041021 | 0.029764 | 0.078597 | 1.030691 | 0.151945 | 5.161e-4 | 3.705e-3 | ok |
| 4 | 1.228905 | 0.327246 | 0.033209 | 0.041702 | 0.029965 | 0.108601 | 0.963151 | 0.174218 | 5.161e-4 | 3.705e-3 | ok |
| 8 | 1.313111 | 0.319275 | 0.033530 | 0.050675 | 0.028922 | 0.168530 | 0.934508 | 0.226616 | 5.161e-4 | 3.705e-3 | ok |
| 16 | 1.512086 | 0.314227 | 0.032208 | 0.061892 | 0.047070 | 0.284502 | 0.912635 | 0.347516 | 5.161e-4 | 3.705e-3 | ok |
| 32 | 3.351659 | 0.327887 | 0.031927 | 0.092738 | 1.414301 | 0.520173 | 0.915199 | 0.575254 | 5.161e-4 | 3.705e-3 | ok |
| 64 | 15.282981 | 0.330892 | 0.034331 | 0.148260 | 12.174444 | 0.993316 | 0.903422 | 1.004251 | 5.161e-4 | 3.705e-3 | ok |
| 128 | 152.886139 | 0.324362 | 0.043104 | 0.259826 | 147.372147 | 1.934954 | 0.904504 | 1.907273 | 5.161e-4 | 3.705e-3 | ok |

Conclusion: `chunk_size=4` remains the global best at T=512. Larger chunks slightly reduce `chunk_gdr`, but they make `w_u`, `chunk_o`, and especially `solve` much worse. The optimal chunk did not move after v9 chunk_o optimization.

## Phase B: chunk_gdr Parameter Sweep

Fixed:

- chunk_size: `4`
- chunk_o: `block_v=4,block_k=16`
- mapping: `current_vk`

`kxv_if_available` is unavailable in current v9; the wrapper only accepts `parallel_mode="vk"`.

| mapping | block_v | block_k | workgroup | full_ms | vLLM_ms | chunk_gdr_ms | chunk_o_v9_ms | output_err | final_state_err | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current_vk | 2 | 64 | 128 | 1.280021 | 0.317231 | 1.012825 | 0.174899 | 5.161e-4 | 3.705e-3 | ok |
| current_vk | 4 | 32 | 128 | 1.517855 | 0.328768 | 1.253823 | 0.175180 | 5.161e-4 | 3.705e-3 | ok |
| current_vk | 4 | 64 | 256 | 1.234594 | 0.312825 | 0.967116 | 0.175300 | 5.161e-4 | 3.705e-3 | ok |
| current_vk | 8 | 32 | 256 | 1.584433 | 0.323600 | 1.313952 | 0.180188 | 5.161e-4 | 3.705e-3 | ok |
| current_vk | 16 | 16 | 256 | 1.970287 | 0.324002 | 1.694196 | 0.176502 | 5.161e-4 | 3.705e-3 | ok |

Best chunk_gdr config remains `block_v=4,block_k=64`. Lower K parallelism is bad, and larger V blocks are slower despite the same 256 workgroup size.

## Phase C: chunk_o Sanity Sweep

Fixed:

- chunk_size: `4`
- chunk_gdr: `block_v=4,block_k=64`

| chunk_o_block_v | chunk_o_block_k | workgroup | full_ms | vLLM_ms | chunk_gdr_ms | chunk_o_v9_ms | output_err | final_state_err | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 16 | 64 | 1.235635 | 0.320716 | 0.965554 | 0.175660 | 5.161e-4 | 3.705e-3 | ok |
| 2 | 32 | 64 | 1.278419 | 0.317832 | 0.963832 | 0.226376 | 5.161e-4 | 3.705e-3 | ok |
| 8 | 16 | 128 | 1.256105 | 0.320156 | 0.966916 | 0.197974 | 5.161e-4 | 3.705e-3 | ok |

Best chunk_o config remains `block_v=4,block_k=16`.

## Final Config

- `chunk_size=4`
- chunk_gdr: `current_vk, block_v=4, block_k=64`
- chunk_o: `vk, block_v=4, block_k=16`

## Phase D: Validation Across T

| T | Avelang v9 full_ms | vLLM full_ms | vLLM/Avelang | cumsum | KKT | solve | w_u | chunk_gdr | chunk_o_v9 | output_err | final_state_err | bottleneck |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.312865 | 0.305053 | 0.975 | 0.030565 | 0.037055 | 0.027922 | 0.056925 | 0.146618 | 0.067180 | 3.730e-4 | 3.709e-3 | chunk_gdr |
| 512 | 1.233713 | 0.307817 | 0.250 | 0.031327 | 0.040340 | 0.027681 | 0.107960 | 0.963992 | 0.172496 | 5.161e-4 | 3.705e-3 | chunk_gdr |
| 1024 | 2.303862 | 0.320716 | 0.139 | 0.028362 | 0.046308 | 0.027000 | 0.173337 | 1.859883 | 0.296600 | 5.428e-4 | 3.419e-3 | chunk_gdr |
| 2048 | 4.576398 | 0.357210 | 0.078 | 0.030646 | 0.058688 | 0.027361 | 0.308458 | 3.678143 | 0.599971 | 6.313e-4 | 4.520e-3 | chunk_gdr |

## Answers

1. Did the chunk_size optimum change after v9 chunk_o?

No. `chunk_size=4` is still best. Bigger chunks reduce `chunk_gdr` only a little, while `w_u`, `chunk_o`, and `solve` grow quickly.

2. Is the current bottleneck still chunk_gdr?

Yes. After v9 chunk_o, `chunk_gdr` dominates the full forward at all tested T. At T=2048, `chunk_gdr=3.678 ms` versus `chunk_o_v9=0.600 ms`.

3. Best config:

- `chunk_size=4`
- chunk_gdr `block_v=4,block_k=64`
- chunk_o `block_v=4,block_k=16`

4. Avelang full vs vLLM full:

Avelang v9 is near vLLM only at T=64. For longer sequences, vLLM is much faster: about 4.0x faster at T=512, 7.2x at T=1024, and 12.8x at T=2048.

5. Next optimization direction:

The next target should be `chunk_gdr`, not `chunk_o`. The current `chunk_gdr` still scales almost linearly with T and is much larger than all other stages after the chunk_o vk optimization.

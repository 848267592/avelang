# Qwen GDN v10 scalar experiment and MFMA feasibility report

## Primary Target

Single-GPU vLLM Qwen3Next TP4 per-rank operator benchmark:

```text
B=1
T=512 for current smoke
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
initial_state=[B,Hv,V,K]
chunk_size=4
chunk_gdr block_v=4, block_k=64
chunk_o   block_v=4, block_k=16
GPU=MI300 / gfx942
```

Legacy `K=64,V=64` is not used here.

## Changed Files

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v10_vllm_layout_fixed.py
test/examples/linear_attention/vllm_compare/test_qwen_gdn_chunked_avelang_v10_vllm_layout_fixed.py
test/examples/linear_attention/vllm_compare/prototype_qwen_gdn_mfma_tile_dot.py
test/examples/linear_attention/vllm_compare/qwen_gdn_v10_scalar_and_mfma_feasibility_report.md
```

Notes:

- Verified `chunk_o_parallel_mode` defaults are restored to `"vk"` in v10.
- v10 `chunk_gdr` keeps the scalar VK lane mapping but batches all token offsets in a chunk into `partial_pred[block_v, chunk_size, block_k]` before the K reduction.
- No production MFMA `chunk_gdr` kernel was integrated yet.

## Correctness

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
python -m pytest -q test_qwen_gdn_chunked_avelang_v10_vllm_layout_fixed.py -s
```

Result:

```text
2 passed in 8.07s
```

The pytest compares:

```text
v10 chunk_gdr_chunk_vk vs v9 chunk_gdr_vk: h, vn, final_state
v10 full forward       vs v9 full forward: output, final_state
```

Smoke correctness at `B=1,T=512,Hk=4,Hv=8,K=128,V=128,chunk=4,BF16`:

```text
h_abs=0
h_rel=0
vn_abs=0
vn_rel=0
final_abs=0
final_rel=0
full output_abs=0
full output_rel=0
full final_abs=0
full final_rel=0
```

## v10 Scalar Benchmark

Command was an inline smoke benchmark in the same container path, using `torch.cuda.Event` timing and `torch.cuda.synchronize()` after each iteration.

Stage breakdown, median over 40 repeats:

| stage | median ms |
|---|---:|
| cumsum | 0.028122 |
| KKT | 0.039699 |
| solve | 0.026760 |
| w_u | 0.107960 |
| chunk_gdr_v9_vk | 0.965354 |
| chunk_gdr_v10_chunk_vk | 0.865084 |
| chunk_o_v9_vk | 0.171054 |

Full forward:

| implementation | median ms |
|---|---:|
| v9 | 1.235434 |
| v10 scalar chunk_vk | 1.130999 |

Speedup:

```text
chunk_gdr v10 vs v9 = 0.965354 / 0.865084 = 1.116x, about 11.6% faster
full v10 vs v9      = 1.235434 / 1.130999 = 1.092x, about 9.2% faster
```

Conclusion for Line A:

```text
v10 correctness passes, but chunk_gdr improvement is below the requested 20%-30% threshold.
Stop investing in this scalar reduction direction. The main bottleneck remains chunk_gdr.
```

This matches the rocprof diagnosis from v9: not VMEM-bound, not scratch spill, not barrier allocation stall; the scalar VK recurrence has too much VALU/SALU/control work and low occupancy.

## Avelang MFMA Validation

Command:

```bash
cd /workspace/project/avelang
python -m pytest -q test/examples/gemm/amdgpu/test_amdgpu_gemm.py -s
```

Result:

```text
1 passed in 3.13s
```

Relevant implementation:

```text
python/avelang_kernels/amdgpu_gemm.py
```

The optimized GEMM path uses:

```text
al.amdgpu.mfma_16x16x16_bf16_f32
raw_buffer_load_x4 / raw_buffer_store_x2
256 work-items = 4 waves
GROUP_M=128, GROUP_N=128, GROUP_K=64
```

So Avelang can express BF16 input + FP32 accumulation MFMA on this MI300 environment.

## Standalone MFMA Tile-Dot Prototype

Added:

```text
test/examples/linear_attention/vllm_compare/prototype_qwen_gdn_mfma_tile_dot.py
```

Command:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
python prototype_qwen_gdn_mfma_tile_dot.py
```

Result:

| operation | status | max_abs | max_rel | shape |
|---|---|---:|---:|---|
| `pred[16,16] = W[16,64] @ H[16,64].T` | ok | 0.125 | 0.00775 | `(16,16)` |
| `pred[16,16] = split K=128 as two 64 GEMMs` | ok | 0.25 | 0.71812 | `(16,16)` |
| `delta_H[16,64] = v_new[16,16].T @ K[16,64]` | ok | 0.0625 | 0.00775 | `(16,64)` |
| `delta_H[16,128] = split K=128 as two 64 GEMMs` | ok | 0.0625 | 0.00775 | `(16,128)` |

Important limitation:

```text
This prototype reuses the existing 128x128 GEMM kernel and pads small tiles to 128x128.
It proves the MFMA math path can express the required tile operations, but it is not a performant chunk_gdr implementation.
```

The high relative error in the split-K pred case is from comparing near-zero elements after BF16-rounded accumulation; max_abs remains within BF16-level smoke tolerance.

## vLLM-style Chunk_GDR Structure

vLLM Triton `chunk_delta_h.py` does not reduce one `(V,K)` lane scalar per token the way v9/v10 do. It keeps a state tile in registers:

```text
b_h1: [BV,64]
b_h2: [BV,64] when K>64
```

For each chunk:

```text
store h tile
b_v = W_chunk[BT,64] @ b_h1.T
if K>64: b_v += W_chunk[BT,64] @ b_h2.T
b_v = u_chunk - b_v
apply decay to b_v and H
b_h1 += (K_chunk_1[64,BT] @ b_v[BT,BV]).T
b_h2 += (K_chunk_2[64,BT] @ b_v[BT,BV]).T
```

For Qwen3Next TP4 target:

```text
K=128 -> two BK=64 tiles
V=128 -> V blocks, likely BV in {16,32}
BT candidates {16,32}; chunk_size=4 is current scalar best but tiled prototype should not be constrained to scalar chunk=4
```

## MFMA Feasibility Answers

1. Can Avelang MFMA express this tile shape?

Yes, for the core matmul pieces. Existing Avelang code already expresses BF16 `A @ B.T` with FP32 accumulation using `mfma_16x16x16_bf16_f32`. The standalone prototype verified both required operations through that path:

```text
pred    = W_chunk @ H_tile.T
delta_H = v_new.T @ K_chunk
```

A production kernel still needs custom layout and state residency instead of padded GEMM calls.

2. Required fragment/layout pieces:

```text
W fragment:      [BT,64] BF16 staged for MFMA A
H fragment:      [BV,64] BF16/BF16-cast state staged as MFMA B for pred
pred/v_new:      [BT,BV] FP32 accumulator, then BF16/FP32 depending update path
K fragment:      [BT,64] BF16 staged transposed for update
delta_H tile:    [BV,64] FP32 accumulator
state tile H1/H2 [BV,64] FP32 resident across chunk loop
h store layout:  [B,num_chunks,Hv,V,K], value-major, same as current vLLM layout
vn layout:       [B,T,Hv,V]
```

One practical issue: MFMA consumes BF16 operands. Current scalar state is FP32. For `pred = W @ H.T`, using MFMA directly means either:

```text
A. keep H tile in BF16 or convert/store a BF16 shadow for MFMA, with possible numerical change, or
B. keep FP32 scalar pred path for H and only MFMA the update, which loses much of the benefit, or
C. find/use an FP32-capable MFMA shape if Avelang exposes it and it is fast enough.
```

vLLM appears to cast the state tile to the dtype of `w` for `tl.dot`, so a BF16 state operand is likely acceptable if correctness tolerance matches vLLM/BF16 behavior.

3. Fixed shapes to support first:

```text
BT=16, BV=16, BK=64, K=128 split as H1/H2
then BT=16, BV=32, BK=64
then BT=32, BV=16 or 32 if register/LDS pressure is acceptable
```

Start with one block handling:

```text
(B, Hv, V block)
loop over time chunks internally
K split fixed to two 64-wide tiles
```

4. Main gap vs current v9/v10 scalar chunk_gdr:

| area | v9/v10 scalar VK | vLLM-style tiled/MFMA target |
|---|---|---|
| mapping | lanes over V and K | tile over BT x BV and BK |
| pred | scalar K reduction per token/value | matrix multiply `W @ H.T` |
| update | scalar state update per lane | matrix multiply `v_new.T @ K` |
| state | per-lane `state[k_items_per_lane]` | tile-resident `H[BV,64]` fragments |
| instruction mix | high VALU/SALU/control | MFMA-heavy, fewer scalar reductions |
| current gain potential | v10 only ~11.6% chunk_gdr | required path for larger gain |

5. Minimal next prototype:

```text
Prototype 1: custom one-wave 16x16x64 Avelang kernel for pred only
  input: W[BT,64], H[BV,64]
  output: pred[BT,BV] FP32
  compare PyTorch
  verify rocprof/disassembly contains MFMA

Prototype 2: extend pred to K=128 split as two 64 tiles
  pred = pred1 + pred2

Prototype 3: custom update kernel
  delta_H[BV,64] = v_new[BT,BV].T @ K_chunk[BT,64]

Prototype 4: combine pred + update in one standalone chunk kernel for one `(B,Hv,Vblock)`
  no solve/cumsum changes
  compare h/vn/final_state vs v9 for T=512

Prototype 5: integrate as v11 `chunk_gdr_tiled_mfma`
  keep v9/v10 scalar fallback
  benchmark T=512/1024/2048
```

## Current Bottleneck

After v10 scalar batching:

```text
chunk_gdr_v10_chunk_vk = 0.865 ms
chunk_o_v9_vk          = 0.171 ms
```

`chunk_gdr` is still about 5x `chunk_o` at T=512, so the next optimization should be tiled/MFMA `chunk_gdr`, not more chunk_o work and not more blind block_v/block_k sweeping.

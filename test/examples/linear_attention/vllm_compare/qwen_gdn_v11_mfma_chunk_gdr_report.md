当前 v11 MFMA 停在正确的位置：`pred` 和 `vn` 已经对，`final_state` 错，说明问题集中在 update：

```text
H += v_decay.T @ K
```

现在不要换方向，不要重新优化 scalar VK，不要 benchmark incorrect v11。请继续完成 v11 MFMA update 修复，直到 correctness 通过，然后再做 benchmark。

## Current known failure

T=16 smoke:

```text
h  max_abs=0
vn max_abs=5.75e-05
final_state max_abs=2.052
```

Against BF16-operand PyTorch reference:

```text
vn vs BF16 ref:          max_abs=1.19e-07
final_state vs BF16 ref: max_abs=2.052
```

K tile error:

```text
cols  0:16  max_abs=0.013
cols 16:32  max_abs=0.378
cols 32:48  max_abs=0.697
cols 48:64  max_abs=0.770
```

Interpretation:

```text
pred path is correct.
vn path is correct.
update path is wrong.
The first 16-column tile is nearly correct, later K-column tiles diverge.
This likely comes from integrated-kernel shared-memory staging / MFMA fragment reuse / accumulator reset / store offset / transposed operand layout, not from the high-level math.
```

## New stop condition

Do not stop just because current v11 final_state fails. That failure is now known and is the target bug.

Only stop if:

```text
1. standalone staged delta kernel cannot be made correct, or
2. compiler/runtime prevents expressing the needed staging, or
3. after isolating the exact failed tile/lane/store pattern.
```

Otherwise continue until:

```text
standalone delta passes
integrated v11 T=16 passes
integrated v11 T=32/T=64/T=512 passes
then benchmark
```

## Step 1: standalone internal-staging delta kernel

Create a new standalone test/prototype, for example:

```text
prototype_qwen_gdn_mfma_delta_staged.py
test_qwen_gdn_mfma_delta_staged.py
```

Implement exactly this computation:

```text
delta[BV,128] = v_decay[BT,BV].T @ k_chunk[BT,128]
```

Start fixed:

```text
BT=16
BV=16
K=128
BK=64
subtile=16
BF16 input
FP32 accumulation
```

Inside the kernel, explicitly stage:

```text
v_decay_source[BT,BV] -> shared/local transposed view v_decay_t[BV,BT]
k[BT,128] -> for each 16-col tile, stage k_tile_t[16,BT]
```

Then compute four 16-column tiles per BK=64:

```text
delta[:,  0:16] = v_decay_t[BV,BT] @ k_tile_t[BT,16]
delta[:, 16:32] = v_decay_t[BV,BT] @ k_tile_t[BT,16]
delta[:, 32:48] = v_decay_t[BV,BT] @ k_tile_t[BT,16]
delta[:, 48:64] = v_decay_t[BV,BT] @ k_tile_t[BT,16]
```

Then repeat for K columns 64:128.

Reference:

```python
expected = v_decay.float().T @ k_chunk.float()
```

Pass criteria:

```text
delta K=64 max_abs close to previous standalone delta result
delta K=128 max_abs close to previous standalone delta result
every 16-column tile correct
```

Important debugging requirements:

```text
- Reset accumulator for every 16x16 output tile.
- Do not reuse accumulator fragments across different K-column tiles unless explicitly intended.
- Insert syncthreads before reading staged shared memory.
- Insert syncthreads before overwriting shared memory for the next tile.
- Check store offsets for cols 0,16,32,48,64,80,96,112.
- Print/report per-tile max_abs for every 16-column tile.
```

## Step 2: replace v11 update with the exact staged delta path

After standalone staged delta passes, replace the v11 integrated update with the same staging logic.

Current intended math:

```text
v_decay[t,v] = v_new[t,v] * exp(g_last - g[t])
H0 = H0 * exp(g_last) + v_decay.T @ K0
H1 = H1 * exp(g_last) + v_decay.T @ K1
```

Do not change pred or vn paths.

In integrated v11:

```text
1. compute pred
2. compute v_new
3. compute decay[t]
4. form/stage v_decay[BT,BV]
5. run the same staged delta_H code as standalone
6. update H0/H1
7. write final_state
```

Make sure the same BF16 operand casting policy is used as in the standalone reference comparison.

## Step 3: correctness tests

Add/extend tests:

```text
test_qwen_gdn_chunked_avelang_v11_mfma_layout_fixed.py
```

Run chunk_gdr-only correctness against scalar v9/v10 fallback.

Required cases:

```text
T=16
T=32
T=64
T=512
with initial_state
without initial_state
B=1,Hk=4,Hv=8,K=128,V=128
BT=16,BV=16,BK=64
```

Check:

```text
h max_abs / max_rel
vn max_abs / max_rel
final_state max_abs / max_rel
```

Do not benchmark until these pass.

## Step 4: benchmark after correctness passes

Only after v11 correctness passes, create or complete:

```text
bench_qwen_gdn_v11_mfma_chunk_gdr.py
qwen_gdn_v11_mfma_chunk_gdr_report.md
```

Benchmark:

```text
v9 scalar best:
  chunk_size=4
  chunk_gdr block_v=4, block_k=64
  chunk_o block_v=4, block_k=16

v10 scalar:
  chunk_size=4

v11 MFMA:
  BT=16,BV=16,BK=64
  chunk_size=16

vLLM:
  same Qwen3Next TP4 per-rank shape
```

Run:

```text
T=512
T=1024
T=2048
```

Report:

```text
chunk_gdr latency
full forward latency
stage breakdown
vLLM latency
speedup vs v9
speedup vs v10
output error
final_state error
current bottleneck
```

## Step 5: rocprof

After benchmark, run rocprof for v11 MFMA chunk_gdr.

Collect:

```text
SQ_INSTS_MFMA
SQ_INSTS_VALU
SQ_INSTS_SALU
SQ_INSTS_VMEM
SQ_INSTS_LDS
SQ_LDS_BANK_CONFLICT
Scratch_Size
VGPR_Count
OccupancyPercent
Dispatch duration
```

Report whether MFMA is actually used.

If `SQ_INSTS_MFMA == 0`, stop and explain.

## Do not do these

```text
- Do not continue scalar VK optimization.
- Do not benchmark v11 before final_state correctness passes.
- Do not delete v9/v10 fallback.
- Do not use old K=64,V=64 for conclusions.
- Do not claim performance improvement from an incorrect kernel.
```
# Qwen GDN v11 MFMA chunk_gdr report

## Status

v11 narrow MFMA `chunk_gdr` was started but **did not pass correctness**. Per the stop condition, I stopped before benchmark/tuning.

Primary target used for the smoke/debug run:

```text
B=1
T=16 first correctness smoke
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
initial_state=[B,Hv,V,K]
chunk_size=BT=16
BV=16
BK=64
GPU=MI300 / gfx942
```

Old `K=64,V=64` was not used.

## Changed Files

```text
test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v11_mfma_layout_fixed.py
test/examples/linear_attention/vllm_compare/qwen_gdn_v11_mfma_chunk_gdr_report.md
```

Existing fallbacks were not deleted:

```text
v9 scalar chunk_gdr_vk
v10 scalar chunk_vk
v9 chunk_o_vk
```

## Implemented v11 Path

Added kernel:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma
```

Narrow wrapper:

```text
qwen_gdn_chunk_gdr_avelang_v11_mfma_layout
qwen_gdn_chunked_avelang_v11_mfma_layout_full
qwen_gdn_chunked_avelang_v11_mfma_layout
```

Supported MFMA path is deliberately narrow:

```text
B=1,Hk=4,Hv=8,K=128,V=128,BF16
chunk_size=16
block_v=16
block_k=64
```

Unsupported configs fall back to v10 scalar `chunk_vk`.

## Compile Issues Fixed

First compile failed because I used local kernel constants like:

```python
batch_size: al.constexpr = 1
chunk_size: al.constexpr = 16
```

Avelang failed resolving `make_layout()` and later produced cascading errors. I fixed this by using direct literal shapes/strides in `al.make_layout`, and using literal chunk stride `16` inside the kernel.

After that, the v11 kernel compiled and ran.

## Correctness Result

Compared v11 MFMA chunk_gdr against v10 scalar chunk_gdr at the same `chunk_size=16`.

T=16 with initial_state:

```text
h  max_abs=0.0,                    max_rel=0.0
vn max_abs=5.753734149e-05,        max_rel=0.384951144
fs max_abs=2.052223444,            max_rel=174118.375
```

`h` is exact, so initial state load and chunk-start `h` store are correct.

`vn` has tiny absolute error and matches a BF16-operand PyTorch reference:

```text
vn vs BF16-reference max_abs=1.192092896e-07, max_rel=1.431936425e-06
```

But `final_state` is wrong even versus the same BF16-operand reference:

```text
final_state vs BF16-reference max_abs=2.052252769, max_rel=58887.859375
```

So the pred path is correct, but the MFMA update path is not.

## Debug Findings

The vLLM-style BF16 reference used for diagnosis:

```python
pred = W0.bf16.float() @ H0.bf16.float().T \
     + W1.bf16.float() @ H1.bf16.float().T
v_new = u - pred
v_decay = (v_new * exp(g_last - g_t)).bf16.float()
H0 = H0 * exp(g_last) + v_decay.T @ K0.float()
H1 = H1 * exp(g_last) + v_decay.T @ K1.float()
```

For one tile `hv=0,vb=0,K0`, error by 16-column group:

```text
cols  0:16  max_abs=0.0134063
cols 16:32  max_abs=0.378202
cols 32:48  max_abs=0.696615
cols 48:64  max_abs=0.770116
```

Interpretation:

```text
The first 16-column update tile is close, but later K tiles diverge.
```

I first suspected dynamic `tile * 16` indexing, so I expanded the update into literal offsets:

```text
0,16,32,48,64,80,96,112
```

The error pattern did not change. That suggests the bug is not only dynamic indexing; it is likely in repeated use of the 16x16 MFMA update fragment inside the larger integrated kernel.

Most likely causes:

```text
1. Repeated MFMA update blocks are not starting from clean accumulator state as intended.
2. Shared-memory view/reuse for k_tile_t / ktile_vec behaves differently inside repeated blocks than in the standalone prototype.
3. The fragment ownership pattern from the standalone delta prototype needs to be integrated as a helper-like isolated tile, not copy-pasted with many local accumulator names.
4. Avelang may be optimizing/reusing local fragment variables unexpectedly across repeated similar blocks.
```

Not the cause:

```text
initial_state load: h matches exactly
pred/vn path: vn matches BF16 PyTorch reference
head grouping: all heads run, but update error appears across heads
old K=64,V=64 toy shape: not used
```

## Why Benchmark Was Not Run

The user-defined stop condition says to stop if v11 MFMA correctness fails. Since `final_state` is wrong, I did not run:

```text
T=512/1024/2048 benchmark
vLLM comparison
v11 rocprof counters
BT/BV tuning
```

Standalone custom MFMA prototypes from the previous step remain valid and had already shown nonzero `SQ_INSTS_MFMA`, but integrated v11 needs correctness first.

## Recommended Next Step

Do not tune this v11 kernel yet.

The next smallest fix is to isolate the update tile in a dedicated standalone Avelang kernel that exactly mirrors the integrated staging pattern:

```text
input:
  v_decay_source [BT,BV] as FP32 or BF16
  k              [BT,128] BF16
output:
  delta          [BV,128] FP32

inside kernel:
  stage v_decay_t[BV,BT]
  stage k_tile_t[16,BT]
  run repeated 16x16 MFMA tiles for all K columns
```

This differs from the already-passing prototype because the passing prototype prepared transposed contiguous tiles from Python. The failing integrated kernel does the transpose/staging internally. The internal staging path must be validated before re-entering full `chunk_gdr`.

Only after that standalone internal-staging delta kernel passes should we re-integrate into:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma
```

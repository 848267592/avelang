# Qwen GDN vLLM-style MFMA staged prototype report

## Target

Primary target remains vLLM Qwen3Next TP4 per-rank:

```text
B=1
T in {512,1024,2048}
Hk=4
Hv=8
K=128
V=128
dtype=BF16
layout=[B,T,H,D]
initial_state=[B,Hv,V,K]
```

This report does not use the old `K=64,V=64` shape for conclusions.

## Baselines Kept

No fallback path was deleted or modified:

```text
v9 scalar chunk_gdr_vk
v10 scalar chunk_vk
v9 chunk_o_vk
```

This work only adds standalone MFMA prototypes. It does not integrate a production v11 forward yet.

## Changed Files

```text
test/examples/linear_attention/vllm_compare/prototype_qwen_gdn_mfma_pred_custom.py
test/examples/linear_attention/vllm_compare/test_qwen_gdn_mfma_pred_custom.py
test/examples/linear_attention/vllm_compare/qwen_gdn_v11_mfma_staged_prototype_report.md
```

## Step 1: Custom MFMA Pred Prototype

Implemented a real custom tile kernel, not padding into the existing 128x128 GEMM:

```text
_mfma_pred_16x16_kernel
```

First target:

```text
BT=16
BV=16
BK=64
input=BF16
acc=FP32
pred[BT,BV] = W[BT,BK] @ H[BV,BK]^T
```

The kernel uses:

```text
al.amdgpu.mfma_16x16x16_bf16_f32
workgroup_size=64
```

For BK=64 it performs two 32-wide fragment batches; for K=128 it performs four batches.

Correctness:

```text
pred_bt16_bv16_bk64_custom, ok=True, max_abs=1.90734863e-06, max_rel=6.65233301e-07
```

## Step 2: K=128 Pred

Same custom pred kernel was extended to Qwen target K=128:

```text
pred[16,16] = W[16,128] @ H[16,128]^T
```

Correctness:

```text
pred_bt16_bv16_k128_custom, ok=True, max_abs=3.81469727e-06, max_rel=2.62542999e-05
```

## Step 3: Custom MFMA Delta_H Prototype

Implemented a smaller reusable custom kernel:

```text
_mfma_matmul_16x16x16_kernel
```

It computes:

```text
C[16,16] = A[16,16] @ B[16,16]^T
```

Then the standalone delta prototype constructs 16-column tiles:

```text
delta_H[BV,BK] = V_new[BT,BV]^T @ K_chunk[BT,BK]
```

For BK=64:

```text
delta_H[:,  0:16]
delta_H[:, 16:32]
delta_H[:, 32:48]
delta_H[:, 48:64]
```

For K=128, it repeats the same 16-column tiling over all 128 columns.

Correctness:

```text
matmul16x16x16_custom, ok=True, max_abs=2.38418579e-07, max_rel=2.20009809e-07
delta_bv16_bk64_custom, ok=True, max_abs=9.53674316e-07, max_rel=4.83907172e-07
delta_bv16_k128_custom, ok=True, max_abs=9.53674316e-07, max_rel=4.69792349e-06
```

Important note:

```text
The delta prototype currently uses Python-side transposed contiguous tiles to validate the custom 16x16 MFMA fragment mapping.
This is acceptable for the staged standalone prototype, but v11 production chunk_gdr must load/stage V_new.T and K_chunk inside one kernel.
```

## Test Commands

Correctness pytest:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
python -m pytest -q test_qwen_gdn_mfma_pred_custom.py -s
```

Result:

```text
2 passed in 9.08s
```

Standalone smoke:

```bash
cd /workspace/project/avelang/test/examples/linear_attention/vllm_compare
python prototype_qwen_gdn_mfma_pred_custom.py
```

Result:

```text
custom_mfma_prototype_status,ok
```

## MFMA Confirmation

rocprof command:

```bash
cd /workspace/project/avelang
rocprofv3 --kernel-trace --stats \
  --pmc SQ_INSTS_MFMA SQ_INSTS_VALU \
  --kernel-include-regex "mfma_(pred|matmul)" \
  -d test/examples/linear_attention/rocprof_outputs/qwen_profile_mfma_pred_custom \
  -o mfma_pred_custom --output-format csv \
  -- python test/examples/linear_attention/vllm_compare/prototype_qwen_gdn_mfma_pred_custom.py
```

Output files:

```text
test/examples/linear_attention/rocprof_outputs/qwen_profile_mfma_pred_custom/mfma_pred_custom_kernel_stats.csv
test/examples/linear_attention/rocprof_outputs/qwen_profile_mfma_pred_custom/mfma_pred_custom_counter_collection.csv
```

Kernel stats:

| kernel | calls | avg ns |
|---|---:|---:|
| `_mfma_pred_16x16_kernel` | 2 | 2203.0 |
| `_mfma_matmul_16x16x16_kernel` | 13 | 2043.15 |

Counter aggregate:

| kernel | Workgroup | VGPR | Accum VGPR | SGPR | Scratch | SQ_INSTS_MFMA | SQ_INSTS_VALU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `_mfma_pred_16x16_kernel` | 64 | 24 | 8 | 16 | 0 | 12 | 33 |
| `_mfma_matmul_16x16x16_kernel` | 64 | 12 | 4 | 16 | 0 | 52 | 338 |

Conclusion:

```text
Custom MFMA is actually used: SQ_INSTS_MFMA is non-zero for both custom kernels.
Scratch remains 0 in these prototypes.
```

## Current Status

Completed:

```text
Step 1: pred[16,16] with BK=64 custom MFMA, correctness passed
Step 2: pred[16,16] with K=128 custom MFMA, correctness passed
Step 3: delta_H[16,64/128] using custom 16x16 MFMA tiles, correctness passed
```

Not done yet:

```text
Step 4: qwen_gdn_chunked_avelang_v11_mfma_layout_fixed.py
Step 5: BT/BV tuning after integrated correctness
```

## Next Action

Build narrow specialized v11 kernel only after these prototypes:

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v11_mfma
B=1, Hk=4, Hv=8, K=128, V=128
BT=16, BV=16, BK=64
chunk_size=BT
```

The v11 kernel should combine the two proven tile operations into the vLLM structure:

```text
for each B,Hv,V block:
    H0[BV,64], H1[BV,64] = initial_state or zero

    for each BT chunk:
        store chunk-start H to h

        pred = W0[BT,64] @ H0.T + W1[BT,64] @ H1.T
        v_new = u_tile - pred
        v_decay[t,v] = v_new[t,v] * exp(g_last - g_t)
        vn = v_new

        H0 = H0 * exp(g_last) + v_decay.T @ K0[BT,64]
        H1 = H1 * exp(g_last) + v_decay.T @ K1[BT,64]

    store final_state
```

Do not tune `BT/BV` before this integrated v11 correctness passes.

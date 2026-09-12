# Qwen GDN v8 chunk_gdr Parallel Optimization Report

Date: 2026-06-10

## 1. Conclusion

这次 v8 直接优化 `chunk_gdr` 的并行 mapping，不再做 decay 预计算或单线程 value tiling。

最终默认路径是：

```text
parallel_mode = "vk"
block_id      -> (B, Hv, V block)
thread_id.x   -> value_idx inside V block
thread_id.y   -> K reduction lane
block_v       = 8
block_k       = auto, K >= 32 时为 32
```

核心结果：

```text
BF16 stage-only chunk_gdr, B=1,T=512,Hk=4,Hv=8,K=64,V=64,chunk=64

v6 chunk_gdr         1.807047 ms
v7 chunk_gdr         1.812007 ms
v8 vblock chunk_gdr  1.807687 ms
v8 vk chunk_gdr      1.359845 ms

speedup vs v6        1.328862x
speedup vs v7        1.332510x
```

也就是说，v8 现在确实让 `chunk_gdr` 变快了，而且不是靠移动 scalar work，而是把 K 方向 dot/recurrent state work 并行化。

但 full forward 只小幅下降：

```text
Avelang v8 full median   35.198536 ms
vLLM full median          0.447202 ms
vLLM / Avelang v8         0.012705x
```

原因是这个大形状下新的最大瓶颈已经转移到 `solve`：

```text
cumsum             0.056800 ms
KKT                0.334241 ms
solve             30.188284 ms
w_u                0.693123 ms
chunk_gdr_v8_vk    1.363845 ms
chunk_o            2.982892 ms
```

所以当前结论是：

```text
v8 已经把 chunk_gdr 从单线程串行瓶颈推进到 1.33x stage speedup。
完整 forward 没有同步大幅变快，是因为 chunk=64/K=64 时 solve 成为主瓶颈。
下一步优先优化 solve，其次 chunk_o；继续压 chunk_gdr 的收益已经不是最大。
```

## 2. Required Context Read

已重新阅读：

- `linear_attn_ave/AGENTS.md`
- `linear_attn_ave/skills/avelang/SKILL.md`
- v6/v7 report 和 rocprof CSV

关键约束：

- Avelang kernel 修改前必须按 correctness-first 流程保留测试护栏。
- 优化要从 scalar pointer kernel 可验证版本推进，不直接跳 raw_buffer/MFMA。
- Qwen grouped heads 必须显式保持 `key_head_idx = value_head_idx // (Hv // Hk)`。
- 不修改上游 Avelang runtime/compiler。

v6 counter 已确认原问题：

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v6_standalone
Workgroup_Size      1
OccupancyPercent    about 3.7%
Scratch_Size        0
SQ_INSTS_VALU       high
```

这说明瓶颈不是 spill，而是 mapping 太串行：一个线程处理一个 `(B,Hv,V)` 的全部 `K/chunk/token` 循环。

## 3. Implementation

修改文件：

- `qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py`
- `bench_qwen_gdn_v8_parallel.py`

新增 v8 kernel：

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v8_vk
```

保留旧 v8 kernel：

```text
_qwen_gdn_chunk_gdr_bf16_kernel_v8_vblock
```

Python wrapper 支持：

```python
parallel_mode="vk"      # new default
parallel_mode="vblock"  # old v8 V-block fallback
use_parallel_chunk_gdr=False  # fallback to v6
```

`vk` kernel 关键实现：

```text
lane_v = thread_id(0)
lane_k = thread_id(1)

value_idx = v_block_idx * block_v + lane_v
owned_k   = lane_k + item * block_k
```

每个 K lane 只维护自己的 strided `state[K]` slice。`pred=sum_k w[k]*state[k]` 使用 shared memory 归约：

```text
partial_pred[block_v, block_k]
chunk_vn[block_v, chunk_size]
```

最终默认参数：

```text
block_v = 8
block_k = 32 when head_dim_k >= 32
workgroup = 8 * 32 = 256 lanes
```

实现中还处理了一个 Avelang lowering 边界：`make_local((1,), f32)` 在当前编译器里动态下标会失败，所以 `k_items_per_lane` 至少取 2。

## 4. Correctness

GPU pytest:

```text
5 passed in 25.06s
```

覆盖：

```text
BF16
with initial_state
without initial_state
partial chunk
grouped heads: Hv > Hk
V tail block
K=32/block_k=32 fast path
vblock fallback
```

主 benchmark correctness：

```text
stage_error,v8_vk,h=1.1920929e-07,vn=2.38418579e-07,final_state=1.1920929e-07
full_error_vs_v6,output=2.98023224e-08,final_state=1.1920929e-07
full_error_vs_vllm,output=0.00107612461,final_state=0.00449794531
```

这些误差来自 K reduction 顺序变化和 BF16/FP32 路径差异，数值在当前测试阈值内。

## 5. Benchmark

环境：

```text
container: qwen_vllm_avelang_rocm722
ROCm:      7.2.2
GPU:       AMD Instinct MI210
dtype:     BF16 q/k/v, FP32 accum/state
shape:     B=1,T=512,Hk=4,Hv=8,K=64,V=64,chunk=64
```

Stage-only final benchmark:

```text
v6_chunk_gdr          1.807047 ms
v7_chunk_gdr          1.812007 ms
v8_vblock_chunk_gdr   1.807687 ms
v8_vk_chunk_gdr       1.359845 ms
```

Speedup:

```text
v8 vk vs v6       1.328862x
v8 vk vs v7       1.332510x
v8 vk vs vblock   1.329333x
```

Parameter sweep showed why the first `block_k=8` attempt was slow:

```text
v8_vk_bv4_bk4      3.032172 ms
v8_vk_bv4_bk8      2.220648 ms
v8_vk_bv4_bk16     1.864168 ms
v8_vk_bv4_bk32     1.493126 ms
v8_vk_bv8_bk4      3.040011 ms
v8_vk_bv8_bk8      2.218569 ms
v8_vk_bv8_bk16     1.696006 ms
v8_vk_bv8_bk32     1.367685 ms
v8_vk_bv16_bk4     3.043532 ms
v8_vk_bv16_bk8     2.123529 ms
v8_vk_bv16_bk16    1.746886 ms
```

Interpretation:

```text
block_k too small -> too many reduction/barrier rounds per useful K work.
block_k=32        -> enough K parallelism with acceptable barrier overhead.
block_v=8         -> better occupancy/workgroup balance than block_v=4 or 16 on this shape.
```

Full operator benchmark:

```text
vLLM median latency       0.447202 ms
Avelang v8 median latency 35.198536 ms
speedup = vLLM / Avelang  0.012705x
```

Full operator remains far slower than vLLM because `solve` dominates at `chunk=64`.

## 6. Rocprof

Trace output:

```text
rocprof_outputs/qwen_profile_v8_vk_bf16/
```

Counter output:

```text
rocprof_outputs/qwen_profile_v8_vk_bf16_counters/
```

v8 vk kernel trace:

```text
Kernel_Name       _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk
Workgroup_Size_X  8
Workgroup_Size_Y  32
Workgroup_Size    256
Grid_Size         16384 global work-items
LDS_Block_Size    3072 bytes
Scratch_Size      0
VGPR_Count        28
Accum_VGPR_Count  4
SGPR_Count        64
AverageNs         1046970.384615
MinNs             846723
MaxNs             1280805
```

Counter medians:

```text
GPUBusy             100.0
OccupancyPercent      7.648936
SQ_INSTS        20433134.5
SQ_INSTS_VALU    5289984.0
SQ_INSTS_VMEM     728064.0
SQ_WAVES             256.0
Wavefronts           256.0
```

This confirms:

```text
Workgroup_Size moved from 1 to 256.
thread_id.x and thread_id.y both participate in computation.
Scratch_Size remains 0.
VGPR did not explode; v8 vk uses fewer VGPRs than v8 vblock trace had used earlier.
Occupancy is still not high, but it is higher than the v6 smoke counter around 3.7%.
```

## 7. Why Full Forward Is Still Slow

After `chunk_gdr` is improved, current full forward bottleneck is no longer `chunk_gdr` for this benchmark shape.

Measured stage breakdown:

```text
solve             30.188284 ms
chunk_o            2.982892 ms
chunk_gdr_v8_vk    1.363845 ms
w_u                0.693123 ms
KKT                0.334241 ms
cumsum             0.056800 ms
```

The `solve` stage is correctness-first and serial over the chunk-local triangular solve. At `chunk_size=64`, its cost explodes and dominates full forward. That explains why `chunk_gdr` improved by about 0.45 ms but full forward only moved modestly.

## 8. Next Step

Do not keep adding complexity to `chunk_gdr` first. The next high-value optimization is:

```text
1. Parallelize/optimize solve for chunk_size=64.
2. Then optimize chunk_o, which is now the second largest measured stage.
3. Return to chunk_gdr only after solve/chunk_o stop dominating.
```

For `solve`, likely directions:

```text
parallelize rows/columns within chunk-local triangular recurrence
specialize chunk_size=64
use shared memory for the local matrix
consider block-level scan/triangular solve structure
```

For `chunk_o`, likely directions:

```text
parallelize V/K contributions instead of one thread scanning full V/K work
reuse q/k/intra weights across value lanes
consider V-block or VK-style mapping similar to this v8 chunk_gdr
```

## 9. Commands Used

Correctness:

```bash
python -m pytest -q \
  /workspace/project/avelang/test/examples/linear_attention/vllm_compare/test_qwen_gdn_chunked_avelang_v8_vllm_layout_fixed.py
```

Benchmark:

```bash
python /workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py \
  --B 1 --T 512 --Hk 4 --Hv 8 --K 64 --V 64 --chunk 64 \
  --warmup 10 --repeat 50 \
  --vk-block-v 8 --vk-block-k 32 \
  --include-vllm
```

Stage breakdown:

```bash
python /workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py \
  --B 1 --T 512 --Hk 4 --Hv 8 --K 64 --V 64 --chunk 64 \
  --warmup 5 --repeat 20 \
  --vk-block-v 8 --vk-block-k 32 \
  --stage-breakdown
```

Rocprof trace:

```bash
rocprofv3 --kernel-trace --stats \
  --kernel-include-regex _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk \
  -d /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v8_vk_bf16 \
  -o v8_vk_bf16 -f csv -- \
  python /workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py \
    --B 1 --T 512 --Hk 4 --Hv 8 --K 64 --V 64 --chunk 64 \
    --warmup 2 --repeat 10 \
    --vk-block-v 8 --vk-block-k 32
```

Rocprof counters:

```bash
rocprofv3 --pmc GPUBusy OccupancyPercent SQ_INSTS SQ_INSTS_VALU SQ_INSTS_VMEM SQ_WAVES Wavefronts \
  --kernel-include-regex _qwen_gdn_chunk_gdr_bf16_kernel_v8_vk \
  -d /workspace/project/avelang/test/examples/linear_attention/rocprof_outputs/qwen_profile_v8_vk_bf16_counters \
  -o v8_vk_bf16_counter -f csv -- \
  python /workspace/project/avelang/test/examples/linear_attention/vllm_compare/bench_qwen_gdn_v8_parallel.py \
    --B 1 --T 512 --Hk 4 --Hv 8 --K 64 --V 64 --chunk 64 \
    --warmup 1 --repeat 3 \
    --vk-block-v 8 --vk-block-k 32
```
#和vllm对比
python /home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/bench_stage1_vllm_chunk_gdn.py
patched vLLM ROCm autotune configs: 12 -> 8 (disabled num_stages=4 for chunk_delta_h)
torch: 2.10.0+rocm7.2.2.git40d237bf
hip: 7.2.53211
device: AMD Instinct MI210
====================================================================================================
B=1, T=64, Hk=2, Hv=4, K=32, V=32, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 64, 4, 32) (1, 64, 4, 32)
state  shapes: (1, 4, 32, 32) (1, 4, 32, 32)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0010410994291305542
state  max abs err: 0.004581227898597717
vLLM ms: {'mean': 0.7791523933410645, 'median': 0.7982419729232788, 'min': 0.533922016620636, 'max': 0.9382410049438477}
Avelang ms: {'mean': 17.757375717163086, 'median': 17.699403762817383, 'min': 16.822444915771484, 'max': 19.416847229003906}
speedup median: 0.045099935772989846
====================================================================================================
B=1, T=512, Hk=4, Hv=8, K=64, V=64, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 512, 8, 64) (1, 512, 8, 64)
state  shapes: (1, 8, 64, 64) (1, 8, 64, 64)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0008772760629653931
state  max abs err: 0.004484026692807674
vLLM ms: {'mean': 0.8363503813743591, 'median': 0.8504070043563843, 'min': 0.5145639777183533, 'max': 1.8219339847564697}
Avelang ms: {'mean': 35.57076644897461, 'median': 35.553409576416016, 'min': 35.077423095703125, 'max': 36.75645446777344}
speedup median: 0.02391914065312298
====================================================================================================
B=1, T=1024, Hk=4, Hv=8, K=64, V=64, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 1024, 8, 64) (1, 1024, 8, 64)
state  shapes: (1, 8, 64, 64) (1, 8, 64, 64)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0009164959192276001
state  max abs err: 0.005059957504272461
vLLM ms: {'mean': 0.8249397277832031, 'median': 0.8227279782295227, 'min': 0.5723260045051575, 'max': 1.8931390047073364}
Avelang ms: {'mean': 40.95771026611328, 'median': 40.90748977661133, 'min': 40.539310455322266, 'max': 43.15776443481445}
speedup median: 0.02011191551283877
====================================================================================================
B=1, T=2048, Hk=4, Hv=8, K=64, V=64, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 2048, 8, 64) (1, 2048, 8, 64)
state  shapes: (1, 8, 64, 64) (1, 8, 64, 64)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0012706965208053589
state  max abs err: 0.005466759204864502
vLLM ms: {'mean': 0.6860946416854858, 'median': 0.6825680136680603, 'min': 0.6385679841041565, 'max': 0.8660910129547119}
Avelang ms: {'mean': 50.515254974365234, 'median': 50.400962829589844, 'min': 50.01872634887695, 'max': 52.83426284790039}
speedup median: 0.013542757426596863
====================================================================================================
B=1, T=512, Hk=4, Hv=8, K=128, V=128, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 512, 8, 128) (1, 512, 8, 128)
state  shapes: (1, 8, 128, 128) (1, 8, 128, 128)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0005464125424623489
state  max abs err: 0.0038022398948669434
vLLM ms: {'mean': 0.757108747959137, 'median': 0.7814520001411438, 'min': 0.5376080274581909, 'max': 0.9276919960975647}
Avelang ms: {'mean': 48.77012634277344, 'median': 48.72200393676758, 'min': 48.35207748413086, 'max': 51.0845947265625}
speedup median: 0.016038995464047995
====================================================================================================
B=1, T=1024, Hk=4, Hv=8, K=128, V=128, chunk_size=64, dtype=torch.bfloat16
output shapes: (1, 1024, 8, 128) (1, 1024, 8, 128)
state  shapes: (1, 8, 128, 128) (1, 8, 128, 128)
output dtype: torch.bfloat16 torch.float32
state  dtype: torch.float32 torch.float32
output max abs err: 0.0006141364574432373
state  max abs err: 0.004225865006446838
vLLM ms: {'mean': 0.8276239037513733, 'median': 0.8385729789733887, 'min': 0.5598490238189697, 'max': 1.118577003479004}
Avelang ms: {'mean': 66.5373764038086, 'median': 66.4967041015625, 'min': 65.88134765625, 'max': 68.46985626220703}
speedup median: 0.012610745003129928
====================================================================================================
summary
B=1 T=64 Hk=2 Hv=4 K=32 V=32 chunk=64 vLLM=0.7982 ms Avelang=17.6994 ms speedup=0.045x out_err=1.041e-03 state_err=4.581e-03
B=1 T=512 Hk=4 Hv=8 K=64 V=64 chunk=64 vLLM=0.8504 ms Avelang=35.5534 ms speedup=0.024x out_err=8.773e-04 state_err=4.484e-03
B=1 T=1024 Hk=4 Hv=8 K=64 V=64 chunk=64 vLLM=0.8227 ms Avelang=40.9075 ms speedup=0.020x out_err=9.165e-04 state_err=5.060e-03
B=1 T=2048 Hk=4 Hv=8 K=64 V=64 chunk=64 vLLM=0.6826 ms Avelang=50.4010 ms speedup=0.014x out_err=1.271e-03 state_err=5.467e-03
B=1 T=512 Hk=4 Hv=8 K=128 V=128 chunk=64 vLLM=0.7815 ms Avelang=48.7220 ms speedup=0.016x out_err=5.464e-04 state_err=3.802e-03
B=1 T=1024 Hk=4 Hv=8 K=128 V=128 chunk=64 vLLM=0.8386 ms Avelang=66.4967 ms speedup=0.013x out_err=6.141e-04 state_err=4.226e-03
root@sigma106:/opt/avelang# 
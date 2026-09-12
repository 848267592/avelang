# Qwen GDN v7 优化记录

## 1. 当前 baseline

当前 baseline 是 `qwen_gdn_chunked_avelang_v6_standalone.py`。它是完整自包含的 correctness baseline，不再在运行时依赖 v3/v4/v5 wrapper。

standalone v6 的功能范围：

- 支持 normal batch 输入。
- 支持 forward only。
- 支持 q/k/v 全 FP32。
- 支持 q/k/v 全 BF16，且 g/beta/initial_state 为 FP32。
- 所有累加、中间状态、output、final_state 保持 FP32。
- 支持 `prefer_optimized=True` 和 `prefer_optimized=False`。
- 不支持 backward、cu_seqlens、packed variable-length input、raw_buffer、shared memory、MFMA、parallel scan 或 full fusion。

standalone v6 的主要 stage：

- `cumsum`
- `KKT`
- `solve`
- `w/u`
- `chunk_gdr`
- `chunk_o`
- `end-to-end`

## 2. Profiling 方法

benchmark 脚本是 `qwen_gdn_v7_benchmark.py`。它对 standalone v6 和 v7 分别计时，并打印 CSV 风格表格。

benchmark shape：

| name | B | T | Hk | Hv | K | V | chunk_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 1 | 16 | 1 | 2 | 4 | 4 | 4 |
| medium | 1 | 64 | 2 | 4 | 8 | 8 | 8 |
| larger_debug | 2 | 128 | 2 | 4 | 16 | 16 | 8 |

默认参数：

- warmup: 5
- repeat: 20
- dtype: FP32 和 BF16 都跑
- timing: 优先使用 `torch.cuda.Event(enable_timing=True)`
- fallback timing: 如果 event 不可用，使用 `time.perf_counter` 配合 `torch.cuda.synchronize`

避免第一次 JIT 编译影响计时的方法：

- 每个 stage 正式计时前先跑 warmup。
- warmup 包含第一次 Avelang JIT 编译。
- warmup 后调用 `torch.cuda.synchronize()`。
- 正式计时只统计后续 repeat 次执行。

## 3. Profiling 结果表格

以下结果来自：

```bash
PYTHONPATH=python:test/examples/linear_attention python3 test/examples/linear_attention/qwen_gdn_v7_benchmark.py
```

standalone v6 stage runtime：

| shape | dtype | cumsum | KKT | solve | w/u | chunk_gdr | chunk_o | end-to-end |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| small | FP32 | 0.032312 | 0.042608 | 0.035040 | 0.055136 | 0.067728 | 0.058936 | 0.315050 |
| small | BF16 | 0.030808 | 0.043888 | 0.030016 | 0.057488 | 0.078361 | 0.054640 | 0.319314 |
| medium | FP32 | 0.030872 | 0.043920 | 0.029672 | 0.056664 | 0.077809 | 0.053080 | 0.318810 |
| medium | BF16 | 0.048744 | 0.051480 | 0.034000 | 0.057568 | 0.068896 | 0.060488 | 0.317338 |
| larger_debug | FP32 | 0.033848 | 0.048144 | 0.028872 | 0.055432 | 0.100585 | 0.052736 | 0.320714 |
| larger_debug | BF16 | 0.030984 | 0.042952 | 0.029200 | 0.057752 | 0.116585 | 0.056096 | 0.325762 |

## 4. 选择优化目标的理由

本轮选择 `chunk_gdr`。

选择理由：

- 在多数 shape 中，standalone v6 的 `chunk_gdr` 是最慢或最接近最慢的 stage。
- `chunk_gdr` 当前 program mapping 是一个 program 负责 `(batch, value_head, value_dim)`。
- 同一个 `(batch, value_head, chunk, token)` 的 `exp(g_last - g_token)` 会跨多个 value_dim 重复计算。
- 计算复杂度主要来自 `num_chunks * chunk_size * head_dim_k * head_dim_v` 级别的状态推进。
- 访存上会重复读取 k、w、u、g，并写 h、vn、final_state。
- 相比 shared memory、MFMA、raw_buffer 等优化，先预计算 decay 是较安全的局部改动。

## 5. 本轮 v7 实际做了什么优化

修改前 program mapping：

- v6 `chunk_gdr` 是一个 program 对应 `(batch, value_head, value_dim)`。
- 每个 value_dim program 内部都会计算 chunk 末端 `exp(g_last)`。
- 每个 value_dim program 内部都会计算每个 token 的 `exp(g_last - g_token)`。

修改后 program mapping：

- 新增 `qwen_gdn_chunk_decay_avelang_v7`。
- decay 预计算 kernel 是一个 program 对应 `(batch, chunk, value_head)`。
- `chunk_gdr` 主 kernel 仍保持一个 program 对应 `(batch, value_head, value_dim)`，保证状态推进顺序和 v6 一致。

减少的重复计算：

- `exp(g_last)` 从每个 value_dim 重复计算，改为每个 `(batch, chunk, value_head)` 计算一次。
- `exp(g_last - g_token)` 从每个 value_dim 重复计算，改为每个 `(batch, token, value_head)` 计算一次。

kernel launch：

- 没有减少 kernel launch。
- 增加了一个 decay 预计算 kernel launch。

local memory：

- 没有增加 chunk_gdr 主 kernel 的 local state 大小。
- 新增全局临时张量 `decay[B,T,Hv]` 和 `last_exp[B,num_chunks,Hv]`。

dtype 和数学公式：

- 没有改变 dtype。
- 没有改变数学公式。
- BF16 路径仍只把 q/k/v 作为 BF16 输入读取，并以 FP32 累加。

## 6. 正确性验证结果

当前正确性测试结果：

- `qwen_gdn_forward_ref` 对齐通过。
- standalone v6 对齐通过。
- FP32 通过。
- BF16 通过。
- `prefer_optimized=True` 通过。
- `prefer_optimized=False` 通过，其中关闭优化时完整回退 standalone v6。
- 比较了全部五个返回张量：`g_cumsum`、`output`、`A_solved`、`chunk_states`、`final_state`。

验证命令：

```bash
PYTHONPATH=python:test/examples/linear_attention python3 -m pytest -q \
  test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v6_standalone.py \
  test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v7.py
```

结果：

```text
37 passed in 70.44s
```

## 7. 性能结果

v7 runtime 与 speedup：

| shape | dtype | v6 end-to-end | v7 end-to-end | end-to-end speedup | v6 chunk_gdr | v7 chunk_gdr | chunk_gdr speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| small | FP32 | 0.315050 | 0.358834 | 0.877982 | 0.067728 | 0.109681 | 0.617505 |
| small | BF16 | 0.319314 | 0.366186 | 0.871999 | 0.078361 | 0.111601 | 0.702151 |
| medium | FP32 | 0.318810 | 0.359914 | 0.885794 | 0.077809 | 0.111809 | 0.695907 |
| medium | BF16 | 0.317338 | 0.364266 | 0.871170 | 0.068896 | 0.111401 | 0.618456 |
| larger_debug | FP32 | 0.320714 | 0.362114 | 0.885671 | 0.100585 | 0.112049 | 0.897687 |
| larger_debug | BF16 | 0.325762 | 0.370722 | 0.878722 | 0.116585 | 0.118169 | 0.986595 |

哪些 shape 有收益：

- 当前默认 debug shape 下没有 end-to-end 收益。
- 当前默认 debug shape 下 `chunk_gdr` 也没有净收益。

哪些 shape 没收益：

- small / medium / larger_debug 的 FP32 和 BF16 都没有收益。

可能原因：

- 预计算确实减少了重复 exp，但额外增加了一个 kernel launch。
- 当前 debug shape 的 `head_dim_v` 还不够大，重复 exp 节省不足以覆盖 launch 和全局临时张量读写。
- `chunk_gdr` 主体仍然是一个 program 对应 `(batch, value_head, value_dim)`，没有改变跨 value_dim 的状态推进粒度。

## 8. 仍然没有做的优化

本轮仍然没有做：

- shared memory
- MFMA
- vectorized load/store
- raw_buffer
- parallel scan
- full fusion
- backward
- cu_seqlens

## 下一轮建议

下一轮不要继续沿用当前 decay 预计算方向，除非 benchmark shape 的 `V` 更大且证明 launch 开销被摊薄。

更值得尝试的方向：

- 对 `chunk_gdr` 探索一个 program 处理 `(batch, value_head)` 并维护 `[K,V]` state 的实现，但必须先确认 local memory 压力和并行度损失。
- 或者转向 `chunk_o`，尝试在不增加 kernel launch 的前提下进一步减少 dot/exp 重复。
- 在做任何进一步优化前，先扩展 benchmark 到更接近真实模型的 shape，再根据 profiling 选择目标。

# Qwen GDN v7 vLLM Layout Optimization Report

Date: 2026-06-09

## 1. Conclusion

新增代码版本：

- `qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py`
- `test_qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py`

这个版本针对当前 profile 中最大的热点 `chunk_gdr` 做了一个 correctness-first 的优化实验：

1. 预计算 `chunk_gdr` 中跨 value 维重复使用的衰减量：
   - `last_exp[b, chunk, hv] = exp(g_cumsum[b, chunk_tail, hv])`
   - `decay[b, token, hv] = exp(g_last - g_cumsum[b, token, hv])`
2. 保持 vLLM/Qwen3Next state layout 不变：
   - `initial_state/final_state: [B, Hv, V, K]`
   - `chunk_states(h): [B, C, Hv, V, K]`
3. 保留 `prefer_optimized=False` 回退到 v6 baseline。
4. 加了一个显式实验开关 `value_tile`：
   - 默认 `value_tile=1`
   - `value_tile=2` 会让一个 program 同时处理两个 V 维，用于实验复用 `k/decay`

重要结论：这个版本已经通过 correctness 验证，也能和 vLLM operator 小形状对齐；但在当前 scalar-local Avelang 映射下，预计算衰减没有带来可观加速，`value_tile=2` 反而明显变慢。因此它应被视为 **vLLM-layout v7 优化实验基线**，不是最终高性能版。

## 2. Why Optimize chunk_gdr First

已有 rocprof 报告显示 v6 的瓶颈集中在 `chunk_gdr`：

- BF16 `chunk_gdr`: 62.54%
- FP32 `chunk_gdr`: 60.21%
- `chunk_o`: 约 19-21%
- `w_u`: 约 7%

所以 KKT、w/u、chunk_o 虽然还能继续优化，但第一刀应该先对准 `chunk_gdr`。

按照 `linear_attn_ave/AGENTS.md` 和 `skills/avelang/SKILL.md` 的要求，本次没有直接上 raw_buffer、shared memory、MFMA 或 scheduler barrier，而是先做 scalar pointer kernel 的正确性优化，并保留慢速/基线回退路径。

## 3. Design

原 v6 `chunk_gdr` 每个 program 处理一个 `(B, Hv, V)`，内部顺序推进所有 chunks，并为每个 value 维重复计算：

```text
g_last_exp = exp(g_last)
decay = exp(g_last - g_token)
```

这些值只依赖 `(B, chunk/token, Hv)`，不依赖 `V`，因此理论上可以跨所有 value 维复用。

新版本拆成：

```text
chunk_decay_precompute -> chunk_gdr_decay
```

其中 `chunk_decay_precompute` 只按 `(B, chunk, Hv)` 运行，生成 `decay` 和 `last_exp`；`chunk_gdr_decay` 继续负责生成 `h/vn/final_state`，但读取预计算衰减。

我还尝试了 `value_tile=2`：

```text
一个 program 处理 (B, Hv, V_TILE=2)
```

这样可以在同一个 program 内跨两个 value 维复用 `k_decay`。结果表明，在当前单线程 program + local state 的 DSL 映射下，state local buffer 增大导致性能显著下降，所以默认没有启用。

## 4. Code Change Log

1. 阅读并确认约束：
   - `linear_attn_ave/AGENTS.md`
   - `linear_attn_ave/skills/avelang/SKILL.md`
   - 目标文件 `qwen_gdn_chunked_avelang_v6_vllm_layout_fixed.py`
   - 已有普通 layout `qwen_gdn_chunked_avelang_v7.py`
   - rocprof CSV 中 `chunk_gdr` 热点

2. 新增 `qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py`：
   - 复用 v6 的 cumsum/KKT/solve/w_u/chunk_o。
   - 新增 `qwen_gdn_chunk_decay_avelang_v7_vllm_layout`。
   - 新增 FP32/BF16 `chunk_gdr` decay kernels。
   - 新增 FP32/BF16 `value_tile` kernels。
   - 新增 vLLM-style wrapper：
     - `qwen_gdn_chunked_avelang_v7_vllm_layout_full`
     - `qwen_gdn_chunked_avelang_v7_vllm_layout`
     - `qwen_gdn_chunk_gated_delta_rule_vllm_compatible`

3. 新增 `test_qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py`：
   - FP32/BF16。
   - grouped heads。
   - partial chunk。
   - multi batch。
   - with/without initial_state。
   - 默认 `value_tile=1` 和显式 `value_tile=2` smoke。

4. 发现 `value_tile=2` correctness 通过但性能很差后，把默认值改为 `value_tile=1`，保留 `value_tile=2` 作为显式实验参数。

## 5. Verification

Host syntax check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -c "from pathlib import Path; paths=[Path('/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py'), Path('/home/jiandongliu/project/avelang/test/examples/linear_attention/vllm_compare/test_qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py')]; [compile(p.read_text(), str(p), 'exec') for p in paths]; print('syntax ok')"
```

Result:

```text
syntax ok
```

Note: direct `python3 -m py_compile` failed because `vllm_compare/__pycache__` is owned by `nobody`, so the check used `compile(..., "exec")` with bytecode writing disabled.

Container environment:

```text
container: qwen_vllm_avelang_rocm722
torch: 2.10.0+rocm7.2.2.git40d237bf
HIP: 7.2.53211
GPU: AMD Instinct MI210
```

Pytest command:

```bash
docker exec \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/workspace/project/avelang/test/examples/linear_attention/vllm_compare:/workspace/project/avelang/test/examples/linear_attention:/opt/avelang/python:/opt/avelang/test/examples/linear_attention \
  -e ROCM_PATH=/opt/rocm-7.2.2 \
  -e PATH=/opt/avelang-rocm722-tools:/opt/rocm-7.2.2/lib/llvm/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  -w /workspace/project/avelang \
  qwen_vllm_avelang_rocm722 \
  /opt/venv/bin/python -m pytest -q \
  /workspace/project/avelang/test/examples/linear_attention/vllm_compare/test_qwen_gdn_chunked_avelang_v7_vllm_layout_fixed.py
```

Result:

```text
.....                                                                    [100%]
5 passed in 18.26s
```

vLLM operator comparison, small BF16 case:

```text
B=1, T=64, Hk=2, Hv=4, K=32, V=32, chunk_size=64
output shapes: (1, 64, 4, 32) vs (1, 64, 4, 32)
state shapes:  (1, 4, 32, 32) vs (1, 4, 32, 32)
output max abs err: 0.0013010576367378235
state max abs err:  0.005337662994861603
```

`chunk_gdr` microbenchmark, BF16 stage-only:

```text
B=1, T=512, Hk=4, Hv=8, K=64, V=64, chunk_size=64
tile1_errs: 0.0 0.0 0.0
tile2_errs: 0.0 0.0 0.0
v6_chunk_gdr_median_ms:       1.809128999710083
v7_tile1_chunk_gdr_median_ms: 1.8188890218734741
v7_tile2_chunk_gdr_median_ms: 7.554758071899414
```

Interpretation:

- Correctness is clean: v7 matches v6 exactly for the measured stage tensors.
- Precomputing decay alone does not beat v6 on this MI210 shape because it adds an extra launch and global memory traffic.
- `value_tile=2` is a useful negative result: single-thread local state tiling increases pressure too much.

## 6. How To Optimize Further

The next optimization should not merely move scalar work around. The profile says `chunk_gdr` is hot because it serializes too much state work in one thread per `(B, Hv, V)`.

Recommended next steps:

1. Keep v6/v7 correctness tests as the oracle.
2. Add a real parallel `chunk_gdr` design:
   - parallelize the K reduction in `pred = sum_k w[k] * state[k]`;
   - use multiple threads per `(B, Hv, V)` or per `(B, Hv, V tile)`;
   - reduce within a block instead of doing the whole K loop in one thread.
3. Stage repeated `w/k/decay` data in shared memory only after the scalar/tiled design is correct.
4. Revisit `value_tile>1` only with thread-level parallelism; doing it inside one scalar program is too register/local-memory heavy.
5. Consider fusing pieces around `chunk_gdr` and `chunk_o` only after the recurrent state contract is stable.
6. Defer raw_buffer/MFMA/scheduler barriers until there is a tested block-level algorithm; they are not the first fix for this bottleneck.

Practical rule: every optimization attempt should report both correctness and a stage benchmark. The `value_tile=2` attempt here is exactly why: it was mathematically right but performance-wrong.

# Avelang Qwen GDN v6 在 vLLM 容器内运行验证报告

验证时间：2026-06-03

## 结论

Avelang Qwen GDN v6 standalone 已经可以在 vLLM 容器 `qwen_vllm_avelang_probe` 内正确运行。

需要注意：只设置 `PYTHONPATH` 和 `LD_LIBRARY_PATH` 时，import 和 py_compile 可以通过，但 pytest 会在 Avelang JIT 的 AMDGPU device link 阶段失败。原因是容器默认存在 `ROCM_PATH=/opt/rocm`，而 `/opt/rocm` 是 ROCm 7.0 / LLVM20。Avelang vLLM 兼容构建产物来自 ROCm 7.2.2 / LLVM22，所以运行 pytest 和 benchmark 时还必须把 `ROCM_PATH` 和 `PATH` 指向 ROCm 7.2.2 工具链。

最终验证环境下：

- `import avelang` 成功
- `qwen_gdn_chunked_avelang_v6_standalone` import 成功
- `py_compile` 成功
- `pytest test_qwen_gdn_chunked_avelang_v6_standalone.py` 结果：`19 passed`
- Avelang v6-only small/medium benchmark 成功，fp32 和 bf16_qkv_fp32_ref 均 correctness pass

本次没有优化 kernel，没有修改 vLLM 源码，没有重建 Avelang。

## 1. vLLM 容器环境

容器：

```text
qwen_vllm_avelang_probe
```

镜像：

```text
rocm/vllm-dev:nightly_main_20260211
```

Python / Torch / HIP / vLLM：

```text
python: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
torch: 2.9.1+git8907517
torch.version.hip: 7.0.51831-a3e329ad8
torch.version.cuda: None
torch.cuda.is_available: True
gpu:
vllm: 0.16.0rc2.dev119+g31d992d21
```

说明：`torch.cuda.get_device_name(0)` 在该 vLLM 容器内返回空字符串，但 `torch.cuda.is_available()` 为 `True`，pytest 和 benchmark 均能实际执行 GPU kernel。

容器默认环境中有：

```text
ROCM_PATH=/opt/rocm
PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:...
```

其中 `/opt/rocm` 是 ROCm 7.0 / LLVM20：

```text
/opt/rocm/llvm/bin/clang:
AMD clang version 20.0.0git ... roc-7.0.0
```

容器内也存在 ROCm 7.2.2 / LLVM22 工具链：

```text
/opt/rocm-7.2.2/lib/llvm/bin/clang:
AMD clang version 22.0.0git ... roc-7.2.2

/opt/avelang-rocm722-tools/clang:
AMD clang version 22.0.0git ... roc-7.2.2
```

## 2. 使用的环境变量

按任务要求，Avelang 使用以下 `PYTHONPATH`：

```text
/workspace/project/avelang/build-vllm-rocm722/python
/workspace/project/avelang/python
/workspace/project/avelang/test/examples/linear_attention
```

实际设置为：

```text
PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention
```

按任务要求，Avelang 使用以下 `LD_LIBRARY_PATH`：

```text
/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib
/usr/lib/x86_64-linux-gnu
/opt/rocm/lib
/opt/rocm/lib64
```

实际设置为：

```text
LD_LIBRARY_PATH=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib:/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/rocm/lib64
```

为了让 Avelang JIT 的 AMDGPU device link 使用 ROCm 7.2.2 / LLVM22，而不是容器默认 ROCm 7.0 / LLVM20，pytest 和 benchmark 还需要额外设置：

```text
ROCM_PATH=/opt/rocm-7.2.2
PATH=/opt/avelang-rocm722-tools:/opt/rocm-7.2.2/lib/llvm/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

## 3. Import probe 结果

命令在容器内打印并确认 `_avelang_bindings` 实际来自 vLLM/ROCm 7.2.2 构建目录：

```text
_avelang_bindings: /workspace/project/avelang/build-vllm-rocm722/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so
```

import 结果：

```text
avelang import ok: 0.1.0 /workspace/project/avelang/python/avelang/__init__.py
qwen_gdn_chunked_avelang_v6_standalone import ok: <function qwen_gdn_chunked_avelang_v6_standalone ...>
```

结论：

- Avelang 可以 import
- `qwen_gdn_chunked_avelang_v6_standalone` 可以 import
- 没有使用 ABI 不兼容的 `/workspace/project/avelang/python/_avelang_bindings*.so`

## 4. py_compile 结果

执行：

```bash
python3 -m py_compile \
  test/examples/linear_attention/qwen_gdn_chunked_avelang_v6_standalone.py \
  test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v6_standalone.py
```

结果：

```text
exit code 0
```

结论：两个文件均通过 py_compile。

## 5. pytest 结果

### 5.1 初次失败：只设置 PYTHONPATH 和 LD_LIBRARY_PATH

初次按任务给出的 `PYTHONPATH` / `LD_LIBRARY_PATH` 运行 pytest，但未覆盖容器默认 `ROCM_PATH=/opt/rocm`。

结果：

```text
14 failed, 5 passed, 1 warning in 23.35s
```

第一个失败 case：

```text
test_qwen_gdn_chunked_v6_standalone_fp32_matches_ref_and_current_v6[
  True-fp32_same_heads_no_initial-1-4-1-1-4-4-4-False
]
```

失败发生在第一颗 Avelang kernel JIT 编译 / AMDGPU device link 阶段，还没有进入输出 tensor 数值比较。因此没有 failing tensor 的 max diff；失败不是 tensor mismatch。

共同错误：

```text
RuntimeError: Failed to generate binary: AMDGPU device linking failed with exit code 1
```

stderr：

```text
error: Not an int attribute (Producer: 'LLVM22.0.0git' Reader: 'LLVM 20.0.0git')
1 error generated.
```

原因：容器默认 `ROCM_PATH=/opt/rocm`，导致 Avelang device linker 使用 ROCm 7.0 / LLVM20 去读取 LLVM22 产物。

### 5.2 修正后成功：ROCM_PATH/PATH 指向 ROCm 7.2.2

修正环境：

```text
ROCM_PATH=/opt/rocm-7.2.2
PATH=/opt/avelang-rocm722-tools:/opt/rocm-7.2.2/lib/llvm/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

先复跑第一个失败 case：

```text
1 passed, 1 warning in 23.72s
```

再跑完整 pytest：

```text
19 passed, 1 warning in 66.59s (0:01:06)
```

pytest warning：

```text
UserWarning: Failed validator: GCN_ARCH_NAME
```

该 warning 来自 Torch HIP tunable validator，不影响本次 Avelang v6 pytest 通过。

结论：在 vLLM 容器中使用 ROCm 7.2.2 工具链环境后，Avelang Qwen GDN v6 standalone pytest 全部通过。

## 6. Avelang v6-only benchmark 结果

使用 `qwen_gdn_compare_benchmark.py` 中已有的输入生成、reference、correctness 和计时 helper，只跑 Avelang v6 standalone，不跑 vLLM Triton 对比。

benchmark 设置：

```text
warmup=2
repeat=5
shapes=small, medium
dtype=fp32, bf16_qkv_fp32_ref
```

结果：

| shape | dtype | B | T | Hk | Hv | K | V | chunk | latency_ms | correctness |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| small | fp32 | 1 | 16 | 1 | 2 | 4 | 4 | 4 | 0.319554 | pass max_abs=2.384e-07 max_rel=8.390e-06 worst=chunk_states |
| small | bf16_qkv_fp32_ref | 1 | 16 | 1 | 2 | 4 | 4 | 4 | 0.336194 | pass max_abs=2.384e-07 max_rel=4.456e-05 worst=final_state |
| medium | fp32 | 1 | 64 | 2 | 4 | 8 | 8 | 8 | 0.322882 | pass max_abs=3.576e-07 max_rel=9.436e-05 worst=chunk_states |
| medium | bf16_qkv_fp32_ref | 1 | 64 | 2 | 4 | 8 | 8 | 8 | 0.324706 | pass max_abs=3.576e-07 max_rel=4.103e-04 worst=chunk_states |

benchmark 过程中同样出现 Torch warning：

```text
UserWarning: Failed validator: GCN_ARCH_NAME
```

结论：small / medium，fp32 / bf16_qkv_fp32_ref 下，Avelang v6 standalone 在 vLLM 容器内可运行，且 correctness 均通过。

## 7. 是否已迁移成功

结论：已成功迁移到 vLLM 容器运行。

准确说法是：Avelang v6 可以在 `qwen_vllm_avelang_probe` 中运行，但运行 pytest/benchmark 时不能只依赖容器默认 ROCm 环境；必须固定使用：

```text
PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention
LD_LIBRARY_PATH=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib:/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/rocm/lib64
ROCM_PATH=/opt/rocm-7.2.2
PATH=/opt/avelang-rocm722-tools:/opt/rocm-7.2.2/lib/llvm/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

## 8. 后续建议

下一步可以开始 operator-level adapter 和 benchmark，但还不要直接改 vLLM 源码。

建议先做一个只读/外置 adapter 脚本，显式设置上述环境变量，然后分别调用：

- vLLM 已有 `fused_recurrent_gated_delta_rule`
- Avelang `qwen_gdn_chunked_avelang_v6_standalone`

这样可以先完成 operator-level 输入输出对齐和性能对比，再决定是否需要进一步集成到 vLLM 调用路径。

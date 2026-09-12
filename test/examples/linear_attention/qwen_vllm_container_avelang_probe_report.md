# Qwen vLLM ROCm 容器内 Avelang 导入探测报告

探测时间：2026-06-02

复核修正时间：2026-06-03

## 结论摘要

已成功拉取并启动 AMD 文章指定的 vLLM ROCm 容器镜像，容器内 vLLM 已包含 `fused_recurrent_gated_delta_rule` Triton kernel 及 Qwen3.5 GDN 相关模型代码。

2026-06-03 复核后确认：Avelang 可以在该 vLLM 容器内导入。此前失败原因是使用了错误的构建产物路径 `python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`。该产物需要更高版本 glibc/libstdc++，不适合当前 vLLM 容器。

正确使用仓库里已经存在的 vLLM/ROCm 7.2.2 构建产物：

```text
/workspace/project/avelang/build-vllm-rocm722/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so
```

并设置对应的 `PYTHONPATH` 与 `LD_LIBRARY_PATH` 后，`import avelang` 和 `qwen_gdn_chunked_avelang_v6_standalone` 均已成功。

因此下一步可以进入 operator-level adapter 和 benchmark。

## 1. Docker 镜像

宿主机原本没有发现 `rocm/vllm-dev:nightly_main_20260211`，也没有明确的 vLLM ROCm 7.2.x 本地镜像；本地已有较接近但不符合目标的镜像包括：

- `rocm/vllm-dev:nightly_main_20250917`
- `rocm/vllm-dev:nightly_main_20251007`
- `rocm/vllm:rocm7.0.0_vllm_0.11.1_20251103`
- `rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210`

随后成功拉取并使用：

- 镜像：`rocm/vllm-dev:nightly_main_20260211`
- Image ID：`sha256:95da041d13426287a539048352fcc23970187c0bb54ab3f849ee794cb6832a43`
- Repo digest：`rocm/vllm-dev@sha256:a21867e715568d05b6e2ecffde64ad82d8aa39832dc2d707587a6110cadef986`

`docker pull` 最终结果：

```text
Status: Downloaded newer image for rocm/vllm-dev:nightly_main_20260211
docker.io/rocm/vllm-dev:nightly_main_20260211
```

## 2. 容器启动与 GPU 可见性

容器名称：

```text
qwen_vllm_avelang_probe
```

容器状态：

```text
qwen_vllm_avelang_probe	rocm/vllm-dev:nightly_main_20260211	Up
```

容器已成功启动，并且 PyTorch 在容器内能看到 ROCm GPU：

```text
available: True
gpu:
```

注意：`torch.cuda.get_device_name(0)` 返回了空字符串。

`rocm-smi --showproductname` 能看到两个 `gfx90a` 设备，但产品名查询失败：

```text
GPU[0]		: get_name, Error when calling libdrm
GPU[0]		: Card Series: 		N/A
GPU[0]		: Card Model: 		0x740f
GPU[0]		: Card Vendor: 		Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]		: Card SKU: 		D67301
GPU[0]		: GFX Version: 		gfx90a
GPU[1]		: get_name, Error when calling libdrm
GPU[1]		: Card Series: 		N/A
GPU[1]		: Card Model: 		0x740f
GPU[1]		: Card Vendor: 		Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]		: Card SKU: 		D67301
GPU[1]		: GFX Version: 		gfx90a
```

结论：容器启动成功，ROCm GPU 对容器可见；由于 `libdrm` 查询失败，容器内没有直接打印出 `MI210` 名称，但设备信息显示为 `gfx90a`。

## 3. 容器内软件版本

```text
python: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
torch: 2.9.1+git8907517
hip: 7.0.51831-a3e329ad8
cuda: None
available: True
gpu:
triton: 3.4.0
vllm: 0.16.0rc2.dev119+g31d992d21
vllm root: /usr/local/lib/python3.12/dist-packages/vllm
```

容器 glibc：

```text
ldd (Ubuntu GLIBC 2.35-0ubuntu3.10) 2.35
```

## 4. Avelang 导入结果

### 4.1 初次失败配置

```text
PYTHONPATH=/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention
```

结果：

```text
avelang import failed: ImportError('libmlir_c_runner_utils.so.22.0git: cannot open shared object file: No such file or directory')
```

完整 traceback 关键位置：

```text
File "/workspace/project/avelang/python/avelang/compiler/code_generator.py", line 5, in <module>
    import _avelang_bindings as _C
ImportError: libmlir_c_runner_utils.so.22.0git: cannot open shared object file: No such file or directory
```

`_avelang_bindings` 的实际路径：

```text
/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so
```

对该扩展执行 `ldd` 显示：

```text
/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found
/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.31' not found
/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.36' not found
/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
libmlir_c_runner_utils.so.22.0git => not found
libmlir_runner_utils.so.22.0git => not found
libmlir_async_runtime.so.22.0git => not found
libmlir_arm_sme_abi_stubs.so.22.0git => not found
libmlir_arm_runner_utils.so.22.0git => not found
libmlir_float16_utils.so.22.0git => not found
```

在容器内搜索以下路径未找到 `libmlir_c_runner_utils.so*`：

```text
/workspace/project/avelang
/opt
/usr/local
```

结论：该失败来自错误的扩展产物路径 `python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`，不是 Avelang 在 vLLM 容器内不可用。

### 4.2 复核后的成功配置

仓库中已有面向 vLLM/ROCm 7.2.2 的构建目录：

```text
/workspace/project/avelang/build-vllm-rocm722
```

其 CMake cache 关键配置：

```text
AVE_LANG_BACKEND=rocm
AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY=/workspace/project/avelang/build-vllm-rocm722/python
CMAKE_C_COMPILER=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/bin/clang
CMAKE_CXX_COMPILER=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/bin/clang++
CMAKE_PREFIX_PATH=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm;/opt/rocm
LLVM_DIR=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib/cmake/llvm
MLIR_DIR=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib/cmake/mlir
Python3=/usr/bin/python3.12, version 3.12.12
```

容器内存在所需 MLIR runtime：

```text
/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib/libmlir_c_runner_utils.so.22.0git
/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib/libmlir_runner_utils.so.22.0git
/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib/libmlir_float16_utils.so.22.0git
```

成功导入命令：

```bash
docker exec -w /workspace/project/avelang \
  -e PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention \
  -e LD_LIBRARY_PATH=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib:/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/rocm/lib64 \
  qwen_vllm_avelang_probe python3 -c 'import avelang; print("avelang import ok", avelang.__version__)'
```

结果：

```text
avelang import ok 0.1.0
```

## 5. qwen_gdn_chunked_avelang_v6_standalone 导入结果

初次失败配置下结果：

```text
qwen gdn v6 import failed: ImportError('libmlir_c_runner_utils.so.22.0git: cannot open shared object file: No such file or directory')
```

使用 `build-vllm-rocm722/python` 和 ROCm 7.2.2 LLVM runtime 后结果：

```text
qwen_gdn_chunked_avelang_v6_standalone import ok
```

结论：`qwen_gdn_chunked_avelang_v6_standalone` 可以在 vLLM 容器内导入。

## 6. vLLM fused_recurrent_gated_delta_rule 定位结果

已找到 `fused_recurrent_gated_delta_rule`。

核心文件：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py
```

关键匹配：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/__init__.py:10:from .fused_recurrent import fused_recurrent_gated_delta_rule
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py:27:def fused_recurrent_gated_delta_rule_fwd_kernel(
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py:178:def fused_recurrent_gated_delta_rule_fwd(
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_recurrent.py:290:def fused_recurrent_gated_delta_rule(
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py:742:            core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py:783:                fused_recurrent_gated_delta_rule(
```

Qwen3.5 相关文件也存在：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5_mtp.py
```

关键匹配：

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py:141:class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py:366:            self.linear_attn = Qwen3_5GatedDeltaNet(
```

## 7. 是否修改源码

本次没有优化 Avelang kernel。

本次没有修改 vLLM 源码。

本次只创建了该探测报告文件。

## 8. 建议下一步

当前 Avelang import/env issue 已通过正确选择构建产物和动态库路径解决，可以进入 operator-level adapter 和 benchmark。

推荐路径：

1. 后续在 vLLM 容器内运行 Avelang 时，优先使用：

```text
PYTHONPATH=/workspace/project/avelang/build-vllm-rocm722/python:/workspace/project/avelang/python:/workspace/project/avelang/test/examples/linear_attention
LD_LIBRARY_PATH=/opt/rocm-7.2.2-llvm-extract/opt/rocm-7.2.2/lib/llvm/lib:/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/rocm/lib64
```

2. 不要优先使用 `/workspace/project/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`，该产物与当前 vLLM 容器 ABI 不匹配。
3. 基于已成功导入的 `build-vllm-rocm722` 环境，构建 vLLM Triton GDN 与 Avelang DSL GDN 的 operator-level adapter 和 benchmark。

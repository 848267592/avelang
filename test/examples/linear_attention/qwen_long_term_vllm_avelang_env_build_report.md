# Qwen Long-Term vLLM + Avelang ROCm 7.2.2 Environment Build Report

Date: 2026-06-09

## Images

- Stage A image: `qwen-vllm-rocm722-base:g31d992d21`
- Stage B image: `qwen-vllm-avelang-rocm722:g31d992d21`
- Stage B image ID: `sha256:fafd32de13a8cb2bdc06a225a2a38caf464478dd6367e403e5637acc7572bff0`
- Stage B created: `2026-06-09T09:01:33.567073726+08:00`

## Runtime Versions

- ROCm image target: `7.2.2`
- HIP runtime reported by PyTorch: `7.2.53211`
- torch: `2.10.0+rocm7.2.2.git40d237bf`
- triton: `3.6.0+rocm7.2.2.git4ed88892`
- vLLM: `0.16.0rc2.dev119+g31d992d21`
- vLLM path: `/opt/venv/lib/python3.12/site-packages/vllm/__init__.py`

## GPU Visibility

- `torch.cuda.is_available()`: `True`
- GPU: `AMD Instinct MI210`

## Avelang Runtime

- avelang path: `/opt/avelang/python/avelang/__init__.py`
- `_avelang_bindings` path: `/opt/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so`
- Binding path check: passed
- Old `build-vllm-rocm722` binding path check: passed, not used
- vLLM FLA import check: `from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule` passed

## LLVM / MLIR / Clang CMake Packages

- LLVMConfig: `/opt/rocm-7.2.2/lib/llvm/lib/cmake/llvm/LLVMConfig.cmake`
- MLIRConfig: `/opt/rocm-7.2.2/lib/llvm/lib/cmake/mlir/MLIRConfig.cmake`
- ClangConfig: `/opt/rocm-7.2.2/lib/llvm/lib/cmake/clang/ClangConfig.cmake`

## Pytest Result

Command:

```bash
PYTHONPATH=/opt/avelang/python:/opt/avelang/test/examples/linear_attention:$PYTHONPATH \
python -m pytest -q /opt/avelang/test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v6_standalone.py -rs
```

Result:

```text
19 passed in 64.19s (0:01:04)
```

Status: passed. No tests were skipped during GPU runtime validation.

## Persistent Container

- Container name: `qwen_vllm_avelang_rocm722`
- Container ID: `3a0a52c0418cbdda996d81c3300709816fe8ee1cfba2ff4a7f1755b9c3130fdc`
- Persistent container validation: passed

Validation output:

```text
torch: 2.10.0+rocm7.2.2.git40d237bf 7.2.53211
gpu visible: True
gpu: AMD Instinct MI210
triton: 3.6.0+rocm7.2.2.git4ed88892
vllm: 0.16.0rc2.dev119+g31d992d21
avelang: /opt/avelang/python/avelang/__init__.py
bindings: /opt/avelang/python/_avelang_bindings.cpython-312-x86_64-linux-gnu.so
```

## Fixes Made

- No Dockerfile changes were needed during this validation.
- No Stage A rebuild was performed.
- No Docker images were deleted.
- No Docker prune command was run.
- No `_avelang_bindings.so` was copied from old containers.
- `build-vllm-rocm722/python` was not used.

## Recommended Next Command

```bash
docker exec -it qwen_vllm_avelang_rocm722 bash
```

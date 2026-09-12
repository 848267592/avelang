# Long-term vLLM + Avelang ROCm 7.2.2 environment plan

This plan is a proposal only. Do not build the images yet, and do not modify
existing containers.

## Current working baseline

- `qwen_vllm_avelang_probe` works and must be preserved.
- Avelang v6 inside the current vLLM container passed 19 pytest tests.
- The working probe currently relies on `build-vllm-rocm722/python` plus special
  `ROCM_PATH`, `PATH`, and `LD_LIBRARY_PATH` handling.
- The current Avelang environment `ljd_substrate` works.
- The long-term base image is:
  `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`.

## Mandatory backup before any future build

Run this before starting the long-term image build:

```bash
docker commit qwen_vllm_avelang_probe qwen_vllm_avelang_probe:working_avelang_v6
```

This creates an image snapshot of the known-good probe container. It should not
delete, restart, rebuild, or otherwise mutate the existing container.

## Proposed split

### Stage A: vLLM-on-ROCm7.2.2 image

Dockerfile: `docker/Dockerfile.vllm_rocm722_base`

Goal: prove that pinned vLLM can build and import on the clean ROCm 7.2.2
PyTorch base before Avelang is added.

Inputs:

- Base image:
  `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`
- vLLM commit:
  `31d992d215a05ad2e4f17653ddff0f515f865914`
- `PYTORCH_ROCM_ARCH=gfx90a`
- `VLLM_TARGET_DEVICE=rocm`

Future build command, not run yet:

```bash
docker build \
  -f docker/Dockerfile.vllm_rocm722_base \
  -t qwen-vllm-rocm722-base:g31d992d21 \
  .
```

Safety checks in Stage A:

- Capture the base `torch` and `triton` module versions and file locations.
- Generate pip constraints from the captured base `torch` and `triton` versions.
- Install vLLM build/common/ROCm requirements with those constraints.
- Re-check that `torch` and `triton` were not downgraded, upgraded, or replaced.
- Build the vLLM wheel with `--no-build-isolation --no-deps`.
- Install the vLLM wheel with `--no-deps`.
- Re-check the base `torch` and `triton` modules after the wheel install.
- Import `vllm`.
- Print `vllm.__version__`.
- Locate `fused_recurrent.py`.
- Locate `fused_recurrent_gated_delta_rule` in installed vLLM Python files.

### Stage B: vLLM+Avelang image

Dockerfile: `docker/Dockerfile.avelang_vllm_rocm722`

Goal: build Avelang natively into `/opt/avelang/python` on top of the validated
Stage A vLLM image, without relying on `build-vllm-rocm722/python`.

Inputs:

- Stage A image tag:
  `qwen-vllm-rocm722-base:g31d992d21`
- Avelang source from this checkout.
- Avelang CMake options verified against the actual CMake files:
  `AVE_LANG_BACKEND=rocm`, `WITH_PYTHON=ON`,
  `AVE_LANG_PYTHON_LIBRARY_OUTPUT_DIRECTORY=/opt/avelang/python`.
- Default CMake prefix proposal:
  `/opt/rocm/llvm;/opt/rocm`.

Future build command, not run yet:

```bash
docker build \
  -f docker/Dockerfile.avelang_vllm_rocm722 \
  --build-arg VLLM_ROCM722_BASE_IMAGE=qwen-vllm-rocm722-base:g31d992d21 \
  -t qwen-vllm-avelang-rocm722:g31d992d21 \
  .
```

Safety checks in Stage B:

- Verify the expected Avelang CMake options against the checked-out
  `CMakeLists.txt`, `python/CMakeLists.txt`, and `setup.py` design.
- Re-check that the base `torch` and `triton` modules match the Stage A capture
  before building Avelang.
- Configure Avelang with the ROCm backend and Python bindings.
- Build the native extension into `/opt/avelang/python`.
- Import `avelang`.
- Import `qwen_gdn_chunked_avelang_v6_standalone`.
- Import `vllm` in the same Python process.
- Confirm no `build-vllm-rocm722` path appears in `sys.path` or relevant env.
- Locate vLLM `fused_recurrent.py`.
- Locate `fused_recurrent_gated_delta_rule`.
- Run:

```bash
python -m pytest -q /opt/avelang/test/examples/linear_attention/test_qwen_gdn_chunked_avelang_v6_standalone.py
```

## Final acceptance criteria

- `vllm` and `avelang` import in the same Python process.
- Avelang v6 standalone pytest passes.
- vLLM `fused_recurrent_gated_delta_rule` is located or importable.
- No `build-vllm-rocm722/python` path is needed.

## Risks

- vLLM requirements may overwrite the base `torch` or `triton`.
- The selected vLLM commit may not run correctly with ROCm 7.2.2 on MI210.
- Triton kernels may target MI300/gfx942 rather than MI210/gfx90a.
- Avelang CMake options must be verified against the actual `CMakeLists.txt`
  files before building.

## Notes for the first future build attempt

- Build Stage A first and stop there if any vLLM check fails.
- Only start Stage B after Stage A imports vLLM and locates the fused recurrent
  pieces.
- Keep `qwen_vllm_avelang_probe` untouched until the new final image satisfies
  all acceptance criteria.
- If Stage A fails during requirements installation because the pinned base
  `torch` or `triton` conflicts with vLLM requirements, do not loosen the guard
  immediately. First inspect which package wants to replace the base stack.
- If Stage B fails during CMake configuration, inspect LLVM/MLIR/Clang package
  discovery before changing Avelang source or CMake options.

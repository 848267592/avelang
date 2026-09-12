# Qwen GDN / KDA recovery runbook

This file is the entry point for an agent restoring the Qwen GDN and KDA
reference environment on a new host. Follow this runbook without requiring the
user to restate the project history.

This branch is a reproducibility snapshot, not a Docker image registry and not
a copy of the official SGLang/vLLM/AITER source trees.

## Snapshot identity

| Item | Value |
|---|---|
| GitHub repository | git@github.com:848267592/avelang.git |
| Branch | backup/qwen-kda-repro-2026-09 |
| Migration base before this README | fa03b7e3ccf61305e11a553acd12a8411e992530 |
| Host project path | /home/jiandongliu/project/avelang |
| Container project path | /workspace/project/avelang |
| Hardware baseline | AMD MI300X, gfx942, 8 GPUs |
| Host driver baseline | ROCm driver 6.16.13 |

The last recovery test cloned this branch into a temporary directory, verified
the core 87 Qwen files and 22 KDA/container files by SHA-256. The historical
archive adds 751 Qwen source/report files; its separate SHA-256 manifest is
checked independently. The recovery test also parsed the selected
Python/JSON/JSONL files, checked shell syntax, and removed the temporary clone.

## Safety rules

Before changing anything, inspect the host and existing Docker state.

- Do not use git reset --hard, git clean, rm -rf on the project, or overwrite an existing checkout.
- Do not stop, delete, recreate, or modify an existing Docker container.
- Do not use Docker --rm; both reference containers must use --restart unless-stopped.
- Do not run docker system prune or upgrade global ROCm, PyTorch, Triton, or the Docker daemon.
- Do not download Kimi/Qwen model weights for operator-level recovery.
- If /dev/kfd, /dev/dri, Docker access, or the video/render groups are unavailable, stop and report the problem; do not modify the host system.

## 1. Clone this branch

Use the canonical host path on the reference server. On another host, choose a
different project path only if necessary and update the bind mount consistently.

~~~bash
AVELANG_ROOT=/home/jiandongliu/project/avelang
AVELANG_BRANCH=backup/qwen-kda-repro-2026-09

if [ -e "$AVELANG_ROOT/.git" ]; then
  git -C "$AVELANG_ROOT" status --short
  git -C "$AVELANG_ROOT" branch --show-current
else
  git clone --branch "$AVELANG_BRANCH" --single-branch \
    git@github.com:848267592/avelang.git "$AVELANG_ROOT"
fi

git -C "$AVELANG_ROOT" switch "$AVELANG_BRANCH"
git -C "$AVELANG_ROOT" rev-parse HEAD
~~~

The migration base recorded above must remain in the branch history. This README
commit advances the branch tip, so obtain the current tip with git ls-remote and
record it before proceeding; do not silently mix versions.

## 2. Verify the migrated files

The repository intentionally contains hash manifests instead of generated
binary artifacts. Run these checks from the Avelang root:

~~~bash
cd "$AVELANG_ROOT"

awk 'NF == 2 && $1 ~ /^[[:xdigit:]]{64}$/ {print}' \
  kda_baseline/QWEN_BACKUP_SHA256.txt | sha256sum -c -

awk 'NF == 2 && $1 ~ /^[[:xdigit:]]{64}$/ {print}' \
  kda_baseline/KDA_BACKUP_SHA256.txt | sha256sum -c -

grep -E '^[0-9a-f]{64}  ' kda_baseline/QWEN_HISTORY_SHA256.txt | sha256sum -c -
~~~

Expected counts are 87 core Qwen/full-graph files, 751 historical
source/report files, and 22 KDA/container reference files. The authoritative
scope documents are:

- kda_baseline/QWEN_BACKUP_BRANCH_ALLOWLIST.md
- kda_baseline/QWEN_HISTORY_ALLOWLIST.md
- kda_baseline/QWEN_HISTORY_SHA256.txt
- kda_baseline/QWEN_BACKUP_BRANCH_MANIFEST.md
- kda_baseline/QWEN_REBUILD_ENV.md
- kda_baseline/KDA_BACKUP_SHA256.txt

The old kda_baseline/QWEN_UPLOAD_INVENTORY.md is intentionally not part of
the recovery branch; the new manifest and allowlist supersede it.

## 3. Re-download the pinned official sources

These source trees are not copied into this branch. Clone them only at the
fixed commits below, and do not overwrite an existing checkout without first
inspecting its status.

| Source | URL | Checkout directory | Commit |
|---|---|---|---|
| SGLang | https://github.com/sgl-project/sglang.git | third_party/sglang | 908226fea2df861769e2720161a75649ae4c6f92 |
| vLLM | https://github.com/vllm-project/vllm.git | third_party/vllm | 40e6042ec83eb8f2971f21043a5da40496bd188a |
| ROCm AITER | https://github.com/ROCm/aiter.git | third_party/aiter | 7bb44998274fa679ece0a37bf8426ab780b1837d |

On a fresh checkout:

~~~bash
cd "$AVELANG_ROOT"
mkdir -p third_party

git clone https://github.com/sgl-project/sglang.git third_party/sglang
git -C third_party/sglang checkout --detach 908226fea2df861769e2720161a75649ae4c6f92

git clone https://github.com/vllm-project/vllm.git third_party/vllm
git -C third_party/vllm checkout --detach 40e6042ec83eb8f2971f21043a5da40496bd188a

git clone https://github.com/ROCm/aiter.git third_party/aiter
git -C third_party/aiter checkout --detach 7bb44998274fa679ece0a37bf8426ab780b1837d
~~~

Confirm each checkout with git -C <dir> rev-parse HEAD. Do not add these
official source directories to the backup branch.

## 4. Host and Docker prerequisites

The reference host was nimrodmi300 with eight MI300X-class devices. Check these
before creating anything:

~~~bash
whoami
hostname
ls -l /dev/kfd
ls -ld /dev/dri
docker version
docker info
docker ps -a
getent group video
getent group render
~~~

The host baseline is ROCm driver 6.16.13, Docker Engine 29.5.1 (API 1.54),
/dev/kfd, and /dev/dri. The host account need not be changed; the container
must receive the device nodes and the existing video/render groups. If either
group or Docker access is missing, stop and report it.

## 5. Pull the exact Docker images

GitHub stores the image references, not the image layers. Pull by the recorded
tag and digest so a moving tag cannot silently change the environment.

~~~bash
SGLANG_IMAGE='lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909'
SGLANG_REF="$SGLANG_IMAGE@sha256:16e1ae7e199418fe4df58f6dffb7b2d6b8382cc998aae0d525f8d8b2f7959c98"

VLLM_IMAGE='vllm/vllm-openai-rocm:kimi-k3'
VLLM_REF="$VLLM_IMAGE@sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b"

docker pull "$SGLANG_REF"
docker pull "$VLLM_REF"
~~~

Recorded runtime versions:

| Container | Python | PyTorch | HIP | Triton | Framework |
|---|---|---|---|---|---|
| SGLang | 3.12.3 | 2.11.0+rocm7.2 | 7.2.26015 | 3.7.0 | 0.5.19.dev20260909+gffe98a4279 |
| vLLM | 3.12.13 | 2.11.0+gitd0c8b1f | 7.2.53211 | 3.7.0 | 0.1.dev19253+g5f76ae224.d20260727 |

## 6. Create or reuse the two persistent containers

First inspect names. If either name already exists, inspect and reuse it; do
not remove or recreate it automatically.

~~~bash
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E '^(ljd_sglang_kda|ljd_vllm_kda)\b' || true
~~~

Only if the corresponding name does not exist, use the following templates.
They deliberately have no --rm, use --restart unless-stopped, and bind the
host project directory so source, scripts, logs, and results remain on the
host:

~~~bash
SGLANG_REF='lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909@sha256:16e1ae7e199418fe4df58f6dffb7b2d6b8382cc998aae0d525f8d8b2f7959c98'
VLLM_REF='vllm/vllm-openai-rocm:kimi-k3@sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b'

docker run -d \
  --name ljd_sglang_kda \
  --restart unless-stopped \
  --ipc=host \
  --shm-size=32g \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/home/jiandongliu/project/avelang,dst=/workspace/project/avelang \
  --workdir /workspace/project/avelang \
  --entrypoint /bin/bash \
  "$SGLANG_REF" \
  -c 'tail -f /dev/null'

docker run -d \
  --name ljd_vllm_kda \
  --restart unless-stopped \
  --ipc=host \
  --shm-size=32g \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --mount type=bind,src=/home/jiandongliu/project/avelang,dst=/workspace/project/avelang \
  --workdir /workspace/project/avelang \
  --entrypoint /bin/bash \
  "$VLLM_REF" \
  -c 'tail -f /dev/null'
~~~

The two containers use the same host disk through the bind mount. Docker image
layers use Docker's configured storage root; they are not the source-of-truth
for the project files.

## 7. Verify the containers and GPU

~~~bash
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E '^(ljd_sglang_kda|ljd_vllm_kda)\b'

for CONTAINER in ljd_sglang_kda ljd_vllm_kda; do
  docker exec "$CONTAINER" bash -lc 'test -d /workspace/project/avelang && test -d /workspace/project/avelang/kda_baseline'
  docker exec "$CONTAINER" python -c 'import torch, triton; print(torch.__version__); print(triton.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_capability(0))'
  docker exec "$CONTAINER" bash -lc 'command -v rocm-smi >/dev/null && rocm-smi --showuniqueid 2>/dev/null || true'
done
~~~

Expected GPU architecture is gfx942 / AMD MI300X. The vLLM image may return an
empty torch.cuda.get_device_name() string; use rocm-smi and the reported GFX
version as the additional check. Do not start a full SGLang or vLLM server for
this operator-level recovery.

## 8. Where the recovered code is

### Qwen GDN / Avelang experiments

~~~text
test/examples/linear_attention/vllm_compare/
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
test/examples/linear_attention/compile_bug/qwen_t8192_native_shaped_qh_wg128/
~~~

The two required Q@H probes are:

~~~text
test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_native_shaped_qh_wg128.py
test/examples/linear_attention/vllm_compare/repro_qwen_gdn_t8192_qh_parity_wg128.py
~~~

The full-graph recovery order is documented by the Stage 2/5B/6A/6R/6S
directories and reports in QWEN_BACKUP_BRANCH_ALLOWLIST.md:

~~~text
codex_qwen_bt64_full_pipeline_stage2/
codex_qwen_bt64_hierarchical_solve_stage5b/
codex_qwen_bt64_full_graph_gap_stage6a/
codex_qwen_bt64_recurrence_reconciliation_stage6r/
codex_qwen_bt64_bf16_recurrence_full_contract_stage6s/
codex_qwen_asm_v0_integration/
~~~

Rebuild generated HSACO/bridge artifacts on the new host with the checked-in
build/capture scripts. Do not copy old .hsaco, .o, .so, profiler output, or
model weights from the old host.

### Historical Qwen source/report archive

The complete v10--v31 experiment trail is an additional source-and-report
archive, not a performance-data dump:

~~~text
kda_baseline/QWEN_HISTORY_ALLOWLIST.md
kda_baseline/QWEN_HISTORY_SHA256.txt
test/examples/linear_attention/vllm_compare/  # selected v10--v31 files
test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/
~~~

It intentionally excludes profiler sessions, sampled CSV/JSON results, tensor
dumps, HSACO/object files, and compiler machine/ISA/IR output. Read the history
allowlist when reconstructing the progression from v10/v11 through v31; use the
X2+Z5B entry and reports as the current long-sequence candidate, not as a
claim that every historical version is production-ready.

### KDA reference and benchmark files

~~~text
kda_baseline/docker_baseline.md
kda_baseline/env_host.txt
kda_baseline/KDA_CODE_PATHS.md
kda_baseline/bench_common.py
kda_baseline/bench_kda_prefill.py
kda_baseline/bench_kda_decode.py
kda_baseline/bench_kda_prefill_aiter.py
kda_baseline/smoke_sglang_kda.py
kda_baseline/smoke_vllm_kda.py
kda_baseline/results/
~~~

SGLang ordinary Triton KDA is under the pinned checkout:

~~~text
third_party/sglang/python/sglang/kernels/ops/attention/fla/kda.py
third_party/sglang/python/sglang/kernels/ops/attention/fla/l2norm.py
third_party/sglang/python/sglang/srt/layers/attention/linear/kernels/kda_triton.py
third_party/sglang/python/sglang/srt/layers/attention/linear/kda_backend.py
~~~

vLLM AMD Kimi-K3 KDA is under the pinned checkout:

~~~text
third_party/vllm/vllm/models/kimi_k3/amd/kda.py
third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py
third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/chunk_intra.py
third_party/vllm/vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py
~~~

## 9. Recovery order for an agent

1. Clone this branch and verify both SHA manifests.
2. Inspect the host device nodes, Docker daemon, groups, and existing container names.
3. Clone SGLang/vLLM/AITER at the pinned commits.
4. Pull the two image digests and create/reuse the two persistent containers with the bind mount.
5. Verify Python, PyTorch, Triton, ROCm and gfx942 inside each container.
6. Rebuild Stage 5B/6R/6S/asm-v0 generated artifacts only when a Qwen full-graph experiment is requested.
7. Run the existing smoke/correctness/benchmark scripts from kda_baseline as needed.
8. Keep all generated outputs under the host project directory; do not add generated artifacts to the recovery branch.

Codex conversation archives are intentionally not part of this code/environment
branch yet. They should be restored separately after the repository and Docker
environment are working.

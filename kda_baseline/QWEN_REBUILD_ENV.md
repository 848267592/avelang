# Qwen / KDA 迁移环境重建清单

> 本文件记录新服务器上重新下载、创建容器和重建 Qwen GDN/KDA 复现环境所需的
> 固定信息。官方依赖源码不复制到备份分支；按 URL + SHA 重新拉取。本文档不
> 执行安装、不修改宿主机 ROCm/driver/Docker daemon。

## 1. 仓库与目录

| 项目 | 固定值 |
|---|---|
| Avelang GitHub fork | `git@github.com:848267592/avelang.git` |
| Avelang upstream | `https://github.com/causalflow-ai/avelang.git` |
| 目标备份分支 | `backup/qwen-kda-repro-2026-09` |
| 宿主机项目目录 | `/home/jiandongliu/project/avelang` |
| 容器挂载目录 | `/workspace/project/avelang` |
| 本地源码根 | `/workspace/project/avelang/test/examples/linear_attention` |
| KDA baseline 文档 | `/workspace/project/avelang/kda_baseline` |

恢复时先 clone Avelang，再将工作树切换到目标备份分支；不要依赖旧的 GitHub 分支。

## 2. 固定的官方源码依赖

| 依赖 | URL | 宿主机 checkout | 固定 commit | 用途 |
|---|---|---|---|---|
| SGLang | `https://github.com/sgl-project/sglang.git` | `third_party/sglang` | `908226fea2df861769e2720161a75649ae4c6f92` | 普通 Triton KDA reference |
| vLLM | `https://github.com/vllm-project/vllm.git` | `third_party/vllm` | `40e6042ec83eb8f2971f21043a5da40496bd188a` | AMD Kimi-K3 KDA reference |
| ROCm AITER | `https://github.com/ROCm/aiter.git` | `third_party/aiter` | `7bb44998274fa679ece0a37bf8426ab780b1837d` | AITER/FlashKDA 辅助 reference |

建议下载命令（新服务器执行时）：

```bash
cd /home/jiandongliu/project/avelang
git clone https://github.com/sgl-project/sglang.git third_party/sglang
git -C third_party/sglang checkout --detach 908226fea2df861769e2720161a75649ae4c6f92

git clone https://github.com/vllm-project/vllm.git third_party/vllm
git -C third_party/vllm checkout --detach 40e6042ec83eb8f2971f21043a5da40496bd188a

git clone https://github.com/ROCm/aiter.git third_party/aiter
git -C third_party/aiter checkout --detach 7bb44998274fa679ece0a37bf8426ab780b1837d
```

这些官方 checkout 不属于本次 Qwen 备份分支的源码迁移闭包；备份分支只记录
固定版本和重建信息。

## 3. 硬件与宿主机基线

| 项目 | 已确认值 |
|---|---|
| 主机名 | `nimrodmi300` |
| GPU | 8 × AMD MI300X-class |
| GFX architecture | `gfx942` |
| ROCm kernel/driver | `6.16.13` |
| Docker Engine | `29.5.1`，API `1.54` |
| Docker root | `/data01/docker` |
| GPU device nodes | `/dev/kfd`、`/dev/dri` |
| host kernel | `5.15.0-190-generic` |

宿主机账号曾缺少 `render` group；恢复时不要擅自修改系统组。容器运行应显式
传递设备节点和 `video`/`render` 组，并在容器内验证 `torch.cuda.is_available()`、
MI300X 和 `gfx942`。

## 4. 两个持久 Docker reference

### 4.1 SGLang

```text
container: ljd_sglang_kda
image: lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909
digest: sha256:16e1ae7e199418fe4df58f6dffb7b2d6b8382cc998aae0d525f8d8b2f7959c98
Python: 3.12.3
PyTorch: 2.11.0+rocm7.2
HIP: 7.2.26015
Triton: 3.7.0
SGLang: 0.5.19.dev20260909+gffe98a4279
```

### 4.2 vLLM

```text
container: ljd_vllm_kda
image: vllm/vllm-openai-rocm:kimi-k3
digest: sha256:5aa7e626ff73672f5ca7aae46754570488c23d33ca1ac90756a1d2d1a3fe099b
Python: 3.12.13
PyTorch: 2.11.0+gitd0c8b1f
HIP: 7.2.53211
Triton: 3.7.0
vLLM: 0.1.dev19253+g5f76ae224.d20260727
```

### 4.3 持久化运行原则

新服务器上创建时必须保持以下约束：

- 两个容器名称保持不变；
- 使用 `--restart unless-stopped`；
- **不使用 `--rm`**；
- 不停止、删除或修改其他既有容器；
- bind mount 宿主机 `/home/jiandongliu/project/avelang` 到容器
  `/workspace/project/avelang`；
- 使用 `/dev/kfd`、`/dev/dri`、`--ipc=host`、`--shm-size=32g`、
  `SYS_PTRACE` 和 unconfined seccomp；
- benchmark 时顺序固定 `HIP_VISIBLE_DEVICES=0`，不同时运行两个容器。

参考创建模板（只有在确认容器名不存在时才执行）：

```bash
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
  lmsysorg/sglang-rocm:v0.5.19-rocm724-mi30x-20260909 \
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
  vllm/vllm-openai-rocm:kimi-k3 \
  -c 'tail -f /dev/null'
```

创建前先用 `docker ps -a --format '{{.Names}}'` 检查名称，避免重复创建。若镜像
tag 发生漂移，优先用上面记录的 digest 拉取并核对版本；不要升级全局 PyTorch、
Triton 或 ROCm。

## 5. Qwen GDN / KDA 复现约束

- 只恢复 operator/kernel 级实验，不下载 Kimi/Qwen 模型权重；
- 先重建 Stage 5B solve HSACO，再执行 Stage 6R capture current-vLLM recurrence；
- 再按脚本构建 Stage 6R、Stage 6S 和 asm-v0 bridge；
- 通过 contracts 中的 dtype/layout/ABI/hash guard 验证，不把旧机器 `.hsaco`、`.so`
  当作可移植源码；
- 两个 Q@H probe 在新机器重新运行并重新生成小型 JSON 证据；
- Avelang kernel 实现不属于本次迁移恢复步骤，先保持源码闭包可复现。

## 6. 相关文档

- `kda_baseline/QWEN_BACKUP_BRANCH_ALLOWLIST.md`：逐文件迁移范围；
- `kda_baseline/QWEN_BACKUP_SHA256.txt`：87 个迁移文件的 hash 冻结；
- `kda_baseline/QWEN_BACKUP_BRANCH_MANIFEST.md`：迁移审计快照和排除项；
- `kda_baseline/docker_baseline.md`：KDA 两容器的实际版本记录；
- `kda_baseline/KDA_CODE_PATHS.md`：SGLang/vLLM KDA 代码调用链；
- `kda_baseline/env_host.txt`：MI300X/ROCm/Docker 宿主机检查记录。


# Qwen gfx942 BT64 Stage 6W: Clustered Eager Confirmation And Intermediate Audit

## 结论

在 2026-07-21 的新鲜、共享环境严格配对 Eager 重测后，Stage 6W / W1 晋级为当前
**Avelang experimental baseline**。该晋级的范围是同一共享 GPU、同一 stream、随机
Williams block 交错下的相对 public-API 排名；它不是跨宿主机的绝对延迟宣称。

冻结的 Stage 6W 路径删除了尾部的 `v_new BF16 -> FP32` cast 和最终
`FP32 output -> BF16` cast；此前小样本的中心趋势显示正收益。但按本轮预先设定的
历史严格 Eager public-API gate 没有通过；但该样本已经显示出数十至上千毫秒的异常
GPU 延迟和外部 context eviction，无法用来判断预期只有 10--60 us 的 Stage 6W 收益。
新鲜的成对重测没有复现这些长尾，且在 T=2048 和 T=8192 均稳定确认 W1 快于 U1。

本轮不修改任何 kernel。新增加的只是确认脚本和静态 intermediate-accounting 脚本。

`KKT FP32 a -> solve` 的 producer-consumer global intermediate 消除仍只是已审计的
候选。它尚未实现；现在可以开展 source/DAG/CTA ownership 可行性审计，但不与其他 kernel
改动并行实现。

## 冻结对象

对比三条完整 Eager public API：

| 名称 | 路径 |
|:--|:--|
| U1 | `qwen_gdn_full_bt64_stage6u_native_bf16_solved_eager` |
| W1 / Stage 6W | `qwen_gdn_full_bt64_stage6w_bf16_chunko_eager` |
| vLLM | `chunk_gated_delta_rule` public API |

固定合约为 gfx942、`B=1,Hk=4,Hv=8,K=V=128,BT=64`、BF16 `q/k/v`、FP32
`g/beta/initial_state`、非零 initial state、同一输入 seed 和同一 current stream。
测量使用一个完整 public API call 周围的 HIP event；没有 CUDA Graph capture/replay。

Stage 6W 代码、recurrence HSACO、KKT、solve、fused W/U 与 compiler 均未改变。

## 2026-07-21 共享环境成对重测

用户确认 U1、W1 与 vLLM 在同一个共享 GPU 环境中按随机顺序交错运行时，外部负载是共同
条件，适合用于相对排名。本次新鲜重测显式使用 `--allow-shared-gpu`，仍保留 preflight
context 和 eviction 记录，并将结论范围写为 `paired_shared_environment`。

每个 T 有 8 个 process-isolated session，每 session 有 8 个完整 Williams block；即每个
T 有 64 个配对 block、1,152 次完整 public-API call。主统计量为每 session 的 paired
median HIP-event gain；block/nested CI 是 sensitivity analysis。HIP event 与 wall-clock
均在每次同一 public API call 周围记录。

| T | U1 session-median mean ms | W1 session-median mean ms | W1 相对 U1 gain us | primary HIP CI us | wall CI us | nested CI us | W1 相对 vLLM |
|--:|--:|--:|--:|:--|:--|:--|:--|
| 2048 | `0.358282` | `0.348283` | `8.994` | `[7.246, 10.545]` | `[7.138, 10.517]` | `[5.671, 10.819]` | 快 `56.685 us`，`0.8607x` |
| 8192 | `0.966426` | `0.939173` | `27.352` | `[25.398, 29.163]` | `[26.100, 29.373]` | `[26.015, 30.168]` | 慢 `172.933 us`，`1.2259x` |

两个长度下 8/8 session 的 HIP 与 wall-clock paired median 均为正，U1-vs-W1 的 primary
gate 全部通过。T=2048 的 vLLM session-median mean 是 `0.404651 ms`，所以本 harness 中
W1 也快于 vLLM；T=8192 的 vLLM 是 `0.766289 ms`，W1 仍落后。因此 W1 是当前最佳
**Avelang** experimental baseline，并非在所有长度都全面超过 vLLM。

## 历史污染样本与统计设计

每个 `T` 独立执行 12 个 process-isolated sessions。每个 session 的两个 warmup
Williams blocks 不计时；随后执行 6 个计时 block。一个完整 block 含六个随机化的
Williams order：

```text
U1 W1 vLLM    W1 vLLM U1    vLLM U1 W1
U1 vLLM W1    vLLM W1 U1    W1 U1 vLLM
```

因此每个实现均出现于每个相对位置，并以相同次数相邻于另两个实现。每个 block 有
18 次完整 API call；每个 `T` 有 72 个配对 block 和 1,296 次计时 call。脚本保存：

- 每 call 的事件时间、session、block、order、position 和时间戳；
- 每 block 的 `U1 - W1`、`vLLM - W1` 配对差值；
- 每 session 的配对差值均值与中位数；
- session 开始/结束的 clock、温度、utilization、容器进程快照与 UTC 时间。

收益定义为 `reference - W1`，正值才代表 Stage 6W 更快。每个比较报告三种
95% bootstrap CI：i.i.d. block、session mean、nested session/block cluster。

这次历史运行的升级门槛是 U1 对 W1 在两个 `T` 的所有三类 CI 下界均大于 0 us。它把
三个相关聚合方式作为并列 gate，偏保守；更重要的是，环境污染已使该历史样本不适合判断。

## 严格确认结果

### U1 对 Stage 6W

| T | sessions | blocks | block mean / median us | block CI us | session CI us | nested CI us | gate |
|--:|--:|--:|--:|:--|:--|:--|:--|
| 2048 | 12 | 72 | `1842.964 / 160.419` | `[-1464.229, 4776.261]` | `[123.018, 3779.325]` | `[-2056.890, 5245.996]` | fail |
| 8192 | 12 | 72 | `-13.598 / 32.208` | `[-102.877, 54.876]` | `[-118.115, 51.297]` | `[-160.278, 70.893]` | fail |

T=2048 的正均值由少数几十毫秒以上的 outlier 主导，nested CI 穿过零；T=8192 的
三类 CI 也跨零。它们只能说明这批非独占样本不足以确认早期正向趋势，不能否定该趋势。

### vLLM 对 Stage 6W

这不是 Stage 6W 的升级 gate，但用于说明完整 public-API 相对关系。

| T | block mean / median us | block CI us | session CI us | nested CI us |
|--:|--:|:--|:--|:--|
| 2048 | `54881.531 / 373.085` | `[18983.948, 97085.539]` | `[-10276.580, 163247.214]` | `[-11884.085, 162116.710]` |
| 8192 | `-826.126 / -177.623` | `[-1219.237, -479.902]` | `[-1796.383, -182.098]` | `[-1768.867, -182.313]` |

T=2048 和 T=8192 都处在同一个受污染环境中，不能用于 W1/U1/vLLM 的 Eager 排名。
它们都不是此前 CUDA Graph replay 或 body benchmark 的替代品。

## 环境有效性审计

本轮脚本在每 session 的开始与结束保存 `amd-smi metric -g 0 --json` 和容器内 `ps`
快照。测量外的只读 `amd-smi process -g 0 --json` 还发现 5 个 GPU context，名称均为
`N/A`；其中一个 context 的累积 `evicted_time` 为 `90,332 ms`。这表明 GPU 不是独占
测量环境，且容器内 `ps` 看不到所有 GPU context。

原始 event 分布也确实异常宽：T=2048 的 U1/W1/vLLM 单次 event 中位数约为
`11.404/11.270/11.649 ms`，最小值约为 `0.335/0.333/0.312 ms`，最大值分别达到
`445.222/390.990/1276.906 ms`。这不是 BT64 算子的正常计算差异，不能据此宣称
Stage 6W 的实际 gain 消失或反转；它只足以阻止在该环境中晋级。

Eager public API 是此项目的权威口径，故 API 内 allocation、cast、dispatch 与返回对象
构造都应保留在计时中。caller-owned intermediate preallocation 只适用于另一个纯设备
pipeline 诊断口径，**不是** W1 Eager ranking 或晋级的前提。

2026-07-21 的重新 preflight 在任何 benchmark child/session 启动前观察到一个外部 GPU
context（PID `83597`，名称 `N/A`，累计 `evicted_time=100656 ms`），因而正确中止，未
产生第二批污染数据。它确认当前机器仍不满足微秒级确认的独占性前提。

更新后的确认脚本现在默认先执行 exclusive-GPU preflight；只有 preflight 看到零个已有
GPU context 才允许开始 session。`--allow-shared-gpu` 则允许当前已采用的严格配对相对
排名，并在产物中标记该范围。主统计量是 **session-level paired median HIP-event gain**
的 bootstrap CI；block 与 nested-cluster CI 仅作 sensitivity analysis。脚本还记录
wall-clock paired gain，并要求其 session-level 方向和 HIP event 一致。

## Stage 6W 更新后的完整图

Stage 6W 现在是 6 个 dispatch：

```text
cumsum -> g_cumsum FP32
KKT -> a FP32
solve -> a_solved BF16
fused W/U -> w_bf16 + u_bf16
immutable recurrence HSACO -> h_bf16 + v_new_bf16 + final_state
chunk-o -> public output BF16
```

已删除的两个 dispatch 是 `v_new BF16 -> FP32` 与最终 `FP32 output -> BF16` cast。

### 全局 intermediate accounting

`write+read` 对单消费者表示一次完整 materialization round trip；多消费者的
`g_cumsum` 使用所有消费者读流量。

| tensor | dtype | T=2048 / T=8192 | producer -> consumer | single use | write+read traffic at 2048 / 8192 | 直接写 consumer 格式 | 额外 dispatch |
|:--|:--|:--|:--|:--|:--|:--|:--|
| `g_cumsum` | FP32 | 0.0625 / 0.25 MiB | cumsum -> KKT,W/U,recurrence,chunk-o | no, 4 | 0.3125 / 1.25 MiB | yes | no |
| `a` | FP32 | 4 / 16 MiB | KKT -> solve | yes | 8 / 32 MiB | yes | no |
| `a_solved_bf16` | BF16 | 2 / 8 MiB | solve -> fused W/U | yes | 4 / 16 MiB | yes | no |
| `w_bf16` | BF16 | 4 / 16 MiB | fused W/U -> immutable recurrence | yes | 8 / 32 MiB | yes | no |
| `u_bf16` | BF16 | 4 / 16 MiB | fused W/U -> immutable recurrence | yes | 8 / 32 MiB | yes | no |
| `h_bf16` | BF16 | 8 / 32 MiB | immutable recurrence -> chunk-o | yes | 16 / 64 MiB | yes | no |
| `v_new_bf16` | BF16 | 4 / 16 MiB | immutable recurrence -> chunk-o | yes | 8 / 32 MiB | yes | no |
| `final_state` | FP32 | 0.5 / 0.5 MiB | recurrence -> public optional output | public | n/a | n/a | no |
| `output_bf16` | BF16 | 4 / 16 MiB | chunk-o -> public output | public | n/a | n/a | no |

## 唯一登记的下一实验

选择 `KKT FP32 a -> solve` 的 producer-consumer global intermediate 消除实验。

- 它在 T=2048 为 4 MiB，一次 write+read 是 8 MiB；T=8192 为 16 MiB 和 32 MiB。
- 两端均为 Avelang source kernel，未跨 immutable recurrence HSACO ABI。
- `h_bf16`、`v_new_bf16`、`w_bf16` 和 `u_bf16` 虽然流量更大，但其边界跨越当前不可变的
  recurrence ABI；本轮不触碰它们。
- `g_cumsum` 有四个消费者，不能把它当作单边 materialization 消除；`a_solved_bf16`
  仅有一半 `a` 的流量。

这里的含义是 KKT producer 与 solve consumer 的 private/tiled handoff 或融合，**不是**
再增加一个 BF16 cast。`a` 在 T=2048 是 4 MiB，不能假定可以整体私有化；后续只能先做
source/DAG/CTA ownership 可行性审计，不能默认融合一定会获益。独占-GPU W1 confirmation
之前不得实现该实验或叠加其他优化。

## 产物与复现

确认脚本：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/confirm_qwen_gdn_bt64_bf16_chunko_stage6w_eager.py \
  --T 2048 8192 --sessions 8 --warmup-blocks 2 --blocks-per-session 6 \
  --bootstrap-samples 10000 --out-dir <out-dir>
```

默认模式会在 session 开始前拒绝任何已有 GPU context，用于绝对延迟声明。
`--allow-shared-gpu` 启用严格配对的共享环境相对排名，并在 summary 中明确写入
`promotion_scope=paired_shared_environment`。静态图审计：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/audit_qwen_gdn_bt64_stage6w_intermediates.py \
  --T 2048 8192 --out-dir <out-dir>
```

实际原始数据位于：

- `codex_qwen_bt64_stage6w_cluster_confirm_t2048/`
- `codex_qwen_bt64_stage6w_cluster_confirm_t8192/`
- `codex_qwen_bt64_stage6w_paired_shared_retest/`
- `codex_qwen_bt64_stage6w_cluster_confirmation_and_intermediate_audit/`

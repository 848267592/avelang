# Stage 6Z Chunk-O 实验复盘

## 结论

本轮没有把 Z1 接入 X2 full graph。不是因为它不正确，也不是因为它 body 不快；它正确且
body 有收益。但 HSACO 生成了 41 个静态 barrier，超过开始前冻结的 19 个上限，所以必须
停止，不能用 isolated 速度绕过资源风险。

## 为什么先做 Z0

Stage6Y 说明长文本 chunk-o 是最大的可恢复差距。旧 Stage6W 的一个 CTA 只处理 V16：

```text
一个 chunk-head 有 8 个 V16 CTA。
每个 CTA 都重新读取 Q，并重新计算 token-token score。
```

因此可以考虑做宽 V ownership，但不能只看到 native CTA 更少就复活 O0。O0 历史上有
AccVGPR 188、LDS 33280 B、19 barrier，收益有限。Z0 先抓真实 native source/IR/ISA。

## Z0 学到的真实结构

native `chunk_fwd_kernel_o` 的真实 ownership：

```text
CTA = [BT64 token, BV64 value, 一个 value head, 一个 chunk]
每个 chunk-head 有 2 个 CTA
BK32，BF16 MFMA32，FP32 accumulator，BF16 直接输出
```

T2048 选择 4-wave/3-stage；T8192 选择 2-wave/2-stage。tile 保持一致，只有 pipeline
参数不同。Z1 遵守规则，只采用长文本的 V64/BK32 ownership，不做两套实现。

关键不是 V64 本身，而是 native TTGIR 的 LDS 生命周期：先放 Q/K/H 做 K reduction，
结束后再放 score 和 V-new。它不会让 source tile、score tile、多个 V accumulator 长期
同时活着。

## Z1 如何实现

四个 wave 覆盖四个 `[row32,value32]` 输出象限。一个 16 KiB phase buffer 被分时复用：

1. Q/H K32 staging，累积 inter-state。
2. source-half 0 score 写入前半 LDS；source-half 1 的 Q/K 使用后半 LDS，避免覆盖。
3. score 完成后，后半 LDS 改装 V-new transpose；最后做 score times V-new。

出现过两次确定性错误，且只修了明确根因：

| 问题 | 原因 | 修复 |
|:--|:--|:--|
| 初始 NaN/大误差 | 只有两个 wave 进入了 workgroup barrier | 所有 wave 均参与 staging/barrier，owner wave 才累积 score |
| intra 大误差 | source-half 1 Q/K 覆盖了 source-half 0 score | second half 改用 phase buffer 上半区 |

没有换 tile、没有换 WG、没有改 compiler、没有加 fallback。

## 结果为什么仍然是 No-Go

数值正确：T64 到 T8192 最大误差最多 `1.526e-5`，小于 `1/128`；zero V-new、output
reuse、invalid dtype 都通过。

body 也正确变快：

| T | Stage6W | Z1 | Z1 加速 |
|---:|---:|---:|---:|
| 2048 | 0.092237 ms | 0.075592 ms | 1.220x |
| 8192 | 0.252154 ms | 0.196411 ms | 1.284x |

但是资源 gate 不是只看 latency：

```text
scratch = 0，spill = 0，AccVGPR = 172，LDS = 28672 B，均通过。
static s_barrier = 41，要求 < 19，失败。
```

所以不创建 Z2 full API，不跑 Eager public 排名，也不改 X2。这个反例说明 CTA 和 MFMA
数量下降不等于 source-level schedule 已经适合 promotion；同步形状同样是硬资源。

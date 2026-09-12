# Qwen gfx942 BT64 Stage 6U 中文实验复盘

## 1. 为什么做这一轮

Stage 6T 已把 W/U 合并为一个 CTA kernel，并直接输出 BF16，但它仍读取 FP32
`a_solved`。为了保留 FP32 solved 中不能被一次 BF16 转换表达的低位信息，F1 把每个
系数拆成 main 与 residual，两部分都跑 MFMA。结果是每 CTA 2048 次 MFMA，而 vLLM
只有 128 次。

Stage 6T-Golden 审计已经把 16 倍工作量拆成：

```text
2x MFMA16/MFMA32 geometry
x 2x main/residual
x 4x lane-group predicated fragments
```

所以这一轮不猜、不同时重写三个因素，只先消除证据最充分的 main/residual 2 倍。
办法是把 solve 到 W/U 的存储边界统一成 BF16，同时仍让 solve 内部使用 FP32。

## 2. Phase A：先审计高级源码

hierarchical solve 的输入、共享 `x/work`、逐行 recurrence、block inverse DAG、MFMA
operand 与 accumulator 全是 FP32。FP32 只在最后写到 global `out`。这说明可以只把
`out_ptr` 和最终 store 改成 BF16，而不改 solve 数学。

审计还确认：

- strict upper 通过开始时清零得到；
- diagonal identity 在所有 strict-lower recurrence 完成后加 1；
- diagonal 与六个 lower block 分阶段写回；
- F1 依次执行 W main、W residual、U main、U residual；
- W 写完之后才创建 U accumulator，没有 W/U 大 accumulator 同时存活问题；
- 重复工作明确来自 residual 源码，不是 compiler 暗中复制。

这些结果决定了实验边界：只改 solved storage dtype，再删除 residual，不碰 KKT、
recurrence、chunk-o、compiler 或汇编。

## 3. Phase B：P-REF 与 P0

P-REF 是当前 FP32 solve 后接一次数值 BF16 cast。它用于回答：“如果 boundary 本来
就是 BF16，应该得到什么 bit pattern？”

P0 则把相同 solve 源码复制成实验 kernel，只做两类改动：

```text
out_ptr: f32 -> bf16
final global store: value -> convert(value, bf16)
```

内部 FP32 recurrence、MFMA、LDS、barrier、DAG、WG 和布局都没变。

最初担心当前运行时是否导出了 `mfma_16x16x4_f32_f32`。真实 source-JIT smoke 表明
当前 Docker binding 已支持它，因此没有修 compiler，也没有伪装旧 HSACO。

P0 在 42 个 case 上与 P-REF bit-exact，覆盖 T=64 到 8192、特殊数值、非默认 stream、
NaN prefill 和重复 output reuse。mismatch=0，max_abs=0。这证明 BF16 solved boundary
本身不会引入额外实现差异。

ISA 中最后写回是 `global_store_short` 和 `global_store_short_d16_hi`。我们没有做 P1
packed store，因为当前 lane 分别负责 diagonal/lower 的离散坐标，不存在一个已经证明
layout 不变的连续 4 元素写法。强行 pack 会把 ownership 改动混进来，破坏单变量原则。

## 4. Phase C：C0 main-only W/U

C0 继承 F1 的 CTA ownership、WG=256、LDS tile 和先 W 后 U 的生命周期顺序。差异是：

```text
输入 A: FP32 -> BF16
保留: W main, U main
删除: W residual coefficient/MFMA, U residual coefficient/MFMA
```

第一次写 C0 时遇到 BF16 coefficient 与 FP32 beta/g 的隐式类型降级编译错误。修复不是
改数学，而是在乘 beta/g 前明确 `al.convert(A_bf16, al.f32)`，最后再把 coefficient
量化为 BF16。之后编译和 correctness 通过。

补充多长度 reference 脚本时还出现两次测试代码错误：先是 `A[token,head,source]` 没
转成 head-first，随后是 `aw[:,head]` 索引错位。两次都发生在 PyTorch reference，kernel
没有改。修正为 `A.permute(1,0,2)` 和 `aw[head]` 后，T=64/512/2048 六个 W/U 检查
全部通过，最大绝对误差 0.0009765625。

动态 counter 给出最关键证据：

```text
F1: 524288 / 256 CTA = 2048 MFMA/CTA
C0: 262144 / 256 CTA = 1024 MFMA/CTA
```

正好减半。C0 的 VALU、SALU、VMEM、LDS 指令也下降，scratch 仍为 0。rocprof trace
中 C0 偶尔反而比 F1 慢，这是 profiler 对短 kernel 的扰动，不能当主性能；本轮始终用
完整 Eager public API 决策。

## 5. Phase D：U0 与 U1

U0 用 FP32 solve + 显式 BF16 cast + C0，目的是隔离数学 contract 与 cast 成本。

U1 用 P0 直接写 BF16 solved + C0：

- 不物化 FP32 solved；
- 没有 solved cast；
- 不物化 FP32 W/U；
- 没有 W/U cast；
- recurrence HSACO hash 不变；
- V-new cast、chunk-o、final cast 不变。

完整 correctness 使用 Eager public API，对比 Stage 6S、F1、U0、U1 和 native vLLM。
常规矩阵、30 个 T=2048 seeds、10 个 T=8192 seeds、3 个 T=16384 smoke 全部通过。
U0 与 U1 的 output/state 完全相同，说明 native BF16 solve store 与显式 cast 在完整图中
也严格等价。

## 6. 权威性能方法

本轮没有使用 CUDA/HIP Graph。每个 sample 都是：

```text
HIP start event
-> 一次完整 public_api(...)
-> HIP end event
-> synchronize
```

public API 内 allocation、cast、dispatch、wrapper glue 和返回对象构造全部计时。五个实现
使用 position-balanced 顺序，5 sessions、warmup=30、repeat=200，同时保存 HIP event
和 wall-clock。

机器在本轮出现明显跨 session 负载/时钟波动，甚至 T=2048 的绝对值高于 T=4096。
因此不能拿本轮绝对值和旧报告直接相减；可靠结论来自同一批次的 paired samples 与 CI。

T=2048：

```text
Stage 6S  1.236274 ms
F1        1.142135 ms
U0        1.106523 ms
U1        1.064339 ms
vLLM      0.884972 ms
```

U1 相对 Stage 6S 配对收益 122.551 us，95% CI [108.741,136.236] us；相对 F1
收益 84.903 us。T=8192/16384 也不退化。U1 gap slope 为 2.052548 us/chunk，Stage 6S
为 2.904197，改善 0.851649 us/chunk。

## 7. 为什么没有继续做 U2

Phase E 把 C0 剩余 8 倍差距继续对齐到源码、IR、ISA 和 PMC：

```text
4x：lane_group 0/1/2/3 在同一 wave 内形成四段 divergent MFMA 调用
2x：MFMA16 16x16 output geometry 相对 native MFMA32 32x32
```

4x 确实是高级源码机会，但“先按 lane 选择 fragment，再只调用一次 wave-uniform MFMA”
尚未在 isolated body 中证明 lowering 和资源不会恶化。如果现在直接改 full U2，就会把
新的 fragment 语义/寄存器风险带进已经通过的 U1。按 gate，U2 标 N/A，而不是隐藏失败。

## 8. 最终判断与下一步

本轮是 CASE A。U1 通过全部 correctness 和性能晋级门槛，是当前最好的 opt-in
experimental Eager candidate；Stage 6S production/default 保持不变。

下一步只做一个动作：在 isolated C0 中验证 predicate collapse，把每个 lane-group 的
fragment 先正确选出来，再尝试每个数学 step 只发一次 wave-uniform MFMA。先验证静态
ISA、动态 MFMA、资源和 isolated correctness，再决定是否创建 full U2。不要同时改
MFMA32 geometry，也不需要 compiler 或 assembly。


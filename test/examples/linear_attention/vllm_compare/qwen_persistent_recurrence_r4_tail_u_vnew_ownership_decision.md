# R4-tail U/v_new 的现有 lane ownership 判定

## 决策

**关闭“只加 I/O packet 标签即可得到 b128”的路线。**

`gfx942_bt64_bv32_joint_v4_tail_issue` 当前在 U/v_new 边界并没有任何 lane
天然持有连续的 4 或 8 个 BF16。它的 owner 是 `V1 × T8(stride 4)`，不是
`T1 × V8`。因此 `persistent_io_packet<8xbf16>` 若只是一项不改变 owner 的
metadata，会错误地声称一个 lane 拥有其实际由八个 lane 分别拥有的元素。

这不是 Triton layout 的缺失问题，也不是在现有 lowering 中丢失了一个已经存在的
V8 寄存器 packet。获得 b128 必须在 U/v_new 边界重分配 consumer owner，并为
pred_partial、vdecay stage 和 update 输入建立新的跨 lane/共享内存转换。失败的
iopacket 候选已经进行了这种重分配，并以大幅动态机器工作回退证明它不是一个安全的
“标签 lowering”。本轮不编码；下一主矛盾应转向已量化的 update operand preparation
（约 +490 VALU/V32/chunk）。

## 审计对象

- R4 production plan：`gfx942_bt64_bv32_joint_v4_tail_issue`
- 固定几何：BT=64，BV=32，WG=128，physical V32 tile。
- production HSACO：
  `5ebff98fd16fe1fa47501fbeb7dd4bd738f716d6445c0d0fd52d71e0621740c7`
- 本文只审计 `emit_audit=False` R4 的 U/v_new consumer owner；不重新审计 Triton，
  不修改 planner、layout 或 kernel。

## R4 的精确 owner 公式

R4 的已编译控制分支是
`repro_qwen_gdn_persistent_recurrence_r4_joint_v4.py` 335--354 行：

```python
for rep_out in al.range(8):
    linear = tid + rep_out * WORKGROUP
    token_off = linear // BV
    local_v = linear - token_off * BV
```

令：

- `l = tid`, `0 <= l < 128`；
- `r = rep_out`, `0 <= r < 8`；
- `q = token_tile`, `0 <= q < 2`；
- `a = floor(l / 32)`，`b = l mod 32`。

则该 lane 在一个 T32×V32 subtile 中拥有的 U、pred、v_new 与 vdecay 元素为：

\[
  (t, v) = (chunk\_start + 32q + a + 4r,\ value\_base + b),\quad r=0..7.
\]

也就是说：

| lane | 其八个 token | V 坐标 |
|---:|---|---:|
| `l=0` | `t0, t0+4, …, t0+28` | 0 |
| `l=7` | `t0, t0+4, …, t0+28` | 7 |
| `l=31` | `t0, t0+4, …, t0+28` | 31 |
| `l=32` | `t0+1, t0+5, …, t0+29` | 0 |

所以一个 lane 的八个 BF16 在 logical memory 中并不连续。对于 head-first=false 的
U/v_new 行主序地址，令 `H_V=8`、`V=128`，相邻 `r` 的字节距离是：

\[
  4 \times H_V \times V \times sizeof(BF16)
  = 4 \times 8 \times 128 \times 2 = 8192\ bytes.
\]

反过来，固定 token `t0+4r` 的连续 V8：

\[
  (t0+4r, V=0..7)
\]

分别由 lanes `0..7` 持有；`V=8..15` 由 lanes `8..15` 持有。连续性存在于**八个
不同 lane 的集合**，不存在于一个 lane 的寄存器集合。

该公式同时覆盖：

```python
pred = pred_partial[0, token_off, local_v] + pred_partial[1, token_off, local_v]
corrected = U[token_idx, global_v] - pred
v_new[token_idx, global_v] = corrected_bf16
vdecay_stage[local_v, token_base + token_off] = ...
```

因此 U load、v_new store、pred consumer 和 vdecay producer 都共享同一 `V1 × T8`
owner；它不是仅发生在最终 store 的地址选择。

## 与失败 iopacket 的公式差分

iopacket 候选故意改成：

\[
  \tau = \lfloor l/4 \rfloor,\quad p=l\bmod4,\quad
  (t,v)=(chunk\_start+32q+\tau,\ value\_base+8p+j),\quad j=0..7.
\]

即每一个 lane 变为 `T1 × V8` owner，因而 `raw_buffer_load_x4` 和
`raw_buffer_store_x4` 的地址在该 lane 内确实连续。这并不是 R4 原有寄存器 fragment
被简单打标签后的结果：它把原来由 lanes `4τ..4τ+3`、并在不同 `r` 中出现的工作，重写为
一个 lane 的 V8 packet，并在该 lane 内显式执行：

1. 8 次 `pred_partial` 读取；
2. 8 次 BF16 boundary 与 vdecay 写入；
3. BF16→u16→u32 的 pack loop；
4. raw b128 load/store 的 exec-predicated lowering。

注意，失败候选并非“一个 wave 串行循环 128 次”；它的 128 个 lane 的确同时分配了不同的
T1×V8 packet。失败原因是**重分配后的每 lane packet loop/pack/谓词成本**，以及该新 owner
和 R4 原 pred/update consumer ownership 不同，而不是一个单一 serial-128 loop。

该结论与同源 PMC/ATT 一致：candidate 的 H/U/v_new packet path 为
1,572,864 / 1,615,360 = 97.37% 的全部 VMEM，且 T=2048 比 R4-tail 慢 3.00×。

## 生产 lowering 交叉核验

production post-planner MLIR 仍保留完整 first-class recurrence op，并以 constexpr
`%false` 选择 R4 control owner；没有已存在的 U/V8 fragment 类型或 I/O packet 属性。
在 production ISA 的 U/v_new late-I/O 区间可见 scalar `global_load_ushort` 以及
`global_store_short` / `global_store_short_d16_hi`（例如 `0x38e4` 以后）；这与上述
V1 owner 一致。W/K 的 typed BF16x8 producer 与 LDS-mediated retile 是另一路已经存在的
packet ownership，不能据此推断 U/v_new 也具有 packet owner。

## 对 AveLang 的实现含义

以下实现是**无效的**：

```text
U scalar load / v_new scalar store
  + persistent_io_packet<8xbf16> metadata
  -> raw_buffer_{load,store}_x4
```

因为标签会承诺一个 lane 的八个元素连续，实际却来自八个 lane。正确实现若要重开 I/O，
必须显式表达：

```text
V1×T8 R4 owner
  -> cooperative V8 packet owner
  -> pred_partial gather + BF16 packet pack
  -> V8 vdecay/update-consumer redistribution
```

这已经是 full-recurrence ownership restructuring；它不满足“不改变 R4 pred/update/state
ownership，只添加 first-class I/O layout/lowering”的前提。故本决策关闭 I/O 路线，不创建
`persistent_io_packet` 候选，后续应只研究 update operand preparation 的 +490 VALU 差额。

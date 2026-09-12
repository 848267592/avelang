# Qwen gfx942 Stage 6Z Z11D: Static V-new Producer-to-Dot Physical Contract

## 结论

**Case C / `STOP_Z11D_MAPPING_NOT_DIRECT`。** 本轮没有实现 compiler candidate，
没有修改 Python kernel、compiler、allocator/RA、selector、X2 或 production，也没有
重新跑 correctness、PMC 或性能。

这不是因为没有找到映射，而是因为映射已经精确到足以否定本轮预注册的 direct
contract：Z5B/Z8W D phase 的 global V-new 连续 packet 与 MFMA consumer 所需的 LDS
fragment 沿着正交轴排列。

```text
global BF16x8: fixed token t, values v..v+7       (contiguous 16 B)
consumer BF16x8: fixed value v, tokens t..t+7     (contiguous in consumer LDS row)
```

把同一个 global BF16x8 的八个元素放进当前 consumer-compatible LDS layout 时，相邻
member 的地址相差 **128 B**。因此它不可能成为一个 `ds_write_b128`、`ds_write_b64`
或少量连续 packed LDS store；必须做 scatter/transpose。selected native 也不是
“packet 原封不动直进 MFMA”：其 TTGIR 在 typed V block 与 `shared4`/dot operand
之间显式出现 `amdg.in_thread_transpose`。

这恰好触发 Z11D 的硬停止条件。若继续创建 `StaticPhysicalTileContract` 或
`legacy|static_direct` A/B，只会把本应禁止的 runtime transpose/scatter 换一个名字
重新实现，重走 C0.5、D0-P、BDV2 已关闭的路线。

机器可检查的完整 mapping 在
[stage6z_z11d_vnew_physical_contract.json](stage6z_z11d_vnew_physical_contract.json)。

## 1. 审计边界

本轮只审计 D phase：

```text
score @ V-new -> intra
```

冻结 Z5B / Z8W 的 `BT64/BV64/BK32/WG256`、每 chunk-head 两个 CTA、BF16 ABI、
MFMA32、K32 reduction order、score operand、Q/H/K/g/output 路径与 caller-owned
output contract。Z5B 是唯一 isolated performance baseline；Z8W 只作其已验证的
waterfall-free H/K packet machine control。

重点不是把 aggregate VMEM 再分摊一次，也不是把 `global_load_ushort` 单独改宽。
问题是：一个宽 V-new packet 是否能在 **不做运行时重排** 的条件下，保持其 identity
穿过 global producer、LDS placement 与 MFMA fragment feeding。

## 2. Z5B/Z8W 的精确物理映射

Z5B source 的 D producer 位于
[`qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py`](../../vllm_compare/qwen_gdn_bt64_native_chunko_stage6z_z5b_direct_q_cache_consumer.py)
的 Phase C；Z8W source 明确注释 C/D/E 与 Z5B byte-for-byte 相同，故本节公式同时
适用于 Z5B 与 Z8W。

定义本 CTA 内的局部逻辑坐标：

```text
t in [0, 63]  : source token
v in [0, 63]  : V block 内的 value coordinate
```

V-new 全局 BF16 元素地址为：

```text
vn[0, chunk_start + t, value_head_idx, value_base + v]
byte = 2 * ((chunk_start + t) * 1024 + value_head_idx * 128 + value_base + v)
```

### 2.1 Logical element -> producer lane

源码 producer 是：

```python
idx = tid + rep * 256
value_offset = idx // 64
token_offset = idx - value_offset * 64
phase[384 + value_offset * 2 + token_offset // 32,
      token_offset % 32] = vn[..., token_offset, ..., value_offset]
```

因此反解一个元素 `(t, v)`：

```text
producer_wave = v mod 4
producer_lane = t
producer_tid  = 64 * (v mod 4) + t
producer_rep  = floor(v / 4)
```

这完全来自 `idx = tid + 256 * rep`，不依赖对 ISA 的猜测。lowered LLVM 中对应的
D producer loop 也保留了 `idx` 的 `/64`、`%64`、`vn` 的 `load bfloat` 和
addrspace(3) `store bfloat` 链。

### 2.2 Logical BF16x8 packet -> producer lanes

令：

```text
P(t,p) = { V-new(t, 8*p+j) | j = 0..7 }
```

它是 input `[token,value]` layout 中连续的 16 B。可是它的八个 member 在当前
Z5B producer 中属于：

```text
producer_wave(j) = j mod 4
producer_lane(j) = t
producer_tid(j)  = 64 * (j mod 4) + t
producer_rep(j)  = 2*p + floor(j/4)
```

所以一个 value-contiguous `BF16x8` 已经跨四个 wave、两个 rep。现有 source 由每个
lane scalar load 一个 BF16；final ISA 的 D region 也确实是 `global_load_ushort` 与
`ds_write_b16` family，而不是一个 producer lane 的 `buffer_load_dwordx4`。

### 2.3 Logical element -> shared byte

当前 phase/shared layout 是 `shared[512,32] bf16`；D phase 用 row `[384,511]`。
对 `(t,v)`：

```text
phase_row  = 384 + 2*v + floor(t/32)
phase_col  = t mod 32
bf16_index = 32*phase_row + phase_col
shared_byte = 2*(32*(384 + 2*v + floor(t/32)) + (t mod 32))
```

把 `P(t,p)` 代入：

```text
dest_byte(j) = 2*(32*(384 + 2*(8*p+j) + floor(t/32)) + (t mod 32))
dest_byte(j+1) - dest_byte(j) = 128 B
```

换句话说，global 上相邻的 `V(t,8p)...V(t,8p+7)` 到当前 LDS 的位置是：

```text
row 384+16p+s, column t mod 32,  s = 0,2,4,...,14
```

它们不是一个 LDS 连续段。`ds_write_b128` 只能写连续 16 B，不能把这八个 BF16
自动送去每隔 128 B 的 cell。

### 2.4 Shared -> consumer lane / fragment / MFMA role

D phase consumer 读取：

```python
v_words = phase_vec[384 + (value_half * 32 + lane_col) * 2 + source_half,
                    kt * 2 + lane_group]
v_frag = view(v_words, (2, 4, 1), bf16)
intra_acc = mfma(v_frag[0], score_frag[0], intra_acc)
intra_acc = mfma(v_frag[1], score_frag[1], intra_acc)
```

对同一 `(t,v)`：

```text
source_half = t >> 5
word        = (t & 31) >> 3
fragment_slot = t & 7
value_half  = v >> 5
lane_col    = v & 31
lane_group  = word & 1
kt          = word >> 1

consumer_lane = 32*lane_group + lane_col
consumer_wave = value_half or value_half+2  (two row halves)
fragment_half = fragment_slot >> 2          (v_frag[0] or v_frag[1])
inner_slot    = fragment_slot & 3
```

`phase_vec` 的一个 `word` 返回 fixed-`v`、连续八个 `t` 的 BF16。也就是说它正是
global packet 的转置方向。数学上 V-new 是 `score @ V-new` 的 logical B operand；
因为该 source 使用 transposed MFMA call convention，它作为 `mfma(v_frag, score_frag)`
的第一个参数出现，不能把“source parameter 0”误读成数学 A operand。

LLVM 和 ISA 也完整保留这一链：phase 的 `<4 x i32>` LDS load 被 bitcast 成
`<8 x bfloat>`，再通过 `extractelement`/`insertelement` 重建两个 4-BF16 fragment；
final ISA 使用 `ds_read_b128` 后喂入 `v_mfma_f32_32x32x8_bf16`。这证明 consumer
端连续的是 token axis，而非 global packet 的 value axis。

## 3. selected native WG256 的物理路径

native selected TTGIR 的相关图是：

```text
amdg.buffer_load tensor<64x64xbf16,#blocked>
  -> amdg.in_thread_transpose
  -> ttg.local_alloc #shared4
  -> ttg.local_load #ttg.dot_op(opIdx=1)
  -> tt.dot
```

`#blocked` 为：

```text
sizePerThread=[2,8], threadsPerWarp=[8,8], warpsPerCTA=[4,1], order=[1,0]
```

对 native logical `[token,value]` V tile，producer encoding 的可机械反解是：

```text
warp  = floor(t / 16)
lane0 = floor(t / 2) mod 8
lane1 = floor(v / 8)
lane  = 8*lane0 + lane1
tid   = 64*warp + lane
token_slot = t mod 2
value_slot = v mod 8
```

故 `P(t,p)` 的 global BF16x8 是一个 native lane packet。它的 byte offset 与 Z5B
同样遵循冻结 ABI 的 `[token,value]` 公式。

但这不是“native 没有转置”的证据。正相反，TTGIR line 360 的
`amdg.in_thread_transpose` 是明确的 transform op；随后才分配 rotating `#shared4`
并 `local_load` 到 dot operand `opIdx=1`。native LLIR 同一区域可看到
`shufflevector` 派生的分离 LDS stores/loads。

冻结 artifact 未包含 Triton layout evaluator 输出的完整 `#linear1 -> dot_op` 展开
表，因此 native 的最终 **consumer lane / fragment slot** 在 JSON 中标记为
`unknown_from_frozen_artifact`，没有凭变量名或经验补写。这个 unknown 不影响停止：
在它之前，TTGIR 已经明确声明需要 `in_thread_transpose`。

## 4. GO/STOP gate

| gate | 结果 | 机械证据 |
|---|---|---|
| 连续 8 BF16 能否形成 16 B global packet | 通过 | `P(t,p)` 在 `[token,value]` layout 连续 |
| packet 到目标 shared 的 map 是否可在编译期写出 | 通过，但为 strided | `dest_byte(j)` 已有闭式公式 |
| 一或少量 packed LDS store 能否直接放到现 consumer layout | **失败** | 相邻 member 间隔 128 B |
| consumer 能否不做 transform 直接使用该 packet | **失败** | fixed-token/value packet 与 fixed-value/token fragment 正交 |
| native 是否提供 direct packet identity 的反例 | 否 | `amdg.in_thread_transpose` 显式存在 |
| 能否避免 runtime scatter/transpose/select/rebuild | **失败** | 需要完整 64x64 transpose relation |

因此最终为：

```text
STOP_Z11D_MAPPING_NOT_DIRECT
```

这里的结论很窄也很有用：它不是说“V-new 永远不能 vectorize”，而是说在冻结的
Z5B/Z8W D-phase global layout、shared consumer layout 和 MFMA feeding 之间，不能做
本任务要求的 **packet-identity-preserving direct contract**。若想让 global x8 宽 load
生效，必须同时改变 producer ownership 或 shared/dot physical layout，并承担 runtime
transpose/scatter；那正是本任务明确禁止重开的路线。

## 5. 未实现项为何是正确行为

由于 hard gate 失败，下列产物按任务规则不存在：

- `AVELANG_STAGE6Z_VNEW_PHYSICAL_CONTRACT=legacy|static_direct` selector；
- `StaticPhysicalTileContract` compiler implementation；
- source-compatible A/B、HSACO hash、machine delta；
- correctness matrix、PMC、T2048/T8192/T16384 benchmark；
- Z8W compiler-control run。

这不是遗漏。任何上述 candidate 都会落入以下被禁止的伪成功：

```text
buffer_load_dwordx4
  -> extract eight BF16
  -> scalar LDS scatter / runtime transpose
  -> old phase layout
  -> fragment rebuild
```

它只把窄 load 的问题搬到更多 LDS/VALU，而没有满足“packet identity 至少部分保持到
shared/consumer”的成功定义。

## 6. Transferability to persistent recurrence

本轮没有改 full recurrence，但审计方法本身可以复用。将来检查
`v_decay / V-new-related value -> update MFMA` 时，一个 generic physical contract 至少
应独立给出：

1. target/MFMA encoding；
2. logical shape、packet width 与 global byte map；
3. producer ownership；
4. shared encoding 与 byte map；
5. consumer lane/fragment/MFMA operand role；
6. packet identity 是否真的跨越 producer 到 consumer。

不能直接把 Stage6Z 的 layout 套到 recurrence：Stage6Z 此处是
`score @ V-new` 的 logical B，global 是 `[token,value]`，consumer 是 fixed-value
token slice；recurrence 的 v-decay/update operand 有不同的 value layout、decay boundary
与 MFMA orientation。未来必须先完成相同的 directness gate：若也需要 transpose/scatter，
就不应假装一个 generic `StaticPhysicalTileContract` 可以消掉该工作；若 mapping 真正
direct，才有理由在 compiler 内增加通用、compile-time-only contract。

## 7. 最终状态

Z5B 仍是唯一 Stage6Z isolated performance baseline。Z11D 是一个已完成的静态证明与
停止决策，不会进入 X2、selector、production、H/K/Q/g variant、packet-width sweep、
scheduler 或 full recurrence。

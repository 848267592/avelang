# Qwen gfx942 BT64: X2 Full Graph with Z5B Chunk-O

## 结论

本轮只把当前 isolated Stage 6Z baseline `Z5B direct-Q-cache-consumer`
接入完整 `X2` 图的最后一个 chunk-o 位置。没有修改 X2 的 cumsum、KKT+solve、
W/U、current-vLLM recurrence bridge，也没有修改任何 compiler、HSACO 或 production
selector。

**X2+Z5B 在 T=1024--16384 均稳定快于原 X2；T=512 因 Z5B 的固定开销慢。**
在本次相同的 Eager public-API 口径中，它在 T=8192 仍略快于 native vLLM，但在
T=16384 落后于 vLLM。因此它是比 X2 更好的长序列 experimental 候选，不应被写成
“所有长度均胜出”或直接改为默认 production dispatch。

## 仅有的替换

新入口：

`vllm_compare/qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py`

```text
cumsum
  -> X2 CTA-local KKT + solve
  -> Stage 6U fused W/U, BF16 solved boundary
  -> hash-guarded current-vLLM BF16 recurrence bridge
  -> Z5B direct-Q-cache chunk-o
```

原 X2 使用的 Stage 6W chunk-o：

`qwen_gdn_chunk_o_bt64_bf16_vnew_bf16_out_stage6w(...)`

本候选只将其改为：

`qwen_gdn_chunk_o_bt64_native_chunko_stage6z_z5b_direct_q_cache(...)`

两者消费同一 BF16 `v_new`、BF16 `h`、FP32 `g`，并直接产生 BF16 public output。
Z5B 固定为 `BT64/BV64/BK32`、`WG256`、两个 CTA / chunk-head；其 launch wrapper
有硬 WG256/no-fallback 检查。

容器与 host 在计时前核对了三个相关源文件 hash：

| source | SHA256 |
|:--|:--|
| X2 full graph | `6629f1c548710c50655b50341c085f83fddd4d464dd655e0b2c9c5253ad895b6` |
| Z5B chunk-o | `e10d1e4b0a5c6afd538f01266cebaabbf8d621893ee95bc704d51370831b6e19` |
| X2+Z5B wrapper | `299fe2eaaa06a84a387f93a0f60b0179611c2f6dcaa58129aeed97ff8134a43b` |

这排除了 Docker 非实时挂载导致 host 与容器跑到不同 source 的可能。

## 正确性

新全图测试：

`vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py`

在 ljd ROCm 7.2.2 / MI300X 容器执行：`47 passed in 29.12s`。

覆盖：

- T=`64/512/2048/8192/16384`；
- random、high-dynamic、cancellation、neutral-gate；
- 有/无 initial state；
- finite；
- T=`64/8192/16384` 的 zero-`V_new` full-output gate；
- 直接对 native vLLM 的四个冻结 public-contract case。

recurrence 在 Z5B 替换前已经结束，因此 `final_state` 对原 X2 是逐元素 FP32
bit-exact。输出不是逐元素 BF16 bit-exact：Z5B 与 Stage 6W 的 chunk-o 使用不同的
合法 MFMA/LDS 累加路径。全矩阵中最大的 X2+Z5B--X2 输出差异为 `0.0009765625`，
小于冻结 public output 阈值 `1/128 = 0.0078125`。

直接对 vLLM 的冻结 cases：

| T / case | output max abs | final-state max abs | gate |
|:--|--:|--:|:--|
| 64 random + state | `0.00048828125` | `0.00560265779` | pass |
| 512 high-dynamic + state | `0.001953125` | `0.0165071487` | pass |
| 2048 neutral-gate, zero state | `0.001953125` | `0.0164036453` | pass |
| 8192 cancellation + state | `0.00000381469727` | `0.0000283084810` | pass |

阈值为 output `<= 1/128`、final state `<= 0.02`。所以本候选在完整 public
contract 下正确；报告不把“没有 bit-exact”隐藏成“bit-exact”。

## Eager Public-API Benchmark

这不是 isolated kernel 或 CUDA/HIP graph replay 测试。每一个 T 在独立 Python
进程中使用完整 Eager public API、current HIP stream、无 Graph；输入固定，编译、
module load 与首次 allocation 在计时前预热。

每个长度执行 5 session；每 session 10 个 block；每 block 随机化六种三方 Williams
顺序。每实现每长度有 300 次完整调用。HIP event 与 wall-clock 都记录，表中为
HIP-event 中位数。`gain` 为 `X2 - X2+Z5B`，正值表示新候选更快；CI 是 block/session
cluster bootstrap 95% CI。

| T | X2 ms | X2+Z5B ms | Z5B/X2 | gain vs X2 us, 95% CI | vLLM ms | Z5B/vLLM |
|--:|--:|--:|--:|:--|--:|--:|
| 512 | `0.193427` | `0.204803` | `1.059x` | `-8.90`, `[-11.68, -5.81]` | `0.348857` | `0.587x` |
| 1024 | `0.233246` | `0.227578` | `0.976x` | `+5.53`, `[3.02, 8.66]` | `0.351882` | `0.647x` |
| 2048 | `0.317913` | `0.291232` | `0.916x` | `+26.39`, `[23.16, 30.20]` | `0.397110` | `0.733x` |
| 4096 | `0.473723` | `0.433223` | `0.915x` | `+40.46`, `[38.00, 43.13]` | `0.517688` | `0.837x` |
| 8192 | `0.847997` | `0.753357` | `0.888x` | `+94.75`, `[92.38, 97.48]` | `0.759887` | `0.991x` |
| 16384 | `1.583108` | `1.392826` | `0.880x` | `+192.25`, `[189.04, 195.66]` | `1.229464` | `1.133x` |

因此：

- T=512：Z5B 在完整图中稳定慢约 `8.9 us`；
- T=1024：开始稳定正收益；
- T=2048：快约 `8.4%`；
- T=8192：快约 `12.6%`，且相对本次 vLLM 样本快约 `4.3 us`，event CI
  `[0.97, 8.05] us`；
- T=16384：相对 X2 快约 `13.7%`，但仍比 vLLM 慢约 `160 us`。

对六个长度的中位数按 BT64 chunk 数做普通最小二乘拟合：

| implementation | intercept ms | slope us/chunk |
|:--|--:|--:|
| X2 | `0.136147` | `5.620101` |
| X2+Z5B | `0.144039` | `4.838862` |
| vLLM | `0.295855` | `3.630467` |

Z5B 让 X2 的拟合 slope 降低约 `0.781 us/chunk`，或约 `13.9%`。它也增加了约
`7.9 us` 固定截距，这与 T=512 的回退一致。vLLM 的 slope 仍更低，因此不要从
T=8192 的小幅领先外推到所有更长长度。

## 排名与下一步

本轮验证了此前不能假设的事实：将 isolated Z5B 接入 X2 后，局部 chunk-o 收益会
转化为完整 Eager 图的实测收益，且没有破坏 public contract。

当前可用的实验性选择是：

- 短序列 `T=512`：保留原 X2 Stage 6W chunk-o；
- `T>=1024` 的已测点：X2+Z5B 更快；
- 是否做长度 selector 是独立的 dispatch-policy 工作，本轮没有修改 selector；
- production/default 仍不变。

原始 JSON/CSV 位于：

`codex_qwen_bt64_x2_z5b_chunko_full_eager/t{512,1024,2048,4096,8192,16384}/`

复现：

```bash
cd /workspace/project/avelang
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  test/examples/linear_attention/vllm_compare/test_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_full.py -s

PYTHONDONTWRITEBYTECODE=1 python3 \
  test/examples/linear_attention/vllm_compare/bench_qwen_gdn_bt64_kkt_solve_handoff_stage6x_z5b_chunko_eager.py \
  --T 2048 --sessions 5 --warmup-blocks 3 --blocks 10 \
  --out-dir /tmp/x2_z5b_eager_t2048
```

## 2026-08-18 Frozen Status

本报告和 `Avelang_Qwen_GDN_gfx942_X2_Z5B_2026-08-18_提交清单.md` 是截至
2026-08-18 的性能提交记忆：

- 单一性能候选：`X2+Z5B`；
- T=512 例外：原 X2 更快；
- production/default：仍是 v24，未在本轮改变；
- 不要把 Z5B 当成完整算子，也不要把它未接入的后续 compiler/persistent-recurrence
  研究文件归为该候选的必需依赖。

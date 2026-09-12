# Stage 5D Frozen Graph Contract

两条 audit-only 图使用相同进程、legacy current stream、同一组输入和预分配
下游 buffer。T=2048 时两图均为 8 个 dispatch：cumsum、KKT、solve、W、U、
asm recurrence、chunk-o、BF16 cast。

唯一不同 dispatch 是 ordinal 2：

| graph | solve symbol | grid | WG |
|:--|:--|:--|:--|
| A | `_qwen_gdn_solve_kernel_v18_parallel` | `256` CTA | `128` |
| B | `_qwen_gdn_solve_bt64_hierarchical_fp32_kernel_v1` | `256` CTA | `256` |

W、U 和 chunk-o 均从同一个 Stage 4 module 发射，constexpr 参数相同；asm
固定为 `qwen_gdn_bt64_gfx942_asm_v0`，grid `(4,8,1)`、WG 256、dynamic LDS
57344 B。asm HSACO SHA256 为
`eedea3f32f445dd29605519f961abcff8474882c28e022588bb3eb0991a6c226`。

精确 shape/dtype/stride/data_ptr、stream 和每个 dispatch 的 grid/WG 在
`graph_a_dispatch_map.json` 与 `graph_b_dispatch_map.json`。

AveLang JIT pointer 不是 constexpr；同进程中 W/U/chunk-o 的相同 symbol 和
相同 constexpr specialization 复用同一 code object。新的 whole-graph rocprof
和 A/B JIT binary hash capture 因 Docker 执行额度阻塞而未完成，故没有伪造两个
独立 hash。

# Accumulator Lifetime

W phase 的 `w_acc0/w_acc1` 在每个 32-column pair 内建立、跨四 source tile 累积、store 后结束。U phase 随后以相同模式使用 `u_acc0/u_acc1`。没有 full private tensor、atomic 或 kernel 内 global roundtrip。

# Ownership Gap

F0/F1 把 Avelang 的 4096 separate W/U CTA 降到 256 fused CTA，等于每个 chunk-head 一个 CTA。该结构已消除重复 launch/CTA ownership；但它没有复制 vLLM 的低 MFMA/VMEM 计数，且 F1 的 BF16 store lowering 增加 VALU。

# Native vLLM W/U Ownership

Stage 6A 的实际 full trace 显示 vLLM 使用单个 `recompute_w_u_fwd_kernel`，T=2048 为 256 CTA、WG=256，没有独立 U dispatch，直接写 BF16 W/U。它的 per-CTA ownership 是一个 chunk/value-head，而不是每个 16-column tile 一次 CTA。

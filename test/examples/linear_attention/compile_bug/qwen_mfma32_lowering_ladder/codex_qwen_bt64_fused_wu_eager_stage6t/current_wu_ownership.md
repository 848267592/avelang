# Current Separate W/U Ownership

T=2048 有 32 chunks 和 8 value heads。旧 W 与旧 U 各有 `32*8*8=2048` 个 CTA，合计 4096；每个 WG=256 的 CTA 只负责一个 16-column tile。W 和 U 因此重复 CTA、`a_solved/beta` 地址工作和 LDS tile 管理。

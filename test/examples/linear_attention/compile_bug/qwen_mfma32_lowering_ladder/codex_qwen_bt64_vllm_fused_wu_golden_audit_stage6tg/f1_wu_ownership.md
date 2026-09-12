# F1 Ownership

{
  "cta": "one CTA per (chunk, value head)",
  "t2048": {
    "chunks": 32,
    "chunk_heads": 256,
    "cta": 256,
    "cta_per_chunk_head": 1,
    "workgroup": 256,
    "waves": 4
  },
  "tile": "four 32-column pairs, four 16-token source tiles, main and residual passes",
  "phase_order": "W phase then U phase",
  "shared_coefficient": false,
  "lane_mapping": "lane_group=lane>>4 selects one of four predicated MFMA16 fragments inside every wave"
}

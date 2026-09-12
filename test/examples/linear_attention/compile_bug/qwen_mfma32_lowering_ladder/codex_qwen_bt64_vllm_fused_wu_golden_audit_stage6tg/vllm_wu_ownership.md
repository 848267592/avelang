# Actual vLLM Ownership

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
  "tile": "two U blocks and two W blocks, each 64x64, MFMA32 32x32x8",
  "phase_order": "U is complete and stored before W begins",
  "shared_coefficient": false,
  "cross_wave_exchange": "LDS/barrier exists in TTGIR/ISA; exact lane permutation not reconstructed",
  "lane_mapping": "N/A: this audit records only TTGIR warpsPerCTA=[2,2] and MFMA32 fragment geometry"
}

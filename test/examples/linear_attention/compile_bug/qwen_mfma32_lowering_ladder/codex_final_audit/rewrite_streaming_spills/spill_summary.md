# Qwen v29 Spill VReg Classification

- Spill records: `107`
- Spill words: `151`

| category | spill words |
|:--|--:|
| fragment REG_SEQUENCE/copy | 72 |
| scalar/vector address formation | 79 |

- Crosses MFMA32 interval: `0` words
- Crosses MFMA16 interval: `0` words
- Crosses both intervals: `0` words
- Frame objects named by spill saves: `107`
- Sum of individual frame-object extents: `604` bytes
- Final private-frame size must be read from the code object: LLVM may color/reuse stack slots.

The detailed per-vreg CSV/JSON intentionally labels LLVM/AveLang source origin as unavailable when LTO stripped debug locations.

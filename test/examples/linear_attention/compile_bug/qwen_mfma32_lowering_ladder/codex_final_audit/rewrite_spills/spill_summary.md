# Qwen v29 Spill VReg Classification

- Spill records: `130`
- Spill words: `190`

| category | spill words |
|:--|--:|
| fragment REG_SEQUENCE/copy | 94 |
| scalar/vector address formation | 96 |

- Crosses MFMA32 interval: `0` words
- Crosses MFMA16 interval: `0` words
- Crosses both intervals: `0` words
- Frame objects named by spill saves: `130`
- Sum of individual frame-object extents: `760` bytes
- Final private-frame size must be read from the code object: LLVM may color/reuse stack slots.

The detailed per-vreg CSV/JSON intentionally labels LLVM/AveLang source origin as unavailable when LTO stripped debug locations.

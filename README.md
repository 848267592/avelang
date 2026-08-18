# Ave

Ave is an experimental Pythonic language for writing GPU kernels with explicit control over execution geometry, memory layouts, and backend intrinsics. It uses a tile-based programming model and compiles `avelang` kernels through a native MLIR pipeline.

The Python package is `ave-lang`; user code imports `avelang`.

## Documentation

Read the documentation at [causalflow.ai/docs/avelang](https://www.causalflow.ai/docs/avelang). It covers installation, tutorials, the language reference, and development notes.

## Status

Ave is alpha-stage software. APIs, generated code, and backend coverage may change.

## Qwen GDN gfx942 research snapshot

This fork contains an experimental AMD MI300X Qwen GDN X2+Z5B integration. See
the [X2+Z5B entry README](QWEN_GDN_X2_Z5B.md) for the code path, compiler intrinsic
change, validation commands, and measured results.

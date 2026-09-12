# Stage 5F Frozen Direct-Common-Out Graph

This audit reuses the Stage 5E graph in one process and one stream. Both paths use the same preallocated solve-output tensor. There are no internal stage events, allocations, copies, fills, dummy kernels, or production-path changes.

```text
cumsum -> KKT -> solve_direct_common_out -> W -> U -> asm-v0 -> chunk-o -> cast
```

The sole A/B difference is the solve implementation and its fixed workgroup. See `graph_a.json`, `graph_b.json`, and `graph_diff.md`.

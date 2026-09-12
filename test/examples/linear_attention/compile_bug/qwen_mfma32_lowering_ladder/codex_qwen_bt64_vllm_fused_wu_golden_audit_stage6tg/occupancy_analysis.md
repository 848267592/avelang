# Occupancy Analysis

F1 static resource pressure is modest: 72 VGPR, 8 AGPR, zero spills. Native normal 4w/2s code object has 180 VGPR and 16 AGPR, also zero spills. The native profiler row has a different WG128 autotune selection and is not normal eager occupancy evidence. Resource pressure is not the primary F1 root cause.

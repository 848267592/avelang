# Stage 5E Experiment Contract

## Mode 0: Original

Each solve writes its separate preallocated output and unchanged downstream
reads that output. There is no solve-output copy.

## Mode 1: Stage 5D Copy Control

Each solve writes a separate output, then a device copy writes canonical
consumer storage outside the timed tail. This mode is context only and is not
the Stage 5E deciding experiment.

## Mode 2: Direct Common Out

Both original solve kernels receive the exact same `solved_common.data_ptr()`
as `out_ptr`. W/U immediately reads `solved_common`. There is no allocation,
copy, fill, zero, cast, synchronization, or other dispatch between solve and W.

Every benchmark session allocates one common tensor before timing; both A/B
share it for the entire session. Five sessions provide five independently
allocated common buffers per T.


# Eager Methodology

Six T values use five independent sessions, 30 complete-public-call warmups per session, and 200 balanced repetitions. Each sample synchronizes the current stream, records a HIP start event, invokes one public API, records an end event, synchronizes, and stores both event and wall-clock duration. Inputs are immutable and pre-created; outputs/intermediates are allocated inside public APIs.

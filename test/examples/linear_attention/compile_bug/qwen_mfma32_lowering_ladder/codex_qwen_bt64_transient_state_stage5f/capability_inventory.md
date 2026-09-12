# Stage 5F Capability Inventory

This is a read-only inventory. Availability does not imply that a mode passes the low-perturbation gate. `available_counters.txt` and `capability_query_raw.json` contain the exact tool output.

| capability | discovered path / status |
|:--|:--|
| `rocprofv3` | `/opt/rocm/bin/rocprofv3` |
| `rocprof` | `/opt/rocm/bin/rocprof` |
| `rocprofv2` | `/opt/rocm/bin/rocprofv2` |
| `amd-smi` | `/opt/rocm/bin/amd-smi` |
| `rocm-smi` | `/opt/rocm/bin/rocm-smi` |

Candidate modes are HIP events, rocprof kernel trace, PMC/counter collection, PC/thread trace if listed, and coarse amd-smi/rocm-smi telemetry. Only a calibrated mode may be used for causal collection.

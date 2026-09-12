# Full Sequence Comparison
This table compares only the same `chunk_delta_h` state-update operator. v24 full-forward includes cumsum/KKT/solve/w_u/chunk_o and therefore is not a like-for-like latency row here.
| T | vLLM wrapper HIP-event median ms | HSACO hash | original rocprof median us | rebuilt rocprof median us |
|--:|--:|:--|--:|--:|
| 64 | 0.058588 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | n/a | n/a |
| 128 | 0.060390 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | n/a | n/a |
| 512 | 0.088191 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | 309.8005 | 308.358 |
| 2048 | 0.193968 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | 345.253 | 331.99350000000004 |
| 8192 | 0.616917 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | 1108.0075000000002 | 1108.8265000000001 |
| 16384 | 1.214906 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | 4490.189 | 3949.709 |

The captured rocprof samples show host/system interference, especially at T=512 and T=16384; original and rebuilt have identical static ISA, resources, and dynamic instruction counts. They are retained as device-trace evidence, not claimed as a performance optimization.

Extracted/rebuilt vs vLLM bit-exact maxima: `{'original': {'h': 0.0, 'v_new': 0.0, 'final_state': 0.0}, 'rebuilt': {'h': 0.0, 'v_new': 0.0, 'final_state': 0.0}}`.

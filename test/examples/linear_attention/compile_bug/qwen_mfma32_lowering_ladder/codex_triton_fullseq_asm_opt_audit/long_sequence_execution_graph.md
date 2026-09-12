# Long Sequence Execution Graph

The real vLLM wrapper launches one `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` dispatch per call. `T` is runtime (`@triton.jit(do_not_specialize=["T"])`); the kernel loops over `ceil(T/64)` chunks inside each program. For B=1,H=8,V=128,BV=32, the grid remains `(4,8,1)` for every tested length.

| T | HSACO SHA256 | config | grid | WG | dynamic LDS | event median ms |
|--:|:--|:--|:--|--:|--:|--:|
| 64 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 0.05858750082552433 |
| 128 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 0.06038999930024147 |
| 512 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 0.08819099888205528 |
| 2048 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 0.1939679980278015 |
| 8192 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 0.6169170141220093 |
| 16384 | `cb1811c318652b89989dbf63ecda595acb82324561084f7b825291b6b56f03f9` | BV=32, w=4, s=2 | `(4,8,1)` | 256 | 57344 | 1.2149055004119873 |

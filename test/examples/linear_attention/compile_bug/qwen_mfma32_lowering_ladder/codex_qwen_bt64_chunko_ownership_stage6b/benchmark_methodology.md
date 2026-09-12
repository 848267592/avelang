# Body Benchmark Method

`bench_qwen_gdn_bt64_chunko_ownership_stage6b.py --mode body` uses the Stage
6A input generator, current stream, CUDA/HIP graph capture, fixed buffers,
warmup=20, repeat=100, five sessions, and the balanced
`current,O0,O1,vLLM,vLLM,O1,O0,current` replay sequence. One HIP event wraps
one captured chunk-o body only. No allocation, compile, or module load is in
the replay measurement. The saved `body_raw.csv`, `body_summary.csv`, and
`body_slopes.csv` are the timing authority; rocprof trace is resource-only.

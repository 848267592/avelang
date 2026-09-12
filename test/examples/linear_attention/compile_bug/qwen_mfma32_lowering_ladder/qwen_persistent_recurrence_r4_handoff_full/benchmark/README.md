# R4 Test-Bench

本目录标记“怎么测”，实际 Python harness 在同级 `avelang/`：

`../avelang/bench_qwen_gdn_persistent_recurrence_r4.py`

它是 fresh-process recurrence-body benchmark，不是最终 Eager public API
排名。每个 worker 独立 Python 进程；输入和输出在计时前预分配；编译、模块
加载和 allocation 不计入 HIP event；不使用 Graph capture；默认 warmup=10、
repeat=50、sessions=5，顺序轮换。

## 推荐命令

```bash
cd /home/jiandongliu/project/avelang
export PYTHONPATH="$PWD/python:$PWD/test/examples/linear_attention/vllm_compare"

python3 test/examples/linear_attention/compile_bug/qwen_mfma32_lowering_ladder/qwen_persistent_recurrence_r4_handoff_full/avelang/bench_qwen_gdn_persistent_recurrence_r4.py \
  --T 512 1024 2048 8192 \
  --sessions 5 --warmup 10 --repeat 50 \
  --out-json /tmp/qwen_r4_body.json
```

会比较七个 body arm：

1. B0 legacy native；
2. R1 joint_v1；
3. R2 joint_v2；
4. R3 joint_v3；
5. R4 joint_v4；
6. current-vLLM Triton recurrence body；
7. Stage 6R external current-vLLM HSACO bridge。

`direct Triton` 和 `external bridge` 都是 control，不是 R4 的 Avelang 编译
产物。若只想运行 R4 correctness，使用：

```bash
python3 .../qwen_persistent_recurrence_r4_handoff_full/avelang/repro_qwen_gdn_persistent_recurrence_r4.py
```

## 看结果时不要混用的指标

- body latency 是 HIP event 的 kernel-body诊断值；
- public Eager API 还会包含上游/下游 kernel、launch、allocator 和 wrapper；
- rocprof 的动态 PMC 和 ISA 的静态 lexical count 是两种不同证据；
- external bridge 的时间反映已编译 Triton HSACO，不代表 Avelang source
  kernel 已经实现同样的 lowering。

# GPU Test Results

Executed in `ljd_qwen_vllm_avelang_rocm722` with the ROCm/AveLang Python
environment:

```text
python -m pytest -q test_qwen_gdn_bt64_native_wu_chunko_mfma_v1.py -s
5 passed in 13.63s

python -m pytest -q test_qwen_gdn_full_bt64_native_wu_o_v1.py -s
3 passed in 28.07s
```

The larger authoritative gate is `stage3_runner.py --random-cases 30`, which
writes `full_correctness.json/csv`: 37 / 37 cases accepted, including T=8192
random and neutral-gate cases.


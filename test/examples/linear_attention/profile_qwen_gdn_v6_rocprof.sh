#!/usr/bin/env bash
set -euo pipefail

shape="larger_debug"
dtype="bf16"
warmup="8"
repeat="80"
seed="2027"
out_root="test/examples/linear_attention/rocprof_outputs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shape)
      shape="$2"
      shift 2
      ;;
    --dtype)
      dtype="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --repeat)
      repeat="$2"
      shift 2
      ;;
    --seed)
      seed="$2"
      shift 2
      ;;
    --out-root)
      out_root="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./test/examples/linear_attention/profile_qwen_gdn_v6_rocprof.sh [options]

Options:
  --shape     small | medium | larger_debug | large_v | larger_tv
  --dtype     fp32 | bf16
  --warmup    Warmup iterations before profiling. Default: 8
  --repeat    Profiled iterations. Default: 80
  --seed      Input seed. Default: 2027
  --out-root  Output root directory. Default: test/examples/linear_attention/rocprof_outputs

Example:
  HIP_VISIBLE_DEVICES=0 ./test/examples/linear_attention/profile_qwen_gdn_v6_rocprof.sh --dtype bf16
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v rocprofv3 >/dev/null 2>&1; then
  echo "rocprofv3 was not found. Run this script inside the ROCm container." >&2
  exit 1
fi

run_name="v6_${shape}_${dtype}"
out_dir="${out_root}/qwen_profile_${run_name}"
mkdir -p "${out_dir}"

export PYTHONPATH="${PYTHONPATH:-python:test/examples/linear_attention}"

rocprofv3 \
  --kernel-trace \
  --hip-runtime-trace \
  --memory-copy-trace \
  --stats \
  --summary \
  --summary-output-file stdout \
  -u usec \
  -d "${out_dir}" \
  -o "${run_name}" \
  -f csv \
  -- python3 test/examples/linear_attention/qwen_gdn_profile_v6_runner.py \
    --shape "${shape}" \
    --dtype "${dtype}" \
    --warmup "${warmup}" \
    --repeat "${repeat}" \
    --seed "${seed}"

echo
echo "rocprofv3 CSV files:"
find "${out_dir}" -maxdepth 1 -type f -name "${run_name}_*.csv" | sort
echo
echo "Kernel stats:"
echo "  ${out_dir}/${run_name}_kernel_stats.csv"

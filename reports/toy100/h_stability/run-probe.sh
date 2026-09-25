#!/usr/bin/env bash
set -euo pipefail
PREP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH_PYTHON=${GAN_PYTHON:-/tmp/pr38-default-env/bin/python}
VARIANT=${1:?choose control, critic_refresh2, average2, or extra_adam}
OUTPUT=${2:?provide a new output directory}
STEPS=${3:-200}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=''
"$BENCH_PYTHON" -u "$PREP_ROOT/stability_runner.py" --variant "$VARIANT" --output "$OUTPUT" --steps "$STEPS"

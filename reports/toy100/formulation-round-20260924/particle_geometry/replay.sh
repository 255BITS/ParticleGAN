#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'Usage: bash replay.sh CANDIDATE GATE NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
PARTICLE_REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PARTICLE_CANDIDATE="$1"
PARTICLE_GATE="$2"
PARTICLE_OUTPUT="$3"
cd "$PARTICLE_REPO"
exec env -u LD_PRELOAD -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 \
  ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 PYTHONHASHSEED=0 \
  /tmp/pr38-default-env/bin/python -u \
  "reports/toy100/particle-geometry/candidates/$PARTICLE_CANDIDATE/probe.py" \
  --repo reports/toy100/particle-geometry/prepared/repos/cuda \
  --config configs/toy100/constraints_simple_regularization.json \
  --task "$PARTICLE_GATE" --backend cuda \
  --initial-state "reports/toy100/cpu-recipe-gpu-port/initialization-fixtures/$PARTICLE_GATE/initial-values.pt" \
  --output "$PARTICLE_OUTPUT"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ -f ../supervisor.md ]]; then
  cat ../supervisor.md
  if rg -q '^STOP|^# STOP' ../supervisor.md; then
    exit 2
  fi
fi
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
log="reports/data-drift-api/runs/$1.log"
shift
ln -sfn "$(basename "$log")" reports/data-drift-api/runs/current.log
exec /tmp/pr38-default-env/bin/python -u "$@" > "$log" 2>&1

#!/usr/bin/env bash
set -euo pipefail
cd /ml2/hypergan/gan-attempts/selected-h-verification/repo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
/tmp/pr38-default-env/bin/python -u reports/toy100/critic_signal_screen.py \
  --declaration /ml2/hypergan/gan-attempts/selected-h-verification/one.json \
  --output /ml2/hypergan/gan-attempts/selected-h-verification/cold-replay \
  --ledger /ml2/hypergan/gan-attempts/selected-h-verification/tests.jsonl \
  --workers 1

#!/usr/bin/env bash
set -euo pipefail
cd /ml2/hypergan/gan-attempts/selected-h-verification/repo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
/tmp/pr38-default-env/bin/python -u reports/toy100/selected_h_remaining.py \
  --declaration /ml2/hypergan/gan-attempts/selected-h-verification/one.json \
  --output /ml2/hypergan/gan-attempts/selected-h-verification/remaining-replay \
  --ledger /ml2/hypergan/gan-attempts/selected-h-verification/tests.jsonl \
  --tasks unipolar two_pole cover_leftover mid_scale_identity vector_anisotropic vector_two_broad vector_spiral unused_token_hold ae_gan_hold \
  --workers 1

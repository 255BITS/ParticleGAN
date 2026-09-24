#!/usr/bin/env bash
set -euo pipefail
# Exact selected H, faithful pure-GAN host plumbing, fresh output required.
# Usage: bash run-one-host.sh TASK /absolute/new/output
TASK=${1:?Specify one frozen task}
OUTPUT=${2:?Specify an unused absolute output directory}
cd /ml2/hypergan/gan-attempts/selected-h-verification/repo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
/tmp/pr38-default-env/bin/python -u reports/toy100/selected_h_remaining.py \
  --declaration /ml2/hypergan/gan-attempts/selected-h-verification/one.json \
  --output "$OUTPUT" --ledger "${OUTPUT}.jsonl" --tasks "$TASK" --workers 1

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
GPU="${GPU:-0}"
RUN_ROOT=runs/cifar_particle_ae/lazy_long
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || { echo 'Long AE-GAN pipeline already running'; exit 1; }
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
trap 'status=$?; echo "LONG PIPELINE EXIT status=$status" >> "$RUN_ROOT/PIPELINE.log"' EXIT
"$PY" -u experiments/follow_grid.py --root "$RUN_ROOT" --log "$RUN_ROOT/PIPELINE.log" -- \
  --configs configs/cifar_particle_ae/lazy_long/n08.yaml \
  --gpus "$GPU" --workers_per_gpu 1 --python "$PY" \
  --trainer experiments/train_cifar_particle_ae.py
# Same-count endpoint audit for comparison with the FID5k learning curve.
if [[ ! -f "$RUN_ROOT/n08/audit5k/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u experiments/audit_cifar_particle_ae.py \
    "$RUN_ROOT/n08" >> "$RUN_ROOT/PIPELINE.log" 2>&1
fi

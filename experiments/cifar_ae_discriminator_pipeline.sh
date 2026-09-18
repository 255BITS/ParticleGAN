#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRACK="${1:-discriminator_diagnosis}"
GPUS="${2:-0,1}"
case "$TRACK" in discriminator_diagnosis|discriminator_smoke) ;; *) exit 2 ;; esac
RUN_ROOT="runs/cifar_particle_ae/$TRACK"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || exit 1
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv/bin/python -u experiments/follow_grid.py --root "$RUN_ROOT" --log "$RUN_ROOT/PIPELINE.log" -- \
  --config_manifest "configs/cifar_particle_ae/$TRACK/manifest.json" --gpus "$GPUS" --workers_per_gpu 1 \
  --python .venv/bin/python --trainer experiments/diagnose_cifar_ae_discriminator.py

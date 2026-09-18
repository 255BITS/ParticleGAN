#!/usr/bin/env bash
# Run from anywhere; follow_grid merges tagged training progress into PIPELINE.log.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
GPUS=1
[[ "$GPUS" != *,* ]] || { echo "Use one GPU for matched timing"; exit 1; }
RUN_ROOT=runs/cifar_particle_ae/bundle_scout
REPORT=reports/cifar-particle-ae/bundle-scout
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || { echo 'Bundle FID scout pipeline already running'; exit 1; }
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
status=0
"$PY" -u experiments/follow_grid.py --root "$RUN_ROOT" --log "$RUN_ROOT/PIPELINE.log" -- \
  --config_manifest configs/cifar_particle_ae/bundle_scout/manifest.json \
  --gpus "$GPUS" --workers_per_gpu 1 --python "$PY" \
  --trainer experiments/train_cifar_ae_bundle.py || status=$?
"$PY" -u experiments/analyze_cifar_ae_bundle.py \
  --config_manifest configs/cifar_particle_ae/bundle_scout/manifest.json \
  --report "$REPORT" >> "$RUN_ROOT/PIPELINE.log" 2>&1 || status=$?
echo "PIPELINE COMPLETE status=$status report=$REPORT/LEADERBOARD.md" | tee -a "$RUN_ROOT/PIPELINE.log"
exit "$status"

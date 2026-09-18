#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TRACK="${1:-growth_scout}"
GPUS="${2:-0,1}"
case "$TRACK" in growth_scout|growth_smoke) ;; *) exit 2 ;; esac
RUN_ROOT="runs/cifar_particle_ae/$TRACK"
REPORT="reports/cifar-particle-ae/$TRACK"
MANIFEST="configs/cifar_particle_ae/$TRACK/manifest.json"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || { echo "$TRACK pipeline already running"; exit 1; }
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
status=0
.venv/bin/python -u experiments/follow_grid.py --root "$RUN_ROOT" --log "$RUN_ROOT/PIPELINE.log" -- \
  --config_manifest "$MANIFEST" --gpus "$GPUS" --workers_per_gpu 1 --python .venv/bin/python \
  --trainer experiments/train_cifar_ae_growth.py || status=$?
if [[ "$TRACK" == growth_scout ]]; then
  .venv/bin/python -u experiments/analyze_cifar_ae_growth.py --config_manifest "$MANIFEST" \
    --report "$REPORT" >> "$RUN_ROOT/PIPELINE.log" 2>&1 || status=$?
fi
echo "PIPELINE COMPLETE status=$status report=$REPORT/LEADERBOARD.md" | tee -a "$RUN_ROOT/PIPELINE.log"
exit "$status"

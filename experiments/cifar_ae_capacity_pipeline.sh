#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
TRACK="${1:?Specify duration_100k or capacity_scout}"
case "$TRACK" in
  duration_100k) GPU=0 ;;
  capacity_scout) GPU=1 ;;
  *) echo "Unknown track: $TRACK"; exit 2 ;;
esac
RUN_ROOT="runs/cifar_particle_ae/$TRACK"
REPORT="reports/cifar-particle-ae/$TRACK"
MANIFEST="configs/cifar_particle_ae/$TRACK/manifest.json"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/pipeline.lock"
flock -n 9 || { echo "$TRACK pipeline already running"; exit 1; }
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
status=0
"$PY" -u experiments/follow_grid.py --root "$RUN_ROOT" --log "$RUN_ROOT/PIPELINE.log" -- \
  --config_manifest "$MANIFEST" --gpus "$GPU" --workers_per_gpu 1 --python "$PY" \
  --trainer experiments/train_cifar_ae_capacity.py || status=$?
"$PY" -u experiments/analyze_cifar_ae_capacity.py --config_manifest "$MANIFEST" \
  --report "$REPORT" >> "$RUN_ROOT/PIPELINE.log" 2>&1 || status=$?
echo "PIPELINE COMPLETE status=$status report=$REPORT/LEADERBOARD.md" | tee -a "$RUN_ROOT/PIPELINE.log"
exit "$status"

#!/usr/bin/env bash
# misgan_pipeline.sh -- run the MisGAN toy grid, then the leaderboards.
#
# Writes grid configs -> runs them on the chosen GPUs (run_grid.py) -> analysis.
# Grid progress, each run's last eval line and the tables all go to ONE file:
#
#     tail -f results/misgan/PIPELINE.log      # the whole study
#     tail -f runs/misgan/<mechanism>__<arm>.log   # one run, one line per eval
#
# Env: GPUS (default 0,1), WORKERS per GPU (default 6), PY (default .venv/bin/python),
#      STEPS (default 7000), FORCE=1 to re-run finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-0,1}"
WORKERS="${WORKERS:-6}"
PY="${PY:-.venv/bin/python}"
STEPS="${STEPS:-7000}"
LOG="results/misgan/PIPELINE.log"
mkdir -p results/misgan runs/misgan
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "=============================================================="
log "misgan toy pipeline start | gpus=$GPUS workers=$WORKERS steps=$STEPS"
log "branch $(git branch --show-current) @ $(git rev-parse --short HEAD)"
"$PY" experiments/misgan_toy.py --write_grid configs/misgan --total_steps "$STEPS" 2>&1 | tee -a "$LOG"
FORCE_FLAGS=()
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAGS=(--force); fi
"$PY" experiments/run_grid.py --config_manifest configs/misgan/manifest.json --gpus "$GPUS" \
    --workers_per_gpu "$WORKERS" --python "$PY" --trainer experiments/misgan_toy.py --echo_last_line \
    ${FORCE_FLAGS[@]+"${FORCE_FLAGS[@]}"} 2>&1 | tee -a "$LOG" || log "grid finished with failures"
"$PY" experiments/analyze_misgan.py 2>&1 | tee -a "$LOG"
log "pipeline finished"

#!/usr/bin/env bash
# sparse_pipeline.sh -- run the sparse conditional mixed-output study end to end.
#
# For each stage: generate configs -> run the grid on the chosen GPU(s) ->
# aggregate into a table. Everything (grid progress, each run's [done] line,
# the stage tables) is appended to ONE file so the study can be followed with
#
#     tail -f results/sparse/PIPELINE.log
#
# Usage:
#     experiments/sparse_pipeline.sh                      # all stages, defaults
#     experiments/sparse_pipeline.sh recipe sparse        # just these stages
#     GPUS=1 WORKERS=6 BASE="ucd_lambda=0.1" experiments/sparse_pipeline.sh sparse
#
# Env: GPUS (default 1), WORKERS per GPU (default 6), PY (default .venv/bin/python),
#      BASE (comma key=value overrides applied LAST to every config of the stage),
#      SEEDS (default 1,2,3), STEPS (default 5000), FORCE=1 to re-run finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS="${GPUS:-1}"
WORKERS="${WORKERS:-6}"
PY="${PY:-.venv/bin/python}"
BASE="${BASE:-}"
SEEDS="${SEEDS:-1,2,3}"
STEPS="${STEPS:-5000}"
LOG="results/sparse/PIPELINE.log"
mkdir -p results/sparse
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
  STAGES=(recipe ucd sparse discrete champion fewshot)
fi

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "=============================================================="
log "sparse-ucd pipeline start | stages: ${STAGES[*]} | gpus=$GPUS workers=$WORKERS steps=$STEPS seeds=$SEEDS base='${BASE}'"
log "branch $(git branch --show-current) @ $(git rev-parse --short HEAD)"
for stage in "${STAGES[@]}"; do
  log "---------------- stage: $stage ----------------"
  GEN_ARGS=(--stage "$stage" --base "$BASE" --seeds "$SEEDS" --total_steps "$STEPS")
  RUN_STAGE=$("$PY" experiments/gen_sparse_configs.py "${GEN_ARGS[@]}" --print_stage)
  "$PY" experiments/gen_sparse_configs.py "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  FORCE_FLAGS=()
  if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAGS=(--force); fi
  "$PY" experiments/run_grid.py --config_manifest "configs/sparse/$RUN_STAGE/manifest.json" --gpus "$GPUS" --workers_per_gpu "$WORKERS" \
      --python "$PY" --trainer experiments/train_sparse.py --echo_last_line "${FORCE_FLAGS[@]}" 2>&1 | tee -a "$LOG"
  "$PY" experiments/analyze_sparse.py --stage "$RUN_STAGE" 2>&1 | tee -a "$LOG"
done
log "pipeline finished"

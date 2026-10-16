#!/usr/bin/env bash
# One rerun lane. Claims the next unclaimed spec (atomic mkdir), takes a GPU lease (one job per GPU),
# and runs the unchanged attempt worker under the task's own declared timeout.
# Usage: lane.sh STATE_ROOT WORKTREE BUDGET_SECONDS [BUDGET_ROOTS (space separated; default STATE_ROOT)]
# LEASE may override the lease helper (fastlease.sh: same locks, 2 s polling).
set -u
ROOT=$1; WT=$2; BUDGET=$3; BUDGET_ROOTS=${4:-$1}
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python
LEASE=${LEASE:-/tmp/claude-1000/-ml2-hypergan/baedb70c-2c24-4e91-92d1-2f2bed2d913c/scratchpad/dig/gpu-lease.sh}
HERE=$WT/reports/forge/dig2-20261016/wallclock-reruns
cd "$WT" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for spec in "$ROOT"/specs/*.json; do
  name=$(basename "$spec" .json)
  dir=$ROOT/attempts/$name
  mkdir "$dir" 2>/dev/null || continue
  # Budget: declared timeouts of every claimed cell must fit the GPU-seconds budget.
  claimed=$($PY $HERE/budget.py $BUDGET_ROOTS)
  if [ "$claimed" -gt "$BUDGET" ]; then echo "$(date -Is) skip $name: claimed $claimed > budget $BUDGET" >> "$ROOT/lanes.log"; rmdir "$dir"; continue; fi
  echo "$(date -Is) claim $name" >> "$ROOT/lanes.log"
  "$LEASE" bash -c '
    set -u
    dir='"$dir"'; spec='"$spec"'; PY='"$PY"'; HERE='"$HERE"'; WT='"$WT"'
    $PY -c "import json,sys;s=json.load(open(sys.argv[1]));s[\"device\"]=\"cuda:$LEASED_GPU\";json.dump(s,open(sys.argv[2],\"w\"),indent=1,sort_keys=True)" "$spec" "$dir/request.json"
    echo "{\"leased_gpu\": $LEASED_GPU, \"start\": $(date +%s.%N)}" > "$dir/lease.json"
    ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr "\n" ";") $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | tr "\n" ";")"; sleep 15; done ) > "$dir/gpu-samples.log" 2>&1 &
    sampler=$!
    $PY -u $WT/experiments/forge/attempt_worker.py supervise "$dir/request.json" 2>&1 | $PY -u $HERE/stamp.py > "$dir/run.log"
    kill $sampler
    echo "{\"leased_gpu\": $LEASED_GPU, \"end\": $(date +%s.%N)}" >> "$dir/lease.json"
  '
  echo "$(date -Is) done $name $(cat "$dir/result.json" 2>/dev/null | head -c 300 | tr '\n' ' ')" >> "$ROOT/lanes.log"
done

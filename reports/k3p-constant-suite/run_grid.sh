#!/usr/bin/env bash
# Grid-search queue: run arms one after another into runs_grid/, one log per arm in logs_grid/<arm>.log.
# usage: run_grid.sh <device> <jobs> <tasks|all> arm1 arm2 ...
set -u
cd "$(dirname "$0")"
dev=$1 jobs=$2 tasks=$3; shift 3
targs=(); [ "$tasks" != all ] && targs=(--tasks "$tasks")
for arm in "$@"; do
  /home/martyn/dev/ParticleGAN/.venv/bin/python run_suite.py --arm "$arm" --device "$dev" --jobs "$jobs" \
    --runs-dir runs_grid --logs-dir logs_grid "${targs[@]}"
done

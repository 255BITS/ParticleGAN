#!/usr/bin/env bash
# Usage: reports/toy100-init-regression/run.sh arm:problem [arm:problem ...]   (max 3 concurrent, cuda:1 only)
# Env: RUNS_DIR (default runs/), LOG_PREFIX. Tail: tail -f reports/toy100-init-regression/logs/<prefix><arm>-<problem>.log
set -u
cd "$(dirname "$0")/../.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
D=reports/toy100-init-regression
mkdir -p $D/logs
for job in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  arm=${job%%:*}; prob=${job##*:}
  .venv/bin/python -u $D/run_arm.py --arm $arm --problem $prob --runs-dir ${RUNS_DIR:-$D/runs} > $D/logs/${LOG_PREFIX:-}$arm-$prob.log 2>&1 &
done
wait

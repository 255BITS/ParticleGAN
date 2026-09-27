#!/usr/bin/env bash
# 100-Gaussian transfer of the simple critic. One process per arm, all on cuda:0.
# Tail with: tail -f reports/simple-critic/toy100/logs/<arm>.log
set -u
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
D=reports/simple-critic/toy100
for arm in "$@"; do
  .venv/bin/python -u $D/run_arm.py --arm $arm > $D/logs/$arm.log 2>&1 &
done
wait

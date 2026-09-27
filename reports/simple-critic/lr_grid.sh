#!/usr/bin/env bash
# Constant-LR grid on sec_nodamp (wgan + R1(1) + secant path(10, t .5) + cap-all(10), Dβ2 .9, A2 0).
# critic LR × mc, G and prior LR × mg (prior stays 2×G); 1/1 cell = sec_nodamp. 2 runs per GPU.
# Tail with: tail -f reports/simple-critic/logs/lr_c*.log ; outcomes: grep -h COMPLETE reports/simple-critic/logs/lr_*.log
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic
BASE="--loss wgan --real r1 --lam-real 1 --path secant --lam-path 10 --path-target 0.5 --path-u 0.1,0.9 --cap all --lam-cap 10 --d-beta2 0.9 --latent-damping 0"
run() { local mc=$1 mg=$2 dev=$3; local arm=lr_c${mc}_g${mg}
  $PY $D/lr_grid.py --lr-c $mc --lr-g $mg --arm $arm --device $dev $BASE > /dev/null 2> $D/logs/$arm.err; }
( run 0.5 0.5 cuda:0; run 0.5 1 cuda:0; run 0.5 2 cuda:0 ) &
( run 1 0.5 cuda:0;   run 1 2 cuda:0;   run 0.25 0.25 cuda:0 ) &
( run 2 0.5 cuda:1;   run 2 1 cuda:1 ) &
( run 2 2 cuda:1;     run 4 4 cuda:1 ) &
wait

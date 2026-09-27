#!/usr/bin/env bash
# Round 3: single changes on top of secant_r1_b2 (wgan + R1(1) + secant path(10, t .5) + cap-all(10), critic beta2 .9).
# Tail with: tail -f reports/simple-critic/logs/<arm>.log ; outcomes: grep -h COMPLETE reports/simple-critic/logs/sec_*.log
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic
BASE="--loss wgan --real r1 --lam-real 1 --path secant --lam-path 10 --path-target 0.5 --path-u 0.1,0.9 --cap all --lam-cap 10 --d-beta2 0.9"
run() { local arm=$1 dev=$2; shift 2
  $PY $D/worker.py --arm $arm --device $dev $BASE "$@" > /dev/null 2> $D/logs/$arm.err; }
( run sec_t1 cuda:0 --path-target 1.0; run sec_drift cuda:0 --real drift+r1 --lam-drift 1e-3 ) &
( run sec_center cuda:1 --lam-center 1; run sec_nodamp cuda:1 --latent-damping 0 ) &
( run sec_lazy4 cuda:1 --lazy-k 4 ) &
wait

# Round 3b: combos of the single changes that helped (fails-outside-transit and arrival both improved):
#   nodamp (A2 off), center(1), t1 (secant target 1.0). drift (fails 116) and lazy_k 4 (never passes) excluded.
( run combo_a cuda:0 --latent-damping 0 --lam-center 1 ) &
( run combo_b cuda:1 --latent-damping 0 --lam-center 1 --path-target 1.0 ) &
wait

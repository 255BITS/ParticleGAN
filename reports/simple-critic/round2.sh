#!/usr/bin/env bash
# Round 2: R1 at real + a path term that does not act at the real endpoint + cap-all.
# Tail with: tail -f reports/simple-critic/logs/<arm>.log
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic
BASE="--loss wgan --real r1 --lam-real 1 --path lower --lam-path 10 --path-target 0.3 --path-u 0.1,0.9 --cap all --lam-cap 10"
run() { local arm=$1 dev=$2; shift 2
  $PY $D/worker.py --arm $arm --device $dev $BASE "$@" > /dev/null 2> $D/logs/$arm.err; }
case "${1:-a}" in
a)
( run int_r1 cuda:0; run secant_r1 cuda:0 --path secant --path-target 0.5 ) &
( run int_r1_hinge cuda:0 --loss hinge ) &
( run int_r1_rp cuda:1 --loss rplogistic ) &
( run int_r1_b2 cuda:1 --d-beta2 0.9 ) &
wait ;;
b)  # chosen after batch a: the spike fix (critic beta2 .9) on the best pre-shift sharpener (secant path)
run secant_r1_b2 cuda:0 --path secant --path-target 0.5 --d-beta2 0.9 ;;
esac

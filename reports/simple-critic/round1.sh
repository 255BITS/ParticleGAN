#!/usr/bin/env bash
# Round 1: 8 arms, 2 lanes per GPU. Tail with: tail -f reports/simple-critic/logs/<arm>.log
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic
BASE="--loss wgan --real drift --lam-real 0.1 --path lower --lam-path 10 --cap all --lam-cap 10"
run() { local arm=$1 dev=$2; shift 2
  $PY $D/worker.py --arm $arm --device $dev $BASE "$@" > /dev/null 2> $D/logs/$arm.err; }
ref() { $PY reports/ka2-default-candidate/constant-lr-api/worker.py --output $D/runs/ka2_stock_ref/raw \
  --device $1 > $D/logs/ka2_stock_ref.log 2> $D/logs/ka2_stock_ref.err; }
( run full cuda:0; run no_path cuda:0 --path none ) &
( run full_r1 cuda:0 --real r1 --lam-real 1; run no_cap cuda:0 --cap none ) &
( run no_real cuda:1 --real none; run full_hinge cuda:1 --loss hinge ) &
( run wgangp_ref cuda:1 --path two_sided --cap none; ref cuda:1 ) &
wait

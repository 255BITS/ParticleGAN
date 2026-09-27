#!/usr/bin/env bash
# Round 5 ring arms (round5_worker.py wraps lr_grid.py + worker.py read-only; curvature diagnostics on).
# Base B_cap3: wgan + R1(1) + secant(10, t .5) + cap-all(10, c=3), Dβ2 .9, A2 0, critic LR ×.5.
# Tail: tail -f reports/simple-critic/logs/{c3_*,rp_*}.log
# Outcomes: grep -h COMPLETE reports/simple-critic/logs/{c3_*,rp_*}.log
# Usage: round5.sh              (stage 1: six arms, 2 per GPU concurrently)
#        round5.sh one ARM DEV "FLAGS"   (a single arm, e.g. a combo)
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic; W="$D/round5_worker.py --lr-c 0.5 --lr-g 1"
OPT="--d-beta2 0.9 --latent-damping 0"
SEC="--real r1 --lam-real 1 --path secant --lam-path 10 --path-target 0.5 --path-u 0.1,0.9"
CAP="--lam-cap 10 --cap-target 3"
B="--loss wgan $OPT $SEC --cap all $CAP"
mkdir -p $D/logs
run() { local arm=$1 dev=$2; shift 2; $PY $W --arm $arm --device $dev "$@" > /dev/null 2> $D/logs/$arm.err; }
if [ "${1:-}" = one ]; then run "$2" "$3" $4; exit; fi
( run c3_r1w cuda:0 $B --lam-real 0.1;              run c3_rate cuda:0 $B --lam-rate 10 ) &
( run c3_capinterp cuda:0 --loss wgan $OPT $SEC --cap interp $CAP ) &
( run c3_capnopath cuda:1 --loss wgan $OPT $SEC --cap ends $CAP;
  run rp_center_min cuda:1 --loss rplogistic $OPT --cap all $CAP --lam-pair-center 1 ) &
( run rp_center cuda:1 --loss rplogistic $OPT $SEC --cap all $CAP --lam-pair-center 1 ) &
wait

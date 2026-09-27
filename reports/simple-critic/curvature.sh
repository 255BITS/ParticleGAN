#!/usr/bin/env bash
# Curvature arms (curvature_worker.py wraps lr_grid.py + worker.py read-only). kappa = 1/(2*0.07) = 7.14.
# B = lr_c0.5_g1: wgan + R1(1) + secant(10, t .5) + cap-all(10, c=1), Dβ2 .9, A2 0, critic LR ×.5.
# Tail: tail -f reports/simple-critic/logs/{B_*,wgan_huber,wgan_margin}.log reports/simple-critic/diag/logs/*.log
# Outcomes: grep -h COMPLETE reports/simple-critic/logs/{B_*,wgan_*}.log
# Usage: curvature.sh            (phase 1: arms 1-5 + observation-only reruns into diag/)
#        curvature.sh combo ARM "EXTRA FLAGS"   (phase 2: one combo on top of B)
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
PY=.venv/bin/python; D=reports/simple-critic; W="$D/curvature_worker.py --lr-c 0.5 --lr-g 1"
OPT="--loss wgan --d-beta2 0.9 --latent-damping 0"
B="$OPT --real r1 --lam-real 1 --path secant --lam-path 10 --path-target 0.5 --path-u 0.1,0.9 --cap all --lam-cap 10"
mkdir -p $D/logs $D/diag
run() { local arm=$1 dev=$2; shift 2; $PY $W --arm $arm --device $dev "$@" > /dev/null 2> $D/logs/$arm.err; }
if [ "${1:-}" = combo ]; then run "$2" cuda:0 $B $3; exit; fi
( run B_margin cuda:0 $B --lam-margin 10; run B_cap3 cuda:0 $B --cap-target 3 ) &
( run B_cap2 cuda:0 $B --cap-target 2;    run B_nnpair cuda:0 $B --nnpair ) &
( run wgan_huber cuda:1 $OPT --lam-huber 10; run wgan_margin cuda:1 $OPT --lam-margin 10 ) &
( $PY $W --arm lr_c0.5_g1 --device cuda:1 $B --output $D/diag/runs/lr_c0.5_g1 --log $D/diag/logs/lr_c0.5_g1.log \
    > /dev/null 2> $D/diag/lr_c0.5_g1.err
  $PY $D/curvature_worker.py --k3p-constant --out-root $D/diag --device cuda:1 > /dev/null 2> $D/diag/k3p_constant.err ) &
wait

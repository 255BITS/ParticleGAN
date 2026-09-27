#!/usr/bin/env bash
# Rerun EVERY ring arm of runs_oldinit/ (pre-#194 random init) under the package's default
# initialization (batch_feature_zero, merged from origin/develop c720645e), each with its recorded
# worker + flags. Seed 0, no seed variants. Skipped: the archived "ref:ka2-constant" row (summarize.py
# reads it from reports/ka2-default-candidate) and the observation-only diag/ reruns.
#
# Queue: cuda:0, at most MAX_JOBS (3) concurrent. An arm with runs/<arm>/result.json is skipped, so
# the script can be restarted. Every run writes an init receipt (runs/<arm>/declaration.json and
# result.json "init_receipt"; line 3 of the log).
#   Tail one arm:     tail -f reports/simple-critic/logs/<arm>.log
#   Queue progress:   tail -f reports/simple-critic/logs/_queue.log
#   Outcomes:         grep -h COMPLETE reports/simple-critic/logs/*.log
#   Leaderboard:      .venv/bin/python reports/simple-critic/summarize.py   (old: --runs-dir runs_oldinit)
# Usage: rerun_all.sh [ARM ...]        (no args = all arms below)
#        DEVICE=cuda:0 MAX_JOBS=3 rerun_all.sh
set -u
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD
unset K3P_INIT
PY=.venv/bin/python; D=reports/simple-critic
DEVICE=${DEVICE:-cuda:0}; MAX_JOBS=${MAX_JOBS:-3}
K3P_ROOT_DIR=${K3P_ROOT_DIR:-/home/martyn/dev/ParticleGAN/.claude/worktrees/k3p-develop}
KA2_WORKER=reports/ka2-default-candidate/constant-lr-api/worker.py
mkdir -p $D/runs $D/logs
Q=$D/logs/_queue.log

# arm  kind  flags   (kind -> worker: worker=worker.py, lr_grid=lr_grid.py, curvature=curvature_worker.py,
#                     round5=round5_worker.py, secant_k3p=secant_k3p_worker.py, k3p=k3p_worker.py, ka2=KA2 worker)
ARMS=$(cat <<'TABLE'
full             worker     --real drift --lam-real 0.1 --path lower --cap all
no_path          worker     --real drift --lam-real 0.1 --cap all
full_r1          worker     --real r1 --path lower --cap all
no_cap           worker     --real drift --lam-real 0.1 --path lower
no_real          worker     --lam-real 0.1 --path lower --cap all
full_hinge       worker     --loss hinge --real drift --lam-real 0.1 --path lower --cap all
wgangp_ref       worker     --real drift --lam-real 0.1 --path two_sided
int_r1           worker     --real r1 --path lower --path-u 0.1,0.9 --path-target 0.3 --cap all
secant_r1        worker     --real r1 --path secant --path-u 0.1,0.9 --path-target 0.5 --cap all
int_r1_hinge     worker     --loss hinge --real r1 --path lower --path-u 0.1,0.9 --path-target 0.3 --cap all
int_r1_rp        worker     --loss rplogistic --real r1 --path lower --path-u 0.1,0.9 --path-target 0.3 --cap all
int_r1_b2        worker     --real r1 --path lower --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.3 --cap all
secant_r1_b2     worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all
sec_t1           worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --cap all
sec_drift        worker     --real drift+r1 --lam-drift 0.001 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all
sec_center       worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --lam-center 1
sec_nodamp       worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
sec_lazy4        worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --lazy-k 4
combo_a          worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0 --lam-center 1
combo_b          worker     --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --cap all --latent-damping 0 --lam-center 1
lr_c0.25_g0.25   lr_grid    --lr-c 0.25 --lr-g 0.25 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c0.5_g0.5     lr_grid    --lr-c 0.5 --lr-g 0.5 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c0.5_g1       lr_grid    --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c0.5_g2       lr_grid    --lr-c 0.5 --lr-g 2 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c1_g0.5       lr_grid    --lr-c 1 --lr-g 0.5 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c1_g2         lr_grid    --lr-c 1 --lr-g 2 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c2_g0.5       lr_grid    --lr-c 2 --lr-g 0.5 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c2_g1         lr_grid    --lr-c 2 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c2_g2         lr_grid    --lr-c 2 --lr-g 2 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
lr_c4_g4         lr_grid    --lr-c 4 --lr-g 4 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0
B_cap2           curvature  --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --cap-target 2 --latent-damping 0
B_cap3           curvature  --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --cap-target 3 --latent-damping 0
B_margin         curvature  --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0 --lam-margin 10
B_nnpair         curvature  --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --latent-damping 0 --nnpair
wgan_huber       curvature  --lr-c 0.5 --lr-g 1 --d-beta2 0.9 --latent-damping 0 --lam-huber 10
wgan_margin      curvature  --lr-c 0.5 --lr-g 1 --d-beta2 0.9 --latent-damping 0 --lam-margin 10
c3_capinterp     round5     --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap interp --cap-target 3 --latent-damping 0
c3_capnopath     round5     --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap ends --cap-target 3 --latent-damping 0
c3_r1w           round5     --lr-c 0.5 --lr-g 1 --real r1 --lam-real 0.1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --cap-target 3 --latent-damping 0
c3_rate          round5     --lr-c 0.5 --lr-g 1 --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --cap-target 3 --latent-damping 0 --lam-rate 10
rp_center        round5     --lr-c 0.5 --lr-g 1 --loss rplogistic --real r1 --path secant --path-u 0.1,0.9 --d-beta2 0.9 --path-target 0.5 --cap all --cap-target 3 --latent-damping 0 --lam-pair-center 1
rp_center_min    round5     --lr-c 0.5 --lr-g 1 --loss rplogistic --d-beta2 0.9 --cap all --cap-target 3 --latent-damping 0 --lam-pair-center 1
sec_anchor       secant_k3p --anchor
sec_guard        secant_k3p --guard
sec_guard_anchor secant_k3p --guard --anchor
k3p_constant     k3p        --arm constant
k3p_stock_ref    k3p        --arm stock
ka2_stock_ref    ka2
TABLE
)

one() {  # arm kind flags...
  local arm=$1 kind=$2; shift 2
  local rc
  case $kind in
    worker)     $PY $D/worker.py --arm $arm --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    lr_grid)    $PY $D/lr_grid.py --arm $arm --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    curvature)  $PY $D/curvature_worker.py --arm $arm --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    round5)     $PY $D/round5_worker.py --arm $arm --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    secant_k3p) $PY $D/secant_k3p_worker.py --arm $arm --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    k3p)        $PY $D/k3p_worker.py --k3p-root $K3P_ROOT_DIR --device $DEVICE "$@" > /dev/null 2> $D/logs/$arm.err ;;
    ka2)        rm -rf $D/runs/$arm/raw   # the KA2 worker refuses an existing output dir
                $PY $KA2_WORKER --output $D/runs/$arm/raw --device $DEVICE "$@" > $D/logs/$arm.log 2> $D/logs/$arm.err \
                  && $PY $D/ka2_ref_result.py $D/runs/$arm ;;
  esac
  rc=$?
  echo "$(date +%F' '%T) done  $arm rc=$rc" >> $Q
}

want=" ${*:-} "
while read -r arm kind flags; do
  [ -z "$arm" ] && continue
  [ "$want" != "  " ] && [[ "$want" != *" $arm "* ]] && continue
  if [ -f $D/runs/$arm/result.json ]; then echo "$(date +%F' '%T) skip  $arm (result exists)" >> $Q; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do wait -n; done
  echo "$(date +%F' '%T) start $arm ($kind $flags)" >> $Q
  # shellcheck disable=SC2086
  one $arm $kind $flags &
done <<< "$ARMS"
wait
echo "$(date +%F' '%T) queue finished" >> $Q

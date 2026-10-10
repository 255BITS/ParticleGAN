#!/bin/bash
# usage: launch.sh <pkg-dir-name> <overrides.json> <label> <task> <gpu> [native_steps]   (packages and overrides live in this directory)
P=$1; O=$2; L=$3; T=$4; G=$5; N=$6
S=/ml2/hypergan/gan-attempts/noout-20260928
OPT='{"eval_output_noise": true, "save_final_state": true}'
[ -n "$N" ] && OPT="{\"eval_output_noise\": true, \"save_final_state\": true, \"native_steps\": $N}"
export CUDA_VISIBLE_DEVICES=$G CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /tmp/pr38-default-env/bin/python /ml2/hypergan/lrfree-20260926/harness/screen.py \
  --package-root $S/$P --overrides $S/$O \
  --task $T --output $S/runs/$L-$T --device cuda:0 \
  --candidate-options "$OPT" --cand $L \
  > $S/logs/$L-$T.log 2>&1

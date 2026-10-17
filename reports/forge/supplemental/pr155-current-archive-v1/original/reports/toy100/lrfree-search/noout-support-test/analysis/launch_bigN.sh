#!/bin/bash
# usage: launch_bigN.sh <pkg-dir-name> <overrides.json> <label> <task> <gpu>   -- same as launch.sh but runs harness-bigN/screen.py (a COPY of the frozen harness whose task specs were edited: vector_unequal_mass with 20,000 particles, batch 2048, 3,000 steps)
P=$1; O=$2; L=$3; T=$4; G=$5
S=/ml2/hypergan/gan-attempts/noout-20260928
export CUDA_VISIBLE_DEVICES=$G CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec /tmp/pr38-default-env/bin/python $S/harness-bigN/screen.py \
  --package-root $S/$P --overrides $S/$O \
  --task $T --output $S/runs/$L-$T --device cuda:0 \
  --candidate-options '{"eval_output_noise": true, "save_final_state": true}' --cand $L \
  > $S/logs/$L-$T.log 2>&1

#!/usr/bin/env bash
# Full 26-task suite for the screen top-3 (screen cells reused: run_nr.py skips tasks whose result.json exists)
# plus a reg_coeff 1.0 check (<arm>_c1) of the same arms on the 7 screen tasks.
# Follow: tail -f logs/*.log   (one line per task)   |   tail -f logs/<arm>/<task>.log (one line per eval)
set -u
cd "$(dirname "$0")"
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
JOBS0=${JOBS0:-5}; JOBS1=${JOBS1:-7}
TOP=nr_dvalcap_anchor,nr_none_anchor,nr_pathcap_anchor
C1=nr_dvalcap_anchor_c1,nr_none_anchor_c1,nr_pathcap_anchor_c1
mkdir -p logs
$PY run_nr.py --arms "$TOP" --tasks all --device cuda:1 --jobs "$JOBS1" > logs/full_cuda1.out 2>&1 &
P1=$!
$PY run_nr.py --arms "$C1" --tasks screen --device cuda:0 --jobs "$JOBS0" > logs/c1_cuda0.out 2>&1 &
P0=$!
wait $P1; S1=$?; wait $P0; S0=$?
echo "full done: full(cuda1) exit=$S1 c1(cuda0) exit=$S0" | tee logs/full_done.txt

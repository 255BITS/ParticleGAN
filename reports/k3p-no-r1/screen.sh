#!/usr/bin/env bash
# 20-arm no-R1 screen on the 7 screen tasks: one run_nr.py launcher per GPU, disjoint arm sets
# (each arm takes a lock). CPU toys (two_pole) run on CPU inside the same pools.
# Follow: tail -f logs/*.log   (one line per task)   |   tail -f logs/<arm>/<task>.log (one line per eval)
set -u
cd "$(dirname "$0")"
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
JOBS0=${JOBS0:-6}; JOBS1=${JOBS1:-6}
A0=nr_none_none,nr_none_hinge,nr_none_oadam,nr_none_anchor,nr_symcap_none,nr_symcap_hinge,nr_symcap_oadam,nr_symcap_anchor,nr_pathcap_none,nr_pathcap_hinge
A1=nr_pathcap_oadam,nr_pathcap_anchor,nr_pairsec_none,nr_pairsec_hinge,nr_pairsec_oadam,nr_pairsec_anchor,nr_dvalcap_none,nr_dvalcap_hinge,nr_dvalcap_oadam,nr_dvalcap_anchor
mkdir -p logs
$PY run_nr.py --arms "$A0" --tasks screen --device cuda:0 --jobs "$JOBS0" > logs/screen_cuda0.out 2>&1 &
P0=$!
$PY run_nr.py --arms "$A1" --tasks screen --device cuda:1 --jobs "$JOBS1" > logs/screen_cuda1.out 2>&1 &
P1=$!
wait $P0; S0=$?; wait $P1; S1=$?
echo "screen done: cuda0 exit=$S0 cuda1 exit=$S1" | tee logs/screen_done.txt

#!/bin/bash
# EMA-centred R1 follow-up (EMA_R1.md): 2 arms x 7 screen tasks.  tail -f logs/e1_*.log
cd "$(dirname "$0")"
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
$PY run_nr.py --arms e1_b0.5_d0.9_w1 --tasks screen --device cuda:0 --jobs 4 &
$PY run_nr.py --arms e1_b0.2_d0.999_w1 --tasks screen --device cuda:0 --jobs 4 &
wait

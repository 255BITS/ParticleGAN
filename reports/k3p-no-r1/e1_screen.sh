#!/bin/bash
# EMA-centred R1 screen (EMA_R1.md): 9 arms x 7 screen tasks, one launcher per GPU.
#   tail -f logs/e1_*.log            # one line per task
#   tail -f logs/e1_<arm>/<task>.log # one line per eval
cd "$(dirname "$0")"
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
$PY run_nr.py --arms e1_b0_d0.999_w1,e1_b0.5_d0.999_w1,e1_b0.8_d0.999_w1,e1_b0.95_d0.999_w1,e1_b1_d0.999_w1 \
  --tasks screen --device cuda:0 --jobs 4 &
$PY run_nr.py --arms e1_b1_d0.99_w1,e1_b1_d0.9_w1,e1_b1_d0.999_w3,e1_b1_d0.999_w10 \
  --tasks screen --device cuda:1 --jobs 4 &
wait

#!/bin/bash
# cuda:1 queue: arms in order, resume-safe (run_suite skips tasks with result.json). tail -f logs/queue_cuda1.log
cd "$(dirname "$0")"
for arm in k3p_const_ams_c3 k3p_stock_ams k3p_nonoise k3p_const; do
  echo "$(date +%T) start $arm" >> logs/queue_cuda1.log
  /home/martyn/dev/ParticleGAN/.venv/bin/python -u run_suite.py --arm $arm --device cuda:1 --jobs 3 >> logs/queue_cuda1.log 2>&1
  echo "$(date +%T) end $arm rc=$?" >> logs/queue_cuda1.log
done
echo "$(date +%T) queue done" >> logs/queue_cuda1.log

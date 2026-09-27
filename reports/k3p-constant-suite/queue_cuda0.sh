#!/bin/bash
# cuda:0 queue: arms in order, resume-safe (run_suite skips tasks with result.json).
# tail -f logs/queue_cuda0.log ; tail -f logs/<arm>.log
cd /home/martyn/dev/ParticleGAN/.claude/worktrees/k3p-constant-fix
JOBS=${JOBS:-3}
for arm in k3p_const_ams k3p_stock k3p_nonoise_ams simple_B_cap3 k3p_stock_oldinit; do
  echo "$(date +%T) start $arm jobs=$JOBS" >> reports/k3p-constant-suite/logs/queue_cuda0.log
  .venv/bin/python reports/k3p-constant-suite/run_suite.py --arm $arm --device cuda:0 --jobs $JOBS
  echo "$(date +%T) end $arm rc=$?" >> reports/k3p-constant-suite/logs/queue_cuda0.log
done
echo "$(date +%T) queue finished" >> reports/k3p-constant-suite/logs/queue_cuda0.log

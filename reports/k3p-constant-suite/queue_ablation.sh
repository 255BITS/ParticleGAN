#!/bin/bash
# Ablation queue (resume-safe: run_suite skips tasks with result.json). tail -f logs/queue_ablation.log ; tail -f logs/<arm>.log
cd "$(dirname "$0")"
PY=/home/martyn/dev/ParticleGAN/.venv/bin/python
Q=logs/queue_ablation.log
run() { echo "$(date +%T) start $1 $2" >> $Q; $PY -u run_suite.py --arm $1 --device $2 --jobs 3 >> $Q 2>&1; echo "$(date +%T) end $1 rc=$?" >> $Q; }
for spec in "$@"; do run ${spec%@*} ${spec#*@} & done
wait
echo "$(date +%T) queue done: $*" >> $Q

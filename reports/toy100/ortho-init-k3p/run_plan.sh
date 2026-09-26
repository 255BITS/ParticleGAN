#!/usr/bin/env bash
# run_plan.sh PLAN JOBS WALL [VARIANT]  -- single instance, idempotent (runner skips finished tasks)
W=/tmp/k3p-ortho-20260925; cd "$W" || exit 1
exec 9>"$W/run.lock"; flock -n 9 || { echo LOCKED; exit 0; }
export CUDA_VISIBLE_DEVICES=GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69 K3P_ORTHO_VARIANT=${4:-qr}
echo "START $(date +%T) plan=$1 variant=${4:-qr}" >> "$W/run.log"
/tmp/k3p-audit-20260925/venv/bin/python -u repo/reports/toy100/ortho-init-k3p/ortho_run.py "$1" "$2" "$3" >> "$W/run.log" 2>&1
echo "EXIT $? $(date +%T) plan=$1" >> "$W/run.log"

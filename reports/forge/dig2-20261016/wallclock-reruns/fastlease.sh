#!/usr/bin/env bash
# fastlease.sh: copy of the workflow gpu-lease.sh (same lock files and log) polling every 2 s instead of 20 s.
# Usage: gpu-lease.sh <command ...>   — waits for a free GPU slot (one job per GPU for this workflow),
# exports LEASED_GPU=<0|1>, then runs the command. Use "$LEASED_GPU" inside the command via: bash -c '... --gpus $LEASED_GPU ...'
D=/tmp/claude-1000/-ml2-hypergan/baedb70c-2c24-4e91-92d1-2f2bed2d913c/scratchpad/dig/locks
while true; do
  for g in 0 1; do
    exec {fd}>"$D/gpu$g.lock"
    if flock -n "$fd"; then
      export LEASED_GPU=$g CUDA_DEVICE_ORDER=PCI_BUS_ID
      echo "[gpu-lease] gpu $g acquired $(date -Is) :: $*" >> "$D/lease.log"
      "$@"; rc=$?
      echo "[gpu-lease] gpu $g released rc=$rc $(date -Is)" >> "$D/lease.log"
      exit $rc
    fi
    exec {fd}>&-
  done
  sleep 2
done

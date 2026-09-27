#!/usr/bin/env bash
# Simple-critic transfer to the sparse-UCD champion: one seed, same 5000-step budget,
# constant LRs, no instance noise. One line per eval in results/sparse-secant/<arm>.log.
#   DEVICE=cuda:1 experiments/sparse_secant.sh [arm ...]
#   tail -f results/sparse-secant/*.log
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda:1}
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(champ_matched sec_nodamp sec_rpbase sec_lazy4)
mkdir -p results/sparse-secant
for a in "${ARMS[@]}"; do
  PYTHONPATH=$PWD PYTHONUNBUFFERED=1 .venv/bin/python experiments/train_sparse.py \
    --config configs/sparse-secant/$a.yaml --device "$DEVICE" > results/sparse-secant/$a.log 2>&1 &
done
wait
.venv/bin/python experiments/sparse_secant_board.py > results/sparse-secant/LEADERBOARD.md
echo "[sparse_secant] done"

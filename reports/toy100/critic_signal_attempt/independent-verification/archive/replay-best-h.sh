#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. This is a replay command, not an extra result.
BENCH_PYTHON=/tmp/pr38-default-env/bin/python
EVIDENCE=reports/toy100/critic_signal_attempt
REPLAY_DIR=${1:-reports/toy100/critic_signal_attempt/replay-best-h}
mkdir "$REPLAY_DIR"
cp -a benchmarks particlegan lib configs pyproject.toml "$REPLAY_DIR/"
tar -xzf "$EVIDENCE/batch-h/source.tar.gz" -C "$REPLAY_DIR"
cp "$EVIDENCE/h-own-hold/critic_signal_continue.py" "$REPLAY_DIR/reports/toy100/"
"$BENCH_PYTHON" - "$EVIDENCE" "$REPLAY_DIR" <<'PY'
import json, sys
from pathlib import Path
evidence, replay = map(Path, sys.argv[1:])
manifest = json.loads((evidence/'batch-h/manifest.json').read_text())
row = next(r for r in manifest['rows'] if r['tag']=='h_n05r06_mixup_c0p01_lr15')
(replay/'one.json').write_text(json.dumps([row], indent=2)+'\n')
PY
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
"$BENCH_PYTHON" -u "$REPLAY_DIR/reports/toy100/critic_signal_screen.py" \
  --declaration "$REPLAY_DIR/one.json" --output "$REPLAY_DIR/cold" \
  --ledger "$REPLAY_DIR/tests.jsonl" --workers 1
"$BENCH_PYTHON" -u "$REPLAY_DIR/reports/toy100/critic_signal_continue.py" \
  --candidate "$REPLAY_DIR/cold/h_n05r06_mixup_c0p01_lr15" \
  --output "$REPLAY_DIR/own-hold" --ledger "$REPLAY_DIR/tests.jsonl"

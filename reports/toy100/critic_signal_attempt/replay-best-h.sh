#!/usr/bin/env bash
set -euo pipefail

evidence=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_python=${BENCH_PYTHON:-/tmp/pr38-default-env/bin/python}
if [[ ! -x "$bench_python" ]]; then
    bench_python=$(command -v "$bench_python")
fi
replay_dir=${1:-"$evidence/replay-best-h"}
mkdir -- "$replay_dir"
replay_dir=$(cd -- "$replay_dir" && pwd)

"$bench_python" - "$evidence" "$replay_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

evidence, replay = map(Path, sys.argv[1:])
inventory = json.loads((evidence / 'publication-manifest.json').read_text())['files']
for relative, expected in inventory.items():
    if hashlib.sha256((evidence / relative).read_bytes()).hexdigest() != expected['sha256']:
        raise RuntimeError(f'Published evidence changed: {relative}')
for name in ('base-code.tar.gz', 'batch-h/source.tar.gz'):
    with tarfile.open(evidence / name) as archive:
        for member in archive.getmembers():
            target = replay / member.name
            if not target.resolve().is_relative_to(replay) or not (member.isfile() or member.isdir()):
                raise RuntimeError(f'Unexpected archive entry: {member.name}')
        archive.extractall(replay, filter='data')
shutil.copy2(evidence / 'h-own-hold/critic_signal_continue.py', replay / 'reports/toy100/critic_signal_continue.py')
manifest = json.loads((evidence / 'batch-h/manifest.json').read_text())
row = next(row for row in manifest['rows'] if row['tag'] == 'h_n05r06_mixup_c0p01_lr15')
(replay / 'one.json').write_text(json.dumps([row], indent=2) + '\n')
PY

unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=''
export PYTHONUNBUFFERED=1
"$bench_python" -u "$replay_dir/reports/toy100/critic_signal_screen.py" \
    --declaration "$replay_dir/one.json" --output "$replay_dir/cold" \
    --ledger "$replay_dir/tests.jsonl" --workers 1
"$bench_python" -u "$replay_dir/reports/toy100/critic_signal_continue.py" \
    --candidate "$replay_dir/cold/h_n05r06_mixup_c0p01_lr15" \
    --output "$replay_dir/own-hold" --ledger "$replay_dir/tests.jsonl"
printf '\nReplay evidence: %s\n' "$replay_dir"

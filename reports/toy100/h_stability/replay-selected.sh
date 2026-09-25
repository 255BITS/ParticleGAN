#!/usr/bin/env bash
# Reconstruct the measured research code without changing the installed package.
set -euo pipefail
evidence=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$evidence/../../.." && pwd)
bench_python=${BENCH_PYTHON:-python3}
bench_python=$(command -v "$bench_python")
replay=${1:?Usage: replay-selected.sh NEW_DIRECTORY [short|cold|long|converged|prepare]}
mode=${2:-short}
case "$mode" in short|cold|long|converged|prepare) ;; *) exit 2 ;; esac
mkdir -- "$replay"
replay=$(cd -- "$replay" && pwd)

"$bench_python" - "$repo" "$replay" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

repo, replay = map(Path, sys.argv[1:])
h = repo / 'reports/toy100/critic_signal_attempt'
inventory = json.loads((h / 'publication-manifest.json').read_text())['files']
for name in ('base-code.tar.gz', 'batch-h/source.tar.gz'):
    archive_path = h / name
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == inventory[name]['sha256'], name
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if (not (replay / member.name).resolve().is_relative_to(replay)
                    or not (member.isfile() or member.isdir())):
                raise RuntimeError(f'Unexpected archive entry: {member.name}')
        archive.extractall(replay, filter='data')
for name in ('critic_signal_attempt', 'h_stability'):
    shutil.copytree(repo / 'reports/toy100' / name, replay / 'reports/toy100' / name,
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('continuous_candidates.py', 'continuous_screen.py', 'critic_signal.py',
             'critic_signal_screen.py', 'selected_h_extension.py', 'selected_h_remaining.py'):
    shutil.copy2(repo / 'reports/toy100' / name, replay / 'reports/toy100' / name)
(replay / 'tests').mkdir(exist_ok=True)
for name in ('test_adam_response.py', 'test_continuous_candidates.py',
             'test_legacy_noise_adapters.py', 'test_legacy_noise_remaining.py', 'test_convergence_gate.py'):
    shutil.copy2(repo / 'tests' / name, replay / 'tests' / name)
manifest = json.loads((h / 'batch-h/manifest.json').read_text())['source_sha256']
for name, expected in manifest.items():
    assert hashlib.sha256((replay / name).read_bytes()).hexdigest() == expected, name
print(f'Prepared isolated replay; all {len(manifest)} frozen training sources match.', flush=True)
PY

unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=''
export PYTHONUNBUFFERED=1
cd -- "$replay"
case "$mode" in
  converged)
    "$bench_python" -u reports/toy100/h_stability/converged_probe.py --output "$replay/results-post-convergence"
    ;;
  short|long)
    steps=200
    if [[ "$mode" == long ]]; then steps=1200; fi
    "$bench_python" -u reports/toy100/h_stability/selected_base_probe.py \
      --output "$replay/results-own-state" --steps "$steps"
    ;;
  cold)
    "$bench_python" -u reports/toy100/h_stability/adam_response_cold.py \
      --declaration reports/toy100/h_stability/eps-net-base/declaration.json \
      --output "$replay/results-cold" --ledger "$replay/tests.jsonl" \
      --workers 1 --tasks mode_hold two_pole
    ;;
esac
printf '\nReplay directory: %s\nInspect saved verdicts; command completion does not imply toy qualification.\n' "$replay"

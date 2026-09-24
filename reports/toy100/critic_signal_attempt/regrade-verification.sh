#!/usr/bin/env bash
# Regrade copied artifacts with the exact independent verifier, without training.
set -euo pipefail
evidence=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bench_python=${BENCH_PYTHON:-/tmp/pr38-default-env/bin/python}
output=${1:?Provide a new output directory}
"$bench_python" - "$evidence" "$output" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

evidence = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
inventory = json.loads((evidence / 'publication-manifest.json').read_text())['files']
for name, entry in inventory.items():
    assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == entry['sha256'], name
shutil.copytree(evidence / 'independent-verification', output)
with tarfile.open(evidence / 'verification-code.tar.gz') as archive:
    archive.extractall(output, filter='data')
PY
output=$(cd -- "$output" && pwd)
unset PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 CUDA_VISIBLE_DEVICES=''
"$bench_python" "$output/repo/reports/toy100/critic_signal_regrade.py" \
    --candidate "$output/cold-replay/h_n05r06_mixup_c0p01_lr15" \
    --output "$output/regrade-cold.json"
for audit in audit-cold-evidence.py audit-remaining-evidence.py audit-controls.py; do
    "$bench_python" -u "$output/$audit"
done
printf '\nRegraded evidence: %s\n' "$output"

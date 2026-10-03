#!/usr/bin/env bash
set -euo pipefail
cd /ml2/hypergan/gan-attempts/selected-h-verification
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ATEN_CPU_CAPABILITY=avx2
/tmp/pr38-default-env/bin/python repo/reports/toy100/critic_signal_regrade.py --candidate cold-replay/h_n05r06_mixup_c0p01_lr15 --output regrade-cold.json
/tmp/pr38-default-env/bin/python -u audit-cold-evidence.py
/tmp/pr38-default-env/bin/python -u audit-remaining-evidence.py
/tmp/pr38-default-env/bin/python -u audit-controls.py

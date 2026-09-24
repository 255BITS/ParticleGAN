#!/usr/bin/env bash
# Canonical toy100 continuous-learning environment. Source it; do not exec it.
#
#   source scripts/toy100_env.sh
#   python -u reports/toy100/gan_followup_probe.py --phase stay --method holdw15 --output DIR
#
# Chooses MKL_CBWR=AVX2,STRICT when a fresh interpreter observes that mode in
# mkl_serv_cbwr_get(-1). Otherwise sets COMPATIBLE. The selector ignores
# LD_PRELOAD so the choice is the native CPU, not a vendor shim. Scoring still
# refuses later if the effective mode in the scoring process does not match.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source scripts/toy100_env.sh; executing it does not export into the caller" >&2
  exit 2
fi

_toy100_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${_toy100_root}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export TOY100_SEED=0
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2
export DNNL_MAX_CPU_ISA=AVX2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export CUDA_VISIBLE_DEVICES=

# Probe AVX2,STRICT before any scoring process initializes MKL.
export MKL_CBWR=AVX2,STRICT
_toy100_cbwr_out=$(
  cd "${_toy100_root}" && env -u LD_PRELOAD python3 -m benchmarks.toy100.canonical_env --select-cbwr
) || return 1
_toy100_cbwr=${_toy100_cbwr_out%%$'\n'*}
_toy100_cbwr_reason=${_toy100_cbwr_out#*$'\n'}
case "${_toy100_cbwr}" in
  AVX2,STRICT|COMPATIBLE) ;;
  *)
    echo "REFUSING TO SCORE: CBWR selector returned '${_toy100_cbwr}'" >&2
    return 2
    ;;
esac
export TOY100_CBWR_REASON="${_toy100_cbwr_reason}"
export MKL_CBWR="${_toy100_cbwr}"
export TOY100_CBWR_MODE="${_toy100_cbwr}"
export TOY100_CANONICAL_ENV=1
echo "toy100 canonical env: MKL_CBWR=${MKL_CBWR} (${TOY100_CBWR_REASON})" >&2
unset _toy100_root _toy100_cbwr

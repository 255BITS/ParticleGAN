#!/bin/bash
# Codex audit commands, both MKL vendor paths, seeds 0 1 2.
# Tail: tail -f reports/toy100/continuous-evidence/dispatch-robustness/driver.log
set -u
set -m
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-/tmp/venv-cpu214/bin/python}"
EVIDENCE="$ROOT/reports/toy100/continuous-evidence/dispatch-robustness"
DRIVER="$EVIDENCE/driver.log"
mkdir -p "$EVIDENCE"
MAX_JOBS="${MAX_JOBS:-3}"

repo_for() {
  case "$1" in
    reachstall|delayg05) echo /tmp/src-pr140 ;;
    holdw15) echo /tmp/src-pr143 ;;
    *) echo "unknown method $1" >&2; return 1 ;;
  esac
}
preload_for() {
  case "$1" in
    intel) echo /tmp/mkl-intel.so ;;
    amd) echo /tmp/mkl-amd.so ;;
    *) echo "unknown path $1" >&2; return 1 ;;
  esac
}

run_one() {
  local path="$1" method="$2" seed="$3" phase="$4"
  local repo preload out log
  repo="$(repo_for "$method")"
  preload="$(preload_for "$path")"
  out="$EVIDENCE/$path/$method/seed$seed/$phase"
  log="$EVIDENCE/$path/$method/seed$seed/${phase}.log"
  mkdir -p "$(dirname "$log")"
  if [[ -f "$log" ]] && grep -q '"event": "DONE"' "$log"; then
    echo "SKIP $path $method seed=$seed $phase" | tee -a "$DRIVER"
    return 0
  fi
  if [[ -e "$out" ]]; then
    echo "RETRY removing partial $path $method seed=$seed $phase" | tee -a "$DRIVER"
    rm -rf "$out"
  fi
  echo "START $(date -Is) $path $method seed=$seed $phase" | tee -a "$DRIVER"
  env -u PYTHONPATH -u ONEDNN_MAX_CPU_ISA -u DNNL_MAX_CPU_ISA \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 CUDA_VISIBLE_DEVICES= \
    LD_PRELOAD="$preload" AUDIT_SEED="$seed" AUDIT_REPO="$repo" \
    PYTHONUNBUFFERED=1 \
    "$PY" -u "$ROOT/reports/toy100/dispatch_seed_run.py" \
      --phase "$phase" --method "$method" --output "$out" \
      >"$log" 2>&1
  local code=$?
  env -u PYTHONPATH -u ONEDNN_MAX_CPU_ISA -u DNNL_MAX_CPU_ISA \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 CUDA_VISIBLE_DEVICES= \
    LD_PRELOAD="$preload" AUDIT_SEED="$seed" AUDIT_REPO="$repo" \
    "$PY" -u "$ROOT/reports/toy100/dispatch_receipt.py" "$out/dispatch.json" \
    >"$EVIDENCE/$path/$method/seed$seed/${phase}.dispatch.json" 2>/dev/null || true
  echo "END $(date -Is) $path $method seed=$seed $phase exit=$code" | tee -a "$DRIVER"
  return 0
}

echo "DRIVER $(date -Is) evidence=$EVIDENCE" | tee -a "$DRIVER"
echo "tail -f $DRIVER" | tee -a "$DRIVER"

# Acquire first. Stay is a second invocation of this script with PHASES=stay,
# and only for gates the summarizer has marked.
PHASES="${PHASES:-cold warm}"
PATHS="${PATHS:-intel amd}"
METHODS="${METHODS:-reachstall delayg05 holdw15}"
SEEDS="${SEEDS:-0 1 2}"
for phase in $PHASES; do
  for path in $PATHS; do
    for method in $METHODS; do
      for seed in $SEEDS; do
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
          wait -n || true
        done
        run_one "$path" "$method" "$seed" "$phase" &
      done
    done
  done
done
wait || true
echo "DRIVER_DONE $(date -Is) phases=$PHASES" | tee -a "$DRIVER"

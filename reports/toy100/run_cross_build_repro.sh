#!/bin/bash
# Cross-build toy100 gates. One JSON line per event. Tail the driver:
#   tail -f reports/toy100/continuous-evidence/cross-build-repro/driver.log
# A single phase log:
#   tail -f reports/toy100/continuous-evidence/cross-build-repro/<build>/<method>/<phase>.log
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EVIDENCE="$ROOT/reports/toy100/continuous-evidence/cross-build-repro"
mkdir -p "$EVIDENCE"
DRIVER="$EVIDENCE/driver.log"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
export PYTHONUNBUFFERED=1
MAX_JOBS="${MAX_JOBS:-3}"

py_for() {
  case "$1" in
    cu130) echo /tmp/venv-cu130/bin/python ;;
    cpu214) echo /tmp/venv-cpu214/bin/python ;;
    cpu213) echo /tmp/venv-cpu213/bin/python ;;
    *) echo "unknown build $1" >&2; return 1 ;;
  esac
}

run_one() {
  local build="$1" method="$2" phase="$3"
  local py
  py="$(py_for "$build")"
  local out="$EVIDENCE/$build/$method/$phase"
  local log="$EVIDENCE/$build/$method/${phase}.log"
  mkdir -p "$(dirname "$log")"
  if [[ -f "$log" ]] && grep -q '"event": "DONE"' "$log"; then
    echo "SKIP $build $method $phase already done" | tee -a "$DRIVER"
    return 0
  fi
  if [[ -e "$out" ]]; then
    echo "RETRY $(date -Is) removing partial $build $method $phase" | tee -a "$DRIVER"
    rm -rf "$out"
  fi
  echo "START $(date -Is) $build $method $phase" | tee -a "$DRIVER"
  "$py" -u -m reports.toy100.gan_followup_probe \
    --phase "$phase" --method "$method" --output "$out" \
    --trace "$out/state.jsonl" >"$log" 2>&1
  local code=$?
  echo "END $(date -Is) $build $method $phase exit=$code" | tee -a "$DRIVER"
  return 0
}

echo "DRIVER $(date -Is) jobs=$MAX_JOBS evidence=$EVIDENCE" | tee -a "$DRIVER"
echo "tail -f $DRIVER" | tee -a "$DRIVER"

# Cold first (acquisition), then warm, then stay. Up to MAX_JOBS at once.
phases=(cold warm stay)
methods=(reachstall delayg05 delayed_g125)
builds=(cu130 cpu214 cpu213)
for phase in "${phases[@]}"; do
  for method in "${methods[@]}"; do
    for build in "${builds[@]}"; do
      while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n || true
      done
      run_one "$build" "$method" "$phase" &
    done
  done
done
wait || true
echo "DRIVER_DONE $(date -Is)" | tee -a "$DRIVER"

#!/usr/bin/env bash
# Re-baseline the three pinned CL gates under the canonical env. One thread.
# Each phase logs JSON lines; tail -f the .log file.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"
OUT=${1:-${ROOT}/reports/toy100/canonical-harness/runs}
mkdir -p "${OUT}"
export TOY100_REPO=${TOY100_REPO:-/tmp/cl143}
run() {
  local name=$1 method=$2 phase=$3
  local dest="${OUT}/${name}/${phase}"
  rm -rf "${dest}"
  mkdir -p "${dest}"
  echo "BEGIN ${name} ${phase} $(date -Is)" | tee -a "${OUT}/driver.log"
  # The probe creates dest itself and refuses a pre-existing directory.
  rm -rf "${dest}"
  TOY100_REPO="${TOY100_REPO}" "${ROOT}/scripts/toy100_cl_gate.sh" \
    --phase "${phase}" --method "${method}" --output "${dest}" \
    > >(tee "${OUT}/${name}-${phase}.log") 2>&1
  echo "END ${name} ${phase} $(date -Is)" | tee -a "${OUT}/driver.log"
}
run 107-reachstall reachstall warm
run 107-reachstall reachstall cold
run 107-reachstall reachstall stay
run 140-delayg05 delayg05 warm
run 140-delayg05 delayg05 cold
run 140-delayg05 delayg05 stay
run 143-holdw15 holdw15 warm
run 143-holdw15 holdw15 cold
run 143-holdw15 holdw15 stay
echo "ALL GATES DONE $(date -Is)" | tee -a "${OUT}/driver.log"

#!/usr/bin/env bash
# One method, full canonical CL gates, seed 0.
# Logs are one JSON object per line: tail -f "$OUT/<method>.log"
#
#   PATH=/tmp/toy100-venv/bin:$PATH scripts/toy100_rebaseline_gates.sh baseline
#
# Stay (1210-2400) runs after a passing cold ring. After a failing ring it
# still runs when that ring took at most TOY100_STAY_CHEAP_SECONDS (default 180).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "${ROOT}"
METHOD=${1:?method}
OUT=${2:-${ROOT}/reports/toy100/canonical-rebaseline/runs}
CHEAP=${TOY100_STAY_CHEAP_SECONDS:-180}
mkdir -p "${OUT}"
LOG="${OUT}/${METHOD}.log"
export PATH="${TOY100_PYTHON_DIR:-/tmp/toy100-venv/bin}:${PATH}"

run_phase() {
  local phase=$1 dest_name=$2
  shift 2
  local dest="${OUT}/${METHOD}/${dest_name}"
  rm -rf "${dest}"
  echo "BEGIN ${METHOD} ${dest_name} $(date -Is)" | tee -a "${OUT}/driver.log" >> "${LOG}"
  set +e
  "${ROOT}/scripts/toy100_cl_gate.sh" \
    --phase "${phase}" --method "${METHOD}" --output "${dest}" "$@" \
    > >(tee -a "${LOG}") 2>&1
  local status=$?
  set -e
  echo "END ${METHOD} ${dest_name} status=${status} $(date -Is)" | tee -a "${OUT}/driver.log" >> "${LOG}"
  return 0
}

run_phase warm warm
run_phase cold cold-traj --tasks trajectory
run_phase cold cold-ring --tasks mode_hold

passed=$(python3 - "${OUT}/${METHOD}/cold-ring/mode_hold.json" <<'PY'
import json, sys
try:
    row = json.load(open(sys.argv[1]))
except FileNotFoundError:
    print("missing")
    raise SystemExit(0)
print("yes" if (row.get("verdict") or {}).get("passed") else "no")
PY
)
seconds=$(python3 - "${LOG}" <<'PY'
import json, sys
seconds = ""
for line in open(sys.argv[1]):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("event") == "COLD" and row.get("task") == "mode_hold":
        seconds = row.get("seconds")
print("" if seconds is None else seconds)
PY
)
run_stay=0
if [[ "${passed}" == "yes" ]]; then
  run_stay=1
elif [[ -n "${seconds}" ]]; then
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" \
    "${seconds}" "${CHEAP}" && run_stay=1 || true
fi
if [[ "${run_stay}" == 1 ]]; then
  run_phase stay stay --steps 2400
else
  echo "SKIP ${METHOD} stay ring_passed=${passed} ring_seconds=${seconds:-na} cheap<=${CHEAP}" \
    | tee -a "${OUT}/driver.log" >> "${LOG}"
fi
echo "METHOD DONE ${METHOD} $(date -Is)" | tee -a "${OUT}/driver.log" >> "${LOG}"

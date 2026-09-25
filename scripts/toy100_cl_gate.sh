#!/usr/bin/env bash
# Score one continuous-learning gate under the canonical env.
# Refuses (exit 2) unless scripts/toy100_env.sh stuck. Logs are line-buffered.
#
#   TOY100_REPO=/path/to/pin scripts/toy100_cl_gate.sh \
#     --phase stay --method holdw15 --output reports/toy100/canonical-harness/runs/holdw15/stay
#
# TOY100_REPO defaults to this checkout. Pins that carry the training factories
# need the guard and the patched probe copied in; this script does that copy
# and does not change factory code.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1091
source "${ROOT}/scripts/toy100_env.sh"
REPO=${TOY100_REPO:-$ROOT}
if [[ "${REPO}" != "${ROOT}" ]]; then
  cp "${ROOT}/benchmarks/toy100/canonical_env.py" "${REPO}/benchmarks/toy100/canonical_env.py"
  cp "${ROOT}/reports/toy100/gan_followup_probe.py" "${REPO}/reports/toy100/gan_followup_probe.py"
fi
cd "${REPO}"
exec python3 -u "${REPO}/reports/toy100/gan_followup_probe.py" "$@"

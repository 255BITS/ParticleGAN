#!/usr/bin/env bash
# One declared study; planning is the default. No legacy engine/permission bypass.
set -euo pipefail
runner_script=$(realpath -- "${BASH_SOURCE[0]}")
runner_root=${PARTICLEGAN_RUNNER_ROOT:-$(cd -- "$(dirname -- "$runner_script")/../.." && pwd)}
runner_python=${GAN_PYTHON:-python3}
cd -- "$runner_root"
exec "$runner_python" -m experiments.forge.codex_runner --root "$runner_root" study "$@"

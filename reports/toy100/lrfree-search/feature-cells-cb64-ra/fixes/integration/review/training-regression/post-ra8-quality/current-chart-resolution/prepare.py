"""Read-only source/input seal before the fixed64/128 chart computations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

AREA = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HOST = Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    files = [AREA/name for name in ('PROTOCOL.md','probe.py','prepare.py')]
    files += [ROOT/name for name in (
        'quality/ra8/READY.json','quality/ra8/COMPOSITION.json','validation-cb64-ra8/source-freeze.json',
        'configs/overrides-CB64-RA8.json','validation-cb64-ra8/learned/SOURCE-FREEZE.json',
        'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-2000.pt',
        'validation-cb64-ra8/learned/training/toy/CB64-RA8/config.json',
        'validation-cb64-ra8/screens/runs/grid100/final-state.pt',
        'validation-cb64-ra8/screens/runs/grid100/result.json',
        'validation-cb64-ra8/screens/runs/grid100/job-header.json',
        'validation-cb64-ra8/screens/runs/grid100/execution-receipt.json',
        'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py',
        'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py',
        'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py',
        'performance/training-regression/count-review/post-ra8-quality/grid-covariance/PREPARATION-FROZEN.json')]
    files += [HOST/'problems.py', HOST/'toy_models.py']
    package = ROOT/'pkg-CB64-RA8/particlegan'
    files += sorted(package.rglob('*.py'))
    ready = json.loads((ROOT/'quality/ra8/READY.json').read_text())
    assert {str(path.relative_to(package)):sha(path) for path in sorted(package.rglob('*.py'))} == ready['package_source_sha256']
    record = dict(status='FROZEN_BEFORE_CHART_COMPUTATION',frozen_utc=datetime.now(timezone.utc).isoformat(),
                  fixed_cells=[64,128],fixed_rank=8,fixed_Q=.05,family='3K+2 commonQ',
                  checkpoint_selection='finaltoy2000 and finalgrid7000 only',
                  source_and_input_sha256={str(path):sha(path) for path in files})
    with (AREA/'SOURCE-FROZEN.json').open('x') as handle:
        handle.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':record['status'],'frozen_files':len(record['source_and_input_sha256']),
                      'frozen_utc':record['frozen_utc'],'source_freeze_sha256':sha(AREA/'SOURCE-FROZEN.json')}))


if __name__ == '__main__':
    main()

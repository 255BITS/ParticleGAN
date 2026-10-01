"""Seal this small contract and all read-only inputs before numerical reads."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

AREA = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE = ROOT / 'validation-cb64-ra8/learned'
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as handle:
        handle.write(json.dumps(value, indent=2) + '\n')


def main():
    package_root = ROOT / 'pkg-CB64-RA8'
    package = package_root / 'particlegan'
    source_map = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}
    h = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        h.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    ready = json.loads((ROOT / 'quality/ra8/READY.json').read_text())
    assert h.hexdigest() == ready['package_sha256']
    assert source_map == ready['package_source_sha256']
    inputs = dict(package_root=str(package_root), package_sha256=h.hexdigest(),
                  package_source_sha256=source_map, numerical_source_sha256=ready['numerical_source_sha256'],
                  checkpoint=str(LANE / 'training/toy/CB64-RA8/checkpoint-2000.pt'),
                  fixed_scorer_seed=314259, fixed_scorer_chunk=256, fixed_chunks_per_branch=2,
                  independent_initial_restores=2, intermediate_reloads=1, training_updates=0,
                  required_actual_coherent_rows=977, required_rows=973)
    write(AREA / 'INPUTS.json', inputs)
    paths = [AREA / name for name in ('PROTOCOL.md', 'sample_reload.py', 'prepare_contract.py', 'INPUTS.json')]
    paths += [ROOT / name for name in (
        'quality/ra8/READY.json', 'quality/ra8/COMPOSITION.json',
        'configs/overrides-CB64-RA8.json', 'validation-cb64-ra8/source-freeze.json',
        'quality/run_positive_lease.py', 'quality/positive-lease-phase/COMPOSITION.json',
        'quality/nested_slot.py', 'gpu_slot.py',
        'integration/review/training-regression/post-ra8-quality/saved-training-parity/receipt.json',
        'integration/review/training-regression/post-ra8-quality/saved-training-parity/FROZEN.json',
        'performance/training-regression/count-review/ra8-prospective/FINAL-RECEIPT.json',
        'performance/training-regression/count-review/ra8-prospective/FINAL-FROZEN.json')]
    paths += [LANE / name for name in (
        'common.py', 'replay.py', 'run_training.py', 'PROTOCOL.md', 'INPUTS.json',
        'SOURCE-FREEZE.json', 'preparation-receipt.json',
        'training/toy/CB64-RA8/config.json', 'training/toy/CB64-RA8/checkpoint-2000.pt')]
    paths += [PREV / 'models_metrics.py']
    paths += list(sorted(package.rglob('*.py')))
    freeze = dict(status='FROZEN_BEFORE_CONTRACT_NUMERICAL_READ',
                  frozen_utc=datetime.now(timezone.utc).isoformat(),
                  file_sha256={str(path): sha(path) for path in paths},
                  package_root=str(package_root), package_sha256=h.hexdigest(),
                  scope='positive-lease sample/reload helper and actual final checkpoint; CUDA NOT RUN')
    write(AREA / 'SOURCE-FROZEN.json', freeze)
    print(json.dumps(dict(status=freeze['status'], frozen_utc=freeze['frozen_utc'],
                         frozen_files=len(freeze['file_sha256']),
                         source_freeze_sha256=sha(AREA / 'SOURCE-FROZEN.json')), sort_keys=True))


if __name__ == '__main__':
    main()

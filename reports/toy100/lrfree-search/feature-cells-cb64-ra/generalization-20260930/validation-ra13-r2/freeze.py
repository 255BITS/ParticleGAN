"""Seal the complete RA13 candidate and unchanged host/scorer inputs once."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
PACKAGE = STUDY / 'pkg-RA13-settled'
CONFIG = STUDY / 'configs/RA13-settled.json'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')
REFERENCE = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/pkg-CB64-RA11/particlegan')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_digest():
    h = hashlib.sha256()
    for path in sorted((PACKAGE / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(PACKAGE / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def verify():
    frozen = json.loads((ROOT / 'SOURCE-FREEZE.json').read_text())
    for path, expected in frozen['hashes'].items():
        assert sha(path) == expected, f'frozen input changed: {path}'
    assert package_digest() == frozen['package_sha256']
    return dict(status='VALID', files=len(frozen['hashes']), package_sha256=frozen['package_sha256'],
                config_sha256=sha(CONFIG), source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'))


def main():
    output = ROOT / 'SOURCE-FREEZE.json'
    assert not output.exists(), 'seal each source revision in a new lane'
    files = set((PACKAGE / 'particlegan').rglob('*.py'))
    files.update(ROOT.glob('*.py'))
    files.update(HARNESS.glob('*.py'))
    for folder in ('hosts', 'tasks'):
        for extension in ('*.py', '*.json', '*.gz'):
            files.update((HARNESS / folder).rglob(extension))
    files.update([CONFIG, ROOT / 'SCREEN-ADAPTER.json', STUDY / 'ROOT-REVIEW.json',
                  STUDY / 'RA12-SOURCE-CLOSURE.json', STUDY / 'RA13-SOURCE-REVIEW.json', STUDY / 'RA13-SOURCE-CLOSURE.json',
                  Path('/ml2/hypergan/gan-attempts/noout-20260928/gif/rotate_gate.py'),
                  REFERENCE / 'initialization.py', REFERENCE / 'qr_bz_pq_init.py',
                  Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/screens/collect.py'),
                  Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929/validation-cb64-ra11/screens/lane.py')])
    receipt = dict(status='FROZEN_NOT_TRAINED', candidate='RA13-settled', package=str(PACKAGE),
        package_sha256=package_digest(), config=str(CONFIG), config_sha256=sha(CONFIG),
        hashes={str(path):sha(path) for path in sorted(files)},
        baseline_pr155_head='f459cb6d6aaaabeb1af076ec53ad7a963618de90',
        scorer_changed=False, schedule_changed=False, thresholds_changed=False,
        calibration='generator/noise x .25 only when auto selects feature_cells',
        selection='finite-resolution feasibility and complete raw-output moment frame',
        reopen_guard='settled network excursion witness plus actual KA2 loss-epoch rebase')
    output.write_text(json.dumps(receipt,sort_keys=True,indent=2) + '\n')
    print(json.dumps(verify(),sort_keys=True))


if __name__ == '__main__':
    main()

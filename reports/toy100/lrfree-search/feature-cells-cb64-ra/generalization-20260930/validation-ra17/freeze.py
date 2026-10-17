"""Immutable RA17 latest suite routing; original quality keeps RA14/RA15 labels."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
PACKAGE = STUDY / 'pkg-RA17-current-pr155'
CONFIG = STUDY / 'configs/RA17-current-pr155.json'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def package_digest():
    h = hashlib.sha256()
    for path in sorted((PACKAGE / 'particlegan').rglob('*.py')):
        h.update(str(path.relative_to(PACKAGE / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def verify():
    frozen = json.loads((ROOT / 'SOURCE-FREEZE.json').read_text())
    for path, value in frozen['hashes'].items():
        assert sha(path) == value, path
    assert package_digest() == frozen['package_sha256']
    return dict(status='VALID', files=len(frozen['hashes']), package_sha256=frozen['package_sha256'],
        config_sha256=sha(CONFIG), source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'))


def main():
    assert not (ROOT / 'SOURCE-FREEZE.json').exists()
    bridge = json.loads((ROOT / 'EXECUTION-BRIDGE.json').read_text())
    assert bridge['status'] == 'PASS_ORIGINAL_FINITE_HORIZON_CURRENT_PR155_SOURCE_EQUIVALENCE_ONLY'
    assert bridge['package_sha256'] == package_digest() == '500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61'
    assert sha(CONFIG) == bridge['config_sha256'] == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
    assert CONFIG.read_bytes() == (STUDY / 'configs/RA15-partial-recovery.json').read_bytes()
    paths = set(map(Path, bridge['read_only_file_sha256']))
    for path, value in bridge['read_only_file_sha256'].items():
        assert sha(path) == value, path
    paths.update(ROOT.glob('*.py'))
    paths.update(ROOT.glob('*.json'))
    paths.update(ROOT.glob('*.md'))
    frozen = dict(status='FROZEN_UNTRAINED_LATEST_SUITE_SOURCE_EQUIVALENCE_LANE', candidate='RA17-current-pr155',
        package=str(PACKAGE), package_sha256=package_digest(), config=str(CONFIG), config_sha256=sha(CONFIG),
        hashes={str(path): sha(path) for path in sorted(paths)},
        RA14_quality_source_equivalent_count=15, RA15_affected_original_quality_count=4,
        full_quality_task_count=19, fresh_RA17_quality_training_planned=False,
        required_actual_RA17_full_suite=True, required_actual_RA17_original_CUDA40_replay=True,
        required_actual_RA17_single_CUDA_default_regression=True,
        scorer_changed=False, schedule_changed=False, thresholds_changed=False,
        new_training_updates=0, GPU_operations=0, original_execution_labels_preserved=True)
    with (ROOT / 'SOURCE-FREEZE.json').open('x') as stream:
        stream.write(json.dumps(frozen, indent=2, sort_keys=True) + '\n')
    print(json.dumps(verify(), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

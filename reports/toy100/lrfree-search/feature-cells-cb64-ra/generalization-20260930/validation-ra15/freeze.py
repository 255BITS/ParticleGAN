"""Seal RA15 routing over the unchanged correctedRA14 hosts and quality gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
PACKAGE = STUDY / 'pkg-RA15-partial-recovery'
CONFIG = STUDY / 'configs/RA15-partial-recovery.json'
HARNESS = Path('/ml2/hypergan/lrfree-20260926/harness')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


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
    assert not (ROOT / 'SOURCE-FREEZE.json').exists(), 'seal every source preparation in a fresh lane'
    original_path = STUDY / 'validation-ra14-r2/SOURCE-FREEZE.json'
    candidate_path = STUDY / 'portability/partial-recovery/SOURCE-FREEZE.json'
    original = json.loads(original_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    assert sha(original_path) == '871322377b4f7330087760298dac5b1869f5bec01ae01d93ee8b3c6956760a5f'
    assert sha(candidate_path) == 'ec1e68eb823e813577d56fa4d86753db75d8ac86f941d97fc9fa860495ec3777'
    assert package_digest() == candidate['package_sha256'] == '741fc3933654b985a74314b730b93be814d408ad46f6f7f6547b561f64619228'
    assert sha(CONFIG) == original['config_sha256'] == 'a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4'
    assert CONFIG.read_bytes() == (STUDY / 'configs/RA14-replay.json').read_bytes()
    bridge = json.loads((ROOT / 'NO-FIRE-EXECUTION-BRIDGE.json').read_text())
    assert bridge['status'] == 'PASS_SOURCE_EQUIVALENCE_ONLY_FRESH_AFFECTED_GATES_PENDING'
    assert bridge['bridged_quality_task_count'] == 15 and len(bridge['required_fresh_quality_tasks']) == 4
    plan = json.loads((ROOT / 'QUALIFICATION-PLAN.json').read_text())
    expected = {}
    for manifest in (original['hashes'], candidate['hashes'], bridge['read_only_file_sha256'], plan['read_only_diagnostic_pins']):
        for path, digest in manifest.items():
            assert path not in expected or expected[path] == digest, path
            expected[path] = digest
    for path, digest in expected.items():
        assert sha(path) == digest, path
    files = set(map(Path, expected))
    files.update((original_path, candidate_path))
    files.update(ROOT.glob('*.py'))
    files.update(ROOT.glob('*.json'))
    files.update(ROOT.glob('*.md'))
    receipt = dict(status='FROZEN_UNTRAINED_AFFECTED_GATES_PENDING', candidate='RA15-partial-recovery',
        package=str(PACKAGE), package_sha256=package_digest(), config=str(CONFIG), config_sha256=sha(CONFIG),
        hashes={str(path): sha(path) for path in sorted(files)},
        original_corrected_lane_source_freeze_sha256=sha(original_path),
        candidate_CPU_source_freeze_sha256=sha(candidate_path),
        no_fire_bridge_sha256=sha(ROOT / 'NO-FIRE-EXECUTION-BRIDGE.json'),
        quality_plan_sha256=sha(ROOT / 'QUALIFICATION-PLAN.json'),
        bridged_tasks=15, required_fresh_tasks=4, full_quality_task_count=19,
        actual_latest_full_suite_status='PENDING', actual_latest_learned_replay_status='PENDING',
        scorer_changed=False, schedule_changed=False, thresholds_changed=False, stream_recipe_changed=False,
        r2_adapters_retained=True, new_gpu_operations=0, new_training_updates=0,
        labels='RA14 executed quality records and RA13 fresh learned training retain actual source labels.')
    with (ROOT / 'SOURCE-FREEZE.json').open('x') as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(verify(), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()

"""Capture exact completed-suite source/tests from the immutable RA16 commit."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'
REPO = Path('/ml2/hypergan/ParticleGAN-ra11-pr155')
COMMIT = 'b25de08cfbeef9fe8aa06522324e132c3e71ae0f'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return sha(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    target = HERE / 'FINALIZER-V3-R3-PREPARATION.json'
    assert not target.exists()
    old_path = HERE / 'FINALIZER-V3-R2-PREPARATION.json'
    old = json.loads(old_path.read_text())
    for name, value in old['helper_file_sha256'].items():
        assert file_sha(HERE / name) == value, name
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'b25de08c']).decode().strip() == COMMIT
    receipt_path = ROOT / 'validation-ra16/FULL-TESTS.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'PASS' and receipt['returncode'] == 0 and 'completed' in receipt
    assert receipt['source_integrity_before']['package_sha256'] == receipt['source_integrity_after']['package_sha256'] == old['latest_package']['package_sha256']
    assert receipt['source_integrity_before']['status'] == receipt['source_integrity_after']['status'] == 'VALID'
    root = HERE / 'ra16-suite-source-snapshot'
    assert not root.exists()
    root.mkdir()
    files = {}
    declared = {**receipt['sources'], **receipt['tests']}
    for original, expected in sorted(declared.items()):
        relative = Path(original).relative_to(REPO)
        raw = subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + str(relative)])
        assert sha(raw) == expected, 'Recorded completed suite source differs from pinned commit: ' + original
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        files[original] = dict(snapshot_path=str(path), source_relative_path=str(relative), sha256=expected, bytes=len(raw))
    fixture_relative = Path('tests/fixtures/feature-auto-base.json')
    fixture_raw = subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + str(fixture_relative)])
    assert sha(fixture_raw) == '9ae50f9fd0e903b7516ac04a72318ba79a35e52076393317511ec29ae19744b4'
    fixture_path = root / fixture_relative
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with fixture_path.open('xb') as stream:
        stream.write(fixture_raw)
    snapshot_path = root / 'SOURCE-SNAPSHOT.json'
    snapshot = dict(status='PASS_EXACT_COMPLETED_SUITE_SOURCE_TESTS_FROM_IMMUTABLE_COMMIT',
        commit=COMMIT, repository=str(REPO), completed_suite_receipt=str(receipt_path),
        completed_suite_receipt_sha256=file_sha(receipt_path), candidate_package_sha256=old['latest_package']['package_sha256'],
        files=files, file_count=len(files), bytes=sum(item['bytes'] for item in files.values()),
        public_fixture=dict(path=str(fixture_path), sha256=sha(fixture_raw)),
        source_snapshots_match_every_completed_suite_declared_SHA=True,
        original_f459_reference_scope_preserved=True, advanced_upstream_source_qualified=False,
        tensor_loads=0, model_calls=0, GPU_operations=0)
    write(snapshot_path, snapshot)
    expected = (HERE / 'finalize_evidence_v3_r2.py').read_text()
    expected = expected.replace("HERE / 'FINALIZER-V3-R2-PREPARATION.json'", "HERE / 'FINALIZER-V3-R3-PREPARATION.json'")
    expected = expected.replace('finalizer_revision=2,', 'finalizer_revision=3,')
    before = "        for path, value in {**receipt.get('sources', {}), **receipt.get('tests', {})}.items():\n            e.bind(path, value)\n"
    after = """        specification = prepared['historical_suite_source_snapshot']
        snapshot = e.read(specification['path'])
        if snapshot is not None:
            e.check(e.inputs[specification['path']] == specification['sha256'], 'Historical suite source snapshot changed')
            e.check(snapshot['completed_suite_receipt_sha256'] == e.inputs[str(suite_lane / 'FULL-TESTS.json')],
                    'Historical suite source snapshot belongs to a different completed suite')
            declared = {**receipt.get('sources', {}), **receipt.get('tests', {})}
            e.check(set(snapshot['files']) == set(declared), 'Historical suite source/test snapshot scope changed')
            for original, value in declared.items():
                item = snapshot['files'].get(original)
                e.check(item is not None and item['sha256'] == value, 'Historical suite source/test SHA mismatch: ' + original)
                if item is not None:
                    e.bind(item['snapshot_path'], value)
"""
    assert expected.count(before) == 1
    expected = expected.replace(before, after)
    assert expected == (HERE / 'finalize_evidence_v3_r3.py').read_text()
    compile(expected, str(HERE / 'finalize_evidence_v3_r3.py'), 'exec')
    bridge = dict(status='PASS_HISTORICAL_COMPLETED_SOURCE_SNAPSHOT_EQUIVALENCE',
        previous_source_sha256=file_sha(HERE / 'finalize_evidence_v3_r2.py'),
        corrected_source_sha256=file_sha(HERE / 'finalize_evidence_v3_r3.py'),
        snapshot_sha256=file_sha(snapshot_path), commit=COMMIT,
        previous_snapshot_preserved=str(HERE / 'final-v3-r2-attempt1'),
        change='Bind completed-suite declared source/test hashes to exact immutable commit snapshots rather than advanced live source files',
        package_or_config_changed=False, original_numerical_gates_or_quality_criteria_changed=False,
        completed_suite_actual_counts_or_replay_checks_changed=False, tensor_loads=0, model_calls=0, GPU_operations=0)
    bridge_path = HERE / 'FINALIZER-V3-R3-SOURCE-BRIDGE.json'
    write(bridge_path, bridge)
    prepared = dict(old)
    names = ['finalize_evidence_v3_r3.py', 'prepare_evidence_v3_r3.py',
        'FINALIZER-V3-R3-PROTOCOL.md', 'FINALIZER-V3-R3-SOURCE-BRIDGE.json', 'FINALIZER-V3-R2-PREPARATION.json']
    prepared['helper_file_sha256'] = dict(old['helper_file_sha256'], **{name: file_sha(HERE / name) for name in names})
    prepared['revision'] = 3
    prepared['historical_suite_source_snapshot'] = dict(path=str(snapshot_path), sha256=file_sha(snapshot_path), commit=COMMIT)
    prepared['qualification_base'] = dict(old['qualification_base'], qualified_RA16_source_commit=COMMIT)
    write(target, prepared)
    print(json.dumps(dict(status=bridge['status'], helper_sha256=file_sha(HERE / 'finalize_evidence_v3_r3.py'),
        preparation_sha256=file_sha(target), source_snapshot_sha256=file_sha(snapshot_path), snapshot_files=len(files)), sort_keys=True))


if __name__ == '__main__':
    main()

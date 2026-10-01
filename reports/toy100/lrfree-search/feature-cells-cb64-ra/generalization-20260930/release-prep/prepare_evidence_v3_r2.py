"""Seal the exact retained cancellation metadata comparison in a fresh helper."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-generalization-20260930')
HERE = ROOT / 'release-prep'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = HERE / 'FINALIZER-V3-R2-PREPARATION.json'
    assert not path.exists()
    original = json.loads((HERE / 'FINALIZER-V3-PREPARATION.json').read_text())
    for name, value in original['helper_file_sha256'].items():
        assert sha(HERE / name) == value, name
    expected = (HERE / 'finalize_evidence_v3.py').read_text()
    expected = expected.replace("HERE / 'FINALIZER-V3-PREPARATION.json'", "HERE / 'FINALIZER-V3-R2-PREPARATION.json'")
    expected = expected.replace("cancelled_receipt.get('status') == 'CANCELLED'",
                                "cancelled_receipt.get('status') == 'CANCELLED_BEFORE_TEST_EXECUTION'")
    expected = expected.replace("paths.update(path for path in HERE.iterdir() if path.is_file() and\n                 path.suffix in ('.json', '.py', '.md'))",
        "paths.update(path for path in HERE.rglob('*') if path.is_file() and\n                 path.suffix in ('.json', '.jsonl', '.log', '.py', '.md', '.patch'))")
    expected = expected.replace('finalizer_version=3,', 'finalizer_version=3, finalizer_revision=2,')
    assert expected == (HERE / 'finalize_evidence_v3_r2.py').read_text()
    compile(expected, str(HERE / 'finalize_evidence_v3_r2.py'), 'exec')
    failure = HERE / 'final-v3-attempt1'
    failed = json.loads((failure / 'QUALIFICATION.json').read_text())
    assert failed['defects'] == ['RA15 cancelled suite history changed']
    assert failed['quality_qualification'] == 'PASS' and failed['pending'] == []
    closed = json.loads((failure / 'FROZEN.json').read_text())
    for name, value in closed['file_sha256'].items():
        assert sha(failure / name) == value, name
    cancelled_path = ROOT / 'validation-ra15/FULL-TESTS.json'
    cancelled = json.loads(cancelled_path.read_text())
    assert cancelled['status'] == 'CANCELLED_BEFORE_TEST_EXECUTION'
    assert cancelled.get('numerical_started') is None and not (ROOT / 'validation-ra15/full-pytest.log').exists()
    receipt = dict(status='PASS_METADATA_ONLY_SOURCE_EQUIVALENCE',
        original_source_sha256=sha(HERE / 'finalize_evidence_v3.py'),
        corrected_source_sha256=sha(HERE / 'finalize_evidence_v3_r2.py'),
        retained_cancelled_receipt_sha256=sha(cancelled_path),
        retained_first_snapshot=str(failure), retained_first_snapshot_sha256=sha(failure / 'FROZEN.json'),
        runtime_validator_change='Exact retained cancellation status comparison only',
        additional_source_changes=['fresh helper preparation path', 'revision metadata', 'retain earlier small release-prep snapshots in archive'],
        package_or_config_changed=False, gates_or_quality_criteria_changed=False,
        suite_or_replay_checks_changed=False, tensor_loads=0, model_calls=0, GPU_operations=0)
    with (HERE / 'FINALIZER-V3-R2-SOURCE-BRIDGE.json').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    prepared = dict(original)
    names = ['finalize_evidence_v3_r2.py', 'FINALIZER-V3-R2-PROTOCOL.md',
             'prepare_evidence_v3_r2.py', 'FINALIZER-V3-R2-SOURCE-BRIDGE.json',
             'FINALIZER-V3-PREPARATION.json']
    prepared['helper_file_sha256'] = dict(original['helper_file_sha256'], **{name: sha(HERE / name) for name in names})
    prepared['revision'] = 2
    prepared['retained_metadata_failure'] = dict(path=str(failure), SHA256=sha(failure / 'FROZEN.json'))
    with path.open('x') as stream:
        stream.write(json.dumps(prepared, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], preparation_sha256=sha(path),
        helper_sha256=sha(HERE / 'finalize_evidence_v3_r2.py')), sort_keys=True))


if __name__ == '__main__':
    main()

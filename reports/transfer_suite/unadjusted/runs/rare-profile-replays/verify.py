"""Verify all 19 exact profile replays without training or selecting again."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.replay_shared_architecture import REPLAY_KEYS, numerical

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
for folder_name, expected_passes in [('initial', 18), ('winner', 19)]:
    folder = ROOT/folder_name
    protocol = json.loads((folder/'protocol.json').read_text())
    with tarfile.open(folder/'source.tar.gz') as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        assert protocol['source_sha256'] and len(names) == len(set(names))
        assert set(names) == set(protocol['source_sha256']), 'Incomplete source manifest'
        for name, expected in protocol['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected

    checks = json.loads((folder/'checks.json').read_text())
    expected_tasks = {job['spec']['name'] for job in plan()}
    assert len(checks) == len(expected_tasks) == 19
    assert {row['task'] for row in checks} == expected_tasks
    index = json.loads((folder/'index.json').read_text())['records']
    assert len(index) == 19
    indexed = {row['spec']['name']: row for row in index}
    assert set(indexed) == expected_tasks
    passed = 0
    for row in checks:
        assert set(row['checks']) == set(REPLAY_KEYS)
        assert all(value is True for value in row['checks'].values())
        reference_bytes = (REPO/row['reference']).read_bytes()
        assert hashlib.sha256(reference_bytes).hexdigest() == row['reference_sha256']
        expected = json.loads(gzip.decompress(reference_bytes))
        raw = gzip.decompress((folder/row['artifact']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == row['uncompressed_sha256']
        assert indexed[row['task']]['uncompressed_sha256'] == row['uncompressed_sha256']
        assert indexed[row['task']]['artifact'] == row['artifact']
        actual = json.loads(raw)
        assert actual['spec']['name'] == row['task']
        assert actual['source_sha256'] == protocol['source_sha256']
        for key in REPLAY_KEYS:
            if key != 'discriminator_variant':
                assert key in actual and key in expected, (row['task'], key)
            assert numerical(actual.get(key)) == numerical(expected.get(key)), (row['task'], key)
        passed += actual['verdict']['passed']
    assert passed == expected_passes
    print(f'{folder_name}: 19/19 exact profile replays verified; {passed}/19 live PASS. No new selection trials.')

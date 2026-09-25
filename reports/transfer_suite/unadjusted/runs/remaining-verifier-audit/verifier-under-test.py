"""Check frozen replay sources, reference bytes and exact numerical evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from benchmarks.transfer_suite.replay_shared_architecture import REPLAY_KEYS, numerical

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
count = 0
for folder in sorted(path for path in ROOT.iterdir() if (path/'protocol.json').is_file()):
    protocol = json.loads((folder/'protocol.json').read_text())
    with tarfile.open(folder/'source.tar.gz') as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        assert protocol['source_sha256'] and len(names) == len(set(names))
        assert set(names) == set(protocol['source_sha256']), 'Incomplete source manifest'
        for name, expected in protocol['source_sha256'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected
    reference = (REPO/protocol['reference']).read_bytes()
    assert hashlib.sha256(reference).hexdigest() == protocol['reference_sha256']
    expected = json.loads(gzip.decompress(reference))
    raw = gzip.decompress((folder/'episode.json.gz').read_bytes())
    checks = json.loads((folder/'checks.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == checks['uncompressed_sha256']
    actual = json.loads(raw)
    assert actual['source_sha256'] == protocol['source_sha256']
    assert set(checks['checks']) == set(REPLAY_KEYS)
    assert all(value is True for value in checks['checks'].values())
    for key in REPLAY_KEYS:
        assert numerical(actual[key]) == numerical(expected[key]), (folder.name, key)
    count += 1
assert count, 'No replay evidence found'
print(f'{count} exact replays verified, including source and reference hashes')

"""Verify retained independent replay bytes and exact numerical comparisons."""
from pathlib import Path
import gzip
import hashlib
import json
import tarfile
ROOT=Path(__file__).resolve().parent

def clean(x):
    if isinstance(x,dict):
        return {k:clean(v) for k,v in x.items() if k not in ('seconds','controller_seconds','confirmed_seconds','stable_from_seconds','created_at')}
    if isinstance(x,list):
        return [clean(v) for v in x]
    return x

protocol=json.loads((ROOT/'protocol.json').read_text())
with tarfile.open(ROOT/'source.tar.gz') as archive:
    for name,sha in protocol['source_sha256'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==sha
checks=json.loads((ROOT/'checks.json').read_text())
for proof in checks:
    raw=gzip.decompress((ROOT/proof['artifact']).read_bytes())
    assert hashlib.sha256(raw).hexdigest()==proof['uncompressed_sha256']
    reference=ROOT.parent/'shared-discriminator-search/cross/episodes'/('shared_c6__'+proof['artifact'])
    assert hashlib.sha256(reference.read_bytes()).hexdigest()==proof['reference_sha256']
    actual,expected=json.loads(raw),json.loads(gzip.decompress(reference.read_bytes()))
    assert all(proof['checks'].values())
    for key in proof['checks']:
        assert clean(actual[key])==clean(expected[key]),(proof['task'],key)
print(f'{len(checks)} exact independent replays verified, including frozen source hashes')

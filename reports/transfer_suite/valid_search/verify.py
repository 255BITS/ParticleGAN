"""Verify current valid-toy reports, archived bytes, curves and recipe corrections."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from benchmarks.transfer_suite.protocol import test_verdict

ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent


def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def verify():
    files=sources=episodes=references=0
    payloads=set()
    for manifest in ROOT.rglob('archive_manifest.json'):
        for item in read(manifest)['files']:
            raw=(manifest.parent/item['path']).read_bytes()
            assert digest(raw)==item['archived_sha256'],(manifest,item['path'])
            if item['gzip_added']:
                assert digest(gzip.decompress(raw))==item['original_sha256'],item['path']
            files+=1
    for archive in ROOT.rglob('source.tar.gz'):
        protocol=archive.parent/'protocol.json.gz'
        if not protocol.exists(): continue
        values=read(protocol)
        if 'source_sha256' not in values: continue
        with tarfile.open(archive) as tar:
            for name,sha in values['source_sha256'].items():
                assert digest(tar.extractfile(name).read())==sha,(archive,name)
                sources+=1
    for path in ROOT.glob('**/episodes/*.json.gz'):
        raw=gzip.decompress(path.read_bytes())
        value=json.loads(raw)
        verdict=test_verdict(value['spec'],value['result'])
        assert verdict==value['verdict'],path
        assert len(value['result']['observations'])==24,path
        assert 'error' not in value['result'],path
        payloads.add(digest(raw));episodes+=1
    for folder in ('adam999_group_fix','coordinated_group_fix'):
        index=read(ROOT/folder/'index.json.gz')['records']
        assert len(index)==1 and index[0]['spec']['name']=='ae_gan_hold'
        value=index[0]
        assert value['verdict']['passed']
        groups=value.get('applied') or [g for optimizer in value['optimizer_specs'] for g in optimizer]
        assert all(g['betas']==[0.,.999] for g in groups),folder
    def check_refs(value):
        nonlocal references
        if isinstance(value,dict):
            if 'artifact' in value and 'sha256' in value:
                path=SUITE/value['artifact']
                assert digest(path.read_bytes())==value['sha256'],path
                references+=1
            for child in value.values(): check_refs(child)
        elif isinstance(value,list):
            for child in value: check_refs(child)
    for path in (ROOT/'leaderboard.json',SUITE/'formulations/leaderboard.json'):
        check_refs(read(path))
    report=read(ROOT/'leaderboard.json')
    assert digest((ROOT/'build.py').read_bytes())==report['source_sha256']
    for row in report['rows']:
        assert row['complete']
        assert len(row['required'])==9 and len(row['data'])==6 and len(row['images'])==4
        if row['name']!='Original recipe + supported D architectures':
            ae=row['required']['ae_gan_hold']['trials']
            assert len(ae)==1 and '_group_fix/' in ae[0]['artifact']
    report=dict(archive_files=files,source_file_instances=sources,complete_episode_files=episodes,
                unique_episode_payloads=len(payloads),leaderboard_references=references,
                ae_group_corrections='Both replacements PASS with all groups Adam(0,.999); original AE results excluded from current comparison.',
                errors=[])
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': verify()

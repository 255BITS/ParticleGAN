"""Preserve byte-exact saved source/result certificates; never trains or grades."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser()
parser.add_argument('--artifacts',type=Path,required=True)
args=parser.parse_args()
result=json.loads((Path(__file__).parent/'publication/results.json').read_text())
archive=args.artifacts/'certificates';manifest={}
for aid,hashes in result['exact_certificate_hashes'].items():
    manifest[aid]={}
    for name,expected in hashes.items():
        assert name in {'request','result','evidence'} and aid.isalnum()
        data=(ROOT/f'reports/forge/attempts/{aid}/{name}.json').read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected
        target=archive/aid/f'{name}.json';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==data
        else:target.write_bytes(data)
        manifest[aid][name]=dict(sha256=expected,bytes=len(data),path=str(target.relative_to(args.artifacts)))
receipt=dict(schema_version=1,scope='exact_saved_training_media_source_certificates',attempts=len(manifest),certificates=manifest)
path=archive/'manifest.json'
encoded=json.dumps(receipt,sort_keys=True,indent=2)+'\n'
if path.exists():assert path.read_text()==encoded
else:path.write_text(encoded)
print(json.dumps(dict(attempts=len(manifest),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))

"""Replay one retained D trial from a full checkout or extracted source archive.

PYTHONPATH=/path/to/extracted/source python reproduce.py run/episode.json.gz --output replay.json
Uses the retained canonical job, not an external historical-report lookup.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import torch
from benchmarks.transfer_suite.shared_discriminator_search import episode

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('artifact',type=Path)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--verify-existing',action='store_true',help='compare an already retained replay without training again')
a=p.parse_args()
if a.output.exists() and not a.verify_existing:raise FileExistsError(a.output)
raw=a.artifact.read_bytes()
old=json.loads(gzip.decompress(raw) if a.artifact.suffix=='.gz' else raw)
root=Path(__import__('benchmarks.transfer_suite.shared_discriminator_search',fromlist=['x']).__file__).resolve().parents[2]
for name,sha in old['source_sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
  raise RuntimeError('source mismatch: '+name)
torch.set_num_threads(1)
job=dict(spec=old['original_spec'],reference=old['reference'],reference_sha256=old['reference_sha256'])
card=old['discriminator_variant']['overrides']['research_discriminator']
if a.verify_existing:
 new=json.loads(a.output.read_text())
else:
 new=episode(job,card)
 new['source_sha256']=old['source_sha256']
 a.output.write_text(json.dumps(new,sort_keys=True,allow_nan=False)+'\n')
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k!='seconds' and not k.endswith('_seconds')}
 if isinstance(value,list):return [clean(v) for v in value]
 return value
assert clean(new['result'])==clean(old['result'])
assert new['applied']==old['applied']
print('Exact metrics, curves, actions and optimizer receipts match (timings excluded).')

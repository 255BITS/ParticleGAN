"""Replay a retained episode using its exact extracted source and canonical spec.

PYTHONPATH=/path/to/extracted/source python reproduce.py EPISODE.json.gz --output NEW.json
No historical report lookup, task-specific recipe or seed override is supported.
"""
import argparse,gzip,hashlib,importlib,json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('artifact',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
raw=a.artifact.read_bytes()
old=json.loads(gzip.decompress(raw) if a.artifact.suffix=='.gz' else raw)
card=old['discriminator_variant']['overrides']['research_discriminator']
module=importlib.import_module('benchmarks.transfer_suite.'+{
 'shared_local_density_v1':'shared_local_density_search',
 'shared_residual_curvature_v1':'shared_residual_curvature_search'}[card['implementation']])
root=Path(module.__file__).resolve().parents[2]
for name,sha in old['source_sha256'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('source mismatch: '+name)
torch.set_num_threads(1)
job=dict(spec=old['original_spec'],reference=old['reference'],reference_sha256=old['reference_sha256'])
new=module.episode(job,card);new['source_sha256']=old['source_sha256']
a.output.write_text(json.dumps(new,sort_keys=True,allow_nan=False)+'\n')
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='seconds' and not k.endswith('_seconds')}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
for key in ['result','recipe','candidate','spec','applied','verdict','ema_verdict']:
 assert clean(new[key])==clean(old[key]),key
print('Exact numerical replay matches; only runtime fields excluded.')

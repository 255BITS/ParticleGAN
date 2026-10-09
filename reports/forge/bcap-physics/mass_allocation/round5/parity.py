"""Check local-v2 singleton parity; old scores remain contextual, never pooled."""
from pathlib import Path
import sys,importlib.util
import torch
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT))
from experiments.forge.contracts import read_json,atomic_json,file_hash
from experiments.forge.tier1_media import _scored_outputs
OUT=Path(__file__).parent
def compare(a,b,path='root'):
 if isinstance(a,torch.Tensor):return [] if isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b) else [path]
 if isinstance(a,dict):
  if not isinstance(b,dict) or a.keys()!=b.keys():return [path+'.keys']
  return [d for k in a for d in compare(a[k],b[k],path+'.'+str(k))]
 if isinstance(a,(list,tuple)):
  if not isinstance(b,(list,tuple)) or len(a)!=len(b):return [path+'.length']
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in compare(x,y,path+'.'+str(i))]
 return [] if a==b else [path]


def norm(x):
 if isinstance(x,dict):return {k:norm(v) for k,v in x.items() if k not in ['kinetic_transport_mode','kinetic_transport_block_size','cpu_rng','cuda_rng','training_state_sha256','primary_state_sha256','confirmed_state_sha256']}
 if isinstance(x,(list,tuple)):return type(x)(norm(v) for v in x)
 return x

def main():
 torch.set_num_threads(1);provenance=read_json(OUT/'provenance.json');oldq=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_transport/queue');old=read_json(oldq/'queue/state.json');checks=[]
 for p in provenance['proofs']:
  if p['role']!='control' or p['task_id']=='grid100':continue
  tid=p['task_id'];job=next((j for j in old['jobs'].values() if j.get('result') and j['definition']['task_id']==tid and any(old['submissions'][r]['request']['candidate']['id']=='projection_transport-round4-transport-v1' for r in j['subscribers'])),None)
  if job is None:continue
  oldrow=job['result']['task_results'][0];newrow=read_json(ROOT/f'reports/forge/attempts/{p["attempt_id"]}/result.json')['task_results'][0];d=oldrow['evidence']['provenance_checkpoint'];assert file_hash(Path(d['artifact_root'])/d['path'])==d['sha256'];a=torch.load(Path(d['artifact_root'])/d['path'],map_location='cpu',weights_only=False);nd=p['provenance_checkpoint'];b=torch.load(Path(nd['artifact_root'])/nd['path'],map_location='cpu',weights_only=False)
  cohort=lambda s:{k:s[k] for k in ['trainer','streams','initialization','prior']}
  diffs=dict(metrics=compare(newrow['metrics'],oldrow['metrics']),observations=compare(newrow['evidence']['observations'],oldrow['evidence']['observations']),numerical_state=compare(norm(cohort(b)),norm(cohort(a))))
  checks.append(dict(task_id=tid,archived_attempt_id=job['result']['attempt_id'],new_attempt_id=p['attempt_id'],archived_source=old['submissions'][next(r for r in job['subscribers'] if old['submissions'][r]['request']['candidate']['id']=='projection_transport-round4-transport-v1')]['request']['source']['digest'],archived_checkpoint_sha256=d['sha256'],new_checkpoint_sha256=nd['sha256'],bitwise_equal=all(not v for v in diffs.values()),mismatches=diffs))
 atomic_json(OUT/'control-parity.json',dict(schema_version=1,qualification_input=False,scope='Exact local-v2 matched-source reproduction of archived singleton; no qualification reuse or independent-seed credit.',checks=checks,optimizer_updates_added=0,sampling_draws_added=0))
 print([(c['task_id'],c['bitwise_equal']) for c in checks]);assert all(c['bitwise_equal'] for c in checks)
if __name__=='__main__':main()

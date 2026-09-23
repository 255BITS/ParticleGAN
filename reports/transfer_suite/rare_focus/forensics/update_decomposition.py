"""Second parity replay: observe actual Adam updates, never modify them."""
import copy,gzip,hashlib,json
from pathlib import Path
from unittest.mock import patch
import torch
from torch.func import functional_call
from benchmarks.transfer_suite import vector_tasks as v
from benchmarks.transfer_suite import smooth_critic_research as smooth
root=Path('/tmp/pr36-rare-forensics')
read=lambda n:json.loads(gzip.decompress((root/n).read_bytes()))
write=lambda n,d:(root/n).write_bytes(gzip.compress((json.dumps(d,sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
ref=read('reference.json.gz');spec=ref['spec'];models={};updates=[];gsteps=0
source=Path(__file__).read_bytes();write('update_plan.json.gz',dict(frozen_before_execution=True,source_sha256=hashlib.sha256(source).hexdigest(),reference_sha256=hashlib.sha256((root/'reference.json.gz').read_bytes()).hexdigest(),scope='One exact replay of same seed0 model, measuring old/new G and old/new prior at every G update1001–1200. No altered update, sampling or diagnostic gradient.'))
def keep(k,ctor):
 def create(*a,**kw):
  out=ctor(*a,**kw);models[k]=out;return out
 return create
original_step=torch.optim.Adam.step
@torch.no_grad()
def step(self,*a,**kw):
 global gsteps
 is_g=any(p is models['p'].z for group in self.param_groups for p in group['params'])
 if is_g:gsteps+=1
 capture=is_g and gsteps>=1001
 if capture:
  rng=torch.get_rng_state().clone();g=models['g'];p=models['p'];old_z=p.z.detach().clone();old_state={k:x.detach().clone() for k,x in g.state_dict().items()};x00=g(old_z)
 result=original_step(self,*a,**kw)
 if capture:
  x10=g(old_z);x11=g(p.z);x01=functional_call(g,old_state,(p.z,))
  assert torch.equal(torch.get_rng_state(),rng)
  updates.append(dict(step=gsteps,old_g_old_z=x00.tolist(),new_g_old_z=x10.tolist(),old_g_new_z=x01.tolist(),new_g_new_z=x11.tolist(),lrs=[group['lr'] for group in self.param_groups]))
 return result
with patch.object(v,'SimpleMLPGenerator',keep('g',v.SimpleMLPGenerator)),patch.object(v,'ParticlePrior',keep('p',v.ParticlePrior)),patch.object(v,'SimpleMLPDiscriminator',smooth.constructor(ref['candidate']['architecture'])),patch.object(torch.optim.Adam,'step',step):
 result=v.run_episode(spec,v.fixed_policy('cosine'),fixed=True)
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k not in ['seconds','controller_seconds','stable_from_seconds','confirmed_seconds']}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
write('update_replay_result.json.gz',result);write('updates.json.gz',updates)
parity=dict(entire_result_except_timing_exact=clean(result)==clean(ref['result']),g_steps=gsteps,observed_updates=len(updates))
write('update_parity.json.gz',parity)
assert parity['entire_result_except_timing_exact'] and gsteps==1200 and len(updates)==200
print('PARITY',parity,'seconds',result['seconds'],flush=True)

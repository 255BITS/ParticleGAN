"""Read-only measurement around one unchanged training run; no diagnostic updates."""
import copy,gzip,hashlib,json,tarfile,time
from pathlib import Path
from unittest.mock import patch
import torch
from benchmarks.transfer_suite import vector_tasks as v
from benchmarks.transfer_suite import smooth_critic_research as smooth
from benchmarks.transfer_suite.protocol import test_verdict
ROOT=Path('/tmp/pr36-rare-forensics')
REF=Path('reports/transfer_suite/valid_search/softplus_refinement/episodes/softplus5_d96_l2_f2__vector_unequal_mass.json.gz')
reference=json.loads(gzip.decompress(REF.read_bytes()));spec=reference['spec']
write=lambda name,data:(ROOT/name).write_bytes(gzip.compress((json.dumps(data,sort_keys=True,allow_nan=False)+'\n').encode(),mtime=0))
source=v.fingerprint();source['source_sha256']['benchmarks/transfer_suite/smooth_critic_research.py']=hashlib.sha256(Path(smooth.__file__).read_bytes()).hexdigest()
plan=dict(purpose='failure forensics, never a GAN candidate or metric change',seed=0,reference=str(REF),reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),spec=spec,measurements='all256 live atoms/latent, exact4096 multiplicities, D input gradients/Hessians, G latent Jacobians, train gradients. Capture at24 scheduled evals; no update from diagnostics. Exact24 live+EMA numerical/action parity required before interpretation.',controls=[],source=source,replay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
write('plan.json.gz',plan);(ROOT/'reference.json.gz').write_bytes(REF.read_bytes())
models={};snapshots=[];calls=0
orig_g=v.SimpleMLPGenerator;orig_p=v.ParticlePrior;orig_score=v.score_samples
make_d=smooth.constructor(reference['candidate']['architecture'])
def keep(key,ctor):
 def new(*args,**kwargs):
  model=ctor(*args,**kwargs);models[key]=model;return model
 return new

def capture(step):
 g,p,d=(models[k] for k in ['g','p','d'])
 state=torch.get_rng_state().clone()
 parameters=[*g.parameters(),*p.parameters(),*d.parameters()]
 grads=[None if x.grad is None else x.grad.clone() for x in parameters]
 with torch.random.fork_rng(devices=[]),torch.enable_grad():
  z=p.z.detach().clone().requires_grad_();x=g(z)
  j=torch.stack([torch.autograd.grad(x[:,q].sum(),z,retain_graph=True)[0] for q in range(2)],1)
  xx=x.detach().clone().requires_grad_();score=d(xx)
  dg=torch.autograd.grad(score.sum(),xx,create_graph=True)[0]
  dh=torch.stack([torch.autograd.grad(dg[:,q].sum(),xx,retain_graph=True)[0] for q in range(2)],1)
  indices=p.sample_indices(4096,generator=torch.Generator().manual_seed(990))
  real=v.sample_target(spec,4096,torch.Generator().manual_seed(991),step)
  real_logits=d(real).detach()
  # Expected Rp-logistic fake coefficient over the declared target draw, diagnostic only.
  scalar=(real_logits[None,:]-score.detach()[:,None]).sigmoid().mean(1)
  ascent=dg.detach()*scalar[:,None]
  latent_ascent=torch.einsum('nij,ni->nj',j,ascent)
  output_prior_ascent=torch.einsum('nij,nj->ni',j,latent_ascent)
  snap=dict(step=step,z=z.detach().tolist(),x=x.detach().tolist(),eval_indices=indices.tolist(),d_score=score.detach().tolist(),d_input_gradient=dg.detach().tolist(),d_input_hessian=dh.detach().tolist(),g_latent_jacobian=j.detach().tolist(),expected_rp_ascent=ascent.tolist(),projected_prior_ascent=output_prior_ascent.tolist(),last_prior_training_gradient=None if p.z.grad is None else p.z.grad.tolist(),parameter_gradient_norms={key:[None if a.grad is None else float(a.grad.norm()) for a in models[key].parameters()] for key in ['g','d']})
 for param,before in zip(parameters,grads):
  assert (param.grad is None and before is None) or (param.grad is not None and before is not None and torch.equal(param.grad,before)), 'diagnostics mutated gradients'
 assert torch.equal(torch.get_rng_state(),state),'diagnostics mutated RNG'
 snapshots.append(snap)
 print('CAPTURE',step,flush=True)

def score(fake,cfg,step):
 global calls
 result=orig_score(fake,cfg,step)
 if calls%2==0:capture(step)
 calls+=1
 return result

def clean(value):
 if isinstance(value,dict):return {k:clean(x) for k,x in value.items() if k not in ['seconds','controller_seconds','confirmed_seconds','stable_from_seconds']}
 if isinstance(value,list):return [clean(x) for x in value]
 return value
with patch.object(v,'SimpleMLPGenerator',keep('g',orig_g)),patch.object(v,'ParticlePrior',keep('p',orig_p)),patch.object(v,'SimpleMLPDiscriminator',keep('d',make_d)),patch.object(v,'score_samples',score):
 result=v.run_episode(spec,v.fixed_policy('cosine'),fixed=True)
write('replay_result.json.gz',result);write('snapshots.json.gz',snapshots)
parity=dict(result_without_timing_exact=clean(result)==clean(reference['result']),all_24_live_ema_exact=clean(result['observations'])==clean(reference['result']['observations']),actions_exact=result['actions']==reference['result']['actions'],gradient_and_rng_unchanged=True)
write('parity.json.gz',parity)
assert all(parity.values()),parity
write('verdict.json.gz',test_verdict(spec,result))
torch.save({key:model.state_dict() for key,model in models.items()},ROOT/'final_models.pt')
with tarfile.open(ROOT/'source.tar.gz','w:gz') as archive:
 for path in source['source_sha256']:archive.add(path,arcname=path)
 archive.add(__file__,arcname='replay.py')
print('EXACT PARITY',parity,'seconds',result['seconds'],flush=True)

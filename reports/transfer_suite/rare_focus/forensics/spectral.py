"""Frozen-model force decomposition, descriptive rather than an intervention."""
import gzip,json,hashlib
from pathlib import Path
import torch
from particlegan import ParticleRegularizer
from benchmarks.transfer_suite import vector_tasks as v
from benchmarks.transfer_suite.smooth_critic_research import constructor
root=Path('/tmp/pr36-rare-forensics');torch.set_num_threads(1)
read=lambda n:json.loads(gzip.decompress((root/n).read_bytes()))
ref=read('reference.json.gz');snap=read('snapshots.json.gz')[-1];spec=ref['spec'];state=torch.load(root/'final_models.pt',weights_only=True)
with torch.random.fork_rng(devices=[]):
 torch.manual_seed(0)
 d=constructor(ref['candidate']['architecture'])(2,96,2,2);d.load_state_dict(state['d'])
x=torch.tensor(snap['x']).requires_grad_();features=d.encode(x);feat=features.detach().requires_grad_();score=d.net(feat).squeeze(-1);df=torch.autograd.grad(score.sum(),feat)[0]
phase=x.unsqueeze(-1)*d.freqs
raw=df[:,:2];sin=df[:,2:6].reshape(-1,2,2);cos=df[:,6:10].reshape(-1,2,2)
harmonics=(sin*phase.cos()-cos*phase.sin())*d.freqs
parts={'raw':raw,'harmonic_pi':harmonics[:,:,0],'harmonic_2pi':harmonics[:,:,1]}
exact=torch.tensor(snap['d_input_gradient']);assert torch.allclose(sum(parts.values()),exact,atol=2e-6,rtol=2e-5)
real=v.sample_target(spec,4096,torch.Generator().manual_seed(991),1200)
coeff=(d(real).detach()[None,:]-score.detach()[:,None]).sigmoid().mean(1)
means=torch.tensor(spec['means']);assignment=torch.cdist(x.detach(),means).argmin(1);rows=[]
z=torch.tensor(snap['z']).requires_grad_();penalty=ParticleRegularizer(weight=spec['prior_reg'])(z);reggrad=torch.autograd.grad(penalty,z)[0]
j=torch.tensor(snap['g_latent_jacobian']);regmove=-torch.einsum('nij,nj->ni',j,reggrad)
# Correct full-support averaging factor for expected GAN gradient per particle.
advprior=torch.tensor(snap['projected_prior_ascent'])/256
for k in range(4):
 mask=assignment==k;xx=x.detach()[mask];delta=xx-xx.mean(0);eig,axes=torch.linalg.eigh(delta.T@delta/len(xx));axis=axes[:,0]
 def rate(direction):return float(2*((delta@axis)*(direction[mask]@axis)).mean())
 entries={name:dict(minor_variance_rate=rate(value*coeff[:,None]),rms_gradient=float(value[mask].square().mean().sqrt())) for name,value in parts.items()}
 rows.append(dict(component=k,force_parts=entries,total_expected_d_minor_rate=rate(exact*coeff[:,None]),raw_prior_gradient_rates=dict(gan=rate(advprior),regularization=rate(regmove),sum=rate(advprior+regmove)),note='D branch rates use output gradient ascent; prior rates include GAN averaging1/256 and regularizer weight. Neither includes Adam second-moment preconditioning or sampled training batches.'))
result=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gradient_decomposition_matches_captured=True,prior_regularization_value=float(penalty),components=rows)
(root/'spectral.json.gz').write_bytes(gzip.compress((json.dumps(result,indent=2)+'\n').encode(),mtime=0))
print(json.dumps(result,indent=2))

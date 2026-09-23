import json,math,sys,time
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-vector-solvability')
import torch
from benchmarks.transfer_suite import vector_tasks as v
from particlegan import ParticlePrior
from lib.toy_models import SimpleMLPGenerator
out=Path('/tmp/pr36-solvability-audit-afe6152')
torch.set_num_threads(1);torch.manual_seed(0)
spec=next(x for x in v.TASKS if x['name']=='vector_unequal_mass')
counts=torch.tensor([141,77,33,5])
support=[]
for mean,cov,count in zip(spec['means'],spec['covariances'],counts):
    angle=torch.arange(int(count))*2*math.pi/int(count)
    unit=math.sqrt(2)*torch.stack([angle.cos(),angle.sin()],1)
    support.append(torch.tensor(mean)+unit@torch.linalg.cholesky(torch.tensor(cov)).T)
support=torch.cat(support)
prior=ParticlePrior(256,4,generator=torch.Generator().manual_seed(0))
with torch.no_grad(): prior.z[:,:2].copy_(support)
model=SimpleMLPGenerator(4,64,2,2)
linears=[x for x in model.net if isinstance(x,torch.nn.Linear)]
with torch.no_grad():
    for layer in linears: layer.weight.zero_();layer.bias.zero_()
    for d in range(2):
        linears[0].weight[2*d,d]=1;linears[0].weight[2*d+1,d]=-1
        linears[1].weight[2*d,2*d]=1/1.2;linears[1].weight[2*d,2*d+1]=-1/1.2
        linears[1].weight[2*d+1,2*d]=-1/1.2;linears[1].weight[2*d+1,2*d+1]=1/1.2
        linears[2].weight[d,2*d]=1/1.2;linears[2].weight[d,2*d+1]=-1/1.2
    fake=model(prior.sample(4096,generator=torch.Generator().manual_seed(990))[0])
start=time.perf_counter();curve=[]
expected=[math.ceil(i*spec['steps']/24) for i in range(1,25)]
for step in expected:
    metric=v.score_samples(fake,spec,step)
    curve.append(dict(metric,step=step,seconds=time.perf_counter()-start))
record={'task':spec['name'],'control':'deterministic_moment_balanced_support256','spec':spec,'no_training':True,'construction':'Allocate 141,77,33,5 uniform particles by largest remainder of target masses. Each component uses a regular polygon at Mahalanobis radius sqrt(2), exactly matching its target mean and covariance before sampling. Set exact existing G64x2 z4 model to identity over first two prior coordinates. This proves representability under these coarse metrics, not Gaussian density identity or adversarial learnability.','support_counts':counts.tolist(),'live':metric,'observations':curve,'stationary_control_sustained':v.sustained(curve,spec['thresholds'],expected_steps=expected),'support':support.tolist()}
(out/'vector_unequal_mass__balanced_support256.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'pass':v.passes(metric,spec['thresholds']),'metric':metric,'convergence':record['stationary_control_sustained']}),flush=True)

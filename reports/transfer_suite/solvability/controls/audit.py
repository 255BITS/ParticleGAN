import datetime,hashlib,json,math,sys,time,tarfile
from pathlib import Path
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-vector-solvability')
import torch
from benchmarks.transfer_suite import vector_tasks as v, stress_tasks as s
from particlegan import ParticlePrior
from lib.toy_models import SimpleMLPGenerator
out=Path('/tmp/pr36-solvability-audit-afe6152')
root=Path('/ml2/hypergan/ParticleGAN-vector-solvability')
torch.set_num_threads(1)
torch.manual_seed(0)
specs=[x for x in v.TASKS+s.TASKS if x['tier']=='ranking']+s.RESERVED_TASKS
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'git_head':'afe615264221eda47c5d1b7fd2cf4082552e30e9','source':v.fingerprint(),'specs':specs,'seed':0,'evaluation_seed':990,'target_score_seed':991,'projection_seed':992,'support_sizes':[256,512,1024],'controls':['direct_target_draw','uniform_empirical_target_support'],'no_training':True,'note':'Cadence has already been evaluated in the published comparison and is now seen. No new reserved vector/image family is inspected. Curves measure a stationary oracle, not convergence of learned weights.'}
protocol['source']['source_sha256']['benchmarks/transfer_suite/stress_tasks.py']=hashlib.sha256((root/'benchmarks/transfer_suite/stress_tasks.py').read_bytes()).hexdigest()
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with tarfile.open(out/'source.tar.gz','w:gz') as archive:
    for name,digest in protocol['source']['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        archive.add(root/name,arcname=name)

def identity_generator():
    model=SimpleMLPGenerator(4,64,2,2)
    linears=[layer for layer in model.net if isinstance(layer,torch.nn.Linear)]
    with torch.no_grad():
        for layer in linears:
            layer.weight.zero_(); layer.bias.zero_()
        linears[0].weight[0,0]=1;linears[0].weight[1,0]=-1
        linears[0].weight[2,1]=1;linears[0].weight[3,1]=-1
        for d in range(2):
            linears[1].weight[2*d,2*d]=1/1.2;linears[1].weight[2*d,2*d+1]=-1/1.2
            linears[1].weight[2*d+1,2*d]=-1/1.2;linears[1].weight[2*d+1,2*d+1]=1/1.2
            linears[2].weight[d,2*d]=1/1.2;linears[2].weight[d,2*d+1]=-1/1.2
    return model

rows=[]
for spec in specs:
    for count in (None,256,512,1024):
        start=time.perf_counter()
        name='direct_target_draw' if count is None else f'finite_support_{count}'
        if count is None:
            fake=v.sample_target(spec,v.EVAL_SAMPLES,torch.Generator().manual_seed(990),spec['steps'])
            support_metadata={}
        else:
            support=v.sample_target(spec,count,torch.Generator().manual_seed(0),spec['steps'])
            prior=ParticlePrior(count,4,generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                prior.z[:,:2].copy_(support)
            indices=prior.sample_indices(v.EVAL_SAMPLES,generator=torch.Generator().manual_seed(990))
            model=identity_generator()
            with torch.no_grad():
                fake=model(prior(indices))
            identity_error=float((fake-support[indices]).abs().max())
            assert identity_error<2e-6
            support_metadata={'support':support.tolist(),'identity_generator_max_error':identity_error,'expressivity':'Exact existing z4 G64x2 MLP configured algebraically as identity on first two latent coordinates; prior coordinates assigned target support. This is no learned or adversarial success.'}
            if spec['kind']=='gaussian_mixture':
                component=torch.cdist(support,torch.tensor(spec['means'])).argmin(1)
                support_metadata['support_counts']=torch.bincount(component,minlength=len(spec['means'])).tolist()
        curve=[]
        expected=[math.ceil(i*spec['steps']/24) for i in range(1,25)]
        for step in expected:
            metrics=v.score_samples(fake,spec,step)
            curve.append(dict(metrics,step=step,seconds=time.perf_counter()-start))
        convergence=v.sustained(curve,spec['thresholds'],expected_steps=expected)
        record={'task':spec['name'],'control':name,'spec':spec,'no_training':True,'live':metrics,'observations':curve,'stationary_control_sustained':convergence,'support_metadata':support_metadata,'seconds':time.perf_counter()-start}
        (out/(spec['name']+'__'+name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        row={'task':spec['name'],'control':name,'pass':v.passes(metrics,spec['thresholds']),'sustained':convergence['stable_from_step'] is not None,'complete':convergence['complete'],'live':{k:metrics[k] for k,_,_ in spec['thresholds']},'support_counts':support_metadata.get('support_counts'),'seconds':record['seconds']}
        rows.append(row)
        (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)

# Counterexample separating HQ/mass from tail covariance: one extreme output.
spec=next(x for x in v.TASKS if x['name']=='vector_unequal_width')
fake=v.sample_target(spec,4096,torch.Generator().manual_seed(990),spec['steps'])
clean=v.score_samples(fake,spec,spec['steps'])
contaminated=fake.clone();contaminated[0]=torch.tensor([-8.,-8.])
corrupt=v.score_samples(contaminated,spec,spec['steps'])
(out/'tail_sensitivity.json').write_text(json.dumps({'spec':spec,'construction':'Replace exactly 1/4096 target samples by [-8,-8], keeping evaluation target/prng fixed. Metric sensitivity diagnostic, not GAN training.','clean':clean,'contaminated':corrupt},indent=2)+'\n')
print('COMPLETE',flush=True)

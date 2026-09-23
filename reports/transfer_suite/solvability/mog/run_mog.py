import datetime,functools,hashlib,json,sys,tarfile,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/ml2/hypergan/ParticleGAN-vector-solvability')
import torch
from particlegan import MoGParticlePrior
from benchmarks.transfer_suite import vector_tasks as v
out=Path('/tmp/pr36-solvability-mog-afe6152')
root=Path('/ml2/hypergan/ParticleGAN-vector-solvability')
specs=[x for x in v.TASKS if x['name'] in ('vector_unequal_mass','vector_unequal_width')]
cards=[{'sigma_rel':sigma,'standardize':standardize} for standardize in (False,True) for sigma in (0.,.025,.1,.3)]
protocol={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'head':'afe615264221eda47c5d1b7fd2cf4082552e30e9','source':v.fingerprint(),'specs':specs,'cards':cards,'seed':0,'policy':v.fixed_policy(),'budget':'Original256 particles,1200 updates, all unchanged live bounds; 16 episodes maximum.','substitution':'Temporarily patch vector_tasks.ParticlePrior constructor to existing MoGParticlePrior with explicit sigma_rel/standardize. Existing sampler controls both component indices and Gaussian noise using original host streams; zero sigma/standardizeFalse should reproduce atom prior. All host updates, eval and penalties unchanged.','no_oracle_training':True}
protocol['source']['source_sha256']['run_mog.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
with tarfile.open(out/'source.tar.gz','w:gz') as archive:
    for name,digest in protocol['source']['source_sha256'].items():
        source=Path(__file__) if name=='run_mog.py' else root/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        archive.add(source,arcname=name)
original_generator=v.SimpleMLPGenerator
rows=[]
for card in cards:
    for spec in specs:
        name=f'mog_sigma{card["sigma_rel"]:g}_std{int(card["standardize"])}'
        print('START',name,spec['name'],flush=True)
        objects={}
        def prior_factory(*args,**kwargs):
            prior=MoGParticlePrior(*args,**kwargs,**card)
            objects['prior']=prior
            objects['initial_sigma']=float(prior.sigma)
            objects['initial_median_distance']=float(prior.d0)
            return prior
        def generator_factory(*args,**kwargs):
            model=original_generator(*args,**kwargs);objects['generator']=model;return model
        with patch.object(v,'ParticlePrior',prior_factory),patch.object(v,'SimpleMLPGenerator',generator_factory):
            result=v.run_episode(spec,protocol['policy'],fixed=True)
        diagnostics={k:value for k,value in objects.items() if not isinstance(value,torch.nn.Module)}
        if 'error' not in result:
            with torch.no_grad():
                centers=objects['generator'](objects['prior'].means())
                assignment=torch.cdist(centers,torch.tensor(spec['means'])).argmin(1)
                diagnostics['generated_component_centers']=centers.tolist()
                diagnostics['prior_component_center_assignment_counts']=torch.bincount(assignment,minlength=len(spec['means'])).tolist()
                diagnostics['distinct_generated_component_centers_per_target']=[len(torch.unique(centers[assignment==k],dim=0)) for k in range(len(spec['means']))]
                diagnostics['limitation']='These are outputs at MoG component means, not all possible noisy outputs or inferred counts from sampled output batches.'
        record={'card_name':name,'card':card,'spec':spec,'policy':protocol['policy'],'protocol':protocol,'result':result,'prior_diagnostics':diagnostics}
        (out/(name+'__'+spec['name']+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        row={'card':name,'task':spec['name'],'live':{k:result['live'].get(k) for k,_,_ in spec['thresholds']},'final_pass':v.passes(result['live'],spec['thresholds']),'convergence':result.get('convergence'),'seconds':result['seconds'],'error':result.get('error'),'prior_diagnostics':{k:value for k,value in diagnostics.items() if k!='generated_component_centers'}}
        rows.append(row)
        (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
print('COMPLETE',flush=True)

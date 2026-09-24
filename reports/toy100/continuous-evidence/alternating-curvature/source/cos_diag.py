import json,sys,torch,statistics as st
from pathlib import Path
sys.path.insert(0,'.')
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
torch.set_num_threads(1)
task,mode=sys.argv[1],sys.argv[2]
config=json.loads(Path('configs/toy100/constraints_simple_regularization.json').read_text())
if mode=='constant':
    config.update(lr_floor=1.,lr_anneal_start=0.);config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
recipe,noise,_=declared_recipe(config)
spec=next(j['spec'] for j in plan() if j['spec']['name']==task)
orig=torch.optim.Adam.step; prev={}; rows={}
def step(opt,closure=None):
    before=[[p.detach().clone() for p in g['params']] for g in opt.param_groups]
    r=orig(opt,closure)
    for gi,(g,b) in enumerate(zip(opt.param_groups,before)):
        role='prior' if g.get('_comparison_prior') else ('g' if len(opt.param_groups)>1 else 'd')
        d=torch.cat([(p.detach()-x).flatten().double() for p,x in zip(g['params'],b)])
        if role in prev and d.norm()>0 and prev[role].norm()>0:
            rows.setdefault(role,[]).append(float(d@prev[role]/(d.norm()*prev[role].norm())))
        prev[role]=d
    return r
torch.optim.Adam.step=step
result,_=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
n=len(rows['g']); bins=[(0,n//6),(n//6,n//3),(n//3,n//2),(n//2,n)]
for role,v in rows.items():
    print(task,mode,role,' '.join(f'[{a}-{b}] {st.mean(v[a:b]):+.3f}' for a,b in bins))
key='identity_mse' if task=='trajectory' else 'hq'
print(' obs',[(o['step'],round(o[key],3)) for o in result['observations']][::3])

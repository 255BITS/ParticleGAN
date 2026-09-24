import json,sys,torch,math,statistics as st
from pathlib import Path
sys.path.insert(0,'.')
from particlegan.gan_loss import GANLoss
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
orig=GANLoss.d_loss; vals=[]
def d_loss(self,r,f):
    out=orig(self,r,f); vals.append((math.log(2)-float(out), float((r-f).mean()))); return out
GANLoss.d_loss=d_loss
result,_=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
n=len(vals); edges=[0,n//12,n//6,n//3,n//2,2*n//3,5*n//6,n]
print(task,mode,'n',n,' '.join(f'[{a}-{b}] adv {st.mean(v[0] for v in vals[a:b]):+.3f} gap {st.mean(v[1] for v in vals[a:b]):+.3f}' for a,b in zip(edges,edges[1:])))
key='identity_mse' if task=='trajectory' else 'hq'
print(' obs',[(o['step'],round(o[key],3)) for o in result['observations']][::3])

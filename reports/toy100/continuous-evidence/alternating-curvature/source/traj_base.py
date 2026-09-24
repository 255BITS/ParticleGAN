import json,sys,torch
from pathlib import Path
sys.path.insert(0,'.')
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_recipe,declared_model_policy
torch.set_num_threads(1)
mode=sys.argv[1]
config=json.loads(Path('configs/toy100/constraints_simple_regularization.json').read_text())
if mode=='constant':
    config.update(lr_floor=1.,lr_anneal_start=0.);config.pop('network_lr_horizon_cap');config.pop('network_lr_floor')
recipe,noise,_=declared_recipe(config)
spec=next(j['spec'] for j in plan() if j['spec']['name']=='trajectory')
result,ctx=run_legacy(spec,recipe,noise,model_policy=declared_model_policy(config))
v=test_verdict(spec,result)
print(mode,v['status'],round(v['metrics'][0]['value'],5),v['convergence']['passing_observations'],v['convergence']['passing_suffix'])
print([(o['step'],round(o['identity_mse'],4)) for o in result['observations']])

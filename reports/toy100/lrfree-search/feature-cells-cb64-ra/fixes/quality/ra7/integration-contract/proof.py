"""Fixed saved-state optimizer API proof; no gradients, steps or new seeds."""
import ast
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA=Path(__file__).resolve().parent
PKG=ROOT/'pkg-CB64-RA7/particlegan'
BASE_PKG=ROOT/'pkg-CB64-RA6/particlegan'
COMPOSITION=ROOT/'quality/ra7/COMPOSITION.json'
OWNER=ROOT/'performance/sampler-regression/cpu-plan-review/post-ra4-quality/anchor-profile/READY.json'
NEW_CONFIG=ROOT/'configs/overrides-CB64-RA7.json'
CHECKPOINT=ROOT/'validation-cb64-ra6/learned/training/toy/CB64-RA6/checkpoint-2000.pt'
CONFIG=ROOT/'configs/overrides-CB64-RA6.json'
READY=ROOT/'quality/ra6/READY.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
paths=[*sorted(PKG.glob('*.py')),*sorted(BASE_PKG.glob('*.py')),CHECKPOINT,CONFIG,NEW_CONFIG,READY,COMPOSITION,OWNER,ROOT/'integration/review/group-count-gpu/result.json',ROOT/'integration/review/group-count-gpu/PHASE-RESULT.json']
before={str(p):sha(p) for p in paths}
composition=json.loads(COMPOSITION.read_text());owner=json.loads(OWNER.read_text())
assert sha(COMPOSITION)=='27dc199147662c489ed33341b5d54a7e477f3e07b2c6d71d6cc00fffa24c8b8f'
assert sha(NEW_CONFIG)==composition['config_sha256']=='08359ed4406148faf6915144ef4265d930829de53775eeafc4f17a11b5667b4c'
package_map={p.name:sha(p) for p in PKG.glob('*.py')}
assert len(package_map)==29 and package_map==composition['source_sha256']==owner['package_source_sha256']
assert all((PKG/filename).read_bytes()==(Path(owner['package_root'])/'particlegan'/filename).read_bytes() for filename in package_map)
assert all(sha(Path(p))==h for p,h in composition['composed_from'].items())
old_config=json.loads(CONFIG.read_text());new_config=json.loads(NEW_CONFIG.read_text())
assert sorted(k for k in old_config if old_config[k]!=new_config[k])==['d_lr_mult','lr','prior_lr_mult']
assert old_config.keys()==new_config.keys()
assert all(new_config[k]==v for k,v in {'lr':.0010625,'prior_lr_mult':8.,'d_lr_mult':4.}.items())
for filename in package_map:
    if filename!='feature_cells.py':assert (PKG/filename).read_bytes()==(BASE_PKG/filename).read_bytes()
rng=torch.get_rng_state().clone()
saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['trainer']
namespace=ModuleType('rate_api_ra7');namespace.__path__=[str(PKG)];sys.modules[namespace.__name__]=namespace
recipes=importlib.import_module(namespace.__name__+'.recipes')
training=importlib.import_module(namespace.__name__+'.training')
initialization=importlib.import_module(namespace.__name__+'.initialization')
continuous=importlib.import_module(namespace.__name__+'.continuous')
prior_module=importlib.import_module(namespace.__name__+'.particle_prior')

def model(weights):
    layers=[]
    for i in (0,2,4):
        w,b=weights[f'{i}.weight'],weights[f'{i}.bias']
        linear=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(linear)
        linear.in_features,linear.out_features=w.shape[1],w.shape[0]
        linear.weight=torch.nn.Parameter(w.clone());linear.bias=torch.nn.Parameter(b.clone())
        layers.append(linear)
        if i!=4:layers.append(torch.nn.LeakyReLU(.2))
    result=torch.nn.Sequential(*layers)
    # These are saved trained parameters. Preserve their declared initialization
    # metadata so the production factory's only_new path does not reinitialize.
    for p in result.parameters():setattr(p,initialization._MARK,True)
    return result

def prior():
    p=prior_module.ParticlePrior.__new__(prior_module.ParticlePrior)
    torch.nn.Module.__init__(p)
    p.z=torch.nn.Parameter(saved['models']['prior']['z'].clone())
    return p

module_ast=ast.parse((PKG/'training.py').read_text())
cls=next(n for n in module_ast.body if isinstance(n,ast.ClassDef) and n.name=='GANTrainer')
ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
def writes(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr==name for t in node.targets for n in ast.walk(t))
start=next(i for i,n in enumerate(ctor.body) if writes(n,'opt_g'))
end=next(i for i,n in enumerate(ctor.body) if writes(n,'roles'))
section=ast.Module(body=deepcopy(ctor.body[start:end+1]),type_ignores=[])
code=compile(ast.fix_missing_locations(section),str(PKG/'training.py'),'exec')
section_sha=hashlib.sha256(ast.dump(section,include_attributes=False).encode()).hexdigest()
base=recipes.Recipe(**saved['recipe'])
proposal=base.replace(lr=.0010625,prior_lr_mult=8.,d_lr_mult=4.)
assert sorted(k for k in base.to_dict() if base.to_dict()[k]!=proposal.to_dict()[k])==['d_lr_mult','lr','prior_lr_mult']
profiles=[];noise_outputs=[]
for name,recipe in [('base',base),('proposal',proposal)]:
    self=training.GANTrainer.__new__(training.GANTrainer)
    self.recipe=recipe;self.G=model(saved['models']['G']);self.D=model(saved['models']['D']);self.prior=prior();self.device=torch.device('cpu');self.dtype=torch.float32;self.optimizer_options=dict(saved['optimizer_options'])
    values_before=[p.detach().clone() for module in [self.G,self.D,self.prior] for p in module.parameters()]
    env=dict(self=self,recipe=recipe,nn=torch.nn,torch=torch,math=math,deepcopy=deepcopy)
    # Execute the exact original contiguous constructor section, not a rewritten
    # parameter-group implementation. It has no stream/birth or optimizer step.
    exec(code,env,env)
    self.lr_settle=continuous.StationarityLR((self.opt_g,self.opt_d),prior_param=self.prior.z,release_rule=recipe.table_release_rule)
    assert self.roles==[['generator','prior','generator'],['critic']]
    assert all(torch.equal(p,q) for p,q in zip(values_before,[p for module in [self.G,self.D,self.prior] for p in module.parameters()]))
    assert len(self.opt_g.state)==len(self.opt_d.state)==0
    self.controller=SimpleNamespace(mobility=saved['controller']['mobility'])
    with torch.no_grad():self.log_output_sigma.copy_(saved['output_noise']['log_sigma'])
    noise=[]
    for label,scale_values,noise_base in [('saved_open',[t['s'] for t in saved['lr_settle'][0]],recipe.output_noise_std),('all_settled',[1./64.]*3,recipe.output_noise_std),('zero_base',[1.]*3,0.)]:
        for tester,s in zip(self.lr_settle.testers[0],scale_values):tester.s=s
        value=self._output_sigma(noise_base)
        noise.append(dict(case=label,scales=scale_values,value=value))
    noise_outputs.append(noise)
    groups=[[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups] for opt in [self.opt_g,self.opt_d]]
    rates=self.initial_lrs
    if name=='base':assert rates==saved['initial_lrs']
    else:assert rates==[[.0010625,.0085,.0010625],[.00425]]
    profiles.append(dict(name=name,rates=rates,roles=self.roles,param_group_sizes=[[len(g['params']) for g in opt.param_groups] for opt in [self.opt_g,self.opt_d]],groups=groups,optimizer_defaults=[self.opt_g.defaults,self.opt_d.defaults],latent_damping_present=self.opt_g.latent_damping is not None,tester_types=[[type(t).__name__ for t in row] for row in self.lr_settle.testers]))
for optimizer_index in (0,1):
    for group_index,(old,new) in enumerate(zip(profiles[0]['groups'][optimizer_index],profiles[1]['groups'][optimizer_index])):
        factor=.25 if optimizer_index==0 and group_index in (0,2) else 1.
        assert new['lr']==old['lr']*factor
        assert {k:v for k,v in old.items() if k!='lr'}=={k:v for k,v in new.items() if k!='lr'}
assert noise_outputs[0]==noise_outputs[1]
# Given identical relative scales, original stationarity scheduling/observation
# divides each applied group LR by its own base. Include every saved endpoint
# scale and all declared power-of-two levels through2^-16; no tester is observed.
scales=sorted({1.,*(2.**-i for i in range(17)),*(t['s'] for row in saved['lr_settle'] for t in row if t is not None and t['s'] is not None)})
for scale in scales:
    for old_rates,new_rates in zip(profiles[0]['rates'],profiles[1]['rates']):
        for old,new in zip(old_rates,new_rates):assert (old*scale)/old==(new*scale)/new
# D's original max(own_s,.75*prior_s) floor remains unchanged for the same s.
for own in scales:
    for table in scales:
        s=max(own,.75*table)
        assert profiles[0]['rates'][1][0]*s==profiles[1]['rates'][1][0]*s
assert torch.equal(rng,torch.get_rng_state())
assert not torch.cuda.is_initialized()
assert before=={str(p):sha(p) for p in paths}
receipt=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),scope='actual composed RA7 source/config plus fixed saved RA6 optimizer-factory/constructor-section API proof; no training or motion-quality claim',source_sha256=before,constructor_section_ast_sha256=section_sha,package_sha256=composition['package_sha256'],package_bytes_exact_gpu_audited_owner=True,all29_package_sources_verified=True,full_config_exact_except_declared_three_fields=True,config_sha256=sha(NEW_CONFIG),output_noise_same_state_cases=noise_outputs[0],profiles=profiles,changed_recipe_fields={k:[base.to_dict()[k],proposal.to_dict()[k]] for k in ['lr','prior_lr_mult','d_lr_mult']},relative_scale_cases=len(scales),critic_prior_floor_pairs=len(scales)**2,saved_step=saved['completed_steps'],saved_generator_scales=[t['s'] for t in saved['lr_settle'][0]],saved_critic_scale=saved['lr_settle'][1][0]['s'],cpu_rng_unchanged=True,cuda_initialized=False,new_seeds=0,optimizer_updates=0,gradients=0,semantics=['G network and learnable log_sigma base rates quarter.','Prior and D base rates are bit-exact preserved; betas, AMSGrad, eps, row damping and critic laws unchanged.','For identical s, applied/base intrinsic ratios are exact; this does not claim future controller verdicts or trajectories stay equal.','Output-noise formula/floor/mode, EMA serving/rate law and all data/evaluation gates are unchanged.','This prospective rate configuration uses original APIs with no new loss, Jacobian solve, feature pass, cache, statistic or serialized key.','New config requires a fresh declared run; original GANTrainer recipe equality rejects cross-config checkpoint load.'],limits=['No G-step support or quality evaluation; count owner handles the separately declared fixed-input motion probe.','Reduced G+sigma rates can slow adaptation within the unchanged budgets; strict learned toy and grid remain required.'])
(AREA/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status='PASS',base_rates=profiles[0]['rates'],proposal_rates=profiles[1]['rates'],scales=len(scales),source_count=len(before),output=str(AREA/'receipt.json'))))

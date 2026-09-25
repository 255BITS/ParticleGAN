"""CUDA FP32 regression: wrapper must preserve updates and reject wrong rates."""
from pathlib import Path
import sys,json,copy,time,os
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'candidate'),json.loads((R/'runtime.json').read_text())['repo']]
import torch
from types import SimpleNamespace
start=time.monotonic();torch.set_default_device('cuda:0');torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
import mechanism as m
from benchmarks.toy100 import schedule
original_step=schedule.step_with_policy;original_action=schedule.policy_rate_action
class TinyTrainer:
 def __init__(self):
  self.recipe=SimpleNamespace(total_steps=7000,lr_anneal_start=.6,lr_floor=.05)
  self.completed_steps=499
  self.p=[torch.nn.Parameter(torch.tensor([v],dtype=torch.float32)) for v in [.2,.3,.4]]
  self.opt_g=torch.optim.Adam([{'params':[self.p[0]],'lr':.00425},{'params':[self.p[1]],'lr':.0085}],betas=(0.,.999),foreach=False,fused=False)
  self.opt_d=torch.optim.Adam([self.p[2]],lr=.00425,betas=(0.,.999),foreach=False,fused=False)
  self.initial_lrs=[[.00425,.0085],[.00425]]
  for opt in (self.opt_g,self.opt_d):
   for g in opt.param_groups:
    for p in g['params']:opt.state[p]={'step':torch.tensor(499.,dtype=torch.float32),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.full_like(p,.1)}
 def step(self,real,**kwargs):
  # Same optimizer-hook ordering as GANTrainer. Real optimizer steps on CUDA.
  for opt in (self.opt_g,self.opt_d):
   for group in opt.param_groups:group['lr']=.0001
  self.opt_d.zero_grad();(self.p[2].square().sum()).backward();self.opt_d.step()
  m._update_phase(.1)
  self.opt_g.zero_grad();sum(p.square().sum() for p in self.p[:2]).backward();self.opt_g.step()
  self.completed_steps+=1
  return {'step':self.completed_steps}
 def state(self):return copy.deepcopy({'p':[p.detach().clone() for p in self.p],'g':self.opt_g.state_dict(),'d':self.opt_d.state_dict(),'step':self.completed_steps,'controller':m._state})
def reset():
 m._state.update(critic=None,critic_ref=None,pending=False,ema=None,depth=0,clips=[],rate_gain=1.,mix_gain=1.,peak=1.,fast=.1,slow=.1,warm=True,quiet=249,shock=0,closed=False,ever_anchored=False)
 m.receipt['critic_steps']=500
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
opts={'network_lr_horizon_cap':1600,'network_lr_floor':.01}
reset();baseline=TinyTrainer();states=[];original_errors=[]
for i in range(3):
 original_step(baseline,None,**opts);states.append(baseline.state())
 try:original_action(baseline,baseline.completed_steps,**opts)
 except RuntimeError as e:original_errors.append(str(e))
assert len(original_errors)==3
import observation_adapter as adapter
adapter.install();reset();observed=TinyTrainer();actions=[];checks={}
for i in range(3):
 before_rng=torch.cuda.get_rng_state();schedule.step_with_policy(observed,None,**opts)
 assert equal(states[i],observed.state()),'wrapper changed training state'
 assert torch.equal(before_rng,torch.cuda.get_rng_state()),'wrapper changed RNG'
 action=schedule.policy_rate_action(observed,observed.completed_steps,**opts);actions.append(action)
 checks[f'update_{i+1}_state_and_rng_identical']=True
saved=observed.opt_g.param_groups[0]['lr'];observed.opt_g.param_groups[0]['lr']*=1.01
try:schedule.policy_rate_action(observed,observed.completed_steps,**opts)
except RuntimeError:checks['wrong_applied_rate_rejected']=True
else:raise AssertionError('invalid rate accepted')
observed.opt_g.param_groups[0]['lr']=saved
try:schedule.policy_rate_action(observed,observed.completed_steps+1,**opts)
except ValueError:checks['wrong_completed_count_rejected']=True
else:raise AssertionError('invalid count accepted')
try:schedule.policy_rate_action(observed,observed.completed_steps,network_lr_horizon_cap=7000,network_lr_floor=.01)
except RuntimeError:checks['changed_policy_arguments_rejected']=True
else:raise AssertionError('changed policy accepted')
for opt in [observed.opt_g,observed.opt_d]:
 for state in opt.state.values():
  assert all(v.is_cuda and v.dtype==torch.float32 for v in state.values() if torch.is_tensor(v))
checks['all_optimizer_state_cuda_fp32']=True
p=R/'runs'/'observation-adapter-regression.json';row={'candidate':'regression','gate':'rp1_observation_adapter_update_invariance','status':'PASS','seconds':time.monotonic()-start,'metrics':{'checks':checks,'original_errors':original_errors,'observed_actions':actions,'scope':'Three deterministic tiny optimizer updates, not a canonical GAN gate','environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS']}},'artifact':str(p)}
p.write_text(json.dumps(row,indent=2)+'\n')
with (R.parents[2].parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps({'status':'PASS','checks':checks,'artifact':str(p)}))

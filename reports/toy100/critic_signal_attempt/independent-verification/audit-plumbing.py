from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).parent/'repo'
sys.path[:0]=[str(ROOT),str(ROOT/'reports/toy100')]
import torch
from benchmarks.locked_shared.hosts.unipolar import ScaleCritic
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from particlegan.grad_regularizers import GradRegularizer
from selected_h_extension import ConditionalCriticView, ARCHIVED_SIGNAL_POLICY, extended_signal_policy

torch.set_num_threads(1)
options=json.loads((Path(__file__).parent/'one.json').read_text())[0]['options']
torch.manual_seed(123)
critic=ScaleCritic(2,torch.tensor([1.,2.]))
policy=NoisePolicy(.029,.5,1.,400,seed=0,output_noise_warmup=.2)
policy.set_step(0)
critic.noise_policy=policy

def make_callable(owner):
    return lambda z,scale=1.: owner.score(z,scale)
function=make_callable(critic)
rng=torch.get_rng_state().clone(); stream=policy.input_stream.get_state().clone()
view=ConditionalCriticView(function)
assert torch.equal(torch.get_rng_state(),rng)
assert torch.equal(policy.input_stream.get_state(),stream)
assert [id(p) for p in view.parameters()]==[id(p) for p in critic.parameters()]
points=torch.tensor([[.1,.2],[.4,-.3]],requires_grad=True)
with ARCHIVED_SIGNAL_POLICY(options):
    a=function(points); ga=torch.autograd.grad(a.sum(),(points,*critic.parameters()),allow_unused=True)
    policy.input_stream.set_state(stream)
    b=view(points); gb=torch.autograd.grad(b.sum(),(points,*critic.parameters()),allow_unused=True)
assert torch.equal(a,b)
assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(ga,gb))
real=points.detach(); fake=-real
reg=GradRegularizer('a_r1r2',coeff=.6)
# The reference provides only the exact modules() interface to the same callable;
# the candidate then evaluates its original mixup implementation unchanged.
function.modules=view.modules
policy.input_stream.set_state(stream); torch.set_rng_state(rng)
with ARCHIVED_SIGNAL_POLICY(options):
    a,_=reg.penalty(function,real,fake)
    ga=torch.autograd.grad(a,tuple(critic.parameters()),allow_unused=True)
    after_a=policy.input_stream.get_state().clone(); global_a=torch.get_rng_state().clone()
policy.input_stream.set_state(stream); torch.set_rng_state(rng)
with extended_signal_policy(options) as receipt:
    b,_=reg.penalty(function,real,fake)
    gb=torch.autograd.grad(b,tuple(critic.parameters()),allow_unused=True)
    after_b=policy.input_stream.get_state().clone(); global_b=torch.get_rng_state().clone()
assert torch.equal(a,b)
assert all((x is None and y is None) or torch.equal(x,y) for x,y in zip(ga,gb))
assert torch.equal(after_a,after_b) and torch.equal(global_a,global_b)
assert receipt['host_extension']['conditional_callable_views']==1
v={'status':'PASS','fixture_only_not_training':True,'forward_equal':True,'input_and_parameter_gradients_equal':True,'penalty_value_and_gradients_equal':True,'global_and_noise_rng_after_equal':True,'construction_does_not_draw_rng':True,'parameter_objects_identical':True,'extension_sha256':hashlib.sha256((ROOT/'reports/toy100/selected_h_extension.py').read_bytes()).hexdigest()}
(Path(__file__).parent/'plumbing-audit.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(v))

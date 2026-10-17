"""Tiny owned-state observational extra-pass control, including exceptions."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
HERE=Path(__file__).resolve().parent
ROOT=Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PACKAGE=HERE/'pkg-PAIR-AVERAGE'
MAIN=ROOT/'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(ROOT/'integration/review/training-regression/post-ra4-quality'))
from particlegan import feature_cells as module
from measure_saved_utils import tensor_state_hash
from contract_utils import fixture,ORIGINAL
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=HERE/'neutrality-receipt.json'
if out.exists():raise SystemExit('Preserve existing receipt.')
sources={str(p):sha(p) for p in [*sorted(PACKAGE.rglob('*.py')),Path(__file__),HERE/'PROTOCOL.md',HERE/'contract_utils.py',MAIN,ORIGINAL]}
path=ROOT/'validation-cb64-ra7/learned/training/toy/CB64-RA7/checkpoint-2000.pt'
inputs={str(path):sha(path)}
global_rng=torch.get_rng_state().clone()
state=torch.load(path,map_location='cpu',weights_only=False)['trainer'];original=tensor_state_hash(state)
namespace=dict(torch=torch,F=F)
defs=[n for n in ast.parse(MAIN.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='forward']
exec(compile(ast.Module(body=defs,type_ignores=[]),str(MAIN),'exec'),namespace)
forward=namespace['forward'];w=state['models']
with torch.no_grad():
    r=forward(state['birth_death']['reservoir'],w['D'],head=True).double()
    q=forward(forward(w['prior']['z'],w['G']),w['D'],head=True).double()
snapshot=module.FeatureCellSnapshot.fit(r,generator=torch.Generator().set_state(state['cpu_rng']),cells=64,rank=8,chunk=256)
snapshot.cache_queries(q)
trainer,bd=fixture(state,module);bd.snapshot=snapshot;trainer.completed_steps=1999
trainer._STREAMS=('latent_generator','penalty_generator','eval_generator','noise_generator')
for name in trainer._STREAMS:setattr(trainer,name,torch.Generator().set_state(state['cpu_rng']))
for root in (trainer.ema_G,trainer.D):
    root.register_buffer('mutable_marker',torch.ones(2))
    root.register_buffer('rebound_marker',torch.ones(2))
    root.register_buffer('none_marker',None)
    list(root.parameters())[0].grad=torch.ones_like(list(root.parameters())[0])
    root.train();root[1].eval()


def capture():
    return dict(rng=torch.get_rng_state().clone(),
        streams={name:getattr(trainer,name).get_state().clone() for name in trainer._STREAMS},
        reaction_stream=bd.stream.get_state().clone(),
        models={name:deepcopy(getattr(trainer,name).state_dict()) for name in ('G','D','ema_G')},
        modes=[[m.training for m in root.modules()] for root in (trainer.G,trainer.D,trainer.ema_G)],
        gradients=[[None if p.grad is None else p.grad.clone() for p in root.parameters()] for root in (trainer.G,trainer.D,trainer.ema_G)],
        nonpersistent=[[sorted(m._non_persistent_buffers_set) for m in root.modules()] for root in (trainer.D,trainer.ema_G)],
        live=trainer.prior.z.detach().clone(),average=trainer.ema_prior.z.detach().clone(),
        optimizer=deepcopy(trainer.opt_g.state[trainer.prior.z]),history=trainer.opt_g.latent_history.clone())


def mutate_owned(root,inputs):
    root.mutable_marker.add_(1)
    root.rebound_marker=root.rebound_marker+2
    root.none_marker=torch.ones(1)
    root.register_buffer('temporary_marker',torch.ones(1),persistent=False)
    torch.rand(())
    for name in trainer._STREAMS:torch.rand((),generator=getattr(trainer,name))
    torch.rand((),generator=bd.stream)
    first,second=list(root.parameters())[:2]
    first.grad.add_(3)
    second.grad=torch.ones_like(second)


checks={}
original_buffers={id(m):(dict(m._buffers),set(m._non_persistent_buffers_set)) for root in (trainer.D,trainer.ema_G) for m in root.modules()}
hooks=[root.register_forward_pre_hook(mutate_owned) for root in (trainer.D,trainer.ema_G)]
before=capture()
bd._record_paired_average(trainer,snapshot)
checks['extra_eval_owned_state_exact']=tensor_state_hash(before)==tensor_state_hash(capture())
checks['buffer_object_maps_exact']=all(set(m._buffers)==set(original_buffers[id(m)][0]) and all(m._buffers[k] is v for k,v in original_buffers[id(m)][0].items()) for root in (trainer.D,trainer.ema_G) for m in root.modules())
checks['nonpersistent_buffer_metadata_exact']=all(m._non_persistent_buffers_set==original_buffers[id(m)][1] for root in (trainer.D,trainer.ema_G) for m in root.modules())
checks['observational_draws_do_not_change_gate']=bd.paired_average['coherent_rows']==985 and bd.paired_average['eligible']
stamp_before=dict(bd.paired_average)
def fail(root,inputs):raise RuntimeError('fixed exceptional eval control')
failure=trainer.ema_G.register_forward_pre_hook(fail)
before=capture();raised=False
try:bd._record_paired_average(trainer,snapshot)
except RuntimeError as error:raised=str(error)=='fixed exceptional eval control'
checks['exception_owned_state_exact']=raised and tensor_state_hash(before)==tensor_state_hash(capture())
checks['exception_keeps_semantic_stamp']=stamp_before==bd.paired_average
failure.remove()
for handle in hooks:handle.remove()
# A supported MLP returning nonfinite coordinates can still capture its head.
# This case must be a conservative false stamp; unrelated custom forward
# exceptions deliberately propagate after restoring owned state.
def nonfinite_output(root,inputs,output):
    value=output.clone();value[0]=float('nan');return value
handle=trainer.ema_G.register_forward_hook(nonfinite_output)
before=capture();bd._record_paired_average(trainer,snapshot);handle.remove()
checks['captured_nonfinite_output_veto']=not bd.paired_average['eligible'] and bd.paired_average['finite_rows']==1020 and bd.paired_average['groups']==0
checks['nonfinite_owned_state_exact']=tensor_state_hash(before)==tensor_state_hash(capture())
checks['checkpoint_tensor_input_unchanged']=original==tensor_state_hash(state)
checks['sources_inputs_unchanged']=sources=={p:sha(Path(p)) for p in sources} and inputs=={p:sha(Path(p)) for p in inputs}
checks['global_rng_unchanged']=torch.equal(global_rng,torch.get_rng_state())
checks['no_cuda']=not torch.cuda.is_initialized()
assert all(checks.values()),[k for k,v in checks.items() if not v]
out.write_text(json.dumps(dict(status='PASS',checks=checks,source_sha256=sources,input_sha256=inputs,
    cpu_only=True,cuda_initialized=False,new_optimizer_steps=0,new_training_steps=0,new_emissions=0,new_seed_experiments=0,
    quality_verdict=None,scope='global CPU RNG, all trainer-owned and reaction streams, registered buffer mappings/values/nonpersistent flags, modes and D/EMA gradients',
    limit='arbitrary unregistered Python state/external RNG is outside the supported owned-state contract'),indent=2)+'\n')
print(json.dumps({'event':'neutrality_contract_complete','status':'PASS','checks':len(checks)}),flush=True)

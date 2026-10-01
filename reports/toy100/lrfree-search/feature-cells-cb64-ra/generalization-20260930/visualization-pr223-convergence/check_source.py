"""Source/scorer law and actual CPU sampling-state guard contracts; no training."""
import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent
import run_capture as run

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--variant',choices=['E22','Atlas'],required=True)
args=ap.parse_args()
contract=run.source_contract(args.variant)
sys.path.insert(0,str(run.PACKAGES[args.variant]));sys.path.insert(1,str(run.HOSTS))
import torch
assert not torch.cuda.is_initialized()
def reject_cuda(*a,**k):raise AssertionError('CUDA forbidden in source/CPU closure')
torch.cuda._lazy_init=reject_cuda
torch.set_num_threads(1)
from torch import nn
from particlegan import GANTrainer, get_recipe, init
from particlegan.particle_prior import ParticlePrior
from native100 import problems,toy_models
import instrumentation as support

with torch.random.fork_rng(devices=[]):
    torch.manual_seed(1234)
    prior=ParticlePrior(20000,2)
    with torch.no_grad():prior.z.uniform_(-5.,5.)
    G=nn.Linear(2,2)
    with torch.no_grad():G.weight.copy_(torch.eye(2));G.bias.zero_()
    D=toy_models.SimpleMLPDiscriminator(2,128,3,3)
    for layer in D.modules():
        if isinstance(layer,nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)
    init.deterministic_orthogonal_(D,seed=1)
    recipe=get_recipe(**{**json.loads(run.CONFIGS[args.variant].read_text()),'num_particles':20000,'z_dim':2,'batch_size':2048})
    trainer=GANTrainer(recipe,G,D,prior=prior,seed=1234,serial_backward=True,optimizer_options={'foreach':False,'fused':False})
    if args.variant=='Atlas':
        # Resolve only the shape/capability contract. No real sample, optimizer
        # update, chart fit or reaction occurs in this CPU witness.
        trainer.policy._feature_selection.observe_shape(torch.empty(1,2))
    stream=torch.Generator(device='cpu').manual_seed(1234)
    draw_source=ast.get_source_segment(run.adapted(args.variant),next(
        n for n in ast.parse(run.adapted(args.variant)).body if isinstance(n,ast.FunctionDef) and n.name=='draw'))
    assert draw_source.count('torch.random.fork_rng(devices=[0])')==1
    draw_source=draw_source.replace('torch.random.fork_rng(devices=[0])','torch.random.fork_rng(devices=[])')
    namespace=dict(torch=torch,trainer=trainer,device='cpu')
    exec('@torch.no_grad()\n'+draw_source,namespace)
    output=HERE/f'cpu-observer-{args.variant.lower()}'
    assert not output.exists();output.mkdir()
    recorder=support.Recorder(trainer,stream,namespace['draw'],lambda step:0.,
        lambda a:torch.tensor([[1.,0.],[0.,1.]]),problems._centers('rotated100',device='cpu',dtype=torch.float32),
        output,args.variant,'full',True,None,None)
    before_rng=torch.get_rng_state().clone()
    records=getattr(trainer.controller,'_latent_application_records',None)
    geometry=getattr(trainer.birth_death,'latent_geometry',None)
    old_entries=None if geometry is None else geometry._entries
    old_work=None if geometry is None else geometry.work
    recorder.capture(0)
    assert torch.equal(before_rng,torch.get_rng_state())
    assert getattr(trainer.controller,'_latent_application_records',None) is records
    if geometry is not None:
        assert geometry._entries is old_entries and geometry.work is old_work
        assert old_work['builds']==0 and not old_entries
    assert trainer.completed_steps==0 and recorder.preservation_checks==1
    assert recorder.frames[0].shape==(4096,2) and recorder.frames[0].dtype.name=='float32'
    # A target-shift event keeps the exact saved cloud and advances no stream.
    recorder._append(recorder.frames[0].copy(),500,0.,'target_shift',torch.from_numpy(recorder.frames[0]))
    assert torch.equal(before_rng,torch.get_rng_state())
    assert (recorder.frames[0]==recorder.frames[1]).all()
    matching=torch.tensor([1.,float('nan'),-0.])
    assert not support.compare(matching,matching.clone())
    assert support.compare(matching,torch.tensor([1.,float('nan'),0.]))
    altered=matching.clone();altered.view(torch.int32)[1]^=1
    assert support.compare(matching,altered)
    assert support.compare(matching,torch.tensor([2.,float('nan'),-0.]))
    assert not torch.cuda.is_initialized()
result=dict(status='CPU_PASS',variant=args.variant,source_contract=contract,
    native_dimensions_and_initialization=True,source_draw_change_for_CPU_witness='fork_rng devices=[] only; CPU tensors and private generator',
    sampling_guard_pass=True,public_semantic_state_identical=True,private_streams_and_global_CPU_rng_identical=True,
    module_modes_parameter_versions_gradients_external_cursor_identical=True,
    lazy_controller_record_object_preserved=True,derived_geometry_original_objects_and_work_preserved=geometry is not None,
    actual_observed_points=4096,target_event_cloud_identical=True,training_updates=0,cuda_initialized=False,
    strict_raw_bytes_NaN_and_signedzero_comparison=True,
    limitation='Actual CUDA window/control parity remains mandatory before full capture.')
support.write_json(HERE/f'CPU-CHECK-{args.variant}.json',result)
print(json.dumps(result,sort_keys=True),flush=True)

"""Independent RP12/RP14/RP15 ring initialization proof. CPU construction only."""
from pathlib import Path
import argparse, ast, hashlib, json, sys, zipfile
from unittest.mock import patch
sys.dont_write_bytecode = True
p=argparse.ArgumentParser(); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
B=a.bundle.resolve(); O=a.output.resolve(); H=B.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hash_bytes=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((B/'bundle-sha256.json').read_text()); declaration=json.loads((B/'declaration.json').read_text())
for rel,wanted in manifest['files'].items():assert sha(B/rel)==wanted,rel
assert {str(p.relative_to(B/'source')):sha(p) for p in (B/'source/particlegan').glob('*.py')}==declaration['package_sha256']
def source_span(text,name):
    nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    assert len(nodes)==1,name
    n=nodes[0]; start=min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])
    return ''.join(text.splitlines(keepends=True)[start-1:n.end_lineno])
frozen=(B/'source/frozen_host.py').read_text(); host_proof={}
archive=Path(declaration['recipe_binding']['original_source_zip'])
assert sha(archive)==declaration['historical_source_zip_sha256']
with zipfile.ZipFile(archive) as z:
    for name,record in declaration['host_source_receipt'].items():
        old=source_span(z.read(record['original_source']).decode(),name)
        fresh=source_span(frozen,name)
        assert old==fresh,name
        assert hash_bytes(fresh.strip().encode())==record['sha256'],name
        host_proof[name]=record['sha256']
port=H.parent/'port-source'/B.name
port_declaration=json.loads((port/'candidate-declaration.json').read_text())
assert sha(port/'candidate-declaration.json')==declaration['candidate_declaration_sha256']
assert declaration['package_sha256']==port_declaration['package_sha256']
expected_recipe={**port_declaration['resolved_recipe'],'num_particles':20000,'z_dim':2,'batch_size':2048}
assert declaration['recipe']==expected_recipe
assert 'particlegan' not in sys.modules
sys.path[:0]=[str(B/'source'),str(B)]
import torch
import particlegan as package
from particlegan import initialization
from particlegan.particle_prior import ParticlePrior
import frozen_host as host
from ring_contract import construct, material
assert not torch.cuda.is_initialized()
assert str(torch.get_default_device())=='cpu'
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
assert str(torch.__version__)==declaration['runtime_expected']['torch']
recipe=package.Recipe(**declaration['recipe'])
assert json.loads(json.dumps(recipe.to_dict()))==declaration['recipe']
assert (recipe.num_particles,recipe.z_dim,recipe.batch_size)==(20000,2,2048)
assert recipe.total_steps is None and recipe.initialization=='batch_feature_zero'
order=[]; init_receipts=[]
G_init=host.SimpleMLPGenerator.__init__; D_init=host.SimpleMLPDiscriminator.__init__; P_init=ParticlePrior.__init__
def wrap_constructor(role,fn):
    def wrapped(self,*args,**kwargs):
        order.append(role)
        return fn(self,*args,**kwargs)
    return wrapped
def wrap_initializer(name,fn):
    def wrapped(*args,**kwargs):
        before=torch.get_rng_state().clone()
        result=fn(*args,**kwargs)
        after=torch.get_rng_state()
        assert torch.equal(before,after),name
        if name=='prior':assert args[1]==1.0
        init_receipts.append(dict(name=name,rng_unchanged=True,key=kwargs.get('key'),std=args[1] if name=='prior' else None))
        return result
    return wrapped
def forbidden(*args,**kwargs):raise AssertionError('CPU proof forbids forward, backward and optimizer steps')
original_initialize=initialization._initialize; original_prior_initialize=initialization._initialize_prior
with patch.object(torch.nn.Module,'_call_impl',forbidden), patch.object(torch.Tensor,'backward',forbidden), patch.object(torch.optim.Adam,'step',forbidden), patch.object(host.SimpleMLPGenerator,'__init__',wrap_constructor('generator',G_init)), patch.object(host.SimpleMLPDiscriminator,'__init__',wrap_constructor('critic',D_init)), patch.object(ParticlePrior,'__init__',wrap_constructor('prior',P_init)), patch.object(initialization,'_initialize',wrap_initializer('network',original_initialize)), patch.object(initialization,'_initialize_prior',wrap_initializer('prior',original_prior_initialize)):
    torch.manual_seed(0)
    initial_rng=torch.get_rng_state().clone()
    trainer=construct(torch,package,host,recipe,'cpu')
    final_rng=torch.get_rng_state().clone()
    first=material(torch,trainer)
    assert order==['generator','critic','prior'],order
    assert host.digest(final_rng)==declaration['expected_initial_fixture']['cpu_rng_sha256']
    assert trainer.completed_steps==trainer.precision.state['updates']==0
    assert trainer.serial_backward is True
    eager_counts=[]
    for optimizer in (trainer.opt_g,trainer.opt_d):
        count=0
        for group in optimizer.param_groups:
            for parameter in group['params']:
                state=optimizer.state[parameter]
                assert {'step','exp_avg','exp_avg_sq'} <= state.keys()
                assert state['step'].shape==() and float(state['step'])==0
                assert all(state[key].device==parameter.device for key in ('step','exp_avg','exp_avg_sq'))
                assert all(not bool(state[key].count_nonzero()) for key in ('exp_avg','exp_avg_sq'))
                count+=1
        eager_counts.append(count)
    assert eager_counts==[9,8],eager_counts
    assert trainer.precision.reference.state_dict().keys()==trainer.D.state_dict().keys()
    assert all(torch.equal(v,trainer.D.state_dict()[k]) for k,v in trainer.precision.reference.state_dict().items())
    for role,seed in [('latent_generator',2),('penalty_generator',3),('eval_generator',4),('noise_generator',5)]:
        assert torch.equal(getattr(trainer,role).get_state(),torch.Generator(device='cpu').manual_seed(seed).get_state())
    expected_z=initialization._qr.qmc_draw(0,(20000,2),('normal',0.,1.),rows_as_points=True).float()
    assert torch.equal(trainer.prior.z,expected_z)
    torch.rand(97)
    second=construct(torch,package,host,recipe,'cpu')
    assert first==material(torch,second),'fresh initialized state depends on constructor RNG'
    assert order==['generator','critic','prior']*2
    assert len(init_receipts)==6 and [r['name'] for r in init_receipts]==['prior','network','network']*2
    assert [r['key'] for r in init_receipts if r['name']=='network']==[0,1,0,1]
    # Same constructor cursor with initialization explicitly disabled; no learner execution.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(initial_rng)
        ordinary=construct(torch,package,host,recipe.replace(initialization=None),'cpu')
        assert torch.equal(torch.get_rng_state(),final_rng),'initializer shifted original constructor RNG consumption'
        assert not torch.equal(ordinary.prior.z,trainer.prior.z)
        assert any(not torch.equal(x,y) for x,y in zip(ordinary.G.parameters(),trainer.G.parameters()))
    assert len(init_receipts)==6
    captured={}
    for role in ('G','D','prior','ema_G','ema_D','ema_prior'):
        module=getattr(trainer,role)
        tensors=list(module.named_parameters())+list(module.named_buffers())
        assert all(t.device.type=='cpu' for _,t in tensors)
        captured[role]=dict(parameters=len(list(module.parameters())),buffers=len(list(module.buffers())),tensors=len(tensors))
    captured['precision_reference']=dict(parameters=len(list(trainer.precision.reference.parameters())),buffers=len(list(trainer.precision.reference.buffers())))
    assert captured['D']['buffers']==captured['ema_D']['buffers']==captured['precision_reference']['buffers']==1
    state_keys=set(trainer.state_dict())
    assert set(first['state'])==state_keys-{'streams','cpu_rng','cuda_rng','device'}
assert not torch.cuda.is_initialized()
imports={}
for name,module in tuple(sys.modules.items()):
    if name=='particlegan' or name.startswith('particlegan.'):
        file=Path(module.__file__).resolve(); rel=str(file.relative_to(B/'source'))
        assert sha(file)==declaration['package_sha256'][rel]
        imports[name]=dict(path=str(file),sha256=sha(file))
result=dict(status='PASS',candidate=declaration['candidate'],manifest_sha256=sha(B/'bundle-sha256.json'),declaration_sha256=sha(B/'declaration.json'),checker_sha256=sha(Path(__file__)),cuda_initialized=False,learner_steps=0,forward_calls=0,backward_calls=0,optimizer_steps=0,recipe=declaration['recipe'],checks={k:True for k in ('all_initial_tensors_repeat_without_seed_reset','initializer_rng_neutral','original_constructor_order','original_prior_std1','all_buffers_and_checkpoint_state_captured')},constructor_order=order,initializer_calls=init_receipts,constructor_rng_sha256=host.digest(final_rng),all_initial_material=first,captured_modules=captured,eager_counts=eager_counts,host_source_proof=host_proof,imported_package=imports,runtime=dict(python=sys.version,torch=str(torch.__version__),device='cpu'),scope='Initialization and immutable source only. Actual CUDA byte equality is mandatory in worker before any updates. No quality or checkpoint continuation claim.')
O.parent.mkdir(parents=True,exist_ok=True); assert not O.exists(); O.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(status=result['status'],candidate=result['candidate'],proof=str(O),sha256=sha(O),cuda_initialized=False,learner_steps=0)))

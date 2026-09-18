"""Unchanged continuation replays the original trainer's complete state."""
import hashlib
import os
import pytest
import torch
from experiments import train_cifar_ae_balance as old
from experiments import train_cifar_ae_scaling as new


@pytest.fixture
def deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
def test_control_preserves_original_full_state(tmp_path, monkeypatch, deterministic_algorithms):
    for trainer in (old, new):
        factory = trainer.build_models
        def deterministic(cfg, factory=factory):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            g, d, e = factory(cfg)
            for head, kernel in zip(d.critic.project, (4, 2, 1)):
                head[5] = torch.nn.AvgPool2d(kernel)
            return g, d, e
        monkeypatch.setattr(trainer, 'build_models', deterministic)
    cfg = {**old.DEFAULTS, 'generator_arch': 'cnn', 'recon_grad': 'encoder_only',
           'arm': 'bounded', 'z_dim': 8, 'width': 8, 'num_particles': 16,
           'batch_size': 2, 'steps': 4, 'reg_every': 4, 'eval_interval': 4,
           'log_interval': 4, 'eval_samples': 0, 'final_samples': 0,
           'recon_samples': 4, 'eval_batch_size': 4, 'out_dir': str(tmp_path / 'parent')}
    old.train(cfg)
    parent = tmp_path / 'parent/checkpoint.pt'
    cfg.update(steps=8, resume_checkpoint=str(parent), resume_sha256=hashlib.sha256(parent.read_bytes()).hexdigest())
    old.train({**cfg, 'out_dir': str(tmp_path / 'old')})
    new.train({**new.DEFAULTS, **cfg, 'out_dir': str(tmp_path / 'new')})
    a, b = [torch.load(tmp_path / name / 'checkpoint.pt', map_location='cpu', weights_only=False) for name in ('old', 'new')]
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a: compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        else:
            assert a == b
    for key in ('G', 'D', 'E', 'prior', 'ema_G', 'ema_E', 'ema_prior', 'optimizer_g', 'optimizer_d', 'rng', 'torch_rng', 'cuda_rng'):
        compare(a[key], b[key])


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
@pytest.mark.parametrize('factor', [8, 16])
def test_actual_parent_expansion_and_resume(tmp_path, factor):
    import copy
    path = new.ROOT/'runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    original = copy.deepcopy(ck)
    cfg = {**new.DEFAULTS, **ck['config'], 'expansion_factor': factor}
    prior = new.MoGParticlePrior(1024, 64, sigma_rel=.025).cuda()
    ep = copy.deepcopy(prior)
    ref = copy.deepcopy(prior); ref.load_state_dict(ck['ema_prior'])
    audit = new.prepare_expansion(ck, prior, ep, cfg)
    for key in ('G','D','E','ema_G','ema_E'):
        for k,v in original[key].items():
            if torch.is_tensor(v): assert torch.equal(v,ck[key][k])
    assert torch.equal(ck['prior']['z'], original['prior']['z'].repeat_interleave(factor,0))
    assert torch.equal(ck['ema_prior']['z'], original['ema_prior']['z'].repeat_interleave(factor,0))
    for pid,state in original['optimizer_g']['state'].items():
        expanded = pid == original['optimizer_g']['param_groups'][2]['params'][0]
        for k,v in state.items():
            expected = v.repeat_interleave(factor,0) if expanded and k != 'step' else v
            assert torch.equal(expected,ck['optimizer_g']['state'][pid][k])
    a,b = new.rng(34002),new.rng(34002)
    za,ia = ref.sample(8192,a); zb,ib = ep.sample(8192,b)
    assert torch.equal(ia,ib//factor) and torch.equal(a.get_state(),b.get_state())
    torch.testing.assert_close(za,zb,rtol=0,atol=2e-6)
    assert len(ib.unique()) > 3000
    # Independent rows and child stream survive serialization.
    prior.track_exposure=True
    z,ids=prior.sample(64,new.rng(22)); z.square().sum().backward()
    assert (prior.z.grad.reshape(1024,factor,64).std(1).abs()>0).any()
    saved=copy.deepcopy(prior.state_dict())
    other=new.MoGParticlePrior(1024,64,sigma_rel=.025).cuda()
    new.activate_expansion(other,factor,0,True);other.load_state_dict(saved)
    a,b=new.rng(123),new.rng(123)
    za,ia=prior.sample(128,a);zb,ib=other.sample(128,b)
    assert torch.equal(za,zb) and torch.equal(ia,ib)
    g,_,_=new.build_models(cfg);g=g.cuda().eval();g.load_state_dict(original['ema_G'])
    a,b=new.rng(34002),new.rng(34002)
    za,_=ref.sample(128,a);zb,_=ep.sample(128,b)
    with torch.no_grad():
        xa,xb=g(za),g(zb)
    error=float((xa-xb).abs().max())
    assert error < 1e-4, error
    audit['max_initial_image_error']=error
    new.write_json(tmp_path/'identity_audit.json',audit)
    print('IDENTITY_AUDIT',audit)


@pytest.mark.skipif(os.environ.get('RUN_CUDA_IMAGE_TESTS') != '1', reason='real CIFAR CUDA test')
@pytest.mark.parametrize('factor', [8, 16])
def test_expanded_full_resume_and_e_only_gradients(tmp_path, monkeypatch, deterministic_algorithms, factor):
    factory = new.build_models
    def deterministic(cfg):
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        g,d,e=factory(cfg)
        for head,kernel in zip(d.critic.project,(4,2,1)):head[5]=torch.nn.AvgPool2d(kernel)
        return g,d,e
    monkeypatch.setattr(new,'build_models',deterministic)
    cfg={**new.DEFAULTS,'generator_arch':'cnn','recon_grad':'encoder_only','arm':'bounded',
         'z_dim':8,'width':8,'num_particles':16,'batch_size':2,'steps':4,'reg_every':4,
         'eval_interval':2,'log_interval':2,'eval_samples':0,'final_samples':0,'recon_samples':4,
         'eval_batch_size':4,'out_dir':str(tmp_path/'parent')}
    new.train(cfg)
    path=tmp_path/'parent/checkpoint.pt'
    cfg.update(expansion_factor=factor,resume_checkpoint=str(path),resume_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    new.train({**cfg,'steps':8,'out_dir':str(tmp_path/'whole')})
    new.train({**cfg,'steps':6,'out_dir':str(tmp_path/'part')})
    path=tmp_path/'part/checkpoint.pt'
    new.train({**cfg,'steps':8,'out_dir':str(tmp_path/'resumed'),'resume_checkpoint':str(path),
               'resume_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    a,b=[torch.load(tmp_path/name/'checkpoint.pt',map_location='cpu',weights_only=False) for name in ('whole','resumed')]
    def compare(x,y):
        if torch.is_tensor(x):assert torch.equal(x,y)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:compare(x[k],y[k])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for v,w in zip(x,y):compare(v,w)
        else:assert x==y
    for key in ('G','D','E','prior','ema_G','ema_E','ema_prior','optimizer_g','optimizer_d','rng','torch_rng','cuda_rng'):
        compare(a[key],b[key])
    g,d,e=[m.cuda() for m in new.build_models(cfg)]
    prior=new.MoGParticlePrior(16,8).cuda();new.activate_expansion(prior,factor,123,True)
    real=torch.randn(2,3,32,32,device='cuda')
    new.reconstruction_loss(g,e,prior,real,cfg).backward()
    assert all(p.grad is None for p in g.parameters()) and prior.z.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in e.parameters())


@pytest.mark.parametrize('factor', [8,16])
def test_cpu_actual_parent_mapping(factor):
    import copy
    torch.set_num_threads(2)
    ck=torch.load(new.ROOT/'runs/cifar_particle_ae/transgan_scout/cnn_e_only/checkpoint_010000.pt',map_location='cpu',weights_only=False)
    original=copy.deepcopy(ck)
    cfg={**new.DEFAULTS,**ck['config'],'expansion_factor':factor}
    prior=new.MoGParticlePrior(1024,64,sigma_rel=.025)
    ep=copy.deepcopy(prior);ref=copy.deepcopy(prior);ref.load_state_dict(ck['ema_prior'])
    audit=new.prepare_expansion(ck,prior,ep,cfg)
    assert prior.num_particles==1024*factor
    for key in ('prior','ema_prior'):
        assert torch.equal(ck[key]['z'],original[key]['z'].repeat_interleave(factor,0))
    a,b=new.rng(1,'cpu'),new.rng(1,'cpu')
    za,ia=ref.sample(2048,a);zb,ib=ep.sample(2048,b)
    assert torch.equal(ia,ib//factor) and torch.equal(a.get_state(),b.get_state())
    torch.testing.assert_close(za,zb,rtol=0,atol=2e-6)
    print('CPU_MAPPING',factor,audit)

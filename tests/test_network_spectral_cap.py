"""CUDA software controls for fixed-scale network spectral clipping."""
from copy import deepcopy
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init
from particlegan.extrapolation import stateless_directions
from particlegan.optim.dualnorm import NormalizedOptimizer, spectral_capped_direction
from experiments.forge.state import state_digest
from benchmarks.toy_audit import gaussian_combined_magnitude as study
from benchmarks.toy_audit.reproducibility import reproducible_execution

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; no CPU fallback')


def test_spectral_cap_limits_small_rank_deficient_and_large_fields():
    scale=.1
    for values in ([.001,0.], [1.,.001], [1.,2.], [0.,0.]):
        gradient=torch.diag(torch.tensor(values, device='cuda:0', dtype=torch.float64))
        expected=torch.diag((torch.tensor(values,device='cuda:0',dtype=torch.float64)/scale).clamp_max(1))
        assert torch.equal(spectral_capped_direction(gradient,scale),expected)
    tiny=torch.tensor([.0003,.0004],device='cuda:0',dtype=torch.float64)
    assert torch.equal(spectral_capped_direction(tiny,scale),tiny/scale)
    large=100*tiny
    assert torch.allclose(spectral_capped_direction(large,scale),large/scale)
    assert torch.allclose(spectral_capped_direction(large*100,scale),large/large.norm())


def test_preview_correction_agree_and_prior_is_unmodified():
    weight=nn.Parameter(torch.zeros(3,2,device='cuda:0',dtype=torch.float64))
    bias=nn.Parameter(torch.zeros(3,device='cuda:0',dtype=torch.float64))
    table=nn.Parameter(torch.zeros(4,2,device='cuda:0',dtype=torch.float64))
    opt=NormalizedOptimizer([dict(params=[weight,bias],role='generator'),dict(params=[table],role='prior')],
                            lr=.03,network_update='spectral_capped',network_gradient_scale=.1)
    weight.grad=torch.tensor([[.1,.02],[.01,.03],[.001,.01]],device='cuda:0',dtype=torch.float64)
    bias.grad=torch.tensor([.001,.002,.003],device='cuda:0',dtype=torch.float64)
    table.grad=torch.full_like(table,.00001)
    rows=torch.tensor([1,3],device='cuda:0')
    opt.set_sampled_rows(table,rows)
    before=state_digest(opt.state_dict())
    direction=stateless_directions(opt)
    assert state_digest(opt.state_dict())==before
    opt.step()
    for p in (weight,bias,table):assert torch.allclose(p,-.03*direction[p],atol=1e-16,rtol=1e-14)
    assert torch.equal(table[[0,2]],torch.zeros(2,2,device='cuda:0',dtype=torch.float64))
    assert torch.allclose(direction[table][rows].norm(dim=1),torch.tensor([.9992933927955095]*2,device='cuda:0',dtype=torch.float64))
    assert all('network_update' not in g for g in opt.param_groups if g['role']=='prior')
    wrong=NormalizedOptimizer([dict(params=[weight,bias],role='generator'),dict(params=[table],role='prior')],lr=.03,
                              network_update='spectral_capped',network_gradient_scale=.2)
    before=state_digest(wrong.state_dict())
    with pytest.raises(ValueError,match='network_gradient_scale'):
        wrong.load_state_dict(opt.state_dict())
    assert state_digest(wrong.state_dict())==before


def build_small(mode):
    r=get_recipe('bcap',optimizer_family='dualnorm',game_update=mode,network_update='spectral_capped',
                 prior_update='row_capped',prior_gradient_scale=.001,
                 num_particles=16,z_dim=2,batch_size=8,total_steps=8)
    g=nn.Sequential(nn.Linear(2,8),nn.LeakyReLU(.2),nn.Linear(8,1)).cuda()
    d=nn.Sequential(nn.Linear(1,8),nn.LeakyReLU(.2),nn.Linear(8,1)).cuda()
    init.deterministic_orthogonal_(g);init.deterministic_orthogonal_(d)
    return GANTrainer(r,g,d,serial_backward=True)


@reproducible_execution
def resume_probe(*,device):
    a=build_small('extrapolation_from_past');b=build_small('simultaneous')
    initial=a.state_dict();other=deepcopy(initial)
    other['recipe']=b.recipe.to_dict();other.pop('extrapolation');b.load_state_dict(other)
    real=torch.linspace(1,3,8,device=device)[:,None]
    a.step(real);b.step(real)
    assert state_digest(a.state_dict()['models'])==state_digest(b.state_dict()['models'])
    for _ in range(2):a.step(real)
    saved=a.state_dict()
    for _ in range(3):a.step(real)
    restored=build_small('extrapolation_from_past');restored.load_state_dict(saved)
    for _ in range(3):restored.step(real)
    assert state_digest(a.state_dict())==state_digest(restored.state_dict())


def test_first_joint_update_and_cuda_checkpoint_resume():resume_probe(device='cuda:0')


@pytest.mark.parametrize('task_id',['gaussian1d_acquisition','ring16_acquisition'])
def test_matched_current_protocol(task_id):
    matched_probe(task_id,device='cuda:0')


@reproducible_execution
def matched_probe(task_id,*,device):
    protocol=study.declaration();states=[]
    for arm in study.ARMS:
        context,trainer,_=study.build(arm,task_id,device)
        assert study.initial_proof(context,task_id,protocol)['matched']
        assert trainer.prior.z.device.type=='cuda'
        assert trainer.recipe.network_update=='spectral_capped'
        assert trainer.recipe.network_gradient_scale==.1
        assert trainer.opt_g.param_groups[1]['algorithm']=='row_capped'
        states.append(context.state_dict()['trainer'])
    for key in ('models','streams','initial_lrs','optimizers'):
        assert len({state_digest(s[key]) for s in states})==1


def test_legacy_defaults_and_unsupported_rules():
    assert 'network_update' not in get_recipe('bcap').to_dict()
    assert 'network_gradient_scale' not in get_recipe('bcap').to_dict()
    for overrides in (dict(optimizer_family='adam'),dict(optimizer_momentum=.5),dict(network_gradient_scale=0)):
        with pytest.raises(ValueError):get_recipe('bcap',**{'optimizer_family':'dualnorm','network_update':'spectral_capped',**overrides})


def test_cuda_required_before_spend(tmp_path):
    with pytest.raises(ValueError,match='requires CUDA'):study.execute(tmp_path/'absent',device='cpu')
    assert not (tmp_path/'absent').exists()


def test_unchanged_default_optimizer_matches_pinned_source():
    import subprocess
    source=subprocess.check_output(['git','show','6e3d8227:particlegan/optim/dualnorm.py'],cwd=study.ROOT,text=True)
    namespace={'__name__':'particlegan.optim._archived_probe','__package__':'particlegan.optim'}
    exec(compile(source,'pinned-default-optimizer','exec'),namespace)
    classes=(namespace['NormalizedOptimizer'],NormalizedOptimizer)
    weights=[];optimizers=[]
    for cls in classes:
        weight=nn.Parameter(torch.eye(2,device='cuda:0'))
        table=nn.Parameter(torch.zeros(4,2,device='cuda:0'))
        opt=cls([dict(params=[weight],role='generator'),dict(params=[table],role='prior')],lr=.012)
        weights.append((weight,table));optimizers.append(opt)
    for _ in range(3):
        for (weight,table),opt in zip(weights,optimizers):
            weight.grad=torch.tensor([[1.,2.],[3.,1.]],device='cuda:0')
            table.grad=torch.ones_like(table)
            opt.set_sampled_rows(table,torch.tensor([1,3],device='cuda:0'))
            opt.step()
        assert all(torch.equal(a,b) for a,b in zip(*weights))
        assert state_digest(optimizers[0].state_dict())==state_digest(optimizers[1].state_dict())


def test_forge_scale_activity_matches_network_rule():
    from experiments.forge.techniques import recipe_field_active,validate_same_technique
    base=get_recipe('bcap',optimizer_family='dualnorm')
    trial=base.replace(network_update='spectral_capped')
    assert not recipe_field_active('network_gradient_scale',base)
    assert recipe_field_active('network_gradient_scale',trial)
    with pytest.raises(ValueError,match='technique'):validate_same_technique(base,trial)
    validate_same_technique(trial,trial.replace(network_gradient_scale=.2))

"""Independent dense-kernel oracle and public optimizer checkpoint checks."""
from copy import deepcopy

import pytest
import torch

from particlegan import Recipe
from particlegan.optim.constraint_geometry import constraint_geometry_backward
from particlegan.optim.sample_force import sample_force_filter


def test_dense_kernel_oracle_preserves_direct_parameter_loss():
    # Deliberately coupled, unequal output sensitivities; full rank, no RNG.
    jacobian=torch.tensor([[3.,1.,0.],[1.,2.,1.],[0.,1.,.2]],dtype=torch.float64)
    p=torch.nn.Parameter(torch.tensor([.4,-.3,.2],dtype=torch.float64))
    output=(jacobian@p).reshape(3,1);target=torch.tensor([[0.],[1.],[-.5]],dtype=torch.float64)
    direct=.17*p.square().sum();loss=.5*(output-target).square().sum()+direct
    force=(output-target).flatten().detach();gradient=jacobian.T@force
    ridge=float(gradient.square().sum()/force.square().sum())
    filtered=torch.linalg.solve(jacobian@jacobian.T+ridge*torch.eye(3,dtype=torch.float64),ridge*force)
    correction,stats=sample_force_filter(loss,(output,),(p,))
    expected=jacobian.T@(filtered-force)
    assert torch.allclose(correction[0],expected,rtol=1e-9,atol=1e-11)
    raw=torch.autograd.grad(loss,p)[0]
    assert torch.allclose(raw+correction[0],jacobian.T@filtered+.34*p.detach(),rtol=1e-9,atol=1e-11)
    assert stats['iterations']<=4 and stats['relative_residual']<1e-10


def recipe():
    return Recipe(name='synthetic-force-check',optimizer_family='dualnorm',optimizer_smoothing=.001,
                  reg_arm='b_cap',constraint_geometry_mode='sample_force',prior_reg=0.,latent_damping_max_rate=0.,reg_anchor_weight=0.,d_guard_ratio=0.,direct_particle_gain=False)


def update(p,opt):
    opt.zero_grad();out=torch.stack((p*3,p*.2)).reshape(2,-1)
    loss=(out-1).square().mean()
    constraint_geometry_backward(loss,opt,(loss,),sample_outputs=(out,));opt.step()


def test_public_factory_checkpoint_restore_and_missing_consumer():
    p=torch.nn.Parameter(torch.tensor([.2,-.1],dtype=torch.float64));opt=recipe().make_generator_optimizer([p])
    update(p,opt);saved=deepcopy(opt.state_dict());value=p.detach().clone()
    update(p,opt);expected=p.detach().clone();expected_state=deepcopy(opt.state_dict())
    q=torch.nn.Parameter(value);restored=recipe().make_generator_optimizer([q]);restored.load_state_dict(saved)
    update(q,restored)
    assert torch.equal(q,expected)
    assert restored.state_dict()['sample_force']==expected_state['sample_force']
    assert restored.sample_force_stats['filtered_steps']==2
    malformed=deepcopy(saved);malformed['sample_force']['stats']['steps']=-1
    with pytest.raises(ValueError):restored.load_state_dict(malformed)
    q_before=q.detach().clone()
    with pytest.raises(ValueError):constraint_geometry_backward(q.square().sum(),restored,(q.square().sum(),))
    assert torch.equal(q,q_before)


def test_zero_output_force_and_invalid_recipe():
    p=torch.nn.Parameter(torch.tensor([.2],dtype=torch.float64));output=p*2
    correction,stats=sample_force_filter((output-output.detach()).square().sum()+p.square().sum(),(output,),(p,))
    assert torch.equal(correction[0],torch.zeros_like(p)) and stats['iterations']==0
    with pytest.raises(ValueError):Recipe(constraint_geometry_mode='sample_force')


def test_public_trainer_actual_nonlinear_joint_network_prior_update():
    from dataclasses import replace
    from particlegan import GANTrainer,init
    with torch.random.fork_rng(devices=[]):
        generator=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.LeakyReLU(.2,inplace=True),torch.nn.Linear(8,1))
        critic=torch.nn.Sequential(torch.nn.Linear(1,8),torch.nn.LeakyReLU(.2,inplace=True),torch.nn.Linear(8,1))
        init.deterministic_orthogonal_(generator);init.deterministic_orthogonal_(critic)
        actual=replace(recipe(),num_particles=16,batch_size=8,z_dim=2,model='gan',conditioning='scalar',
            encoder_mode='none',loss='non_saturating',total_steps=2,prior_kind='mog')
        trainer=GANTrainer(actual,generator,critic,max_steps=2,seed=0)
        real=torch.linspace(-1,1,8).reshape(8,1)
        for _ in range(2):
            result=trainer.step(real)
            assert torch.isfinite(result['loss_g'])
        assert trainer.opt_g.sample_force_stats['filtered_steps']==2
        assert trainer.opt_g.sample_force_stats['cg_iterations']<=8
        assert trainer.state_dict()['completed_steps']==2

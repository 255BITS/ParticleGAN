"""Public default binding and exact promotion of the measured discriminator."""
import pytest
import torch
from torch import nn

from particlegan import LinearSkipDiscriminator, Recipe, get_recipe
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor


def test_public_discriminator_preserves_research_initialization_and_cap_gradients():
    card = next(c for c in ARCHITECTURES if c['name'] == 'linear_skip_d96_beta5')
    torch.manual_seed(0)
    reference = constructor(card)(hidden_dim=96)
    torch.manual_seed(0)
    promoted = LinearSkipDiscriminator()
    assert reference.state_dict().keys() == promoted.state_dict().keys()
    assert all(torch.equal(v, promoted.state_dict()[k]) for k, v in reference.state_dict().items())
    rng = torch.Generator().manual_seed(0)
    real, fake = torch.randn(16, 2, generator=rng), torch.randn(16, 2, generator=rng)
    for model in (reference, promoted):
        with torch.no_grad():
            model.main.net[-1].weight.mul_(100.)
            model.skip.weight.fill_(2.)
        loss = get_recipe().make_gradient_penalty()(model, real, fake)
        assert loss > 0
        (loss + model(real).mean()).backward()
    assert torch.equal(reference(real), promoted(real))
    for a, b in zip(reference.parameters(), promoted.parameters()):
        assert a.grad is not None and torch.equal(a.grad, b.grad)


def test_default_optimizers_and_losses_bind_the_winning_recipe():
    recipe = get_recipe(num_particles=16)
    g, d = nn.Linear(4, 2), LinearSkipDiscriminator()
    trainer = recipe.make_trainer(g, d)
    assert [group['lr'] for group in trainer.opt_g.param_groups] == [.00425, .0085]
    assert [group['lr'] for group in trainer.opt_d.param_groups] == [.00425]
    assert all(group['betas'] == (0., .99) for opt in (trainer.opt_g, trainer.opt_d) for group in opt.param_groups)
    assert trainer.loss.mode == 'rp' and trainer.loss.loss_type == 'logistic'
    result = trainer.step(torch.randn(8, 2))
    assert torch.equal(result['loss_g'], result['loss_gan'] + .05 * result['prior_regularization'])
    assert Recipe(**trainer.state_dict()['recipe']) == recipe


@pytest.mark.parametrize('name', ['mog', 'ddgan', 'denoising', 'ddgan_mog', 'ae_gan', 'vae_gan', 'ae_ddgan'])
def test_other_domain_recipes_keep_their_selected_core(name):
    recipe = get_recipe(name)
    assert (recipe.reg_coeff, recipe.reg_kappa, recipe.prior_reg, recipe.betas) == (1., 1., 1., (0., .999))


@pytest.mark.parametrize('kwargs', [dict(in_dim=0), dict(hidden_dim=0), dict(n_hidden=0),
                                   dict(fourier=-1), dict(beta=0), dict(beta=float('nan'))])
def test_reference_discriminator_rejects_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        LinearSkipDiscriminator(**kwargs)


def test_historical_stock_comparison_preserves_its_original_settings(tmp_path):
    from benchmarks.locked_shared.grid_study import load_example, resolved_kwargs

    kwargs = resolved_kwargs(load_example(), tmp_path, 'stock', 'cpu', {})
    assert (kwargs['lr'], kwargs['beta1'], kwargs['beta2']) == (.0006, 0., .999)
    assert (kwargs['reg_coeff'], kwargs['reg_kappa'], kwargs['lambda_ep']) == (1., 1., 1.)

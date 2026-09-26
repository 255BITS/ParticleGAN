"""Mathematical contracts for the public, loop-independent PyTorch primitives."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import (
    DDGAN, GANLoss, GaussianPrior, ParticlePrior,
    ParticleRegularizer, Recipe, UCD, get_recipe, learning_rate_scale,
    ucd_labels, ucd_loss,
)
from particlegan.grad_regularizers import GradientPenalty


def test_rp_logistic_loss():
    real = torch.tensor([2., -1.])
    fake = torch.tensor([-.5, .5], requires_grad=True)
    loss = GANLoss()
    torch.testing.assert_close(loss.d_loss(real, fake), F.softplus(fake - real).mean())
    torch.testing.assert_close(loss.g_loss(fake, real), F.softplus(real - fake).mean())
    with pytest.raises(ValueError, match="real_logits"):
        loss.g_loss(fake)


def test_explicit_generators_leave_global_rng_untouched():
    rng = torch.Generator().manual_seed(11)
    state = torch.get_rng_state().clone()
    prior = ParticlePrior(num_particles=12, z_dim=2, generator=rng)
    prior.sample(5, generator=rng)
    gaussian = GaussianPrior(z_dim=2).double()
    draws, indices = gaussian.sample(5, generator=rng)
    process = DDGAN().double()
    process.forward_pair(draws, torch.ones(5, dtype=torch.long), generator=rng)
    assert torch.equal(state, torch.get_rng_state())
    assert draws.dtype == torch.float64 and indices is None
    assert sum(b.numel() for b in gaussian.buffers()) == 0
    assert not list(gaussian.parameters())


def test_penalty_early_form_value_derivative_and_callable_match():
    # Before the critic LR anneals (s == 1): R1 on reals plus a cap on fakes, RMS units.
    discriminator = nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        discriminator.weight.copy_(torch.tensor([[3., 4.]]))
    real = torch.zeros(3, 2, dtype=torch.float64)
    fake = torch.ones_like(real, requires_grad=True)
    regularizer = GradientPenalty()
    penalty, stats = regularizer.penalty(discriminator, real, fake)
    w = torch.tensor([3., 4.], dtype=torch.float64, requires_grad=True)
    expected = 0.5 * (w.square().sum() / 2 + (w.norm() / 2 ** .5 - 1).relu().square())
    torch.testing.assert_close(penalty, expected.detach())
    torch.testing.assert_close(regularizer(discriminator, real, fake), penalty)
    assert stats["applied"] and stats["center"] == 1 and stats["s"] == 1 and stats["phase"] == "a"
    penalty.backward()
    expected.backward()
    torch.testing.assert_close(discriminator.weight.grad, w.grad.unsqueeze(0))
    assert fake.grad is None
    lazy = GradientPenalty(lazy_k=2)
    assert lazy(discriminator, real, fake, step=1).item() == 0
    torch.testing.assert_close(lazy(discriminator, real, fake, step=2), 2 * penalty)


@pytest.mark.parametrize("rows", [0, 1])
def test_small_particle_cloud_returns_differentiable_zero(rows):
    z = torch.ones(rows, 3, dtype=torch.float64, requires_grad=True)
    value = ParticleRegularizer()(z)
    assert value.item() == 0 and value.dtype == z.dtype
    value.backward()
    assert z.grad.shape == z.shape


def test_particle_regularizer_matches_explicit_covariance():
    z = torch.tensor([[0., 1.], [1., 2.], [2., 0.]], dtype=torch.float64)
    cov = torch.cov(z.T)
    expected = F.relu(1 - torch.sqrt(z.var(dim=0) + 1e-4)).mean() + cov[0, 1] ** 2
    torch.testing.assert_close(ParticleRegularizer(weight=.2)(z), .2 * expected)


def test_schedule_forward_rng_and_reverse_posterior():
    process = DDGAN(alpha_bar=[1., .9, .5], dtype=torch.float64)
    x = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    t = torch.tensor([1, 2, 2])
    rng = torch.Generator().manual_seed(19)
    reference = torch.Generator().manual_seed(19)
    prev, xt = process.forward_pair(x, t, rng)
    ab_prev = process.ab[t - 1, None]
    expected_prev = ab_prev.sqrt() * x + (1 - ab_prev).sqrt() * torch.randn(x.shape, dtype=x.dtype, generator=reference)
    expected_xt = process.alpha[t, None].sqrt() * expected_prev + process.beta[t, None].sqrt() * torch.randn(x.shape, dtype=x.dtype, generator=reference)
    torch.testing.assert_close(prev, expected_prev)
    torch.testing.assert_close(xt, expected_xt)
    torch.testing.assert_close(prev[0], x[0])
    clean = x.clone().requires_grad_()
    noise = torch.ones_like(x, requires_grad=True)
    reverse = process.reverse(clean, xt, t, noise)
    alpha, beta = .5 / .9, 1 - .5 / .9
    torch.testing.assert_close(reverse[1:], math.sqrt(.9) * beta / .5 * clean[1:]
                              + math.sqrt(alpha) * .1 / .5 * xt[1:]
                              + math.sqrt(beta * .1 / .5) * noise[1:])
    torch.testing.assert_close(reverse[0], clean[0])
    grads = torch.autograd.grad(reverse.sum(), (clean, noise))
    torch.testing.assert_close(grads[0], process.A[t, None].expand_as(x))
    torch.testing.assert_close(grads[1], process.posterior_var[t, None].sqrt().expand_as(x))


@pytest.mark.parametrize("schedule", [[], [1], [[1, .5]], [1, .5, .6], [1, 0], [1, float("nan")]])
def test_schedule_rejects_invalid_sequences(schedule):
    with pytest.raises(ValueError):
        DDGAN(schedule)


@pytest.mark.parametrize("times", [torch.tensor([0, 1]), torch.tensor([1, 5]), torch.tensor([1., 2.]), torch.tensor([[1], [2]])])
def test_schedule_rejects_invalid_timesteps(times):
    with pytest.raises(ValueError):
        DDGAN().forward_pair(torch.zeros(2, 2), times)


class RecordingLogits(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.linear = nn.Linear(2, heads)
        self.kwargs = None

    def forward(self, x, **kwargs):
        self.kwargs = kwargs
        return self.linear(x)


def test_ucd_network_conditioning_and_joint_score_selection():
    x, xt = torch.ones(2, 2), torch.zeros(2, 2)
    labels, t = torch.tensor([0, 1]), torch.tensor([1, 2])
    network = RecordingLogits(2)
    critic = UCD(network, num_classes=2)
    score, logits = critic(x, labels, xt=xt, t=t)
    assert set(network.kwargs) == {"xt", "t"}
    torch.testing.assert_close(score, logits[torch.arange(2), labels])
    joint_net = RecordingLogits(4)
    joint = UCD(joint_net, 2, target="time_class", num_steps=2)
    score, logits = joint(x, labels, xt=xt, t=t)
    assert set(joint_net.kwargs) == {"xt"}
    torch.testing.assert_close(joint.ucd_labels(labels, t), torch.tensor([0, 3]))
    torch.testing.assert_close(score, logits[torch.arange(2), torch.tensor([0, 3])])
    with pytest.raises(ValueError, match="out of range"):
        joint(x, labels, xt=xt, t=torch.tensor([1, 3]))
    with pytest.raises(ValueError, match="out of range"):
        ucd_labels(torch.tensor([-1, 0]), num_classes=2)


def test_ucd_ce_keeps_both_input_gradient_paths():
    real = torch.tensor([[1., 2.], [3., 1.]], requires_grad=True)
    fake = (-real.detach()).requires_grad_()
    targets = torch.tensor([0, 1])
    loss = ucd_loss(real, fake, targets)
    expected = .02 * (F.cross_entropy(real, targets) + F.cross_entropy(fake, targets))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert real.grad is not None and fake.grad is not None


def test_caller_validated_hot_paths_do_not_convert_tensors_to_python(monkeypatch):
    schedule = DDGAN(validate_args=False)
    critic = UCD(RecordingLogits(4), 2, target="time_class", num_steps=2, validate_args=False)
    x, t, labels = torch.ones(2, 2), torch.tensor([1, 2]), torch.tensor([0, 1])

    def forbid_scalar_sync(self):
        raise AssertionError("unexpected tensor-to-host bounds check")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", forbid_scalar_sync)
        prev, xt = schedule.forward_pair(x, t)
        schedule.reverse(x, xt, t, torch.zeros_like(x))
        critic(prev, labels, xt=xt, t=t)

    # Even caller-validated paths retain inexpensive dtype/shape checks.
    with pytest.raises(ValueError, match="LongTensor"):
        schedule.forward_pair(x, t.float())


def test_recipe_factories_resolve_overrides_and_filter_frozen_parameters():
    recipe = get_recipe(model='ddgan', num_classes=4, conditioning='ucd', z_dim=2, num_particles=8, lr=.001,
                        reg_coeff=.3, prior_reg=.4, betas=[0., .9])
    assert recipe.total_steps == 7000 and recipe.num_classes == 4
    assert recipe.betas == (0., .9)
    assert Recipe(**recipe.to_dict()) == recipe
    prior = recipe.make_prior()
    assert prior.z.shape == (8, 2)
    critic_opt = recipe.make_critic_optimizer(nn.Linear(2, 1), ema_critic=nn.Linear(2, 1))
    assert recipe.make_critic_penalty(critic_opt).regularizer.coeff == .3
    assert recipe.make_prior_regularizer().weight == .4
    generator, discriminator = nn.Linear(2, 2), nn.Linear(2, 1)
    generator.bias.requires_grad_(False)
    opt_g, opt_d = recipe.make_optimizers(generator, discriminator, prior)
    assert [g["lr"] for g in opt_g.param_groups] == [.001, .002]
    assert opt_d.param_groups[0]["lr"] == .001
    assert all(p is not generator.bias for g in opt_g.param_groups for p in g["params"])
    assert opt_g.param_groups[1]["params"] == [prior.z]
    for frozen in (GaussianPrior(2), ParticlePrior(8, 2, learnable=False)):
        assert len(recipe.make_optimizers(generator, discriminator, frozen)[0].param_groups) == 1
    assert recipe.replace(lr=.002).lr == .002 and recipe.lr == .001
    with pytest.raises(TypeError):
        get_recipe(typo=True)
    with pytest.raises(TypeError):
        get_recipe(gan_mode="ra")  # one formulation: no loss/penalty switches
    with pytest.raises(ValueError):
        get_recipe(reg_coeff=-1)


def test_explicit_component_choices_preserve_shared_defaults():
    default = get_recipe()
    assert default == Recipe()
    ddgan = get_recipe(model='ddgan', conditioning='ucd', num_classes=2)
    assert (ddgan.model, ddgan.conditioning, ddgan.num_classes) == ('ddgan', 'ucd', 2)
    assert ddgan.lr == default.lr and ddgan.reg_coeff == default.reg_coeff
    assert Recipe(**ddgan.to_dict()) == ddgan


def test_delayed_cosine_endpoints():
    assert learning_rate_scale(1, 100) == 1
    assert learning_rate_scale(60, 100) == 1
    assert learning_rate_scale(80, 100) == pytest.approx(.525)
    assert learning_rate_scale(100, 100) == .05
    assert learning_rate_scale(200, 100) == .05

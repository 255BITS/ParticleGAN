"""Independent pipeline use cases: no ParticleGAN training loop is involved."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from particlegan import (
    DDGAN, GANLoss, GradientPenalty, ParticlePrior, ParticleRegularizer,
    Recipe, UCD, get_recipe, ucd_loss,
)


def test_regularizer_augments_an_existing_objective_without_a_prior():
    features = nn.Parameter(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, 0.1]]))
    reconstruction = features.square().mean()
    extra = ParticleRegularizer()(features)
    assert features.grad is None
    base_gradient = torch.autograd.grad(reconstruction, features, retain_graph=True)[0]
    (reconstruction + 0.4 * extra).backward()
    assert not torch.allclose(features.grad, base_gradient)
    assert torch.isfinite(features.grad).all()


def test_teacher_student_composes_losses_and_owns_optimizer_steps():
    teacher = nn.Linear(3, 2)
    student = nn.Linear(4, 2)
    critic = nn.Linear(2, 1)
    prior = ParticlePrior(num_particles=16, z_dim=4)
    # An existing pipeline can own all optimizer setup and network definitions.
    optimizer = torch.optim.SGD([*student.parameters(), *prior.parameters()], lr=0.1)
    before_teacher = copy.deepcopy(teacher.state_dict())
    before_student = copy.deepcopy(student.state_dict())
    before_prior = prior.z.detach().clone()
    with torch.no_grad():
        target = teacher(torch.ones(6, 3))
    ids = torch.tensor([0, 2, 4, 6, 8, 10])
    fake = student(prior(ids))
    critic.requires_grad_(False)
    adversarial = GANLoss().g_loss(critic(fake), critic(target).detach())
    loss = F.mse_loss(fake, target) + 0.1 * adversarial
    loss = loss + 0.2 * ParticleRegularizer()(prior(ids))
    assert all(p.grad is None for p in student.parameters())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert not torch.equal(student.weight, before_student["weight"])
    assert not torch.equal(prior.z[ids], before_prior[ids])
    torch.testing.assert_close(prior.z[1::2], before_prior[1::2])
    for key, tensor in teacher.state_dict().items():
        torch.testing.assert_close(tensor, before_teacher[key])


def test_inference_checkpoint_needs_only_generator_and_prior(tmp_path):
    generator = nn.Linear(4, 2).double().eval()
    prior = ParticlePrior(num_particles=12, z_dim=4).double().eval()
    ids = torch.tensor([1, 3, 7, 11])
    with torch.inference_mode():
        expected = generator(prior(ids))
    path = tmp_path / "student.pt"
    torch.save({"generator": generator.state_dict(), "prior": prior.state_dict()}, path)
    restored_g = nn.Linear(4, 2).double().eval()
    restored_p = ParticlePrior(num_particles=12, z_dim=4).double().eval()
    state = torch.load(path, weights_only=True)
    restored_g.load_state_dict(state["generator"])
    restored_p.load_state_dict(state["prior"])
    with torch.inference_mode():
        actual = restored_g(restored_p(ids))
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float64 and not actual.requires_grad


def test_ddgan_inference_runs_without_critic_or_training_state():
    process = DDGAN().double()
    prior = ParticlePrior(num_particles=12, z_dim=4).double()
    generator = nn.Linear(4, 2).double().eval()
    rng = torch.Generator().manual_seed(19)
    global_rng = torch.get_rng_state()
    with torch.inference_mode():
        xt = torch.randn(5, 2, dtype=torch.float64, generator=rng)
        for step in range(process.steps, 0, -1):
            z, _ = prior.sample(len(xt), generator=rng)
            clean = generator(z)
            t = torch.full((len(xt),), step, dtype=torch.long)
            noise = torch.randn(xt.shape, dtype=xt.dtype, generator=rng)
            xt = process.reverse(clean, xt, t, noise)
    torch.testing.assert_close(xt, clean)
    assert torch.isfinite(xt).all()
    assert torch.equal(torch.get_rng_state(), global_rng)


def test_ucd_and_ddgan_losses_fit_an_external_discriminator_update():
    class LogitNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(5, 4)

        def forward(self, x, *, xt, t):
            return self.layer(torch.cat((x, xt, t[:, None].to(x)), dim=1))

    critic = UCD(LogitNetwork(), num_classes=4)
    process = DDGAN()
    generator = nn.Linear(4, 2)
    prior = ParticlePrior(num_particles=16, z_dim=4)
    labels = torch.arange(8) % 4
    t = torch.arange(8) % process.steps + 1
    rng = torch.Generator().manual_seed(11)
    real, xt = process.forward_pair(torch.randn(8, 2), t, rng)
    z, _ = prior.sample(8, generator=rng)
    fake = process.reverse(generator(z), xt, t, torch.randn_like(xt))
    sr, lr = critic(real, labels, xt=xt, t=t)
    sf, lf = critic(fake.detach(), labels, xt=xt, t=t)
    objective = GANLoss().d_loss(sr, sf) + ucd_loss(lr, lf, labels)
    objective += GradientPenalty(kappa=0)(
        lambda x: critic(x, labels, xt=xt, t=t)[0], real, fake.detach())
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
    assert all(p.grad is None for p in generator.parameters())
    assert prior.z.grad is None


def test_loaded_toml_dicts_can_be_passed_directly_to_constructors():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads('''
[particlegan]
name = "denoising"
z_dim = 8
num_particles = 32
num_classes = 3
betas = [0.0, 0.999]
alpha_bar = [1.0, 0.5, 0.01]
[prior]
num_particles = 32
z_dim = 8
[loss]
loss_type = "logistic"
mode = "rp"
''')
    recipe = get_recipe(**config["particlegan"])
    prior = ParticlePrior(**config["prior"])
    loss = GANLoss(**config["loss"])
    assert recipe.model == "ddgan" and recipe.total_steps == 56_000
    assert recipe.z_dim == prior.z_dim == 8
    assert recipe.num_classes == 3
    assert isinstance(recipe.betas, tuple) and isinstance(recipe.alpha_bar, tuple)
    assert loss.mode == "rp"
    assert Recipe(**recipe.to_dict()) == recipe
    with pytest.raises(TypeError):
        get_recipe(**{**config["particlegan"], "typo": True})

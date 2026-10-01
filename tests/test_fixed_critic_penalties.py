"""Fixed public penalties vs pinned v0.7.0 kernels; no scientific training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, MoGParticlePrior, Recipe
from particlegan.grad_regularizers import GradientPenalty
from experiments.forge.api import FormulationContext
from experiments.forge.state import state_digest


@pytest.fixture(scope="module")
def release_penalty():
    fixture = json.loads((Path(__file__).parent / "fixtures/release07-public-api.json").read_text())
    assert fixture["commit"] == "180d18f400335fb295611d624b48a4e072ae3bae"
    entry = fixture["files"]["particlegan/grad_regularizers.py"]
    assert entry["sha256"] == "da03f7653b3d8718fbfa086b0c6eb20bf06943b1f6a8e360b6e14ebd66dabd3f"
    assert hashlib.sha256(entry["source"].encode()).hexdigest() == entry["sha256"]
    module = types.ModuleType("release07_penalty_reference")
    exec(compile(entry["source"], "pinned-v0.7.0/grad_regularizers.py", "exec"), module.__dict__)
    return module.GradientPenalty


class _PolynomialCritic(nn.Module):
    def __init__(self, dimension, spatial_output=False, dtype=torch.float64):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.1, .8, dimension, dtype=dtype))
        self.offset = nn.Parameter(torch.linspace(-.4, .3, dimension, dtype=dtype))
        self.spatial_output = spatial_output

    def forward(self, x):
        x = x.flatten(1)
        score = (self.weight * x.square() + self.offset * x).sum(1)
        if self.spatial_output:
            factors = torch.arange(1, 7, dtype=x.dtype, device=x.device).reshape(1, 2, 3)
            return score[:, None, None] * factors
        return score


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
@pytest.mark.parametrize("shape,spatial", [((4,), False), ((1, 2, 2), False), ((1, 2, 2), True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_kernel_and_critic_gradients_match_release(release_penalty, arm, shape, spatial, dtype):
    dimension = 4
    real = torch.linspace(-1.1, 1.4, 5 * dimension, dtype=dtype).reshape((5,) + shape).requires_grad_()
    fake = real.detach().flip(0).add(.37).requires_grad_()
    reference = _PolynomialCritic(dimension, spatial, dtype)
    current = deepcopy(reference)
    expected, old_stats = release_penalty(arm, coeff=6., kappa=1.25).penalty(reference, real, fake)
    actual, stats = GradientPenalty(coeff=6., kappa=1.25, arm=arm).penalty(current, real, fake)
    assert torch.equal(actual, expected)
    for key in ("applied", "pen", "center"):
        assert stats[key] == old_stats[key]
    assert stats["phase"] == arm and "s" not in stats and "prox" not in stats
    expected.backward()
    actual.backward()
    for left, right in zip(reference.parameters(), current.parameters()):
        assert torch.equal(left.grad, right.grad)
    assert real.grad is None and fake.grad is None  # critic penalty owns detached inputs


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_fixed_zero_gradient_and_lazy_schedule_match_release(release_penalty, arm):
    critic = _PolynomialCritic(4)
    with torch.no_grad():
        for p in critic.parameters():
            p.zero_()
    real, fake = torch.zeros(3, 4, dtype=torch.float64), torch.ones(3, 4, dtype=torch.float64)
    current = GradientPenalty(coeff=2., kappa=0., lazy_k=3, arm=arm)
    reference = release_penalty(arm, coeff=2., kappa=0., lazy_k=3)
    for step in range(1, 7):
        actual, stats = current.penalty(critic, real, fake, step)
        expected, old_stats = reference.penalty(critic, real, fake, step)
        assert torch.equal(actual, expected)
        assert stats["applied"] == old_stats["applied"] == (step % 3 == 0)
        if step % 3:
            assert not actual.requires_grad
        else:
            critic.zero_grad()
            actual.backward()
            assert all(torch.isfinite(p.grad).all() for p in critic.parameters())
    assert current.record.calls == 2 and not current.record.anchor_started


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_fixed_factory_has_no_anchor_phase_or_ema_evaluation(arm):
    recipe = Recipe(reg_arm=arm, reg_every=2)
    critic = _PolynomialCritic(4)
    ema = deepcopy(critic)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=ema)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    before = deepcopy(ema.state_dict())
    def fail_ema(*args):
        raise AssertionError("fixed penalty evaluated the K3P anchor")
    ema.register_forward_pre_hook(fail_ema)
    real, fake = torch.ones(3, 4, dtype=torch.float64), torch.zeros(3, 4, dtype=torch.float64)
    for step in range(1, 5):
        # Exercise a recorded LR ratio that would enter K3P's late phase.
        optimizer.param_groups[0]["lr"] = recipe.lr if step == 1 else recipe.lr * .01
        value = penalty(critic, real, fake)
        stats = penalty.last_stats
        assert stats["applied"] == (step % 2 == 0)
        if stats["applied"]:
            assert stats["phase"] == arm and "s" not in stats and "prox" not in stats
        optimizer.zero_grad()
        (critic(real).mean() + value).backward()
        optimizer.step()
    assert not optimizer.record.anchor_started
    assert "blend_weight" not in penalty.diagnostics()
    assert all(torch.equal(before[k], v) for k, v in ema.state_dict().items())
    # The fixed kernels also work when no EMA module exists at all.
    no_ema = recipe.make_critic_optimizer(critic)
    assert torch.isfinite(recipe.make_critic_penalty(no_ema)(critic, real, fake))


@pytest.mark.parametrize("arm", ["a_r1r2", "b_cap"])
def test_public_mog_recipe_selector_and_checkpoint_resume(arm):
    options = dict(reg_arm=arm, num_particles=8, z_dim=2, batch_size=4, total_steps=4,
                   prior_kind="mog", standardize=False, input_noise_std=0., output_noise_std=0.)
    context = FormulationContext(recipe_overrides={k: v for k, v in options.items()
                                                 if k not in ("prior_kind", "standardize")})
    assert context.recipe.reg_arm == arm
    def build(seed):
        torch.manual_seed(seed)
        recipe = Recipe(**options)
        prior = MoGParticlePrior(num_particles=8, z_dim=2, sigma=.025, standardize=False)
        return GANTrainer(recipe, nn.Linear(2, 2), nn.Linear(2, 1), prior=prior, seed=0)
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 5
    full = build(0)
    full.step(real)
    checkpoint = full.state_dict()
    expected = full.step(real, collect_stats=True)
    expected_state = full.state_dict()
    resumed = build(9)
    resumed.load_state_dict(checkpoint)
    actual = resumed.step(real, collect_stats=True)
    for key in ("loss_d", "loss_g", "penalty"):
        assert torch.equal(actual[key], expected[key])
    assert actual["penalty_stats"] == expected["penalty_stats"]
    assert state_digest(resumed.state_dict()) == state_digest(expected_state)
    with pytest.raises(ValueError, match="recipe"):
        resumed.load_state_dict({**checkpoint, "recipe": {**checkpoint["recipe"], "reg_arm": "k3p"}})


def test_penalty_selector_preserves_existing_positional_coeff_and_rejects_unimplemented_options():
    assert GradientPenalty(2., .7, 3).coeff == 2.
    assert GradientPenalty().arm == "k3p"  # historical low-level kernel default
    assert Recipe().reg_arm is None and Recipe().effective_critic_formulation == "ka2"
    with pytest.raises(ValueError, match="arm"):
        Recipe(reg_arm="e_interp")
    with pytest.raises(TypeError, match="unknown critic penalty"):
        Recipe()._penalty_options(norm="l1")
    with pytest.raises(TypeError, match="unknown critic penalty"):
        Recipe()._penalty_options(method="finite_difference")

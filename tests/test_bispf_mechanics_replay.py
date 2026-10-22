"""Numerical method and full public-API transaction/continuation controls."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, init


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def trainer(mode="secant_extragradient", generator=None):
    recipe = get_recipe("bispf_optimized", num_particles=16, z_dim=2,
                        prior_kind="mog", sigma_rel=.025, standardize=False,
                        batch_size=8, total_steps=8, output_energy_weight=1.,
                        bispf_generator_step=mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        g = generator or nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 2))
        d = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
        prior = recipe.make_prior()
        for i, module in enumerate((g, d, prior)):
            init.deterministic_orthogonal_(module, seed=i)
        return GANTrainer(recipe, g, d, prior=prior, seed=0)


def batch():
    return torch.linspace(-1., 1., 16).reshape(8, 2)


def test_public_step_consumes_original_streams_once_and_notifies_hooks_once():
    baseline, candidate = trainer("ordinary"), trainer()
    calls = dict(g=0, d=0, real=0, prior=0)
    def g_hook(*args):
        calls["g"] += 1
    def d_hook(*args):
        calls["d"] += 1
    def real():
        calls["real"] += 1
        return batch().flip(0)
    original_sample = candidate.prior.sample
    def sampled(*args, **kwargs):
        calls["prior"] += 1
        return original_sample(*args, **kwargs)
    candidate.prior.sample = sampled
    candidate.opt_g.register_step_post_hook(g_hook)
    candidate.opt_d.register_step_post_hook(d_hook)
    base = baseline.step(batch(), generator_real=batch().flip(0))
    output = candidate.step(batch(), generator_real=real)
    assert calls == dict(g=1, d=1, real=1, prior=2)
    assert output["generator_replay"]["field_evaluations"] >= 2
    assert candidate.completed_steps == 1
    assert candidate.opt_g.constraint_geometry_stats["steps"] == 1
    assert all(value["step"] == 1 for value in candidate.opt_g.state.values())
    assert candidate.opt_d.record.observed_steps == 1
    equal(baseline.state_dict()["streams"], candidate.state_dict()["streams"])
    equal(baseline.D.state_dict(), candidate.D.state_dict())
    assert torch.equal(base["loss_g"], output["loss_g"])
    assert any(not torch.equal(a, b) for a, b in zip(baseline.G.parameters(), candidate.G.parameters()))


def test_public_resume_reproduces_complete_state_losses_and_acceptance():
    first = trainer()
    for _ in range(2):
        first.step(batch(), collect_stats=True)
    checkpoint = first.state_dict()
    a = first.step(batch(), collect_stats=True)
    samples_a, state_a = first.sample(16), first.state_dict()
    resumed = trainer()
    resumed.load_state_dict(checkpoint)
    b = resumed.step(batch(), collect_stats=True)
    samples_b, state_b = resumed.sample(16), resumed.state_dict()
    equal(a, b)
    equal(samples_a, samples_b)
    equal(state_a, state_b)


def test_probe_exception_restores_whole_update_including_critic_and_records(monkeypatch):
    model = trainer()
    before = model.state_dict()
    original_forward, calls = model.G.forward, []
    def fail_on_probe(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("fixed probe exception")
        return original_forward(*args, **kwargs)
    monkeypatch.setattr(model.G, "forward", fail_on_probe)
    with pytest.raises(RuntimeError, match="fixed probe exception"):
        model.step(batch())
    equal(before, model.state_dict())
    assert model.policy._phase == "ready"
    assert all(p.requires_grad for p in model.D.parameters())


class BufferedGenerator(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 2))
        self.register_buffer("calls", torch.zeros(()))
    def forward(self, value):
        self.calls.add_(1)
        return super().forward(value)


def test_unknown_buffer_mutation_rejects_and_rolls_back_the_complete_step():
    model = trainer(generator=BufferedGenerator())
    before = model.state_dict()
    with pytest.raises(ValueError, match="model buffer"):
        model.step(batch())
    equal(before, model.state_dict())


def test_unknown_stochastic_forward_rejects_and_restores_rng(monkeypatch):
    model = trainer()
    before = model.state_dict()
    original_forward = model.G.forward
    def stochastic(value):
        return original_forward(value) + torch.rand_like(value) * .01
    monkeypatch.setattr(model.G, "forward", stochastic)
    with pytest.raises(ValueError, match="RNG"):
        model.step(batch())
    equal(before, model.state_dict())


@pytest.mark.parametrize("layer", [nn.Dropout(.2), nn.BatchNorm1d(8)])
def test_known_stochastic_or_running_buffer_model_rejected_before_training(layer):
    with pytest.raises(ValueError, match="stochastic or mutable-buffer"):
        trainer(generator=nn.Sequential(nn.Linear(2, 8), layer, nn.Linear(8, 2)))


def test_fixed_fourier_buffers_allowed_and_untouched():
    # Ordinary frozen Fourier coordinates are not running statistics.
    model = trainer()
    model.D.register_buffer("immutable_coordinates", torch.arange(3.))
    before = model.D.immutable_coordinates.clone()
    model.step(batch())
    assert torch.equal(model.D.immutable_coordinates, before)


def test_final_movement_stats_describe_after_correction_and_observer_is_single():
    model = trainer()
    old = [p.detach().clone() for group in model.opt_g.param_groups for p in group["params"]]
    output = model.step(batch(), collect_stats=True)
    parameters = [p for group in model.opt_g.param_groups for p in group["params"]]
    squared = sum(float((p.detach().double() - value.double()).square().sum())
                  for p, value in zip(parameters, old))
    assert output["generator_replay"]["applied_l2"] == pytest.approx(math.sqrt(squared), abs=1e-14)


def test_global_optimizer_hooks_do_not_observe_probe_clocks():
    import torch.optim.optimizer as implementation
    calls = []
    model = trainer()
    handle = implementation.register_optimizer_step_post_hook(lambda optimizer, *args: calls.append(optimizer))
    try:
        model.step(batch())
    finally:
        handle.remove()
    assert calls == [model.opt_d, model.opt_g]


def test_default_metadata_retains_original_serialized_recipe():
    original = get_recipe("bispf_optimized", standardize=False)
    assert "bispf_generator_step" not in original.to_dict()
    changed = original.replace(bispf_generator_step="secant_extragradient")
    assert changed.to_dict()["bispf_generator_step"] == "secant_extragradient"


def quadratic_fixture(monkeypatch, *, probe="ordinary"):
    generator = nn.Sequential(nn.Linear(2, 2))
    generator[0].weight.requires_grad_(False)
    model = trainer(generator=generator)
    with torch.no_grad():
        generator[0].weight.zero_()
        generator[0].bias.fill_(1e-5)
    anchor = generator[0].bias.detach().clone()
    original_losses, calls = model._generator_losses, []
    def objective(*args):
        result = original_losses(*args)
        parameter = model.G[0].bias
        if not calls or probe == "ordinary":
            potential = .5 * parameter.square().sum()
        elif probe == "nonfinite":
            potential = parameter.sum() * float("nan")
        else:
            # Fixed adversarial probe control; not a learned-quality task.
            potential = -4 * (anchor * parameter).sum()
        calls.append(1)
        result["loss_g"], result["loss_gan"] = potential, potential
        return result
    monkeypatch.setattr(model, "_generator_losses", objective)
    return model


def test_stiff_normalized_quadratic_shortens_and_contracts_with_verified_bound(monkeypatch):
    model = quadratic_fixture(monkeypatch)
    old = model.G[0].bias.detach().clone()
    output = model.step(batch())
    stats = output["generator_replay"]
    assert stats["accepted"] and 0 < stats["applied_scale"] < 1
    assert 1 < stats["trials"] <= 9
    assert stats["consistency_lhs"] <= stats["consistency_rhs"]
    assert model.G[0].bias.detach().norm() < old.norm()


@pytest.mark.parametrize("probe", ["opposed", "nonfinite"])
def test_failed_bounded_search_preserves_G_prior_bits_but_commits_one_clock(monkeypatch, probe):
    model = quadratic_fixture(monkeypatch, probe=probe)
    originals = [p.detach().clone() for group in model.opt_g.param_groups for p in group["params"]]
    output = model.step(batch())
    stats = output["generator_replay"]
    assert stats["trials"] == 9 and not stats["accepted"] and stats["applied_scale"] == 0.
    for current, old in zip((p for group in model.opt_g.param_groups for p in group["params"]), originals):
        assert torch.equal(current.detach().view(torch.int32), old.view(torch.int32))
    assert model.completed_steps == 1
    assert model.opt_g.state[model.G[0].bias]["step"] == 1
    if probe == "nonfinite":
        assert stats["nonfinite_trials"] == 9


def test_constant_zero_force_leaves_exact_zeros_and_uses_no_trial_forward(monkeypatch):
    model = trainer()
    original = model._generator_losses
    def zero(*args):
        result = original(*args)
        potential = sum(p.sum() * 0 for group in model.opt_g.param_groups for p in group["params"])
        result["loss_g"], result["loss_gan"] = potential, potential
        return result
    monkeypatch.setattr(model, "_generator_losses", zero)
    old = [p.detach().clone() for group in model.opt_g.param_groups for p in group["params"]]
    output = model.step(batch())
    assert output["generator_replay"]["trials"] == 0
    for current, before in zip((p for group in model.opt_g.param_groups for p in group["params"]), old):
        assert torch.equal(current.detach().view(torch.int32), before.view(torch.int32))


def test_frozen_opponent_bilinear_has_no_extragradient_circulation_correction(monkeypatch):
    # L(x,y)=xy with the critic/opponent held at y=1 during the G phase.
    # The predictor gradient is unchanged, unlike joint-player extragradient.
    ordinary, candidate = trainer("ordinary"), trainer()
    for model in (ordinary, candidate):
        original = model._generator_losses
        def linear(*args, owner=model, original=original):
            result = original(*args)
            potential = next(owner.G.parameters()).sum()
            result["loss_g"], result["loss_gan"] = potential, potential
            return result
        monkeypatch.setattr(model, "_generator_losses", linear)
    ordinary.step(batch())
    stats = candidate.step(batch())["generator_replay"]
    assert stats["trials"] == 1 and stats["applied_scale"] == 1.
    assert stats["consistency_lhs"] == 0.
    assert stats["protection_delta_l2"] == 0.
    equal(ordinary.G.state_dict(), candidate.G.state_dict())
    equal(ordinary.prior.state_dict(), candidate.prior.state_dict())


def test_nonfinite_anchor_total_loss_rolls_back_before_proposals(monkeypatch):
    model = trainer()
    original, before = model._generator_losses, model.state_dict()
    def nonfinite(*args):
        result = original(*args)
        result["loss_g"] = result["loss_g"] * float("nan")
        return result
    monkeypatch.setattr(model, "_generator_losses", nonfinite)
    with pytest.raises(ValueError, match="finite anchor"):
        model.step(batch())
    equal(before, model.state_dict())


def test_forge_owns_field_and_refuses_unconsumed_component_path():
    from experiments.forge.api import CapabilityError, FormulationContext, task_policy_blockers
    from experiments.forge.boundaries import RECIPE_FIELD_OWNERS, validate_registry
    validate_registry()
    assert RECIPE_FIELD_OWNERS["bispf_generator_step"] == "technique"
    recipe = trainer().recipe.replace(sigma_rel=0.)
    with pytest.raises(CapabilityError, match="does not consume"):
        FormulationContext(recipe_overrides=recipe.to_dict(), execution_path="public_components",
                           prior={"kind": "mog", "sigma": .025, "standardize": False})
    task = dict(adapter="transfer_behavior", execution=dict(execution_path="public_components"))
    assert task_policy_blockers(task, dict(recipe_overrides=recipe.to_dict()))


@pytest.mark.parametrize("delta", [dict(optimizer_momentum=.5), dict(output_noise_std=.1),
                                  dict(constraint_geometry_mode="none"), dict(encoder_mode="ae")])
def test_unsupported_replay_recipe_rejected_before_reservation(delta):
    with pytest.raises(ValueError, match="secant_extragradient"):
        get_recipe("bispf_optimized", bispf_generator_step="secant_extragradient", **delta)

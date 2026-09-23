"""The optional noise scalar is a real G parameter on every custom host."""

import math

import pytest
import torch
from torch import nn

from benchmarks import learned_lr_evaluation as bridge
from benchmarks.locked_shared import mode_hold, trajectory, two_pole
from benchmarks.locked_shared.hosts import (
    ae_gan_hold, cover_leftover, mid_scale_identity, residual_student,
    unipolar, unused_token_hold,
)
from benchmarks.smart_descent import evaluate
from benchmarks.transfer_suite.compare_defaults import optimizer_defaults
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy, wrap_output
from particlegan import get_recipe


def _run(host, policy, monkeypatch):
    if host == "two_pole":
        monkeypatch.setattr(two_pole, "TOY_STEPS", 2)
        return two_pole.train(noise_policy=policy)
    if host == "trajectory":
        monkeypatch.setitem(trajectory.PROTOCOL, "steps", 2)
        return trajectory.train(noise_policy=policy)
    if host == "residual_student":
        monkeypatch.setitem(residual_student.PROTOCOL, "steps", 2)
        return residual_student.train(noise_policy=policy)
    if host == "unipolar":
        return unipolar.run_arm("locked_rpgan", steps=2, noise_policy=policy)
    if host == "ae_gan_hold":
        return ae_gan_hold.train(ae_gan_hold.HoldConfig(
            name="learnable_probe", steps=2, batch=16, n_particles=16,
        ), noise_policy=policy)
    if host == "cover_leftover":
        return cover_leftover.fit_cover_leftover(
            cover_leftover.CoverRecipe(steps=2), noise_policy=policy,
        )
    if host == "unused_token_hold":
        return unused_token_hold.train(
            unused_token_hold.UnusedHoldRecipe(steps=2), noise_policy=policy,
        )
    if host == "mid_scale_identity":
        return mid_scale_identity.run_arm("locked", steps=2, noise_policy=policy)
    if host == "mode_hold":
        return mode_hold.train_mode_hold(
            mode_hold.ModeHoldRecipe(steps=2), noise_policy=policy,
        )
    raise AssertionError(host)


@pytest.mark.parametrize("host", (
    "two_pole", "trajectory", "residual_student", "unipolar",
    "ae_gan_hold", "cover_leftover", "unused_token_hold",
    "mid_scale_identity", "mode_hold",
))
def test_all_legacy_hosts_own_and_update_one_learnable_output_scalar(
    host, monkeypatch,
):
    policy = NoisePolicy(
        0.029, 0.5, 0.5, 2, output_noise_learnable=True,
    )
    _run(host, policy, monkeypatch)
    receipt = policy.receipt()
    assert receipt["output_noise_learnable"] is True
    assert receipt["output_scale_parameter_count"] == 1
    assert receipt["output_scale_optimizer_owned"] is True
    assert receipt["generator_base_parameters"] > 0
    assert (receipt["generator_total_parameters"]
            == receipt["generator_base_parameters"] + 1)
    assert receipt["step_calls"] == 2
    assert len(receipt["output_sigma_effective_step_trace"]) == 2
    assert receipt["output_train_elements"] > 0
    assert receipt["input_train_elements"] > 0
    assert math.isfinite(receipt["output_scale_final"])
    assert receipt["output_scale_final"] > 0
    assert math.isclose(receipt["output_sigma_effective_final_evaluation"],
                        receipt["output_scale_final"], rel_tol=1e-7)
    assert not math.isclose(receipt["output_scale_final"], 0.029, abs_tol=1e-8)
    if host in ("mode_hold", "cover_leftover"):
        assert receipt["output_scale_ema_final"] > 0
        assert receipt["output_sigma_effective_ema_final_evaluation"] > 0


def test_scalar_gradient_belongs_to_generator_only_and_eval_preserves_rng():
    torch.manual_seed(9)
    policy = NoisePolicy(0.029, 0.5, 0.5, 4, output_noise_learnable=True)
    generator = wrap_output(nn.Linear(2, 2), policy)
    discriminator = nn.Linear(2, 1)
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.001)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    policy.register_generator_optimizer(opt_g, opt_d)
    scalar = policy.output_scale.raw_scale
    policy.set_step(0)
    with policy.discriminator():
        fake_d = generator(torch.ones(16, 2))
    discriminator(fake_d).sum().backward()
    assert scalar.grad is None
    opt_g.zero_grad(set_to_none=True)
    fake_g = generator(torch.ones(16, 2))
    fake_g.sum().backward()
    assert scalar.grad is not None and scalar.grad.abs() > 0
    before = float(policy.output_scale().detach())
    opt_g.step()
    assert float(policy.output_scale().detach()) != before
    torch.manual_seed(23)
    global_state = torch.random.get_rng_state().clone()
    input_state = policy.input_stream.get_state().clone()
    with policy.evaluation(2):
        first = generator(torch.ones(16, 2))
        policy.input(torch.ones(16, 2))
    with policy.evaluation(2):
        second = generator(torch.ones(16, 2))
    assert torch.equal(first, second)
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert torch.equal(input_state, policy.input_stream.get_state())


def test_learnable_scalar_multiplies_the_shared_warmup_without_rng_at_zero():
    policy = NoisePolicy(
        0.029, 0.5, 0.5, 10, output_noise_warmup=0.5,
        output_noise_learnable=True,
    )
    base = torch.zeros(8, 2)
    torch.manual_seed(29)
    saved_rng = torch.random.get_rng_state().clone()
    policy.set_step(0)
    assert policy.output(base) is base
    assert torch.equal(saved_rng, torch.random.get_rng_state())
    with torch.no_grad():
        policy.output_scale.raw_scale.add_(0.2)
    learned = float(policy.output_scale().detach())
    policy.set_step(1)
    expected = 0.029 * learned / 0.029 / 5
    assert math.isclose(policy.receipt()["output_sigma_effective_step_trace"][1],
                        expected, rel_tol=1e-6)
    policy.set_step(5)
    assert math.isclose(policy.receipt()["output_sigma_effective_step_trace"][2],
                        learned, rel_tol=1e-6)


def test_two_pole_direct_particles_keep_prior_role_while_scalar_is_generator(
    monkeypatch,
):
    monkeypatch.setattr(two_pole, "TOY_STEPS", 2)
    policy = NoisePolicy(0.029, 0.5, 0.5, 2, output_noise_learnable=True)
    recipe = get_recipe()
    applied = []
    with optimizer_defaults(recipe, applied):
        control = evaluate.FixedControl({"schedule": "cosine"}, 2)
        with bridge.control_host_schedules(control):
            two_pole.train(noise_policy=policy)
    roles = {item["role"]: item for item in applied}
    assert set(roles) == {"d", "prior", "g"}
    assert roles["g"]["parameters"] == 1
    assert roles["prior"]["parameters"] == policy.generator_base_parameters
    assert math.isclose(roles["g"]["lr"], recipe.lr)
    assert math.isclose(roles["prior"]["lr"], recipe.lr * recipe.prior_lr_mult)
    assert policy.receipt()["output_scale_optimizer_owned"]


@pytest.mark.parametrize("learnable,std", [(True, 0.0), (1, 0.029)])
def test_invalid_legacy_learnable_policy_is_rejected(learnable, std):
    with pytest.raises(ValueError, match="output_noise_learnable"):
        NoisePolicy(std, 0.5, 0.5, 2, output_noise_learnable=learnable)

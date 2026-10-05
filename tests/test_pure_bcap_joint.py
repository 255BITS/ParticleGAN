"""Joint BiGAN loss routing through the retained public word fixture.

Only two CPU updates exercise software gradient ownership; these are not
acquisition measurements, qualification runs, or a scientific campaign.
"""
from copy import deepcopy

import pytest
import torch

from benchmarks.toy_audit.api_images import WordFixture
from experiments.forge.api import FormulationContext


LOSSES = ("relativistic", "non_saturating", "hinge", "wasserstein", "least_squares")


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_fixture(loss="relativistic"):
    components = FormulationContext(
        recipe_preset="bcap", recipe_overrides={"loss": loss, "num_particles": 5,
            "z_dim": 2, "batch_size": 256, "total_steps": 20000},
        prior={"kind": "particle_cloud", "sigma": 0., "standardize": False,
               "learnable": True, "exception_reason": "Retained joint word fixture uses point particles."},
        execution_path="public_components", seed=0, device="cpu")
    return WordFixture(device="cpu", seed=0, recipe_name=None,
                       max_steps=2, components=components)


def assert_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_equal(first[name], second[name])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("loss", LOSSES)
def test_joint_word_updates_train_generator_encoder_and_prior_with_a_frozen_critic(loss, monkeypatch):
    fixture = make_fixture(loss)
    assert type(fixture.opt_g) is type(fixture.opt_d) is torch.optim.Adam
    original = fixture.loss.joint_g_loss
    generator_phase = {}
    before = {role: deepcopy(module.state_dict()) for role, module in
              (("generator", fixture.G), ("encoder", fixture.E), ("prior", fixture.prior))}

    def observe_joint_loss(fake_logits, real_logits):
        assert fake_logits.requires_grad and real_logits.requires_grad
        assert not any(parameter.requires_grad for parameter in fixture.D.parameters())
        generator_phase["critic"] = deepcopy(fixture.D.state_dict())
        generator_phase["critic_gradients"] = [parameter.grad.detach().clone()
                                                for parameter in fixture.D.parameters()]
        return original(fake_logits, real_logits)

    monkeypatch.setattr(fixture.loss, "joint_g_loss", observe_joint_loss)

    def before_generator_step(optimizer, args, kwargs):
        assert not any(parameter.requires_grad for parameter in fixture.D.parameters())
        assert_equal(generator_phase["critic"], fixture.D.state_dict())
        for parameter, previous in zip(fixture.D.parameters(), generator_phase["critic_gradients"]):
            assert torch.equal(parameter.grad, previous)
        for module in (fixture.G, fixture.E, fixture.prior):
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in module.parameters())
            assert any(torch.count_nonzero(parameter.grad) for parameter in module.parameters())

    handle = fixture.opt_g.register_step_pre_hook(before_generator_step)
    try:
        for step in (1, 2):
            observed = fixture.step()
            assert fixture.completed_steps == step
            assert all(torch.isfinite(value) for name, value in observed.items() if name != "step")
            assert all(parameter.requires_grad for parameter in fixture.D.parameters())
            for module in (fixture.G, fixture.E, fixture.prior):
                assert all(int(fixture.opt_g.state[parameter]["step"]) == step
                           for parameter in module.parameters())
            assert all(int(fixture.opt_d.state[parameter]["step"]) == step
                       for parameter in fixture.D.parameters())
            assert [group["lr"] for group in fixture.opt_g.param_groups] == [.00425, .00425, .0085]
            assert fixture.opt_d.param_groups[0]["lr"] == .00425
    finally:
        handle.remove()
    for role, module in (("generator", fixture.G), ("encoder", fixture.E), ("prior", fixture.prior)):
        assert any(not torch.equal(tensor, before[role][name])
                   for name, tensor in module.state_dict().items()), role


def test_default_joint_loss_keeps_the_exact_previous_paired_word_trajectory(monkeypatch):
    changed = make_fixture()
    previous = make_fixture()
    # The old retained host called g_loss(fake_joint, real_joint) directly.
    monkeypatch.setattr(previous.loss, "joint_g_loss", previous.loss.g_loss)
    for _ in range(2):
        assert_equal(changed.step(), previous.step())
    assert_equal(changed.state_dict(), previous.state_dict())

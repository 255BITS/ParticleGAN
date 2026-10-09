"""Optimizer observers preserve public training and measure actual update support.

These bounded software checks make no scientific acquisition claim.
"""
from copy import deepcopy
import hashlib
import json
import math

import pytest
import torch
from torch import nn

from experiments.forge.optimizer_diagnostics import OptimizerDiagnostics, attach
from particlegan import GANTrainer, get_recipe
from particlegan.init import deterministic_orthogonal_
from test_dualnorm_optimizers import assert_state_equal


def trainer_for(family):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        recipe = get_recipe("bcap", loss="relativistic", optimizer_smoothing=0.,
                            optimizer_convolution="none", optimizer_family=family,
                            optimizer_momentum=.5 if family == "dualnorm" else 0.,
                            lr=.01, d_lr_mult=1.5, prior_lr_mult=3.,
                            z_dim=2, num_particles=8, batch_size=4, total_steps=6,
                            prior_kind="mog", sigma_rel=.025, standardize=False)
        generator = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Dropout(.2), nn.Linear(4, 2))
        critic = nn.Sequential(nn.Linear(2, 4), nn.BatchNorm1d(4), nn.Tanh(),
                               nn.Dropout(.25), nn.Linear(4, 1))
        prior = recipe.make_prior()
        for module, seed in ((generator, 0), (critic, 1), (prior, 2)):
            deterministic_orthogonal_(module, seed=seed)
        return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                          model_generator=torch.Generator().manual_seed(17))


def batch(step):
    return torch.tensor([[-1., -.5], [-.2, .1], [.4, .8], [1., 1.5]]) + .01 * step


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_environment_adds_no_hooks_or_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS", raising=False)
    trainer = trainer_for("adam")
    before = trainer.state_dict()
    hook_counts = [(len(optimizer._optimizer_step_pre_hooks), len(optimizer._optimizer_step_post_hooks))
                   for optimizer in (trainer.opt_g, trainer.opt_d)]
    forwards = len(trainer.D._forward_pre_hooks)
    assert attach({"G": trainer.opt_g, "D": trainer.opt_d}, trainer.D,
                  tmp_path, [1, 3, 6], prior=trainer.prior) is None
    assert hook_counts == [(len(optimizer._optimizer_step_pre_hooks), len(optimizer._optimizer_step_post_hooks))
                           for optimizer in (trainer.opt_g, trainer.opt_d)]
    assert len(trainer.D._forward_pre_hooks) == forwards
    assert_state_equal(before, trainer.state_dict())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("family", ["adam", "dualnorm"])
def test_six_actual_api_updates_preserve_models_optimizer_and_every_rng_with_diagnostics(family, monkeypatch, tmp_path):
    monkeypatch.setenv("PARTICLEGAN_FORGE_OPTIMIZER_DIAGNOSTICS", "1")
    baseline = trainer_for(family)
    initial = baseline.state_dict()
    expected_results = [baseline.step(batch(step)) for step in range(6)]
    expected = baseline.state_dict()

    observed = trainer_for(family)
    observed.load_state_dict(initial)
    hooks_before = [(len(opt._optimizer_step_pre_hooks), len(opt._optimizer_step_post_hooks))
                    for opt in (observed.opt_g, observed.opt_d)]
    critic_hooks_before = len(observed.D._forward_pre_hooks)
    diagnostics = attach({"G": observed.opt_g, "D": observed.opt_d}, observed.D,
                         tmp_path, [1, 3, 6], prior=observed.prior)
    actual_results = [observed.step(batch(step)) for step in range(6)]
    receipt = diagnostics.receipt()
    # Includes model buffers, gradient histories, global CPU/CUDA RNGs and
    # every checkpointed sampling, prior-noise, penalty and model stream.
    assert_state_equal(expected, observed.state_dict())
    assert_state_equal(expected_results, actual_results)
    path = tmp_path / receipt["path"]
    rows = read_rows(path)
    assert [(row["step"], row["optimizer"]) for row in rows] == [
        (1, "D"), (1, "G"), (3, "D"), (3, "G"), (6, "D"), (6, "G"),
    ]
    assert receipt["rows"] == 6
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["qualification_input"] is False
    assert receipt["sampling_draws_added"] == receipt["optimizer_updates_added"] == 0
    assert hooks_before == [(len(opt._optimizer_step_pre_hooks), len(opt._optimizer_step_post_hooks))
                            for opt in (observed.opt_g, observed.opt_d)]
    assert len(observed.D._forward_pre_hooks) == critic_hooks_before
    critic_rows = [row for row in rows if row["optimizer"] == "D"]
    assert all("critic_input_gradient_mean_real" in row and "critic_input_gradient_mean_fake" in row
               for row in critic_rows)
    generator_rows = [row for row in rows if row["optimizer"] == "G"]
    expected_support_kind = "sampled_indices" if family == "dualnorm" else "nonzero_gradient_support"
    assert all(row["prior"]["row_support_kind"] == expected_support_kind for row in generator_rows)
    if family == "dualnorm":
        assert all(row["prior"]["max_outside_support_row_displacement"] == 0. for row in generator_rows)
        assert observed.opt_g._sampled_rows == {}


def test_prior_measurements_use_actual_unique_sampled_indices_before_the_optimizer_clears_them(tmp_path):
    recipe = get_recipe("bcap", optimizer_family="dualnorm", lr=.02,
                        prior_lr_mult=3., z_dim=2, num_particles=5, standardize=False)
    generator, encoder, critic = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)
    prior = recipe.make_prior()
    opt_g, opt_d = recipe.make_optimizers(generator, critic, prior, encoder=encoder)
    diagnostics = OptimizerDiagnostics({"G": opt_g, "D": opt_d}, critic, tmp_path, [1], prior=prior)
    for parameter in list(generator.parameters()) + list(encoder.parameters()):
        parameter.grad = torch.ones_like(parameter)
    # A full-table regularizer supplies gradients to all five rows, while only
    # the unique observed rows own this row-normalized update.
    prior.z.grad = torch.tensor([[1., 2.], [3., 4.], [8., 9.], [-5., 0.], [6., 7.]])
    opt_g.set_sampled_rows(prior.z, torch.tensor([1, 1, 3]))
    previous = prior.z.detach().clone()
    opt_g.step()
    assert opt_g._sampled_rows == {}
    diagnostics.receipt()
    row, = read_rows(tmp_path / "optimizer-diagnostics.jsonl")
    displacement = (prior.z.detach() - previous).norm(dim=1)
    assert row["prior"] == {
        "row_support_kind": "sampled_indices", "support_rows": 2,
        "mean_support_row_displacement": pytest.approx(float(displacement[[1, 3]].mean())),
        "max_outside_support_row_displacement": 0.,
    }
    assert torch.equal(prior.z[[0, 2, 4]], previous[[0, 2, 4]])
    assert set(row["players"]) == {"G", "prior"}
    g_layers = [layer for layer in row["layers"] if layer["player"] == "G"]
    assert {layer["component"] for layer in g_layers} == {"generator", "encoder"}
    assert row["players"]["G"]["relative_update_sum"] == pytest.approx(
        sum(layer["relative_update"] for layer in g_layers))


def test_spectral_and_input_gradient_probe_preserves_mixed_modes_buffers_grads_and_rng(tmp_path):
    critic = nn.Sequential(nn.Linear(2, 2, bias=False), nn.BatchNorm1d(2),
                           nn.Dropout(.5), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        critic[0].weight.copy_(torch.tensor([[2., 0.], [0., .5]]))
        critic[3].weight.copy_(torch.tensor([[3., 4.]]))
    critic.train()
    critic[1].eval()
    for parameter in critic.parameters():
        parameter.grad = torch.full_like(parameter, .123)
    optimizer = get_recipe("bcap").make_critic_optimizer(critic)
    diagnostics = OptimizerDiagnostics({"D": optimizer}, critic, tmp_path, [1])
    diagnostics.inputs = [batch(0), batch(1)]
    before = deepcopy(critic.state_dict())
    gradients = [parameter.grad.clone() for parameter in critic.parameters()]
    modes = [module.training for module in critic.modules()]
    rng = torch.get_rng_state().clone()
    metrics = diagnostics._critic_metrics()
    assert_state_equal(before, critic.state_dict())
    assert [module.training for module in critic.modules()] == modes
    assert torch.equal(rng, torch.get_rng_state())
    for parameter, gradient in zip(critic.parameters(), gradients):
        assert torch.equal(parameter.grad, gradient)
    assert metrics["critic_weight_spectral_norms"] == pytest.approx([2., 5.])
    assert metrics["critic_spectral_product"] == pytest.approx(10.)
    assert metrics["critic_log_spectral_product"] == pytest.approx(math.log(10.))
    expected_gradient = math.sqrt(40. / (1. + critic[1].eps))
    assert metrics["critic_input_gradient_mean_real"] == pytest.approx(expected_gradient)
    assert metrics["critic_input_gradient_mean_fake"] == pytest.approx(expected_gradient)
    for handle in diagnostics.handles:
        handle.remove()


def test_input_gradient_labels_capture_the_current_critic_phase_and_ignore_previous_generator_phase(tmp_path):
    class QuadraticCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1., 2.]]))

        def forward(self, inputs):
            return (inputs * self.weight).square().sum(dim=1, keepdim=True)

    critic = QuadraticCritic()
    optimizer = get_recipe("bcap").make_critic_optimizer(critic)
    diagnostics = OptimizerDiagnostics({"D": optimizer}, critic, tmp_path, [2])
    # Finish update 1. Its post-hook clears the observation buffer, and the
    # optimizer record now points to update 2 as the next observation.
    critic.train()
    with torch.no_grad():
        critic(torch.tensor([[10., 20.]]))
        critic(torch.tensor([[30., 40.]]))
    optimizer.step()
    assert diagnostics.inputs == []
    # The generator part of update 1 evaluates D(fake) and then D(real).
    # Neither is a real/fake observation from update 2's critic phase.
    critic.eval()
    with torch.no_grad():
        critic(torch.tensor([[50., 60.]]))
        critic(torch.tensor([[70., 80.]]))
    assert diagnostics.inputs == []
    real = torch.tensor([[1., 2.], [3., 4.]])
    fake = torch.tensor([[-1., .5], [2., -.5]])
    critic.train()
    with torch.no_grad():
        critic(real)
        critic(fake)
    assert len(diagnostics.inputs) == 2
    assert torch.equal(diagnostics.inputs[0], real)
    assert torch.equal(diagnostics.inputs[1], fake)
    optimizer.step()
    diagnostics.receipt()
    row, = read_rows(tmp_path / "optimizer-diagnostics.jsonl")
    assert row["step"] == 2 and row["optimizer"] == "D"
    # grad_x sum((x * w)^2) = 2*x*w^2, so labels have distinct expected values.
    for label, inputs in (("real", real), ("fake", fake)):
        expected = (2 * inputs * critic.weight.detach().square()).norm(dim=1).mean()
        assert row["critic_input_gradient_mean_" + label] == pytest.approx(float(expected))

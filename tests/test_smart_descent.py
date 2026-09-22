"""Causal timing, exact baseline behavior, and raw-SGD update contracts."""
import math
import json

import pytest
import torch

from particlegan import ParticleRegularizer, learning_rate_scale
from benchmarks.smart_descent.controller import GradientFeedback, control_regularization
from benchmarks.smart_descent.study import policy, ring_objective


def test_zero_policy_matches_cosine_adam_and_preserves_rng():
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    reference = torch.nn.Parameter(p.detach().clone())
    opt = torch.optim.Adam([p], lr=.02, betas=(0., .99))
    ref = torch.optim.Adam([reference], lr=.02, betas=(0., .99))
    card = policy(torch.zeros(2, 2, 5))
    card["interval"] = 1
    controller = GradientFeedback(card, 10)
    for step in range(10):
        opt.zero_grad()
        p.square().sum().backward()
        before = torch.get_rng_state().clone()
        controller.step(opt, step, role="g")
        assert torch.equal(before, torch.get_rng_state())
        opt.step()
        ref.zero_grad()
        reference.square().sum().backward()
        ref.param_groups[0]["lr"] = .02 * learning_rate_scale(step, 10, .6, .05)
        ref.step()
        assert torch.equal(p, reference)


def test_raw_sgd_has_exact_unpreconditioned_gradient_direction():
    p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([3.], dtype=torch.float64))
    opt = torch.optim.SGD([dict(params=[p], lr=.02), dict(params=[q], lr=.06)])
    weights = torch.zeros(2, 2, 5)
    weights[0, 0, 0] = math.log(2.)
    controller = GradientFeedback(policy(weights, "constant"), 2)
    (p.square().sum() + q.square().sum()).backward()
    expected = [v.detach() - lr * math.sqrt(2.) * v.grad for v, lr in ((p, .02), (q, .06))]
    controller.step(opt, 0, role="g")
    opt.step()
    for value, target in zip((p, q), expected):
        torch.testing.assert_close(value, target, rtol=1e-8, atol=1e-8)
    assert not opt.state
    assert opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"] == pytest.approx(3.)


def test_regularization_action_affects_next_loss_only_and_restores_patch():
    z = torch.nn.Parameter(torch.tensor([[.1, .2], [.3, -.1], [-.2, .4]]))
    regularizer = ParticleRegularizer(weight=.05)
    original = regularizer(z)
    weights = torch.zeros(2, 2, 5)
    weights[0, 1, 0] = math.log(2.)
    controller = GradientFeedback(policy(weights), 2)
    opt = torch.optim.SGD([z], lr=.01)
    with control_regularization(controller):
        previous_loss = regularizer(z)
        previous_loss.backward()
        gradient = z.grad.clone()
        controller.step(opt, 0, role="g")
        assert torch.equal(gradient, z.grad)
        assert torch.equal(previous_loss, original)
        assert regularizer(z).item() == pytest.approx(original.item() * math.sqrt(2.))
    assert torch.equal(regularizer(z), original)


def test_feedback_ablation_and_nonfinite_errors():
    weights = torch.ones(2, 2, 5)
    weights[:, :, 0] = 0
    card = policy(weights, "constant")
    card["interval"] = 1
    controller = GradientFeedback(card, 2, ablation="bias_only")
    p = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([p], lr=.01)
    p.grad = torch.ones_like(p)
    controller.step(optimizer, 0, role="d")
    assert optimizer.param_groups[0]["lr"] == .01
    assert controller.regularization_scale("d") == 1.
    p.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        controller.step(optimizer, 1, role="d")


def test_missing_or_failed_ring_cannot_win():
    assert ring_objective({}) == 1000.
    assert ring_objective({"error": "failed"}) == 1000.


def test_freeze_binds_fitting_sources_and_original_transfer(tmp_path, monkeypatch):
    from benchmarks.smart_descent import freeze
    source = tmp_path / "search"
    source.mkdir()
    numerical = "particlegan/training.py"
    renderer = "benchmarks/smart_descent/evaluate.py"
    weights = torch.ones(2, 2, 5)
    report = {"selected": "candidate", "fresh_transfer": [{"name": "originally_reserved"}],
              "protocol": {"source_sha256": {numerical: "same_math", renderer: "old_renderer"}},
              "rows": [{"name": "candidate", "policy": policy(weights), "toys": {}}]}
    (source / "search.json").write_text(json.dumps(report))
    monkeypatch.setattr(freeze.study, "row_summary", lambda row: {
        "bounds": 29, "stable": 9, "mean_confirmation_fraction": .5})
    monkeypatch.setattr(freeze.study, "fingerprint", lambda: {
        "source_sha256": {numerical: "same_math", renderer: "new_renderer"}})
    monkeypatch.setattr(freeze.study, "TRANSFER", [{"name": "changed_after_fitting"}])
    output = tmp_path / "frozen"
    freeze.run([source], output)
    frozen = json.loads((output / "frozen.json").read_text())
    assert frozen["fresh_transfer"] == report["fresh_transfer"]
    assert frozen["fitting_source_sha256"][renderer] == "old_renderer"
    assert frozen["evaluation_source_sha256"][renderer] == "new_renderer"
    assert frozen["numerical_source_sha256"] == {numerical: "same_math"}
    monkeypatch.setattr(freeze.study, "fingerprint", lambda: {
        "source_sha256": {numerical: "changed_math", renderer: "new_renderer"}})
    rejected = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="numerical source changed"):
        freeze.run([source], rejected)
    assert not rejected.exists()

"""Checks for the conditional target and noise-table intervention."""
import torch
import pytest

from lib.denoising_toy import GaussianGrid, DiffusionSchedule, DrawSource, ToyDiscriminator, grid_metrics
from experiments.train_denoising import DEFAULTS, validate


def test_reverse_coefficients_and_noise_gradient():
    s = DiffusionSchedule(DEFAULTS["alpha_bar"])
    x0, xt = torch.randn(12, 2), torch.randn(12, 2)
    eta = torch.randn(12, 2, requires_grad=True)
    t = torch.ones(12, dtype=torch.long)
    out = s.reverse(x0, xt, t, eta)
    torch.testing.assert_close(out, x0)
    assert torch.count_nonzero(torch.autograd.grad(out.sum(), eta)[0]) == 0
    t.fill_(2)
    out = s.reverse(x0, xt, t, eta)
    grad = torch.autograd.grad(out.sum(), eta)[0]
    torch.testing.assert_close(grad, s.posterior_var[2].sqrt().expand_as(eta))


def test_posterior_bayes_matches_direct_density_and_component_moments():
    toy = GaussianGrid()
    xt = torch.tensor([[.2, -.3], [1.2, .5]])
    c = torch.tensor([0, 3])
    a = .4
    w, m, v = toy.posterior(xt, c, a)
    mu = toy.means[toy.by_class[c]]
    marginal_sd = (a * toy.std**2 + 1 - a)**.5
    dist = torch.distributions.Normal(a**.5 * mu, marginal_sd)
    ref = dist.log_prob(xt[:, None]).sum(-1).softmax(-1)
    torch.testing.assert_close(w, ref)
    # Independent Gaussian precision formula for the within-component posterior.
    prec = 1 / toy.std**2 + a / (1 - a)
    mean = (mu / toy.std**2 + a**.5 * xt[:, None] / (1 - a)) / prec
    torch.testing.assert_close(m, mean)
    torch.testing.assert_close(v, torch.full_like(v, 1 / prec))


def test_oracle_reverse_recovers_clean_marginal():
    torch.set_num_threads(1)
    toy = GaussianGrid()
    n = 30000
    rng = torch.Generator().manual_seed(71)
    c = torch.zeros(n, dtype=torch.long)
    clean = toy.sample(c, rng)
    a = .5
    xt = a**.5 * clean + (1 - a)**.5 * torch.randn(clean.shape, generator=rng)
    recovered = toy.oracle_clean(xt, c, a, rng)
    # Posterior resampling should preserve the joint's x0 marginal.
    torch.testing.assert_close(recovered.mean(0), clean.mean(0), atol=.025, rtol=0)
    torch.testing.assert_close(torch.cov(recovered.T), torch.cov(clean.T), atol=.075, rtol=0)
    nearest = torch.cdist(recovered, toy.means).argmin(1)
    assert (toy.labels[nearest] == c).float().mean() > .999


def test_fixed_and_learned_tables_start_identically_and_draw_discretely():
    fixed = DrawSource("fixed", 16, 2, 123, "cpu")
    learned = DrawSource("learned", 16, 2, 123, "cpu")
    torch.testing.assert_close(fixed.table, learned.table)
    r1, r2 = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2)
    x, ids = learned.sample(128, r1)
    torch.testing.assert_close(x, fixed.sample(128, r2)[0])
    x.sum().backward()
    assert learned.table.grad is not None
    assert len(torch.unique(x, dim=0)) <= 16
    assert len(list(fixed.parameters())) == 0


def test_ucd_keeps_transition_condition_but_class_only_selects_score():
    cfg = {**DEFAULTS, "d_mode": "ucd"}
    d = ToyDiscriminator(cfg)
    x, xt, t = torch.randn(5, 2), torch.randn(5, 2), torch.ones(5, dtype=torch.long)
    score0, logits0 = d(x, torch.zeros(5, dtype=torch.long), xt, t)
    score1, logits1 = d(x, torch.ones(5, dtype=torch.long), xt, t)
    torch.testing.assert_close(logits0, logits1)
    torch.testing.assert_close(score0, logits0[:, 0])
    torch.testing.assert_close(score1, logits0[:, 1])
    assert not torch.equal(logits0, d(x, torch.zeros(5, dtype=torch.long), xt + 2, t)[1])


def test_invalid_schedule_and_meaningless_one_shot_noise_are_rejected():
    with pytest.raises(ValueError):
        DiffusionSchedule([1, .4, .5])
    with pytest.raises(ValueError):
        validate({**DEFAULTS, "model": "gan", "noise": "learned"})


def test_joint_ucd_hides_time_and_class_and_preserves_candidate_cap_gradients():
    from lib.denoising_toy import FixedConditionCritic
    from lib.grad_regularizers import GradRegularizer
    from torch.nn import functional as F
    cfg = {**DEFAULTS, "ucd_target": "time_class"}
    d = ToyDiscriminator(cfg)
    c = torch.arange(4).repeat(4)
    t = torch.arange(1, 5).repeat_interleave(4)
    x, xt = torch.randn(16, 2), torch.randn(16, 2)
    score, logits = d(x, c, xt, t)
    assert logits.shape == (16, 16)
    torch.testing.assert_close(d.ucd_labels(c, t), torch.arange(16))
    torch.testing.assert_close(score, logits.diag())
    # Changing either label only selects a head; neither enters the backbone.
    torch.testing.assert_close(logits, d(x, c.flip(0), xt, t.flip(0))[1])
    assert not torch.equal(logits, d(x, c, xt + 2, t)[1])
    penalty, _ = GradRegularizer('b_cap', 1, kappa=0).penalty(
        FixedConditionCritic(d, c, xt, t), x, torch.randn_like(x), 1)
    assert penalty > 0 and torch.isfinite(penalty)
    (penalty + F.cross_entropy(logits, d.ucd_labels(c, t))).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in d.parameters())
    assert xt.grad is None


@pytest.mark.parametrize('model,mode,target', [('gan','ucd','time_class'), ('ddgan','concat','time_class'), ('ddgan','ucd','bad')])
def test_invalid_joint_ucd_configs_are_rejected(model, mode, target):
    cfg = {**DEFAULTS, 'model':model, 'd_mode':mode, 'ucd_target':target}
    with pytest.raises(ValueError):
        validate(cfg)
    with pytest.raises(ValueError):
        ToyDiscriminator(cfg)


def test_metrics_detect_wrong_classes_and_center_collapse():
    toy = GaussianGrid()
    nearest = torch.arange(100).repeat_interleave(25)
    x = toy.means[nearest]
    c = toy.labels[nearest]
    rng = torch.Generator().manual_seed(123)
    real = toy.sample(c, rng)
    right = grid_metrics(x, c, toy, real)
    wrong = grid_metrics(x, (c + 1) % 4, toy, real)
    assert right["hq"] == 1 and right["modes"] == 100
    assert right["per_mode_cov_eig_max_ratio"] == 0
    assert wrong["hq"] == 1 and wrong["joint_hq"] == 0
    assert wrong["conditional_mode_tv"] == pytest.approx(1.0, abs=1e-6)


def test_terminal_gaussian_approximation_has_small_moment_error():
    toy = GaussianGrid()
    a = DEFAULTS["alpha_bar"][-1]
    for means in toy.means[toy.by_class]:
        terminal_mean = a**.5 * means.mean(0)
        cov = torch.cov(means.T, correction=0) + toy.std**2 * torch.eye(2)
        terminal_cov = a * cov + (1 - a) * torch.eye(2)
        assert terminal_mean.norm() < .008
        assert (terminal_cov - torch.eye(2)).abs().max() < .001

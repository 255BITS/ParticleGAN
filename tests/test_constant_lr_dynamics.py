"""The constant-LR dynamics, each at the one setting named in its module."""
import hashlib
import importlib.util
import os
import sys

import pytest
import torch
from torch import nn

import benchmarks.legacy.grad_regularizers as legacy_mod
import particlegan.grad_regularizers as k3p_mod
from benchmarks.legacy.grad_regularizers import GradientPenalty as LegacyPenalty
from particlegan.dynamics import optimistic, pair_chord, unit_rms
from particlegan.dynamics.shared_batch import shared_batch_update
from particlegan.grad_regularizers import GradientPenalty as K3PPenalty

_LEGACY_PENALTY = legacy_mod.GradRegularizer.penalty
_K3P_PENALTY = k3p_mod.GradientPenalty.penalty


def _restore_penalties() -> None:
    legacy_mod.GradRegularizer.penalty = _LEGACY_PENALTY
    k3p_mod.GradientPenalty.penalty = _K3P_PENALTY
    finder = pair_chord._FINDER
    if finder is not None and finder in sys.meta_path:
        sys.meta_path.remove(finder)
    pair_chord._FINDER = None


def _linear():
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.1, 0.0]]))
    return layer


def test_unit_rms_is_scale_free_at_the_group_lr():
    torch.manual_seed(0)
    parameter = nn.Parameter(torch.zeros(4))
    opt = torch.optim.Adam([parameter], lr=0.00425)
    grad = torch.tensor([3.0, 0.0, 0.0, 0.0])
    rms = grad.square().mean().sqrt()
    expected = -0.00425 * grad / (rms + opt.param_groups[0]["eps"])
    try:
        unit_rms.install()
        parameter.grad = grad.clone()
        opt.step()
        moved = parameter.detach().clone()
        parameter.grad = grad * 10
        with torch.no_grad():
            parameter.zero_()
        opt.step()
    finally:
        unit_rms.uninstall()
    assert torch.allclose(moved, expected)
    assert torch.allclose(parameter.detach(), expected, atol=1e-10, rtol=0)


def test_pair_chord_matches_the_endpoint_term():
    real = torch.randn(8, 2)
    fake = torch.randn(8, 2)
    critic = _linear()
    legacy = LegacyPenalty("a_r1r2", coeff=1.0, kappa=1.0)
    base = legacy.penalty(critic, real, fake, collect_stats=False)[0].detach()
    capped = LegacyPenalty("b_cap", coeff=1.0, kappa=1.0)
    cap_before = capped.penalty(critic, real, fake, collect_stats=False)[0].detach()
    try:
        pair_chord.install()
        legacy_pen = legacy.penalty(critic, real, fake, collect_stats=False)[0]
        k3p = K3PPenalty(coeff=1.0, kappa=1.0)
        k3p_pen = k3p.penalty(critic, real, fake, collect_stats=False)[0]
        cap_after = capped.penalty(critic, real, fake, collect_stats=False)[0].detach()
        # Inplace activations, as on the ring critic. The chord has to backward.
        activated = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2, inplace=True), nn.Linear(8, 1))

        class _Critic(nn.Module):
            def forward(self, x):
                return activated(x).squeeze(-1)

        inplace = LegacyPenalty("a_r1r2", coeff=1.0, kappa=1.0)
        loss = inplace(_Critic(), real, fake)
        loss.backward()
    finally:
        _restore_penalties()
    # ||w||^2 = 0.01. Legacy endpoint term is (c/2) mean||g||^2 = 0.005.
    # K3P real term is (c/2) mean(||g||^2 / d) = 0.0025, and the fake cap is inactive.
    assert torch.allclose(legacy_pen.detach(), base + 0.005)
    assert torch.allclose(k3p_pen.detach(), torch.tensor(0.005))
    assert torch.allclose(cap_before, cap_after)
    assert loss.detach().isfinite()


def test_shared_batch_reuses_the_critic_draw(monkeypatch):
    from tests.test_k3p_trainer import _reals, _trainer

    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    trainer = _trainer()
    draws = {"n": 0}
    original = trainer.prior.sample

    def sample(*args, **kwargs):
        draws["n"] += 1
        return original(*args, **kwargs)

    trainer.prior.sample = sample
    paired = {"n": 0}

    def generator_real():
        paired["n"] += 1
        return _reals(1)[0]

    trainer.step(_reals(1)[0], generator_real=generator_real)
    assert draws["n"] == 2 and paired["n"] == 1

    monkeypatch.setenv("K3P_DYNAMICS", "shared_batch")
    trainer.step(_reals(1)[0], generator_real=generator_real)
    assert draws["n"] == 3 and paired["n"] == 1
    assert shared_batch_update() is True


def test_shared_batch_off_matches_two_draws(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    assert shared_batch_update() is False
    assert os.environ.get("K3P_DYNAMICS") is None


def _digest(*tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _k3p_groups():
    """Network and prior groups at K3P's constant rates and betas."""
    generator = nn.Parameter(torch.tensor([0.2, -0.5, 0.1], dtype=torch.float64))
    prior = nn.Parameter(torch.tensor([0.4, -0.3], dtype=torch.float64))
    critic = nn.Parameter(torch.tensor([-0.2, 0.7], dtype=torch.float64))
    opt_g = torch.optim.Adam(
        [{"params": [generator], "lr": 0.00425}, {"params": [prior], "lr": 0.0085}],
        lr=0.00425, betas=(0.0, 0.999), eps=1e-8,
    )
    opt_d = torch.optim.Adam([critic], lr=0.00425, betas=(0.0, 0.999), eps=1e-8)
    gradients = (
        (torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64),
         torch.tensor([0.5, -0.1], dtype=torch.float64),
         torch.tensor([0.2, -0.6], dtype=torch.float64)),
        (torch.tensor([-0.7, 0.1, 0.4], dtype=torch.float64),
         torch.tensor([-0.2, 0.8], dtype=torch.float64),
         torch.tensor([-0.5, 0.3], dtype=torch.float64)),
        (torch.tensor([0.1, 0.2, -0.3], dtype=torch.float64),
         torch.tensor([0.4, 0.4], dtype=torch.float64),
         torch.tensor([0.05, -0.2], dtype=torch.float64)),
    )
    return (generator, prior, critic), (opt_g, opt_d), gradients


def _expected_optimistic(start, gradients, lr, betas=(0.0, 0.999), eps=1e-8):
    """Paper update ``-lr * (2 m_t - m_{t-1})`` with Adam's bias correction."""
    beta1, beta2 = betas
    moment = torch.zeros_like(start)
    second = torch.zeros_like(start)
    previous = torch.zeros_like(start)
    value = start.clone()
    for t, grad in enumerate(gradients, start=1):
        moment = beta1 * moment + (1.0 - beta1) * grad
        second = beta2 * second + (1.0 - beta2) * grad.square()
        b1 = 0.0 if beta1 == 0.0 else beta1 ** t
        b2 = 0.0 if beta2 == 0.0 else beta2 ** t
        direction = (moment / (1.0 - b1)) / ((second / (1.0 - b2)).sqrt() + eps)
        value = value - lr * (2.0 * direction - previous)
        previous = direction.clone()
    return value


def test_optimistic_matches_two_directions_minus_previous():
    (generator, prior, critic), (opt_g, opt_d), gradients = _k3p_groups()
    try:
        optimistic.install()
        for grad_g, grad_p, grad_d in gradients:
            generator.grad = grad_g.clone()
            prior.grad = grad_p.clone()
            critic.grad = grad_d.clone()
            opt_g.step()
            opt_d.step()
    finally:
        optimistic.uninstall()
    assert torch.allclose(
        generator, _expected_optimistic(torch.tensor([0.2, -0.5, 0.1], dtype=torch.float64),
                                        [row[0] for row in gradients], 0.00425),
        atol=1e-12, rtol=0)
    assert torch.allclose(
        prior, _expected_optimistic(torch.tensor([0.4, -0.3], dtype=torch.float64),
                                    [row[1] for row in gradients], 0.0085),
        atol=1e-12, rtol=0)
    assert torch.allclose(
        critic, _expected_optimistic(torch.tensor([-0.2, 0.7], dtype=torch.float64),
                                     [row[2] for row in gradients], 0.00425),
        atol=1e-12, rtol=0)


def test_optimistic_reaches_k3p_subclasses():
    """D and the generator/prior optimizers call Adam through ``_adam_step``."""
    from particlegan.k3p import K3PCriticAdam, K3PGeneratorAdam

    critic = nn.Linear(2, 1, bias=False).double()
    particle = nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float64))
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[0.1, -0.2]], dtype=torch.float64))
    start_d = critic.weight.detach().clone()
    start_p = particle.detach().clone()
    opt_d = K3PCriticAdam(critic.parameters(), critic=critic, lr=0.00425, betas=(0.0, 0.999))
    opt_g = K3PGeneratorAdam([particle], lr=0.0085, betas=(0.0, 0.999))
    grads_d = (torch.tensor([[0.4, -0.3]], dtype=torch.float64),
               torch.tensor([[-0.2, 0.5]], dtype=torch.float64))
    grads_p = (torch.tensor([0.6, -0.1], dtype=torch.float64),
               torch.tensor([-0.4, 0.2], dtype=torch.float64))
    try:
        optimistic.install()
        for grad_d, grad_p in zip(grads_d, grads_p):
            critic.weight.grad = grad_d.clone()
            particle.grad = grad_p.clone()
            opt_d.step()
            opt_g.step()
    finally:
        optimistic.uninstall()
    assert torch.allclose(
        critic.weight, _expected_optimistic(start_d, grads_d, 0.00425), atol=1e-12, rtol=0)
    assert torch.allclose(
        particle, _expected_optimistic(start_p, grads_p, 0.0085), atol=1e-12, rtol=0)


def test_flag_off_parameter_hash_matches_adam_and_on_is_deterministic():
    gradients = [torch.tensor([0.3, -0.2, 0.4]), torch.tensor([-0.5, 0.1, 0.2]),
                 torch.tensor([0.05, -0.3, 0.1])]

    def run(use_optimistic: bool):
        parameter = nn.Parameter(torch.tensor([0.2, -0.4, 0.1]))
        opt = torch.optim.Adam([parameter], lr=0.00425, betas=(0.0, 0.999))
        if use_optimistic:
            optimistic.install()
        try:
            for grad in gradients:
                parameter.grad = grad.clone()
                opt.step()
            state = opt.state[parameter]
            return (
                _digest(parameter, state["exp_avg"], state["exp_avg_sq"]),
                _digest(parameter),
                tuple(state.keys()),
            )
        finally:
            if use_optimistic:
                optimistic.uninstall()

    off_a = run(False)
    off_b = run(False)
    assert off_a[0] == off_b[0]
    assert "optimistic_prev_direction" not in off_a[2]
    # Importing the module must not install it.
    assert optimistic._ORIGINAL is None
    on_a = run(True)
    on_b = run(True)
    assert on_a[0] == on_b[0]
    assert on_a[1] != off_a[1]


def test_sitecustomize_installs_only_when_the_flag_is_set(monkeypatch):
    path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "toy100", "constant-lr-dynamics", "sitecustomize.py",
    )
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    before = torch.optim.Adam.step
    spec = importlib.util.spec_from_file_location("clr_sitecustomize_off", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert torch.optim.Adam.step is before
    assert optimistic._ORIGINAL is None

    monkeypatch.setenv("K3P_DYNAMICS", "optimistic")
    spec = importlib.util.spec_from_file_location("clr_sitecustomize_on", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        assert getattr(torch.optim.Adam.step, "_k3p_optimistic", False) or optimistic._ORIGINAL is not None
        parameter = nn.Parameter(torch.zeros(2))
        opt = torch.optim.Adam([parameter], lr=0.00425, betas=(0.0, 0.999))
        parameter.grad = torch.tensor([1.0, -1.0])
        before_param = parameter.detach().clone()
        opt.step()
        # First step is -2*lr*sign(g), not Adam's -lr*sign(g).
        assert torch.allclose(parameter.detach(), before_param - 2 * 0.00425 * torch.tensor([1.0, -1.0]))
    finally:
        optimistic.uninstall()


def test_scaled_penalty_accepts_ema_critic_and_matches_constant_lr_k3p():
    """Unequal mass calls ``penalty(..., ema_critic=)`` on a penalty with no ``arm``."""
    from particlegan.grad_regularizers import GradientPenalty
    import torch.optim.optimizer as optimizer_module

    real = torch.tensor([[0.2, -0.4], [0.5, 0.1], [-0.3, 0.7]])
    fake = torch.tensor([[-0.1, 0.2], [0.4, -0.5], [0.0, 0.3]])
    critic = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[0.3, -0.2]]))
    library = GradientPenalty(coeff=1.0, kappa=1.0)
    library_pen = library.penalty(critic, real, fake, 1, False, ema_critic=lambda x: critic(x))[0].detach()

    pre = dict(optimizer_module._global_optimizer_pre_hooks)
    post = dict(optimizer_module._global_optimizer_post_hooks)
    original = GradientPenalty.penalty
    path = os.path.join(
        os.path.dirname(__file__), "..", "reports", "toy100",
        "gap-fill-20260925", "sources", "k3p", "mechanism.py",
    )
    spec = importlib.util.spec_from_file_location("k3p_mechanism_unequal", os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        penalty = GradientPenalty(coeff=1.0, kappa=1.0)
        assert not hasattr(penalty, "arm")
        value, stats = module.scaled_penalty(
            penalty, critic, real, fake, 1, False, ema_critic=lambda x: critic(x),
        )
        assert torch.allclose(value.detach(), library_pen)
        assert stats == {}
        # The call CriticPenalty makes: positional collect_stats, keyword ema_critic.
        again = penalty.penalty(critic, real, fake, 1, True, ema_critic=lambda x: critic(x))
        assert torch.allclose(again[0].detach(), library_pen)
        assert again[1]["applied"] is True
    finally:
        GradientPenalty.penalty = original
        optimizer_module._global_optimizer_pre_hooks.clear()
        optimizer_module._global_optimizer_pre_hooks.update(pre)
        optimizer_module._global_optimizer_post_hooks.clear()
        optimizer_module._global_optimizer_post_hooks.update(post)


@pytest.fixture(autouse=True)
def _clear_dynamics(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    yield
    optimistic.uninstall()

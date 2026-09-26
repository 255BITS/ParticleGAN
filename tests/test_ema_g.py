"""Shadow EMA of G and the particles. Flag off matches the live baseline."""
import hashlib
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from particlegan import ParticlePrior
from particlegan.dynamics import ema_g


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    ema_g.reset()


def _models():
    generator = nn.Linear(2, 1, bias=False)
    prior = ParticlePrior(4, 2, init_std=0.1)
    with torch.no_grad():
        generator.weight.copy_(torch.tensor([[1.0, 0.0]]))
        prior.z.fill_(0.5)
    ema_params = [generator.weight.detach().clone()]
    with torch.no_grad():
        ema_params[0].copy_(torch.tensor([[3.0, 0.0]]))
    ema_z = prior.z.detach().clone()
    with torch.no_grad():
        ema_z.fill_(0.25)
    return generator, prior, ema_params, ema_z


def test_flag_off_measures_once_and_keeps_the_callers_decay():
    generator, prior, ema_params, ema_z = _models()
    calls = {"n": 0}

    def measure():
        calls["n"] += 1
        return {"modes": 1, "hq": 0.1}

    assert ema_g.decay_or(0.995) == 0.995
    assert ema_g.score(1, generator, prior, ema_params, ema_z, measure)["modes"] == 1
    assert calls["n"] == 1
    assert ema_g.points() == []
    assert not ema_g.critic_fakes()


def test_score_records_the_average_and_restores_live_weights(monkeypatch):
    monkeypatch.setenv("K3P_DYNAMICS", "ema_g")
    generator, prior, ema_params, ema_z = _models()
    seen = []

    def measure():
        weight = float(generator.weight.detach()[0, 0])
        seen.append((weight, float(prior.z.detach()[0, 0])))
        high = weight > 2
        return {"modes": 8 if high else 1, "hq": 0.95 if high else 0.2}

    out = ema_g.score(4, generator, prior, ema_params, ema_z, measure)
    assert out["modes"] == 1
    assert seen == [(1.0, 0.5), (3.0, 0.25)]
    assert float(generator.weight.detach()[0, 0]) == 1.0
    assert float(prior.z.detach()[0, 0]) == 0.5
    row = ema_g.points()[-1]
    assert row["live_modes"] == 1 and row["ema_modes"] == 8 and row["ema_pass"]
    ema_g.score(4, generator, prior, ema_params, ema_z, measure)
    assert len(ema_g.points()) == 1


def test_averaged_fake_is_the_swapped_generator(monkeypatch):
    monkeypatch.setenv("K3P_DYNAMICS", "ema_g_fake")
    assert ema_g.critic_fakes()
    generator, prior, ema_params, ema_z = _models()
    stream = torch.Generator().manual_seed(0)
    fake = ema_g.averaged_fake(generator, prior, ema_params, ema_z, 4, stream)
    assert float(generator.weight.detach()[0, 0]) == 1.0
    assert float(prior.z.detach()[0, 0]) == 0.5
    # Linear(2, 1) with weight [3, 0] on particles at 0.25 is 0.75 everywhere.
    assert torch.allclose(fake, torch.full((4, 1), 0.75))


def test_ring_hold_and_stay_columns():
    ema_g.reset()
    ring = []
    for step in sorted({__import__("math").ceil(i * 1200 / 24) for i in range(1, 25)}):
        ring.append({"step": step, "live_modes": 4, "live_hq": 0.2, "live_pass": False,
                     "ema_modes": 8, "ema_hq": 0.95, "ema_pass": True})
    live = ema_g.ring_summary(ring, "live")
    averaged = ema_g.ring_summary(ring, "ema")
    assert live["pass"] is False and averaged["pass"] is True and averaged["passing_suffix"] == 24

    hold = [{"step": step, "live_modes": 8, "live_hq": 1.0, "live_pass": True,
             "ema_modes": 8, "ema_hq": 1.0, "ema_pass": True} for step in range(1, 2601)]
    # 1201..2600 is 1400 checks: 200 to converge and 1200 to hold.
    assert ema_g.hold_summary(hold, "ema")["pass"] is True
    broken = [dict(row) for row in hold]
    # Step 1500 is inside the 1,200-check hold that starts after step 1400.
    broken[1499] = dict(broken[1499], ema_modes=3, ema_hq=0.1, ema_pass=False)
    fail = ema_g.hold_summary(broken, "ema")
    assert fail["pass"] is False and fail["status"] == "POST_CONVERGENCE_FAIL"

    stay = [{"step": step, "live_modes": 1, "live_hq": 0.0, "live_pass": False,
             "ema_modes": 8, "ema_hq": 0.95, "ema_pass": True}
            for step in range(1210, 2410, 10)]
    assert ema_g.stay_summary(stay, "ema") == {"stay": "120/120", "pass": True, "checks": 120,
                                               "passing_checks": 120}
    assert ema_g.stay_summary(stay, "live")["pass"] is False


def _digest(modules) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, parameter in module.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train(monkeypatch, flag, steps=8):
    from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
    from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, train_mode_hold
    from benchmarks.locked_shared.observation import recording
    from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy

    found = {}
    originals = {
        "g": SimpleMLPGenerator.__init__,
        "d": SimpleMLPDiscriminator.__init__,
        "p": ParticlePrior.__init__,
    }

    def wrap(key, original):
        def init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            found[key] = self
        return init

    monkeypatch.setattr(SimpleMLPGenerator, "__init__", wrap("g", originals["g"]))
    monkeypatch.setattr(SimpleMLPDiscriminator, "__init__", wrap("d", originals["d"]))
    monkeypatch.setattr(ParticlePrior, "__init__", wrap("p", originals["p"]))
    if flag:
        monkeypatch.setenv("K3P_DYNAMICS", flag)
    else:
        monkeypatch.delenv("K3P_DYNAMICS", raising=False)
    ema_g.reset()
    policy = NoisePolicy(0.029, 0.5, 0.1, steps, seed=0, output_noise_warmup=0.2)
    recipe = ModeHoldRecipe(steps=steps, particle_l2=0.0, vicreg_weight=0.0)
    with recording(steps):
        train_mode_hold(recipe, noise_policy=policy)
    digest = _digest((found["g"], found["d"], found["p"]))
    curve = [(row["live_modes"], row["ema_modes"]) for row in ema_g.points()]
    return digest, curve


def test_flag_off_matches_ema_g_live_parameters_and_repeats(monkeypatch):
    off_a, curve_off = _train(monkeypatch, None)
    off_b, _ = _train(monkeypatch, None)
    on_a, curve_on = _train(monkeypatch, "ema_g")
    on_b, curve_on_b = _train(monkeypatch, "ema_g")
    assert off_a == off_b
    assert on_a == on_b == off_a
    assert curve_off == []
    assert curve_on == curve_on_b
    assert curve_on
    print(f"parameter_sha256 {off_a}")


def test_scaled_penalty_accepts_ema_critic_and_matches_k3p():
    import particlegan.grad_regularizers as regularizers
    original = regularizers.GradientPenalty.penalty
    k3p = Path(__file__).resolve().parents[1] / "reports/toy100/gap-fill-20260925/sources/k3p"
    sys.path.insert(0, str(k3p))
    import mechanism
    try:
        torch.manual_seed(0)
        critic = nn.Linear(2, 1)
        real, fake = torch.randn(4, 2), torch.randn(4, 2)
        penalty = regularizers.GradientPenalty(coeff=1.0, kappa=1.0, lr_floor=0.0)
        anchor = lambda x: critic(x)
        expected = original(penalty, critic, real, fake, 1, True, ema_critic=anchor)
        got = penalty.penalty(critic, real, fake, 1, False, ema_critic=anchor)
        assert torch.allclose(got[0], expected[0])
        assert mechanism.scaled_penalty is not original
    finally:
        regularizers.GradientPenalty.penalty = original
        sys.modules.pop("mechanism", None)

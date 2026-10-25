"""Declared BISPF behavior: shape/step gates, SBNP weights, thermal ownership and streams.

Software contracts only; no quality claim.
"""
from dataclasses import asdict, fields
import json

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.bispf import (
    BISPFEngine,
    BISPFGates,
    BISPFLossAdapter,
    ContinuousThermalCooler,
    SBNPPriorRegularizer,
    SinkhornRelativisticLoss,
    StratifiedBlueNoisePrior,
    _BISPFPenaltyKernel,
    _LEGACY_THERMAL_CHANNEL,
    bind_thermal_source,
)
from particlegan.init import deterministic_orthogonal_


@pytest.fixture(autouse=True)
def isolated_thermal_channel():
    previous = dict(_LEGACY_THERMAL_CHANNEL)
    _LEGACY_THERMAL_CHANNEL.clear()
    yield
    _LEGACY_THERMAL_CHANNEL.clear()
    _LEGACY_THERMAL_CHANNEL.update(previous)


class _Reference:
    def __init__(self, d, count):
        self.moment_tether_mean = torch.zeros(d)
        self.moment_tether_covariance = torch.eye(d) * .8
        self.moment_tether_count = torch.tensor(count)


def _rows(n, d, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)


# --- #4: SBNP weight ---------------------------------------------------------

def test_sbnp_legacy_regimes_are_declared_and_disabling_them_applies_the_declared_weight():
    rows, ref = _rows(48, 8), _Reference(8, 48)
    legacy = [SBNPPriorRegularizer(ref, weight=w)(rows) for w in (.05, 1.)]
    # Historical small-latent regime: the declared weight is replaced by 1.0.
    assert torch.equal(legacy[0], legacy[1])
    assert SBNPPriorRegularizer(ref, weight=.05).effective_weight(rows) == 1.
    for gates in (BISPFGates(sbnp_small_latent_weight=None), BISPFGates(shape_gates=False)):
        low, high = (SBNPPriorRegularizer(ref, weight=w, gates=gates)(rows) for w in (.05, 1.))
        torch.testing.assert_close(low, .05 * high)
        assert SBNPPriorRegularizer(ref, weight=.05, gates=gates).effective_weight(rows) == .05

    unanchored = _rows(200, 3, seed=1)
    legacy = [SBNPPriorRegularizer(weight=w)(unanchored) for w in (.01, 1.)]
    assert torch.equal(legacy[0], legacy[1])  # historical unanchored coercion of (0, .1)
    gates = BISPFGates(sbnp_unanchored_weight_enabled=False)
    low, high = (SBNPPriorRegularizer(weight=w, gates=gates)(unanchored) for w in (.01, 1.))
    torch.testing.assert_close(low, .01 * high)


def test_sbnp_small_latent_repulsion_constants_are_declared():
    rows, ref = _rows(48, 8, seed=2), _Reference(8, 48)
    legacy = SBNPPriorRegularizer(ref, weight=1.)(rows)
    same = SBNPPriorRegularizer(ref, weight=1., gates=BISPFGates(
        sbnp_small_latent_repel_mult=25., sbnp_small_latent_min_k_nn=10, sbnp_small_latent_margin=1.))(rows)
    assert torch.equal(legacy, same)
    changed = SBNPPriorRegularizer(ref, weight=1., gates=BISPFGates(sbnp_small_latent_repel_mult=1.))(rows)
    assert not torch.equal(legacy, changed)


# --- #5: shape/step gates ----------------------------------------------------

class _Critic(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 1))
        deterministic_orthogonal_(self.net, seed=0)

    def forward(self, x):
        return self.net(x.flatten(1))


def _late_thermal_scale(gates):
    xr = torch.randn(128, 2, generator=torch.Generator().manual_seed(3))
    kernel = _BISPFPenaltyKernel(gates=gates)
    for _ in range(40):
        kernel._observe_sinkhorn(.001, 1.)
    kernel.record.calls = 1700
    kernel.penalty(_Critic(2), xr, xr + 1e-4, step=0)
    return kernel.record._thermal_scale


def test_late_thermal_gate_and_master_switch():
    assert _late_thermal_scale(BISPFGates()) == 1e-5
    assert _late_thermal_scale(BISPFGates(late_thermal_scale=1e-3)) == 1e-3
    assert _late_thermal_scale(BISPFGates(late_thermal_enabled=False)) != 1e-5
    assert _late_thermal_scale(BISPFGates(shape_gates=False)) != 1e-5
    assert _late_thermal_scale(BISPFGates(late_thermal_min_call=2000)) != 1e-5


def test_simplex_gate_controls_chords_and_gauge():
    xr = torch.randn(32, 180, generator=torch.Generator().manual_seed(4))
    stats = {}
    for name, gates in (("legacy", BISPFGates()), ("global", BISPFGates(shape_gates=False)),
                        ("raised", BISPFGates(simplex_min_dim=200))):
        _, stats[name] = _BISPFPenaltyKernel(gates=gates).penalty(
            _Critic(180), xr, xr.flip(0), step=0, collect_stats=True)
    assert stats["legacy"]["gauge_loss"] == 0.  # discrete simplex: no gauge
    assert stats["global"]["gauge_loss"] > 0. and stats["raised"]["gauge_loss"] > 0.


def test_dense_generator_rescale_gate():
    def grad(gates):
        loss_fn = SinkhornRelativisticLoss(gates=gates)
        source = type("Source", (), {"_thermal_scale": .1})()
        loss_fn.thermal_source = source
        fake = torch.linspace(-1, 1, 1024).unsqueeze(1).requires_grad_(True)
        real = torch.linspace(-.5, 1.5, 1024).unsqueeze(1)
        return torch.autograd.grad(loss_fn.g_loss(fake, real), fake)[0]

    legacy, off = grad(BISPFGates()), grad(BISPFGates(dense_g_rescale_enabled=False))
    s = .1
    expected = s / (1 - s * s) ** .5 * .001
    torch.testing.assert_close(legacy, expected * off)
    assert torch.equal(off, grad(BISPFGates(shape_gates=False)))


def test_gates_validate_names_and_values():
    with pytest.raises(ValueError, match="unknown BISPF gates"):
        BISPFGates.from_overrides(True, {"missing": 1})
    with pytest.raises(ValueError, match="late_thermal_min_call"):
        BISPFGates(late_thermal_min_call=-1)
    with pytest.raises(ValueError, match="simplex_enabled"):
        BISPFGates(simplex_enabled=1)
    with pytest.raises(ValueError, match="gauge_ramp_calls"):
        BISPFGates(gauge_ramp_calls=0)
    off = BISPFGates(shape_gates=False)
    assert not any(off.active(spec.name[:-len("_enabled")])
                   for spec in fields(off) if spec.name.endswith("_enabled"))


def test_recipe_declares_gates_for_every_bispf_component():
    default = get_recipe()
    assert "bispf_shape_gates" not in default.to_dict()
    assert "bispf_gate_overrides" not in default.to_dict()
    assert default.bispf_gates() == BISPFGates()

    recipe = get_recipe("bispf", bispf_shape_gates=False,
                        bispf_gate_overrides={"late_thermal_min_call": 10, "simplex_min_dim": 99})
    assert recipe.bispf_gate_overrides == (("late_thermal_min_call", 10), ("simplex_min_dim", 99))
    restored = Recipe(**json.loads(json.dumps(recipe.to_dict())))
    assert restored == recipe
    gates = recipe.bispf_gates()
    assert not gates.shape_gates and gates.late_thermal_min_call == 10
    assert recipe.make_loss().gates == gates
    assert recipe.make_prior_regularizer().gates == gates
    critic = nn.Linear(2, 1)
    penalty = recipe.make_critic_penalty(recipe.make_critic_optimizer(critic))
    assert penalty.regularizer.gates == gates
    assert not penalty.regularizer.cooler.tau_batch_scaling
    with pytest.raises(ValueError, match="unknown BISPF gates"):
        get_recipe("bispf", bispf_gate_overrides={"no_such_gate": True})
    with pytest.raises(ValueError, match="bispf_shape_gates"):
        get_recipe("bispf", bispf_shape_gates=0)


# --- #12: thermal state ownership ---------------------------------------------

def _trainer(**penalty_options):
    recipe = get_recipe("bispf_optimized", num_particles=64, z_dim=2, batch_size=32,
                        total_steps=8, reg_every=2)
    generator = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    critic = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    prior = recipe.make_prior()
    for component in (generator, critic, prior):
        deterministic_orthogonal_(component, seed=0)
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0, penalty_options=penalty_options)


def _batch(step):
    angle = torch.linspace(0, 6.283, 32) + step
    return torch.stack((angle.cos(), angle.sin()), 1) * 2


def _same_trajectory(left, right):
    assert torch.equal(left.prior.z, right.prior.z)
    for a, b in zip((*left.G.parameters(), *left.D.parameters()), (*right.G.parameters(), *right.D.parameters())):
        assert torch.equal(a, b)


def test_two_trainers_in_one_process_do_not_share_thermal_state():
    # A large cooling threshold drives one trainer's thermal scale near its floor.
    # With reg_every=2 the penalty skips odd steps, so before the fix each
    # trainer's prior regularizer read the other's last published scale there.
    solo = []
    for options in ({"tau_cooling": 50.}, {}):
        trainer = _trainer(**options)
        for step in range(4):
            trainer.step(_batch(step))
        solo.append(trainer)
    cold, live = _trainer(tau_cooling=50.), _trainer()
    for step in range(4):
        cold.step(_batch(step))
        live.step(_batch(step))
    assert cold.opt_d.record._thermal_scale < 1e-3
    assert live.prior_regularizer.thermal_source is live.opt_d.record
    assert not _LEGACY_THERMAL_CHANNEL
    _same_trajectory(solo[0], cold)
    _same_trajectory(solo[1], live)


def test_bound_readers_ignore_the_legacy_channel():
    record = type("Record", (), {"_thermal_scale": 1.})()
    rows = _rows(200, 2, seed=5)
    reg, loss_fn = SBNPPriorRegularizer(weight=1.), SinkhornRelativisticLoss()
    prior = StratifiedBlueNoisePrior(num_particles=16, z_dim=2, seed=0)
    bind_thermal_source(record, reg, loss_fn, prior)
    assert record.legacy_thermal_channel is False
    assert reg.thermal_source is loss_fn.thermal_source is prior.thermal_source is record
    expected = torch.autograd.grad(reg(rows), rows)[0]
    _LEGACY_THERMAL_CHANNEL[rows.device] = 1e-5
    assert torch.equal(torch.autograd.grad(reg(rows), rows)[0], expected)
    unbound = SBNPPriorRegularizer(weight=1.)
    assert not torch.equal(torch.autograd.grad(unbound(rows), rows)[0], expected)


# --- #13: OT barycentric target ----------------------------------------------

def test_column_normalized_ot_target_is_translation_equivariant():
    gen = torch.Generator().manual_seed(6)
    x_real, x_fake = torch.randn(48, 2, generator=gen), torch.randn(48, 2, generator=gen)
    shift = torch.tensor([3., -2.])
    legacy, normalized = BISPFEngine(), BISPFEngine(ot_target_column_normalize=True)
    for engine, equivariant in ((legacy, False), (normalized, True)):
        base = engine.ot_target(x_real, x_fake)
        moved = engine.ot_target(x_real + shift, x_fake + shift)
        assert torch.allclose(moved, base + shift, atol=1e-5) is equivariant
    with pytest.raises(TypeError):
        BISPFEngine(ot_target_column_normalize=1)


# --- #14: adapter training stream --------------------------------------------

def test_loss_adapter_draws_only_from_its_declared_stream():
    critic = _Critic(2)
    gen = torch.Generator().manual_seed(7)
    x_real, x_fake = torch.randn(64, 2, generator=gen) * 3, torch.randn(64, 2, generator=gen)

    def run(adapter, steps=2):
        return [adapter.d_loss(critic, x_real, x_fake).detach() for _ in range(steps)]

    global_state = torch.get_rng_state()
    first = BISPFLossAdapter(BISPFEngine(sigma_d_max=.1), seed=3)
    losses = run(first)
    assert torch.equal(torch.get_rng_state(), global_state)
    assert first.last_d_stats["sigma_d"] > 0
    saved = first.state_dict()
    continued = run(first)

    second = BISPFLossAdapter(BISPFEngine(sigma_d_max=.1), seed=3)
    assert all(torch.equal(a, b) for a, b in zip(run(second), losses))
    resumed = BISPFLossAdapter(BISPFEngine(sigma_d_max=.1))
    resumed.engine.load_state_dict(second.engine.state_dict())
    resumed.load_state_dict(saved)
    assert all(torch.equal(a, b) for a, b in zip(run(resumed), continued))

    explicit = torch.Generator().manual_seed(3)
    third = BISPFLossAdapter(BISPFEngine(sigma_d_max=.1), seed=99)
    third_losses = [third.d_loss(critic, x_real, x_fake, generator=explicit).detach() for _ in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(third_losses, losses))
    assert third.generator is None


# --- #6: cooler -------------------------------------------------------------

def test_cooler_scale_is_the_documented_smoothstep_at_the_declared_query():
    cooler = ContinuousThermalCooler(tau_cooling=.02, s_min=1e-5)
    cooler.update(.01)
    u = .01 / .02
    assert cooler.scale == pytest.approx(1e-5 + (1 - 1e-5) * (3 * u * u - 2 * u ** 3))
    assert cooler.scale == cooler.scale_for(128, 2) == cooler.scale_for(512, 3)
    small = cooler.scale_for(32, 2)
    assert small < cooler.scale  # small batches use a larger threshold, so they cool further
    flat = ContinuousThermalCooler(tau_cooling=.02, s_min=1e-5, tau_batch_scaling=False)
    flat.update(.01)
    assert flat.scale_for(32, 2) == flat.scale and flat.effective_tau(16, 1) == .02
    queried = ContinuousThermalCooler(tau_cooling=.02, s_min=1e-5, default_batch_size=32)
    queried.update(.01)
    assert queried.scale == small
    old = cooler.state_dict()
    for name in ContinuousThermalCooler._DECLARED:
        old.pop(name)
    old["eps_smooth"] = .001
    restored = ContinuousThermalCooler()
    restored.load_state_dict(old)
    assert restored.scale == cooler.scale
    assert "eps_smooth" not in restored.state_dict()

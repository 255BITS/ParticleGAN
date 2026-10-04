"""CPU software contracts for a fixed-width, complete routed AE-MoG bank.

These tiny public updates test ownership/restoration, not toy convergence or
scientific qualification. Independent MoG structural control remains refused.
"""
from copy import deepcopy
import io
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import RoutedBatch, RoutedRows, UpdatePolicy, get_recipe
from particlegan.particle_prior import MoGParticlePrior, ParticlePrior


@pytest.fixture(autouse=True)
def private_cpu_scope():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            same(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float) and math.isnan(right)
    else:
        assert type(left) is type(right) and left == right


class Router(nn.Module):
    def __init__(self, particles):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(particles, dtype=torch.float64))


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(2, 8), nn.Tanh())
        self.head = nn.Linear(8, 1)

    def forward(self, value):
        return self.head(self.features(value))


def complete_forward(models, context, candidate, routing):
    # An explicit dense AE software fixture: actual E produces query/offset,
    # actual MoG width bounds the offset, and every structural counterfactual
    # reruns the decoder from the candidate mass and named mixed-code site.
    query, offset = models["encoder"](context).chunk(2, dim=1)
    logits = -torch.cdist(query, candidate.table.detach()).square() / .25
    code = routing.mix("ae_bank", logits)
    return models["generator"](code + 3 * models["prior"].sigma * (offset / 3).tanh())


def paired_features(models, context, samples, targets):
    return models["critic"].features(samples - targets)


def recipe(**changes):
    fields = dict(prior_kind="mog", encoder_mode="ae", sigma_rel=0., standardize=False,
                  row_policy="routed_paired", num_particles=6, z_dim=2, batch_size=8)
    fields.update(changes)
    return get_recipe("atlas", **fields)


def owners(options, *, prior_kind="mog"):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        G, E, D = nn.Linear(2, 2).double(), nn.Linear(2, 4).double(), Critic().double()
    rng = torch.Generator().manual_seed(432)
    fields = dict(num_particles=options.num_particles, z_dim=options.z_dim,
                  dtype=torch.float64, generator=rng)
    prior = (MoGParticlePrior(**fields, sigma=.025, standardize=False)
             if prior_kind == "mog" else ParticlePrior(**fields))
    R = Router(options.num_particles)
    return G, E, D, prior, R


def make_policy(options=None, *, change=None):
    options = recipe() if options is None else options
    G, E, D, prior, R = owners(options, prior_kind="particles" if change == "cloud" else "mog")
    rows = RoutedRows(model_forward=complete_forward, sites=("ae_bank",), features=paired_features,
                      probe_budget=2, reservoir_size=8, min_observations=1,
                      output_error_guard=True, max_output_error_increase=0., max_output_context_harm=0.)
    table = prior.z
    if change == "missing_prior": prior = None
    elif change == "missing_encoder": E = None
    elif change == "wrong_encoder": E = object()
    elif change == "wrong_table": table = nn.Parameter(table.detach().clone())
    elif change == "zero": prior.set_sigma(0.)
    elif change == "negative": prior.sigma.fill_(-.025)
    elif change == "nonfinite": prior.sigma.fill_(torch.nan)
    elif change == "sigma_grad": prior.sigma.requires_grad_(True)
    elif change == "sigma_parameter":
        width = prior.sigma.clone()
        del prior._buffers["sigma"]
        prior.register_parameter("sigma", nn.Parameter(width))
    elif change == "calibrated": prior.sigma_rel = .025
    elif change == "bool_sigma_rel": prior.sigma_rel = False
    elif change == "standardized": prior.standardize = True
    elif change == "bad_standardize_type": prior.standardize = 0
    elif change == "lying_standardize":
        prior.standardize = True
        prior.get_extra_state = lambda: {"sigma_rel": 0., "standardize": False}
    elif change == "spacing": prior.d0.fill_(1.)
    elif change == "disabled_noise": prior._noise_enabled = False
    elif change == "foreign_metadata": prior.get_extra_state = lambda: {"sigma_rel": 0., "standardize": False, "foreign": 1}
    elif change == "legacy_routing":
        rows = RoutedRows(route=lambda models, context, candidate: (context @ candidate.table.T + candidate.log_mass).softmax(1),
                          generate=lambda models, context, candidate, weights: models["generator"](candidate.codes),
                          features=paired_features)
    elif change == "empty_sites": rows.sites = ()
    elif change == "wrong_routing": rows = object()
    groups = [{"params": list(G.parameters())}]
    if isinstance(E, nn.Module): groups.append({"params": list(E.parameters())})
    groups.append({"params": [table], "lr": options.lr * options.prior_lr_mult})
    opt_g = (torch.optim.SGD(groups, lr=options.lr) if change == "sgd" else
             options.make_generator_optimizer(groups, latent_table=table, foreach=False))
    opt_d = options.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    before = len(opt_g.param_groups)
    try:
        policy = UpdatePolicy(options, G, D, prior=prior, table=table, encoder=E, router=R,
                              generator_optimizer=opt_g, critic_optimizer=opt_d,
                              routed_rows=None if change == "missing_routing" else rows,
                              row_semantics="conditional", seed=24)
    except (ValueError, TypeError):
        # Early law/owner errors must be detected before learned output noise
        # adds an optimizer group. Transport validation happens later.
        if change not in ("sgd",): assert len(opt_g.param_groups) == before
        raise
    fit = torch.linspace(-.7, .65, 16, dtype=torch.float64).reshape(8, 2)
    guard = torch.tensor([[-.9, .3], [.81, -.4], [.1, .92], [-.12, -.83]], dtype=torch.float64)
    target = lambda context: context @ context.new_tensor([[.8, .1], [-.2, .6]]) + .1
    return SimpleNamespace(policy=policy, fit=fit, targets=target(fit), guard=guard, guard_targets=target(guard))


def update(loop):
    p = loop.policy
    batch = RoutedBatch(loop.fit, loop.targets, loop.guard, loop.guard_targets)
    noise = p.begin_step(loop.targets, routed=batch)
    loss = p.recipe.make_loss()
    with torch.no_grad():
        prediction = p.routed_generate(loop.fit, sigma=0, perturb=True)
        real = noise.output_sigma * torch.randn(loop.targets.shape, dtype=p.dtype, generator=p.noise_generator)
        fake = real + prediction - loop.targets
    p.observe_critic_pair(real, fake)
    adversarial = loss.d_loss(p.D(real), p.D(fake))
    penalty = p.penalty(p.D, real, fake)
    p.opt_d.zero_grad(set_to_none=True)
    p.before_critic_backward()
    (adversarial + penalty).backward()
    p.opt_d.step()
    p.after_critic_step()
    flags = [v.requires_grad for v in p.D.parameters()]
    try:
        p.D.requires_grad_(False)
        prediction = p.routed_generate(loop.fit, sigma=0, perturb=True)
        real = noise.output_sigma * torch.randn(loop.targets.shape, dtype=p.dtype, generator=p.noise_generator)
        with torch.no_grad(): real_logits = p.D(real.detach())
        game = loss.g_loss(p.D(real + prediction - loop.targets), real_logits)
        p.opt_g.zero_grad(set_to_none=True)
        p.before_generator_backward()
        game.backward()
        p.after_generator_backward(loss_gan=game.detach(), loss_critic=adversarial.detach())
        p.opt_g.step()
        p.after_generator_step()
    finally:
        for value, flag in zip(p.D.parameters(), flags): value.requires_grad_(flag)
    event = p.finish_step()
    return {"critic": float(adversarial.detach()), "generator": float(game.detach()), "event": event}


@pytest.mark.parametrize("encoder", ["none", "ae", "hard", "categorical"])
def test_independent_mog_birth_death_is_still_refused(encoder):
    with pytest.raises(ValueError, match="particle_birth_death requires prior_kind='particles'"):
        recipe(row_policy="independent", encoder_mode=encoder)


@pytest.mark.parametrize("changes", [
    {"encoder_mode": "none"}, {"encoder_mode": "hard"}, {"encoder_mode": "categorical"},
    {"sigma_rel": .025}, {"sigma_rel": False}, {"standardize": True}, {"model": "ddgan"},
    {"conditioning": "conditional", "num_classes": 2},
])
def test_other_routed_mog_laws_are_not_admitted(changes):
    with pytest.raises(ValueError, match="routed MoG requires"):
        recipe(**changes)


@pytest.mark.parametrize("change", [
    "missing_prior", "cloud", "wrong_table", "missing_encoder", "wrong_encoder",
    "zero", "negative", "nonfinite", "sigma_grad", "sigma_parameter", "calibrated",
    "bool_sigma_rel", "standardized", "bad_standardize_type", "lying_standardize", "spacing", "disabled_noise",
    "foreign_metadata", "missing_routing", "legacy_routing", "empty_sites", "wrong_routing",
])
def test_actual_fixed_width_owner_and_complete_context_are_required(change):
    with pytest.raises((ValueError, TypeError)):
        make_policy(change=change)


def test_actual_mog_cannot_be_mislabelled_as_particle_cloud():
    with pytest.raises(ValueError, match="explicit scalar fixed-width AE recipe"):
        make_policy(recipe(prior_kind="particles", encoder_mode="none"))


def test_existing_transport_and_protected_guard_checks_are_not_bypassed():
    with pytest.raises(ValueError, match="transport supports Adam"):
        make_policy(change="sgd")
    loop = make_policy()
    update(loop)  # Public first-shape selection precedes the bad-batch probe.
    before = loop.policy.state_dict()
    with pytest.raises(ValueError, match="overlap"):
        loop.policy.begin_step(loop.targets, routed=RoutedBatch(loop.fit, loop.targets, loop.fit.clone(), loop.targets.clone()))
    same(before, loop.policy.state_dict())
    with pytest.raises(ValueError, match="RoutedBatch"):
        loop.policy.begin_step(loop.targets)
    same(before, loop.policy.state_dict())


def test_declared_routing_site_must_actually_run():
    loop = make_policy()
    loop.policy._routed_rows.model_forward = lambda models, context, candidate, routing: models["generator"](context)
    with pytest.raises(ValueError, match="routing site"):
        loop.policy.routed_generate(loop.fit, sigma=0, perturb=False)


def test_real_routed_ae_lifecycle_checkpoint_and_next_update_match():
    loop = make_policy()
    p = loop.policy
    assert p.birth_death is p.routed_control and p.row_evidence is p.routed_control.evidence
    assert p.row_semantics == "conditional" and p.prior.z is p.table
    assert p.recipe.particle_birth_death and p.recipe.row_evidence_gate
    initial = p.table.detach().clone()
    update(loop)
    saved = p.state_dict()
    unchanged = deepcopy(saved)
    assert saved["models"]["prior"]["_extra_state"] == {"sigma_rel": 0., "standardize": False}
    assert saved["routing"]["probe_clock"]["observed_updates"] == 1
    assert saved["routing"]["evidence"]["counters"]["updates"] == 1
    assert p.table.grad is not None and not torch.equal(initial, p.table)
    expected_output = p.served_model().routed_forward(loop.guard)
    same(saved, p.state_dict())
    expected_row = update(loop)
    expected = p.state_dict()
    expected_served = p.served_model().routed_forward(loop.guard)
    buffer = io.BytesIO()
    torch.save(saved, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
    resumed = make_policy()
    resumed.policy.load_state_dict(checkpoint)
    same(saved, resumed.policy.state_dict())
    same(expected_output, resumed.policy.served_model().routed_forward(resumed.guard))
    same(expected_row, update(resumed))
    same(expected, resumed.policy.state_dict())
    same(expected_served, resumed.policy.served_model().routed_forward(resumed.guard))
    same(unchanged, saved)
    with pytest.raises(ValueError, match="independent|routed"):
        resumed.policy.served_model().sample(2)


@pytest.mark.parametrize("family", ["models", "averages"])
@pytest.mark.parametrize("change", [
    "missing", "foreign", "scalar", "sigma_rel", "sigma_rel_bool", "sigma_rel_int",
    "standardize", "standardize_int", "metadata_tensor", "sigma", "sigma_nan", "sigma_grad", "d0",
])
def test_malformed_mog_metadata_or_fixed_width_restore_is_atomic(family, change):
    loop = make_policy()
    update(loop)
    p = loop.policy
    before = p.state_dict()
    selected = p.served_snapshot()
    bad = deepcopy(before)
    prior = bad[family]["prior"]
    if change == "missing": del prior["_extra_state"]
    elif change == "foreign": prior["_extra_state"]["foreign"] = 1
    elif change == "scalar": prior["_extra_state"] = .0
    elif change == "sigma_rel": prior["_extra_state"]["sigma_rel"] = .025
    elif change == "sigma_rel_bool": prior["_extra_state"]["sigma_rel"] = False
    elif change == "sigma_rel_int": prior["_extra_state"]["sigma_rel"] = 0
    elif change == "standardize": prior["_extra_state"]["standardize"] = True
    elif change == "standardize_int": prior["_extra_state"]["standardize"] = 0
    elif change == "metadata_tensor": prior["_extra_state"] = torch.zeros(2)
    elif change == "sigma": prior["sigma"].fill_(.03)
    elif change == "sigma_nan": prior["sigma"].fill_(torch.nan)
    elif change == "sigma_grad": prior["sigma"].requires_grad_(True)
    else: prior["d0"].fill_(.1)
    with pytest.raises(ValueError): p.load_state_dict(bad)
    same(before, p.state_dict())
    same(selected, p.served_snapshot())


def test_arbitrary_non_tensor_module_state_remains_unsupported():
    class ExtraEncoder(nn.Linear):
        def get_extra_state(self): return {"sigma_rel": 0., "standardize": False}
        def set_extra_state(self, state): pass
    loop = make_policy()
    # Rebind an otherwise valid actual encoder before constructing its public
    # owner; foreign metadata cannot masquerade as the MoG prior exception.
    encoder = ExtraEncoder(2, 4).double()
    encoder.load_state_dict({**loop.policy.encoder.state_dict(), "_extra_state": encoder.get_extra_state()})
    G, D, prior, R, options = loop.policy.G, loop.policy.D, loop.policy.prior, loop.policy.router, loop.policy.recipe
    opt_g = options.make_generator_optimizer([{"params": list(G.parameters())},
        {"params": list(encoder.parameters())}, {"params": [prior.z]}], latent_table=prior.z, foreach=False)
    opt_d = options.make_critic_optimizer(D, ema_critic=deepcopy(D), foreach=False)
    p = UpdatePolicy(options, G, D, prior=prior, encoder=encoder, router=R,
                     generator_optimizer=opt_g, critic_optimizer=opt_d, routed_rows=loop.policy._routed_rows)
    before = p.state_dict()
    with pytest.raises(ValueError, match="encoder tensor _extra_state"):
        p.load_state_dict(before)
    same(before, p.state_dict())


def test_cpu_controls_do_not_initialize_cuda():
    assert not torch.cuda.is_initialized()

"""Dense conditional attribution, protected guards and structural transport."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import GANLoss
from particlegan.routing import RoutedBatch, RoutedRows


class Router(nn.Module):
    def __init__(self, n=4, *, extra=False):
        super().__init__()
        self.query = nn.Linear(2, 2, bias=False).double()
        self.log_mass = nn.Parameter(torch.tensor([-.9, 0., 0., 0.], dtype=torch.float64))
        if extra:
            self.head = nn.Parameter(torch.zeros(n, 2, dtype=torch.float64))
            self.frozen_key = nn.Parameter(torch.arange(n * 2, dtype=torch.bfloat16).reshape(n, 2) * .01,
                                           requires_grad=False)
            self.register_buffer("ordinal", torch.arange(n, dtype=torch.long))
            self.register_buffer("enabled", torch.ones(n, dtype=torch.bool))
        with torch.no_grad():
            self.query.weight.copy_(torch.eye(2, dtype=torch.float64) * .04)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 3, bias=False).double()
        self.score = nn.Linear(3, 1, bias=False).double()
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[.5], [1.], [-1.5]], dtype=torch.float64))
            self.score.weight.copy_(torch.tensor([[1., 1., -1.]], dtype=torch.float64))

    def features(self, x):
        return self.hidden(x).tanh()

    def forward(self, x):
        return self.score(self.features(x))


def route(models, context, candidate):
    query = models["router"].query(context)
    logits = query @ candidate.table.T / 2 ** .5
    if "head" in candidate.row_state:
        logits = logits + context @ candidate.row_state["head"].T
        logits = logits + query @ candidate.row_state["frozen_key"].to(query.dtype).T
    return (logits + candidate.log_mass).softmax(1)


def generate(models, context, candidate, weights):
    return models["generator"](candidate.codes)


def features(models, context, samples, targets):
    return models["critic"].features(samples - targets)


def components(*, extra=False, **options):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        table = nn.Parameter(torch.tensor([[-2., -.6], [1., .1], [1.1, .2], [.9, .3]], dtype=torch.float64))
        router, critic = Router(extra=extra), Critic()
        decoder = nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            decoder.weight.copy_(torch.tensor([[1., .1]], dtype=torch.float64))
    table_opt = torch.optim.Adam([table], lr=.001, amsgrad=True, foreach=False)
    router_opt = torch.optim.Adam([p for p in router.parameters() if p.requires_grad], lr=.001, amsgrad=True, foreach=False)
    # Populate optimizer moments without synthetic evidence or changing weights.
    for optimizer in (table_opt, router_opt):
        for group in optimizer.param_groups:
            old_lr, group["lr"] = group["lr"], 0.
            for p in group["params"]:
                p.grad = torch.ones_like(p)
            optimizer.step()
            group["lr"] = old_lr
        optimizer.zero_grad()
    table_opt.latent_history = torch.arange(table.numel(), dtype=table.dtype).reshape_as(table)
    models = {"generator": decoder, "critic": critic, "router": router}
    averages = {name: deepcopy(module).eval().requires_grad_(False)
                for name, module in models.items() if name != "critic"}
    avg_table = table.detach().clone()
    spec = RoutedRows(route=route, generate=generate, features=features,
                      row_parameters=("head", "frozen_key") if extra else (),
                      row_buffers=("ordinal", "enabled") if extra else (),
                      probe_budget=4, reservoir_size=32, min_observations=8, **options)
    controller = SimpleNamespace(previous_gradient=torch.ones(2), alignment=.5, last_cosine=.7)
    control = spec.bind(models=models, averaged_models=averages, table=table, averaged_table=avg_table,
                        optimizers=[router_opt, table_opt], table_optimizer=table_opt,
                        controller=controller, seed=11)
    return control


def batch(*, guard_target=1.2):
    fit = torch.linspace(.1, .8, 32, dtype=torch.float64)
    guard = torch.linspace(.85, 1.5, 32, dtype=torch.float64)
    fit = torch.stack((fit, .3 + fit * .2), 1)
    guard = torch.stack((guard, .3 + guard * .2), 1)
    return RoutedBatch(fit, torch.full((32, 1), 1.2, dtype=fit.dtype),
                       guard, torch.full((32, 1), guard_target, dtype=guard.dtype))


def observe(control, observation):
    for optimizer in control.optimizers:
        optimizer.zero_grad()
    output = control.generate(observation.context)
    critic = control.models["critic"]
    # A genuine adversarial backward supplies gradients; no evidence primers.
    loss = GANLoss().g_loss(critic(output - observation.targets), critic(torch.zeros_like(output)))
    loss.backward()
    control.observe_backward()
    control.begin(observation)


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    else:
        assert left == right


def test_tied_key_value_routing_retains_both_gradient_paths_and_mixed_code_jitter():
    control, observation = components(), batch()
    output = control.generate(observation.context)
    actual = torch.autograd.grad(output.sum(), control.table)[0]
    query = control.models["router"].query(observation.context)
    weights = (query @ control.table.T / 2 ** .5 + control.candidate().log_mass).softmax(1)
    manual = control.models["generator"](weights @ control.table)
    expected = torch.autograd.grad(manual.sum(), control.table)[0]
    detached_keys = (query @ control.table.detach().T / 2 ** .5 + control.candidate().log_mass).softmax(1)
    incomplete = torch.autograd.grad(control.models["generator"](detached_keys @ control.table).sum(), control.table)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, incomplete)
    assert actual.ne(0).any(1).all()  # genuinely dense gradients, no argmax routing
    original = control.candidate(copy=True)
    perturbed = control.generate(observation.context, perturb_fn=lambda codes: codes + .2)
    manual = control.models["generator"](weights @ control.table + .2)
    torch.testing.assert_close(perturbed, manual)
    assert torch.equal(control.table, original.table)


def test_duplicate_split_preserves_parent_mass_and_exact_deleted_bank_function():
    control, observation = components(extra=True), batch()
    original = control.candidate(copy=True)
    deleted = control._delete(original, 0)
    split = control._split(original, 0, 2, torch.zeros(2, dtype=control.table.dtype))
    for context in (observation.context, observation.guard_context):
        torch.testing.assert_close(control.generate(context, candidate=deleted),
                                   control.generate(context, candidate=split), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(split.log_mass[[0, 2]].exp().sum(), original.log_mass[2].exp())
    assert torch.equal(split.row_state["head"][0], original.row_state["head"][2])
    assert torch.equal(split.row_state["frozen_key"][0], original.row_state["frozen_key"][2])
    assert split.row_state["ordinal"].dtype == torch.long
    assert split.row_state["enabled"].dtype == torch.bool


def test_observed_guarded_move_improves_paired_error_and_resets_coupled_state():
    control, observation = components(extra=True), batch()
    observe(control, observation)
    original, avg_original = control.candidate(copy=True), control.candidate(averaged=True, copy=True)
    moments = {name: deepcopy(optimizer.state[tensor]) for name, (tensor, _, optimizer) in control._bindings.items()
               if optimizer is not None}
    history = control.table_optimizer.latent_history.clone()
    rng = torch.get_rng_state().clone()
    event = control.maybe_apply()
    assert event["accepted"] and event["moves"] == 2 and event["child"] == 0
    assert event["variant"] == "antisymmetric"
    assert event["guard_error_after"] < event["guard_error_before"] * .3
    assert event["average_guard_error_after"] < event["average_guard_error_before"] * .3
    assert max(event["max_context_harm"], event["average_max_context_harm"]) <= 0
    parent, child = event["parent"], event["child"]
    assert not torch.equal(control.table[parent], original.table[parent])
    assert not torch.equal(control.table[parent], control.table[child])
    torch.testing.assert_close(control.table[[parent, child]].mean(0), original.table[parent])
    torch.testing.assert_close(control.averaged_table[[parent, child]].mean(0), avg_original.table[parent])
    rows = control.moved_rows
    untouched = torch.ones(len(control.table), dtype=torch.bool)
    untouched[rows] = False
    for name, (tensor, _, optimizer) in control._bindings.items():
        if optimizer is None:
            continue
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert not optimizer.state[tensor][key][rows].any()
            assert torch.equal(optimizer.state[tensor][key][untouched], moments[name][key][untouched])
        assert torch.equal(optimizer.state[tensor]["step"], moments[name]["step"])
    assert not control.table_optimizer.latent_history[rows].any()
    assert torch.equal(control.table_optimizer.latent_history[untouched], history[untouched])
    assert control.evidence.counters["global_resets"] == 1
    assert not control.evidence.valid and not control.evidence.effect_weight.any()
    assert not control.evidence.M.any() and not control.evidence.flag.any()
    assert control.controller.previous_gradient is None and control.controller.alignment == 0
    assert torch.equal(torch.get_rng_state(), rng)


def test_protected_contexts_only_accept_or_reject_and_averages_are_also_guarded():
    accepted, rejected = components(), components()
    observe(accepted, batch())
    observe(rejected, batch(guard_target=0.))
    before = rejected.candidate(copy=True)
    event_a, event_b = accepted.maybe_apply(), rejected.maybe_apply()
    assert event_a["accepted"] and not event_b["accepted"]
    assert (event_a["child"], event_a["parent"], event_a["variant"]) == (event_b["child"], event_b["parent"], event_b["variant"])
    assert torch.equal(rejected.table, before.table) and torch.equal(rejected.candidate().log_mass, before.log_mass)
    assert rejected.evidence.effect[0] > 0 and rejected.evidence.effect[1:].lt(0).all()
    # Both high-mass useful rows and the bad row have dense responsibilities;
    # paired deletion harm, rather than the responsibilities, establishes support.
    assert rejected.evidence.diagnostics()["support"][0] == 0
    assert rejected.evidence.diagnostics()["support"][1:].gt(0).all()

    average_rejected = components()
    observe(average_rejected, batch())
    with torch.no_grad():
        average_rejected.averaged_models["generator"].weight.mul_(4)
    event = average_rejected.maybe_apply()
    assert not event["accepted"] and event["guard_gain"] > 0 and event["average_guard_gain"] < 0


def test_replay_effective_contexts_do_not_count_repeated_reservoir_visits():
    control = components(improvement_margin=100.)
    observe(control, batch())
    control.maybe_apply()
    first = control.evidence.effective_contexts.clone()
    control.rows_since_eval = 8
    control.maybe_apply()
    torch.testing.assert_close(control.evidence.effective_contexts, first, rtol=0, atol=0)


def test_exact_resume_before_structural_event_keeps_evidence_identity():
    control = components(extra=True)
    observe(control, batch())
    saved = control.state_dict()
    assert "row_state" not in saved and "row_ownership" in saved  # weights remain owner-owned
    event = control.maybe_apply()
    expected = control.state_dict()
    restored = components(extra=True)
    identity = restored.evidence
    restored.load_state_dict(saved)
    assert restored.evidence is identity
    actual = restored.maybe_apply()
    same(event, actual)
    same(expected, restored.state_dict())
    same(control.table, restored.table)
    same(control.candidate().row_state, restored.candidate().row_state)
    same(control.averaged_table, restored.averaged_table)
    for owner, recovered in zip(control.optimizers, restored.optimizers):
        same(owner.state_dict(), recovered.state_dict())


def test_guard_overlap_and_malformed_observations_are_rejected_atomically():
    control, observation = components(), batch()
    before = control.state_dict()
    for invalid in (RoutedBatch(observation.context, observation.targets, observation.context.clone(), observation.targets.clone()),
                    RoutedBatch(observation.context, observation.targets, observation.guard_context, observation.guard_targets[:1]),
                    RoutedBatch(observation.context, observation.targets, observation.guard_context * float("nan"), observation.guard_targets)):
        with pytest.raises(ValueError):
            control.begin(invalid)
        same(before, control.state_dict())
    control.begin(observation)
    before = control.state_dict()
    with pytest.raises(ValueError, match="overlap"):
        control.begin(RoutedBatch(observation.guard_context.clone(), observation.targets,
                                 observation.guard_context + 10, observation.guard_targets))
    same(before, control.state_dict())
    bad = deepcopy(before)
    bad["pools"]["guard_context"] = bad["pools"]["fit_context"].clone()
    with pytest.raises(ValueError, match="overlap"):
        control.load_state_dict(bad)
    same(before, control.state_dict())


def test_callbacks_must_honor_mass_bias_and_deterministic_paired_evaluation():
    control = components()
    control.spec.route = lambda models, context, candidate: (models["router"].query(context) @ candidate.table.T).softmax(1)
    control.begin(batch())
    before = control.table.detach().clone()
    with pytest.raises(ValueError, match="candidate.log_mass"):
        control.maybe_apply()
    assert torch.equal(control.table, before)
    stochastic = components()
    stochastic.spec.features = lambda models, context, samples, targets: features(models, context, samples, targets) + torch.randn(len(context), 3)
    stochastic.begin(batch())
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="deterministic"):
        stochastic.maybe_apply()
    assert torch.equal(torch.get_rng_state(), rng)


def test_storage_alias_and_bad_latent_history_owners_reject_before_binding():
    control = components()
    router = control.models["router"]
    del router.log_mass
    router.register_buffer("log_mass", control.table.detach()[:, 0])
    arguments = {"models": control.models, "averaged_models": control.averaged_models,
                 "table": control.table, "averaged_table": control.averaged_table,
                 "optimizers": control.optimizers, "table_optimizer": control.table_optimizer}
    with pytest.raises(ValueError, match="storage"):
        control.spec.bind(**arguments)
    normal = components()
    normal.table_optimizer.latent_history = torch.zeros(1, 2, dtype=normal.table.dtype)
    with pytest.raises(ValueError, match="latent_history"):
        normal.spec.bind(models=normal.models, averaged_models=normal.averaged_models, table=normal.table,
                         averaged_table=normal.averaged_table, optimizers=normal.optimizers,
                         table_optimizer=normal.table_optimizer)

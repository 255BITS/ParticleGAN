"""Opt-in raw paired-output guards supplement learned-feature acceptance."""
from copy import deepcopy
import io
import math
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan import GANLoss, RoutedBatch, RoutedRows


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            same(a, b)
    elif isinstance(left, float) and math.isnan(left):
        assert math.isnan(right)
    else:
        assert left == right


def cpu_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.fixture(autouse=True)
def serial_backward():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(previous)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log_mass", torch.zeros(2, dtype=torch.float64))


class ProjectedCritic(nn.Module):
    """A learned feature that currently ignores the second output channel."""
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 1, bias=False).double()
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[1., 0.]], dtype=torch.float64))

    def forward(self, residual):
        return self.projection(residual).flatten(1).mean(1, keepdim=True)


def route(models, context, candidate):
    return candidate.log_mass.expand(len(context), -1).softmax(-1)


def generate(models, context, candidate, weights):
    return models["generator"](candidate.codes)


def model_forward(models, context, candidate, routing):
    logits = context.new_zeros((*context.shape[:-1], len(candidate.table)))
    first = routing.mix("first", logits)
    second = routing.mix("second", logits + first[..., :1] * 0.)
    output = models["generator"](second)
    return torch.stack((output[..., 0], output[..., 1] * context[..., 1]), dim=-1)


def features(models, context, samples, targets):
    return models["critic"].projection(samples - targets).flatten(1)


def components(*, output_error_guard=False, bad_fast=True, bad_average=True,
               tokens=False, **options):
    # Deleting row 1 improves the critic's visible channel from 1 to 0.
    # When the invisible channel is present, raw output MSE worsens .5 -> 2.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        decoder = nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            decoder.weight.copy_(torch.eye(2, dtype=torch.float64))
        models = {"generator": decoder, "router": Router(), "critic": ProjectedCritic()}
    bad = [[0., 2.], [2., -2.]]
    good = [[0., 0.], [2., 0.]]
    table = nn.Parameter(torch.tensor(bad if bad_fast else good, dtype=torch.float64))
    average = torch.tensor(bad if bad_average else good, dtype=torch.float64)
    averaged_models = {name: deepcopy(module).eval().requires_grad_(False)
                       for name, module in models.items() if name != "critic"}
    optimizer = torch.optim.Adam([table], lr=0., amsgrad=True, foreach=False)
    spec = RoutedRows(features=features, model_forward=model_forward, sites=("first", "second"),
                      probe_budget=2, reservoir_size=8, min_observations=8, split_scale=0.,
                      output_error_guard=output_error_guard, **options) if tokens else RoutedRows(
        route=route, generate=generate, features=features, probe_budget=2, reservoir_size=8,
        min_observations=8, split_scale=0., output_error_guard=output_error_guard, **options)
    controller = SimpleNamespace(previous_gradient=torch.ones(2), alignment=.5, last_cosine=.7)
    control = spec.bind(models=models, averaged_models=averaged_models, table=table,
                        averaged_table=average, optimizers=[optimizer], table_optimizer=optimizer,
                        controller=controller, seed=11)
    if tokens:
        fit = torch.stack((torch.arange(8, dtype=table.dtype), torch.ones(8, dtype=table.dtype)), -1)
        guard = fit.clone()
        guard[:, 0] += 100.
        fit, guard = fit[:, None].expand(-1, 3, -1).clone(), guard[:, None].expand(-1, 3, -1).clone()
        guard[-1, :, 1] = torch.tensor([1., 2., 3.], dtype=table.dtype)
        target = torch.zeros(8, 3, 2, dtype=table.dtype)
    else:
        fit = torch.arange(16, dtype=table.dtype).reshape(8, 2)
        guard = fit + 100.
        target = torch.zeros_like(fit)
    observation = RoutedBatch(fit, target, guard, target.clone())
    return control, observation


def observe(control, observation):
    prediction = control.generate(observation.context)
    critic = control.models["critic"]
    GANLoss().g_loss(critic(prediction - observation.targets), critic(torch.zeros_like(prediction))).backward()
    control.observe_backward()
    # Populate actual Adam/AMSGrad history without changing this counterexample.
    control.table_optimizer.step()
    control.table_optimizer.zero_grad()
    control.begin(observation)


def owner_state(control):
    return {"models": {name: deepcopy(module.state_dict()) for name, module in control.models.items()},
            "averages": {name: deepcopy(module.state_dict()) for name, module in control.averaged_models.items()},
            "table": control.table.detach().clone(), "average_table": control.averaged_table.clone(),
            "optimizer": deepcopy(control.table_optimizer.state_dict()),
            "controller": deepcopy(vars(control.controller))}


def test_feature_improvement_can_worsen_output_but_opt_in_rejects_without_owner_mutation(monkeypatch):
    legacy, observation = components()
    guarded, guarded_observation = components(output_error_guard=True)
    observe(legacy, observation)
    observe(guarded, guarded_observation)
    legacy_event = legacy.maybe_apply()
    assert legacy_event["accepted"] and legacy_event["moves"] == 2
    assert legacy_event["guard_error_before"] == 1. and legacy_event["guard_error_after"] == 0.
    assert (legacy.generate(observation.guard_context) - observation.guard_targets).square().mean() == 2.

    before = owner_state(guarded)
    flags = [module.training for module in (*guarded.models.values(), *guarded.averaged_models.values())]
    measurement_requests = []
    original = guarded._measure

    def measured(context, targets, candidate, **kwargs):
        measurement_requests.append((bool(kwargs.get("with_output_error")), candidate.averaged,
                                     bool(torch.equal(context, guarded_observation.guard_context))))
        assert all(not module.training for module in (*guarded.models.values(), *guarded.averaged_models.values()))
        return original(context, targets, candidate, **kwargs)

    monkeypatch.setattr(guarded, "_measure", measured)
    rng = torch.get_rng_state().clone()
    event = guarded.maybe_apply()
    assert event["feature_guard_accepted"] and not event["output_guard_accepted"]
    assert not event["accepted"] and event["moves"] == 0
    assert event["guard_output_mse_before"] == .5 and event["guard_output_mse_after"] == 2.
    assert event["average_guard_output_mse_before"] == .5 and event["average_guard_output_mse_after"] == 2.
    assert sum(output for output, _, _ in measurement_requests) == 4
    assert all(protected for output, _, protected in measurement_requests if output)
    assert [averaged for output, averaged, _ in measurement_requests if output] == [False, False, True, True]
    assert guarded.counters["guard_rejections"] == 1 and guarded.counters["moves"] == 0
    assert guarded.moved_rows is None and guarded.moved_parameters == {}
    same(before, owner_state(guarded))
    same(rng, torch.get_rng_state())
    assert flags == [module.training for module in (*guarded.models.values(), *guarded.averaged_models.values())]


@pytest.mark.parametrize("bad_fast,bad_average", [(True, False), (False, True)])
def test_output_guard_requires_both_clean_fast_and_averaged_outputs(bad_fast, bad_average):
    control, observation = components(output_error_guard=True, bad_fast=bad_fast, bad_average=bad_average)
    observe(control, observation)
    before = owner_state(control)
    event = control.maybe_apply()
    assert event["feature_guard_accepted"] and not event["output_guard_accepted"]
    assert event["guard_output_mse_increase"] == (1.5 if bad_fast else -.5)
    assert event["average_guard_output_mse_increase"] == (1.5 if bad_average else -.5)
    same(before, owner_state(control))


@pytest.mark.parametrize("mean_bound,context_bound,accepted", [(0., 100., False), (3., 3., False), (3., 9., True)])
def test_full_model_guard_means_all_tokens_channels_then_checks_each_context(mean_bound, context_bound, accepted):
    control, observation = components(output_error_guard=True, tokens=True,
                                     max_output_error_increase=mean_bound,
                                     max_output_context_harm=context_bound)
    observe(control, observation)
    before = owner_state(control)
    event = control.maybe_apply()
    assert event["feature_guard_accepted"]
    assert event["output_guard_accepted"] is accepted and event["accepted"] is accepted
    assert event["guard_output_mse_before"] == .5
    assert event["guard_output_mse_after"] == pytest.approx(35 / 12)
    assert event["guard_output_mse_increase"] == pytest.approx(29 / 12)
    assert event["max_output_context_harm"] == pytest.approx(53 / 6)
    assert event["average_guard_output_mse_after"] == pytest.approx(35 / 12)
    if not accepted:
        same(before, owner_state(control))


def test_output_guard_does_not_replace_the_learned_feature_guard():
    control, observation = components(output_error_guard=True, bad_fast=False, bad_average=False)
    # Deletion helps fitting features; a different protected target makes that
    # same change harmful. Permissive output tolerances cannot waive features.
    observation.guard_targets[:, 0] = 2.
    control.spec.max_output_error_increase = control.spec.max_output_context_harm = 100.
    observe(control, observation)
    before = owner_state(control)
    event = control.maybe_apply()
    assert event["output_guard_accepted"] and not event["feature_guard_accepted"]
    assert not event["accepted"] and event["moves"] == 0
    same(before, owner_state(control))


@pytest.mark.parametrize("options", [{"output_error_guard": 1}, {"max_output_error_increase": .1},
                                     {"max_output_context_harm": .1},
                                     {"output_error_guard": True, "max_output_error_increase": float("nan")},
                                     {"output_error_guard": True, "max_output_context_harm": -.1}])
def test_output_guard_configuration_is_explicit_and_finite(options):
    with pytest.raises(ValueError, match="routed output|routed max_output"):
        components(**options)


def test_opt_in_config_roundtrips_and_incompatible_restore_is_atomic():
    legacy, _ = components()
    enabled, observation = components(output_error_guard=True,
                                     max_output_error_increase=.125, max_output_context_harm=.25)
    assert not {"output_error_guard", "max_output_error_increase", "max_output_context_harm"} & legacy.spec.to_dict().keys()
    assert {name: enabled.spec.to_dict()[name] for name in
            ("output_error_guard", "max_output_error_increase", "max_output_context_harm")} == {
        "output_error_guard": True, "max_output_error_increase": .125, "max_output_context_harm": .25}
    observe(enabled, observation)
    enabled.maybe_apply()
    saved = cpu_roundtrip(enabled.state_dict())
    restored, _ = components(output_error_guard=True, max_output_error_increase=.125, max_output_context_harm=.25)
    restored.load_state_dict(saved)
    same(saved, restored.state_dict())
    for destination in (legacy, components(output_error_guard=True)[0]):
        before = destination.state_dict()
        owners = owner_state(destination)
        with pytest.raises(ValueError, match="configuration"):
            destination.load_state_dict(saved)
        same(before, destination.state_dict())
        same(owners, owner_state(destination))


def test_opted_in_paired_game_exact_cpu_checkpoint_resume_and_served_outputs():
    api = runpy.run_path(str(EXAMPLES / "e22_routed_sites.py"))
    loop = api["make_loop"](output_error_guard=True)
    assert loop.config["output_error_guard"] is True
    rows, saved = [], None
    for index in range(8):
        rows.append(api["update"](loop))
        if index == 0:
            saved = cpu_roundtrip(api["checkpoint"](loop))
    assert loop.policy.routed_control.counters["splits"] > 0
    assert any(row["move"] and row["move"].get("moves", 0)
               and row["move"]["feature_guard_accepted"] and row["move"]["output_guard_accepted"] for row in rows[1:])
    expected = api["checkpoint"](loop)
    output = loop.policy.served_model().routed_forward(loop.test_context)
    restored = api["make_loop"](output_error_guard=True)
    api["restore"](restored, saved)
    for row in rows[1:]:
        same(row, api["update"](restored))
    same(expected, api["checkpoint"](restored))
    same(output, restored.policy.served_model().routed_forward(restored.test_context))
    legacy = api["make_loop"]()
    assert "output_error_guard" not in legacy.config
    before = api["checkpoint"](legacy)
    with pytest.raises(ValueError, match="configuration"):
        legacy.policy.load_state_dict(saved["policy"])
    same(before, api["checkpoint"](legacy))


@pytest.mark.parametrize("mode", ["frozen", "no_rows"])
def test_controls_off_never_read_guard_metrics_even_when_opted_in(mode, monkeypatch):
    api = runpy.run_path(str(EXAMPLES / "e22_routed_sites.py"))
    loop = api["make_loop"](mode=mode, output_error_guard=True)
    control = loop.policy.routed_control

    def forbidden(*args, **kwargs):
        pytest.fail("row controls are disabled; no feature/output guard may run")

    monkeypatch.setattr(control, "_measure", forbidden)
    for _ in range(2):
        api["update"](loop)
    assert control.fit_fill == control.guard_fill == 0
    assert control.counters["probes"] == control.counters["proposals"] == 0


def test_evidence_only_never_reads_protected_outputs_when_opted_in(monkeypatch):
    api = runpy.run_path(str(EXAMPLES / "e22_routed_paired.py"))
    build = api["make_loop"]
    row_constructor = build.__globals__["RoutedRows"]
    monkeypatch.setitem(build.__globals__, "RoutedRows",
                        lambda **kwargs: row_constructor(output_error_guard=True, **kwargs))
    loop = build(recipe_overrides={"particle_birth_death": False})
    control = loop.policy.routed_control
    original = control._measure
    calls = []

    def measured(context, targets, candidate, **kwargs):
        assert not kwargs.get("with_output_error", False)
        assert not (context[:, None] == loop.guard_context[None]).all(-1).any()
        calls.append(context.detach().clone())
        return original(context, targets, candidate, **kwargs)

    monkeypatch.setattr(control, "_measure", measured)
    for _ in range(3):
        api["update"](loop)
    assert calls and control.counters["probes"] > 0
    assert control.counters["proposals"] == control.counters["moves"] == 0
    assert control.last["evidence_only"]
    assert not any("guard_output" in key for key in control.last)

"""Structural probe cadence counts training updates independently of samples."""
from copy import deepcopy
from functools import partial
import io
from pathlib import Path
import runpy

import pytest
import torch
from torch import nn

from particlegan import GANLoss, RoutedBatch, RoutedRows


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
        self.register_buffer("log_mass", torch.tensor([-.9, 0., 0., 0.], dtype=torch.float64))


def route(models, context, candidate):
    return (.04 * context @ candidate.table.T + candidate.log_mass).softmax(1)


def generate(models, context, candidate, weights):
    return models["generator"](candidate.codes)


def features(models, context, samples, targets):
    return models["critic"](samples - targets)


def make_control(*, probe_interval=1, min_observations=8, probe_budget=2):
    with torch.random.fork_rng(devices=[]):
        decoder = nn.Linear(2, 1, bias=False).double()
        critic = nn.Linear(1, 3, bias=False).double()
        with torch.no_grad():
            decoder.weight.copy_(torch.tensor([[1., .1]], dtype=torch.float64))
            critic.weight.copy_(torch.tensor([[.5], [1.], [-1.5]], dtype=torch.float64))
    table = nn.Parameter(torch.tensor([[-2., -.6], [1., .1], [1.1, .2], [.9, .3]], dtype=torch.float64))
    models = {"generator": decoder, "critic": critic, "router": Router()}
    averages = {name: deepcopy(module).requires_grad_(False) for name, module in models.items() if name != "critic"}
    optimizer = torch.optim.Adam([table], lr=.001, foreach=False)
    rows = RoutedRows(route=route, generate=generate, features=features,
                      probe_interval=probe_interval, min_observations=min_observations,
                      probe_budget=probe_budget, reservoir_size=32, improvement_margin=100.)
    return rows.bind(models=models, averaged_models=averages, table=table,
                     averaged_table=table.detach().clone(), optimizers=(optimizer,), table_optimizer=optimizer)


def observation(step, size):
    x = torch.linspace(.1, .7, size, dtype=torch.float64) + step * .001
    fit = torch.stack((x, .3 + x * .2), 1)
    guard = fit + 100
    target = torch.full((size, 1), 1.2, dtype=torch.float64)
    return RoutedBatch(fit, target, guard, target.clone())


def update(control, step, size=16, *, mutate=False):
    batch = observation(step, size)
    control.begin(batch)
    for module in control.models.values():
        module.zero_grad(set_to_none=True)
    control.table_optimizer.zero_grad(set_to_none=True)
    prediction = control.generate(batch.context)
    critic = control.models["critic"]
    loss = GANLoss().g_loss(critic(prediction - batch.targets).tanh().sum(1, keepdim=True),
                          critic(torch.zeros_like(prediction)).tanh().sum(1, keepdim=True))
    loss.backward()
    control.observe_backward()
    return control.maybe_apply(mutate=mutate)


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


def roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("size,min_observations", [(32, 1), (32, 8), (2, 8)])
def test_probe_interval_counts_updates_not_contexts_and_gradient_evidence_keeps_updating(size, min_observations):
    control = make_control(probe_interval=4, min_observations=min_observations)
    events = []
    for step in range(1, 13):
        event = update(control, step, size)
        if event is not None:
            events.append(step)
        assert control.evidence.counters["updates"] == step
        assert control.probe_clock["observed_updates"] == step
        assert int(control.evidence.touches.min()) == step
    assert events == [4, 8, 12]
    assert control.counters["evals"] == 3 and control.counters["probes"] == 6
    assert control.counters["proposals"] == control.counters["moves"] == 0
    assert control.probe_clock["last_probe_update"] == 12


def test_context_eligibility_delays_pass_without_rounding_to_cadence_modulo():
    control = make_control(probe_interval=4, min_observations=5)
    for step in range(1, 5):
        assert update(control, step, 1) is None
    assert control.rows_since_eval == 4 and control.probe_clock["last_probe_update"] == 0
    assert update(control, 5, 1)["evidence_only"]
    assert control.probe_clock["last_probe_update"] == 5
    for step in range(6, 9):
        assert update(control, step, 8) is None
    assert update(control, 9, 8)["evidence_only"]
    assert control.probe_clock["last_probe_update"] == 9  # not 8 or 12
    assert control.maybe_apply(mutate=False) is None  # no second pass in this update


def test_variable_batches_and_skipped_calls_preserve_accumulated_context_eligibility():
    control = make_control(probe_interval=4)
    sizes = (1, 1, 12, 1, 1, 1, 1, 5)
    events = []
    for step, size in enumerate(sizes, 1):
        event = update(control, step, size)
        if event is not None:
            events.append(step)
        if step == 3:
            assert control.rows_since_eval == 14
            assert control.counters["evals"] == 0
        if step == 7:
            assert control.rows_since_eval == 3
    assert events == [4, 8]
    assert control.counters["evals"] == 2


def test_probe_cursor_and_counterfactual_freshness_advance_only_on_scheduled_passes():
    control = make_control(probe_interval=3, probe_budget=1)
    stamps = torch.full((4,), -1, dtype=torch.long)
    for step in range(1, 13):
        old_effects = control.evidence.effect_sum.clone()
        event = update(control, step)
        if step % 3 == 0:
            row = step // 3 - 1
            stamps[row] = step // 3
            assert event is not None
            assert control.probe_cursor == (row + 1) % 4
        else:
            assert event is None
            same(old_effects, control.evidence.effect_sum)
        same(stamps, control.evidence.last_probe)
        assert control.counters["probes"] == step // 3
    assert control.evidence.effect_contexts.eq(32).all()


def test_birth_death_candidate_evaluations_follow_the_same_probe_cadence():
    control = make_control(probe_interval=3, probe_budget=4)
    passes = []
    for step in range(1, 10):
        previous_proposals = control.counters["proposals"]
        event = update(control, step, mutate=True)
        if event is None:
            assert control.counters["proposals"] == previous_proposals
        else:
            passes.append(step)
            assert control.counters["proposals"] > previous_proposals
            assert event["moves"] == 0  # deliberately large fit improvement margin
    assert passes == [3, 6, 9]
    assert control.counters["evals"] == 3 and control.counters["probes"] == 12


@pytest.mark.parametrize("cut", (3, 7))
def test_exact_resume_at_first_and_subsequent_cadence_boundaries_without_trainer_callback(cut):
    control = make_control(probe_interval=4)
    for step in range(1, cut + 1):
        update(control, step, size=step % 3 + 8)
    saved = roundtrip(control.state_dict())
    restored = make_control(probe_interval=4)
    restored.load_state_dict(saved)
    same(saved, restored.state_dict())
    for step in range(cut + 1, 13):
        event = update(control, step, size=step % 3 + 8)
        actual = update(restored, step, size=step % 3 + 8)
        same(event, actual)
        same(control.state_dict(), restored.state_dict())
    assert control.counters["evals"] == 3


def test_probe_interval_configuration_and_default_checkpoint_compatibility():
    default = make_control()
    assert "probe_interval" not in default.spec.to_dict()
    explicit_default = make_control(probe_interval=1)
    same(default.spec.to_dict(), explicit_default.spec.to_dict())
    assert make_control(probe_interval=7).spec.to_dict()["probe_interval"] == 7
    for invalid in (0, -1, 1.5, True, "4", None):
        with pytest.raises(ValueError, match="probe_interval"):
            make_control(probe_interval=invalid)
    update(default, 1)
    default.begin(observation(2, 16))
    saved = default.state_dict()
    legacy = deepcopy(saved)
    del legacy["probe_clock"]
    restored = make_control()
    restored.load_state_dict(legacy)
    assert restored.probe_clock == {"observed_updates": 0, "last_probe_update": 0}
    expected, actual = default.refresh_evidence(), restored.refresh_evidence()
    same(expected, actual)
    after, restored_after = default.state_dict(), restored.state_dict()
    del after["probe_clock"], restored_after["probe_clock"]
    same(after, restored_after)
    nondefault = make_control(probe_interval=4)
    incomplete = nondefault.state_dict()
    del incomplete["probe_clock"]
    with pytest.raises(ValueError, match="schema"):
        nondefault.load_state_dict(incomplete)


@pytest.mark.parametrize("clock", [None, {}, {"observed_updates": 1},
                                   {"observed_updates": 1, "last_probe_update": 0, "extra": 1},
                                   {"observed_updates": True, "last_probe_update": 0},
                                   {"observed_updates": 2, "last_probe_update": .5},
                                   {"observed_updates": -1, "last_probe_update": 0},
                                   {"observed_updates": 1, "last_probe_update": 2}])
def test_malformed_current_probe_clock_rejects_before_mutating_control(clock):
    control = make_control(probe_interval=4)
    update(control, 1)
    before = control.state_dict()
    malformed = deepcopy(before)
    malformed["probe_clock"] = clock
    with pytest.raises(ValueError, match="probe clock"):
        control.load_state_dict(malformed)
    same(before, control.state_dict())


@pytest.mark.parametrize("evidence", (True, False))
def test_policy_evidence_only_cadence_or_controls_off_no_passes_and_exact_replay(evidence):
    # Run the documented actual paired-error external loop with the new public
    # RoutedRows argument. Its factory otherwise owns the unchanged algorithm.
    path = Path(__file__).resolve().parents[1] / "examples" / "e22_routed_paired.py"
    api = runpy.run_path(str(path))
    api["make_loop"].__globals__["RoutedRows"] = partial(RoutedRows, probe_interval=4)
    options = {"particle_birth_death": False, "row_evidence_gate": evidence}
    loop = api["make_loop"](recipe_overrides=options, batch_size=8)
    assert loop.policy.birth_death is None
    rows, saved = [], None
    for step in range(1, 13):
        rows.append(api["update"](loop))
        if step == 7:
            saved = roundtrip(api["checkpoint"](loop))
    control = loop.policy.routed_control
    if evidence:
        assert control.evidence.counters["updates"] == 12
        assert control.counters["evals"] == 3 and control.counters["probes"] == 24
        assert control.probe_clock == {"observed_updates": 12, "last_probe_update": 12}
        assert loop.policy.row_evidence.valid
        assert [row["step"] for row in rows if row["move"] is not None] == [4, 8, 12]
    else:
        assert control.evidence.counters["updates"] == control.counters["evals"] == control.counters["probes"] == 0
        assert control.probe_clock == {"observed_updates": 0, "last_probe_update": 0}
        assert control.fit_context is None and control.guard_context is None
    assert control.counters["proposals"] == control.counters["moves"] == 0
    restored = api["make_loop"](recipe_overrides=options, batch_size=8)
    api["restore"](restored, saved)
    for expected in rows[7:]:
        same(expected, api["update"](restored))
    same(api["checkpoint"](loop), api["checkpoint"](restored))

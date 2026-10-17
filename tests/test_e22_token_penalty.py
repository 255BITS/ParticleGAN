"""Token-local KA2 units preserve pooled scores and complete contextual forwards."""
from copy import deepcopy
import io
import math
from pathlib import Path
import runpy
import sys

import pytest
import torch
from torch import nn

from particlegan import get_recipe
from particlegan.ka2 import WARMUP_CALLS


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "e22_routed_sites.py"


@pytest.fixture(scope="module")
def api():
    return runpy.run_path(str(EXAMPLE))


@pytest.fixture(autouse=True)
def serial_backward():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        torch.set_num_threads(previous)


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


class ContextCritic(nn.Module):
    """Nonlinear pooled scores couple tokens; flattening before D changes D."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3., -2.], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(.4, dtype=torch.float64))
        self.gain = nn.Parameter(torch.tensor(.2, dtype=torch.float64))

    def forward(self, error):
        assert error.ndim == 3
        pooled = (error.mean(1) * self.weight).sum(-1, keepdim=True) + self.bias
        return pooled + self.gain * pooled.square()


def inputs(tokens):
    real = torch.tensor([[.1, .2], [-.2, .3], [.4, -.1]], dtype=torch.float64)
    fake = torch.tensor([[1., -1.], [.5, .2], [-.4, -.2]], dtype=torch.float64)
    return tuple(values[:, None].repeat(1, tokens, 1) for values in (real, fake))


def pair(*, phase="a", lazy_k=1, kappa=.3):
    critic = ContextCritic()
    ema = deepcopy(critic)
    with torch.no_grad():
        ema.weight.mul_(.7)
        ema.bias.mul_(.6)
        ema.gain.mul_(.5)
    recipe = get_recipe("e22_routed", num_particles=8, z_dim=2, batch_size=3,
                        row_evidence_gate=False, particle_birth_death=False)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=ema, foreach=False)
    penalty = recipe.make_critic_penalty(optimizer, collect_stats=True,
                                         coeff=1.7, kappa=kappa, lazy_k=lazy_k)
    if phase == "blend":
        # Isolate the phase's mathematical units at a valid post-warmup record.
        # The trained fixture below reaches call800 with real optimizer steps.
        optimizer.record.calls = WARMUP_CALLS - 1
        optimizer.record.observed_steps = WARMUP_CALLS * lazy_k - 1
        optimizer.record.anchor_started = True
        optimizer.record.last_sur = .1
    return critic, optimizer, penalty


def measured(api, tokens, *, phase="a", units="token", lazy_k=1, kappa=.3):
    critic, optimizer, penalty = pair(phase=phase, lazy_k=lazy_k, kappa=kappa)
    if lazy_k > 1 and phase == "a":
        optimizer.record.observed_steps = lazy_k - 1
    real, fake = inputs(tokens)
    value = api["apply_critic_penalty"](penalty, critic, real, fake, units=units)
    gradients = torch.autograd.grad(value, tuple(critic.parameters()))
    return value.detach(), gradients, deepcopy(penalty.last_stats), optimizer.record


def test_view_preserves_full_context_grouping_scores_and_gradient_units(api):
    critic = ContextCritic()
    error = torch.arange(24, dtype=torch.float64).reshape(3, 4, 2) * .03
    view = api["TokenPenaltyView"](critic, 4)
    assert dict(view.named_children()) == {"critic": critic}
    flat = error.flatten(0, 1).clone().requires_grad_(True)
    actual = view(flat)
    expected = critic(flat.reshape(3, 4, 2)).repeat_interleave(4, dim=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Repetition multiplies the summed-score derivative by T, then KA2 sees
    # per-token dimension C. The critic itself still receives whole contexts.
    gradient = torch.autograd.grad(actual.sum(), flat)[0]
    context = error.clone().requires_grad_(True)
    original_gradient = torch.autograd.grad(critic(context).sum(), context)[0]
    torch.testing.assert_close(gradient.reshape_as(error), original_gradient * 4, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["a", "blend"])
@pytest.mark.parametrize("tokens", [1, 8, 128])
def test_replicated_tokens_keep_A_B_caps_EMA_proximity_and_parameter_gradients(api, phase, tokens):
    reference, reference_gradients, reference_stats, _ = measured(api, 1, phase=phase, units="context")
    actual, gradients, stats, record = measured(api, tokens, phase=phase)
    torch.testing.assert_close(actual, reference, rtol=2e-12, atol=1e-12)
    for gradient, expected in zip(gradients, reference_gradients):
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, expected, rtol=3e-12, atol=1e-12)
    assert stats["phase"] == phase and stats["applied"]
    assert record.calls == (1 if phase == "a" else 800)
    critic, _, _ = pair()
    for values in inputs(1):
        values.requires_grad_(True)
        gradient = torch.autograd.grad(critic(values).sum(), values)[0].flatten(1)
        assert (gradient.norm(dim=1) / math.sqrt(2) > .3).all()  # A fake caps active
        assert (gradient.norm(dim=1) > .3).all()  # both B L2 caps active
    if phase == "blend":
        assert stats["prox"] > 0
        assert stats["prox"] == pytest.approx(reference_stats["prox"], rel=2e-12)
    else:
        assert stats["prox"] == 0.


@pytest.mark.parametrize("tokens", [8, 128])
def test_original_context_real_R1_scaling_remains_an_explicit_legacy_convention(api, tokens):
    # Disable caps to isolate real R1. Original context units weaken it by T².
    one, _, _, _ = measured(api, 1, units="context", kappa=1e6)
    context, _, _, _ = measured(api, tokens, units="context", kappa=1e6)
    token, _, _, _ = measured(api, tokens, kappa=1e6)
    torch.testing.assert_close(context * tokens ** 2, one, rtol=2e-12, atol=1e-12)
    torch.testing.assert_close(token, one, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize("phase", ["a", "blend"])
def test_lazy_calls_skip_without_advancing_applied_clock_and_scale_all_terms(api, phase):
    critic, optimizer, penalty = pair(phase=phase, lazy_k=3)
    real, fake = inputs(128)
    boundary = 3 if phase == "a" else 2400
    calls = optimizer.record.calls
    for completed in (boundary - 3, boundary - 2):
        optimizer.record.observed_steps = completed
        state = deepcopy(optimizer.record.state_dict())
        skipped = api["apply_critic_penalty"](penalty, critic, real, fake)
        assert skipped == 0 and not skipped.requires_grad
        assert penalty.last_stats["applied"] is False
        same(state, optimizer.record.state_dict())
    optimizer.record.observed_steps = boundary - 1
    actual = api["apply_critic_penalty"](penalty, critic, real, fake)
    gradients = torch.autograd.grad(actual, tuple(critic.parameters()))
    reference, reference_gradients, stats, _ = measured(api, 128, phase=phase)
    torch.testing.assert_close(actual, reference * 3, rtol=2e-12, atol=1e-12)
    for gradient, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(gradient, expected * 3, rtol=3e-12, atol=1e-12)
    assert optimizer.record.calls == calls + 1
    assert penalty.last_stats["prox"] == pytest.approx(stats["prox"], rel=2e-12)


@pytest.fixture(scope="module", params=("token", "context"))
def trained_boundary(api, request):
    assert WARMUP_CALLS == 800
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        loop = api["make_loop"](mode="no_rows", tokens=2, particles=8, batch_size=4,
                                 penalty_units=request.param)
        loop.policy.penalty.collect_stats = True
        tail, phases, saved = [], [], None
        with torch.autograd.set_multithreading_enabled(False):
            for index in range(806):
                row = api["update"](loop)
                phases.append(deepcopy(loop.policy.penalty.last_stats))
                if index == 797:
                    saved = cpu_roundtrip(api["checkpoint"](loop))
                if index >= 798:
                    tail.append(row)
        yield {"loop": loop, "tail": tail, "phases": phases, "saved": saved,
               "final": api["checkpoint"](loop), "output": loop.policy.served_model().routed_forward(loop.test_context)}
    finally:
        torch.set_num_threads(previous)


def test_actual_paired_training_crosses_800_and_resumes_exactly_in_both_unit_conventions(api, trained_boundary):
    fixture, loop = trained_boundary, trained_boundary["loop"]
    assert [row["phase"] for row in fixture["phases"][:799]] == ["a"] * 799
    assert all(row["phase"] == "blend" for row in fixture["phases"][799:])
    assert fixture["phases"][799]["prox"] == 0.
    assert any(row["prox"] > 0 for row in fixture["phases"][800:])
    assert loop.policy.opt_d.record.calls == loop.policy.opt_d.record.observed_steps == 806
    assert loop.policy.opt_d.record.anchor_started
    assert loop.policy.routed_control.counters["probes"] == 0
    restored = api["make_loop"](mode="no_rows", tokens=2, particles=8, batch_size=4,
                                 penalty_units=loop.config["penalty_units"])
    restored.policy.penalty.collect_stats = True
    saved = deepcopy(fixture["saved"])
    if loop.config["penalty_units"] == "context":
        # A pre-metadata checkpoint must restore only to the original game.
        del saved["config"]["penalty_units"]
        del saved["config"]["max_context_harm"]
    api["restore"](restored, saved)
    for expected in fixture["tail"]:
        same(expected, api["update"](restored))
    same(fixture["final"], api["checkpoint"](restored))
    same(fixture["output"], restored.policy.served_model().routed_forward(restored.test_context))


def test_restore_rejects_unit_change_before_weights_optimizers_or_rng_mutate(api):
    token = api["make_loop"](mode="no_rows")
    assert token.config["penalty_units"] == "token"
    context = api["make_loop"](mode="no_rows", penalty_units="context")
    for _ in range(3):
        api["update"](context)
    saved = cpu_roundtrip(api["checkpoint"](context))
    del saved["config"]["penalty_units"]
    del saved["config"]["max_context_harm"]
    before = api["checkpoint"](token)
    with pytest.raises(ValueError, match="penalty units"):
        api["restore"](token, saved)
    same(before, api["checkpoint"](token))
    restored = api["make_loop"](mode="no_rows", penalty_units="context")
    api["restore"](restored, saved)
    same(api["update"](context), api["update"](restored))
    same(api["checkpoint"](context), api["checkpoint"](restored))
    before = api["checkpoint"](context)
    with pytest.raises(ValueError, match="penalty units"):
        api["restore"](context, api["checkpoint"](token))
    same(before, api["checkpoint"](context))


def test_token_game_output_guards_make_real_decisions_reject_atomically_and_resume(api, monkeypatch):
    loop = api["make_loop"](output_error_guard=True)
    control, observations = loop.policy.routed_control, []
    original = control.maybe_apply

    def owners():
        policy = loop.policy
        return deepcopy({"table": policy.table.detach(), "average_table": policy.averaged_table,
                         "models": {name: model.state_dict() for name, model in policy._training_modules().items()},
                         "averages": {name: model.state_dict() for name, model in policy._average_modules().items()},
                         "optimizers": [optimizer.state_dict() for optimizer in policy.optimizers],
                         "controller": policy.controller.state_dict()})

    def observed(*args, **kwargs):
        before = owners()
        event = original(*args, **kwargs)
        if event is not None and "output_guard_accepted" in event:
            observations.append(deepcopy(event))
            assert event["accepted"] == (event["feature_guard_accepted"] and event["output_guard_accepted"])
            if not event["accepted"]:
                same(before, owners())
                assert event["moves"] == 0 and control.moved_rows is None
        return event

    monkeypatch.setattr(control, "maybe_apply", observed)
    rows, saved = [], None
    for index in range(8):
        rows.append(api["update"](loop))
        if index == 0:
            saved = cpu_roundtrip(api["checkpoint"](loop))
    assert loop.config["penalty_units"] == "token"
    assert observations and any(not event["accepted"] for event in observations)
    expected = api["checkpoint"](loop)
    output = loop.policy.served_model().routed_forward(loop.test_context)
    restored = api["make_loop"](output_error_guard=True)
    api["restore"](restored, saved)
    for row in rows[1:]:
        same(row, api["update"](restored))
    same(expected, api["checkpoint"](restored))
    same(output, restored.policy.served_model().routed_forward(restored.test_context))


@pytest.mark.parametrize("initialization", ["conformance", "api"])
def test_cli_preserves_legacy_units_and_harm_allowance_and_rejects_explicit_mismatch(
        api, monkeypatch, tmp_path, capsys, initialization):
    context = api["make_loop"](mode="no_rows", penalty_units="context",
                                initialization=initialization, max_context_harm=1e-4)
    for _ in range(3):
        api["update"](context)
    saved = api["checkpoint"](context)
    saved["config"] = dict(saved["config"])
    del saved["config"]["penalty_units"]
    del saved["config"]["max_context_harm"]
    source, destination = tmp_path / "legacy.pt", tmp_path / "continued.pt"
    torch.save(saved, source)
    for _ in range(2):
        api["update"](context)
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--steps", "2", "--resume", str(source), "--output", str(destination)])
    api["main"]()
    continued = torch.load(destination, map_location="cpu", weights_only=True)
    same(api["checkpoint"](context), continued)
    assert '"penalty_units": "context"' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--steps", "1", "--resume", str(source), "--penalty-units", "token"])
    with pytest.raises(SystemExit) as error:
        api["main"]()
    assert error.value.code == 2
    assert "conflicts with checkpoint units context" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--steps", "1", "--resume", str(source), "--max-context-harm", "0"])
    with pytest.raises(SystemExit) as error:
        api["main"]()
    assert error.value.code == 2
    assert "conflicts with checkpoint allowance" in capsys.readouterr().err


def test_API_guard_defaults_to_zero_and_records_explicit_allowance_without_changing_initialization(api):
    strict = api["make_loop"](initialization="api")
    permissive = api["make_loop"](initialization="api", max_context_harm=1e-4)
    conformance = api["make_loop"]()
    for loop, allowance in ((strict, 0.), (permissive, 1e-4), (conformance, 1e-4)):
        assert loop.config["max_context_harm"] == allowance
        assert loop.policy.routed_control.spec.max_context_harm == allowance
    same(strict.policy.table, permissive.policy.table)
    for name, module in strict.policy._training_modules().items():
        same(module.state_dict(), permissive.policy._training_modules()[name].state_dict())
    for _ in range(2):
        left, right = api["update"](strict), api["update"](permissive)
        for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest"):
            same(left[key], right[key])
        for row, allowance in ((left, 0.), (right, 1e-4)):
            event = row["move"]
            if event and event.get("accepted"):
                assert max(event["max_context_harm"], event["average_max_context_harm"]) <= allowance + 1e-12
    before = api["checkpoint"](strict)
    with pytest.raises(ValueError, match="max_context_harm"):
        api["restore"](strict, api["checkpoint"](permissive))
    same(before, api["checkpoint"](strict))


@pytest.mark.parametrize("tokens", [0, -1, True, 1.5])
def test_view_rejects_invalid_token_count(api, tokens):
    with pytest.raises(ValueError, match="positive integer"):
        api["TokenPenaltyView"](ContextCritic(), tokens)


def test_view_rejects_ambiguous_context_reconstruction(api):
    view = api["TokenPenaltyView"](ContextCritic(), 8)
    for shape in ((2, 8, 2), (7, 2), (0, 2), (8, 0)):
        with pytest.raises(ValueError, match="flat"):
            view(torch.zeros(shape))

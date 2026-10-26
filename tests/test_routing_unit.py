"""Direct tests for ``RoutedExecution``, ``RoutedEvidence`` and ``RoutedRowControl``.

The e22 routed tests drive these classes through whole policies; here each is
constructed on a tiny CPU bank so the site protocol, evidence statistics,
reservoir bookkeeping, evidence-only refresh, guarded split and checkpoint
validation are pinned individually.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particlegan.routing import RoutedBatch, RoutedCandidate, RoutedEvidence, RoutedExecution, RoutedRows

N, D, C = 4, 2, 3


def candidate(log_mass=None):
    table = torch.arange(N * D, dtype=torch.float64).reshape(N, D)
    mass = torch.zeros(N, dtype=torch.float64) if log_mass is None else log_mass
    return RoutedCandidate(table, mass, {"log_mass": mass})


# RoutedExecution --------------------------------------------------------------

def test_mix_adds_log_mass_softmaxes_and_averages_usage_over_sites():
    mass = torch.tensor([0., -torch.inf, 0., 0.], dtype=torch.float64)
    routing = RoutedExecution(("a", "b"), candidate(mass), 2, None)
    logits_a = torch.zeros(2, N, dtype=torch.float64)
    logits_b = torch.tensor([[[5., 0., 0., 0.]] * 3, [[0., 0., 0., 5.]] * 3], dtype=torch.float64)
    codes_a = routing.mix("a", logits_a)
    expected = torch.tensor([1 / 3, 0., 1 / 3, 1 / 3], dtype=torch.float64)
    torch.testing.assert_close(codes_a, (expected @ candidate().table).expand(2, D))
    codes_b = routing.mix("b", logits_b)
    assert codes_b.shape == (2, 3, D)
    usage = routing.finish()
    weights_b = (logits_b + mass).softmax(-1).mean(1)
    torch.testing.assert_close(usage, (expected.expand(2, N) + weights_b) / 2)
    routing.close()
    with pytest.raises(ValueError, match="only during its complete model forward"):
        routing.mix("a", logits_a)
    with pytest.raises(ValueError, match="only during its complete model forward"):
        routing.finish()


def test_sites_must_occur_once_in_declared_order():
    routing = RoutedExecution(("a", "b"), candidate(), 2, None)
    with pytest.raises(ValueError, match="declared order"):
        routing.mix("b", torch.zeros(2, N, dtype=torch.float64))
    routing.mix("a", torch.zeros(2, N, dtype=torch.float64))
    with pytest.raises(ValueError, match="every declared routing site"):
        routing.finish()
    routing.mix("b", torch.zeros(2, N, dtype=torch.float64))
    with pytest.raises(ValueError, match="declared order"):
        routing.mix("b", torch.zeros(2, N, dtype=torch.float64))


@pytest.mark.parametrize("logits,match", [
    (torch.zeros(3, N, dtype=torch.float64), "finite floating"),             # wrong batch
    (torch.zeros(2, N + 1, dtype=torch.float64), "finite floating"),         # wrong row count
    (torch.zeros(2, N, dtype=torch.long), "finite floating"),                # integer logits
    (torch.full((2, N), float("nan"), dtype=torch.float64), "finite floating"),
])
def test_mix_rejects_malformed_logits(logits, match):
    with pytest.raises(ValueError, match=match):
        RoutedExecution(("a",), candidate(), 2, None).mix("a", logits)


def test_mix_rejects_a_site_with_no_remaining_mass_and_shape_changing_perturbations():
    dead = torch.full((N,), -torch.inf, dtype=torch.float64)
    with pytest.raises(ValueError, match="retain finite mass"):
        RoutedExecution(("a",), candidate(dead), 2, None).mix("a", torch.zeros(2, N, dtype=torch.float64))
    routing = RoutedExecution(("a",), candidate(), 2, lambda codes: codes[:, :1])
    with pytest.raises(ValueError, match="preserve the mixed-code shape"):
        routing.mix("a", torch.zeros(2, N, dtype=torch.float64))
    shifted = RoutedExecution(("a",), candidate(), 2, lambda codes: codes + 1)
    out = shifted.mix("a", torch.zeros(2, N, dtype=torch.float64))
    torch.testing.assert_close(out, candidate().table.mean(0).expand(2, D) + 1)


# RoutedEvidence ---------------------------------------------------------------

SPEC = SimpleNamespace(min_observations=3, min_effect=1e-6, persistence_threshold=.75)


def test_constant_gradients_are_fully_persistent_and_alternating_ones_are_not():
    evidence = RoutedEvidence(torch.zeros(N, D), SPEC)
    for step in range(6):
        grad = torch.zeros(N, D)
        grad[0] = torch.tensor([1., 2.])
        grad[1] = torch.tensor([1., 2.]) * (-1) ** step
        evidence.update(grad)
    persistence = evidence.gradient_persistence
    assert persistence[0] == pytest.approx(1.)
    assert persistence[1] < .2
    assert evidence.touches.tolist() == [6, 6, 0, 0]
    assert evidence.W[2] == 0 and evidence.counters["updates"] == 6
    with pytest.raises(ValueError, match="full dense table gradient"):
        evidence.update(torch.zeros(N - 1, D))


def test_rows_are_flagged_only_with_enough_contexts_effect_and_persistence():
    evidence = RoutedEvidence(torch.zeros(N, D), SPEC)
    for _ in range(4):
        evidence.update(torch.ones(N, D))
    weights = torch.full((8, N), 1. / N, dtype=torch.float64)
    base = torch.ones(8, dtype=torch.float64)
    evidence.observe_effect(0, weights, base, base * 3, evaluation=1)  # deleting row 0 hurts: support
    evidence.observe_effect(1, weights, base, base * 1, evaluation=1)  # no effect
    assert evidence.effect[0] == pytest.approx(-.5) and evidence.effect[1] == 0
    assert evidence.effective_contexts[0] == pytest.approx(8.)
    assert evidence.flag.tolist() == [True, False, False, False]
    assert evidence.valid and evidence.fraction == .25
    diagnostics = evidence.diagnostics()
    assert diagnostics["support"][0] == pytest.approx(.5) and diagnostics["law"] == "conditional_paired_diagnostic"


def test_evidence_reset_and_checkpoint_roundtrip():
    evidence = RoutedEvidence(torch.zeros(N, D), SPEC)
    evidence.update(torch.ones(N, D))
    saved = evidence.state_dict()
    other = RoutedEvidence(torch.zeros(N, D), SPEC)
    other.load_state_dict(saved)
    assert torch.equal(other.M, evidence.M) and other.counters == evidence.counters
    evidence.reset()
    assert not evidence.M.any() and bool((evidence.last_probe == -1).all())
    assert evidence.counters["global_resets"] == 1 and evidence.counters["resets"] == N
    for corrupt, match in ((lambda s: s.pop("valid"), "invalid routed evidence state"),
                           (lambda s: s["tensors"].update(M=torch.zeros(N + 1, D, dtype=torch.float64)),
                            "does not match the bank"),
                           (lambda s: s.update(fraction=2.), "diagnostics")):
        state = deepcopy(saved)
        corrupt(state)
        with pytest.raises(ValueError, match=match):
            other.check_state(state)


# RoutedRowControl -------------------------------------------------------------

class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(C, N)
        self.register_buffer("log_mass", torch.zeros(N))


def spec(**options):
    return RoutedRows(
        route=lambda models, context, cand: (models["router"].key(context) + cand.log_mass).softmax(-1),
        generate=lambda models, context, cand, weights: cand.codes,
        features=lambda models, context, samples, targets: samples,
        reservoir_size=8, min_observations=4, probe_budget=N, **options)


def control(**options):
    torch.manual_seed(0)
    router = Router()
    table = nn.Parameter(torch.randn(N, D))
    optimizer = torch.optim.Adam([table, *router.parameters()], lr=.01)
    rows = spec(**options)
    bound = rows.bind(models={"router": router}, averaged_models={"router": deepcopy(router)},
                      table=table, averaged_table=table.detach().clone(), optimizers=(optimizer,),
                      table_optimizer=optimizer, seed=3)
    return bound, table, optimizer, router


def batch(offset, size=4):
    context = torch.arange(size * C, dtype=torch.float32).reshape(size, C) / 10 + offset
    return context, torch.sin(context[:, :D])


def routed_batch(step):
    context, targets = batch(step)
    guard_context, guard_targets = batch(100 + step)
    return RoutedBatch(context, targets, guard_context, guard_targets)


def test_bind_validates_table_ownership_and_separate_averages():
    torch.manual_seed(0)
    router, table = Router(), nn.Parameter(torch.randn(N, D))
    optimizer = torch.optim.Adam([table], lr=.01)
    common = dict(models={"router": router}, averaged_models={"router": deepcopy(router)}, optimizers=(optimizer,))
    with pytest.raises(ValueError, match="own the table exactly once"):
        spec().bind(table=table, averaged_table=table.detach().clone(), table_optimizer=torch.optim.Adam([table]), **common)
    with pytest.raises(ValueError, match="separate, frozen"):
        spec().bind(table=table, averaged_table=table.detach(), table_optimizer=optimizer, **common)
    single = nn.Parameter(torch.randn(1, D))
    with pytest.raises(ValueError, match="at least two trainable table rows"):
        spec().bind(table=single, averaged_table=single.detach().clone(), table_optimizer=optimizer, **common)


def test_begin_fills_ring_reservoirs_and_rejects_leaking_guard_contexts():
    bound, *_ = control()
    for step in range(3):
        bound.begin(routed_batch(step))
    assert (bound.fit_fill, bound.guard_fill, bound.fit_cursor) == (8, 8, 4)
    torch.testing.assert_close(bound.fit_context[:4], batch(2)[0])
    assert bound.probe_clock["observed_updates"] == 3
    context, targets = batch(50)
    with pytest.raises(ValueError, match="must not overlap"):
        bound.begin(RoutedBatch(context, targets, context.clone(), targets))
    with pytest.raises(ValueError, match="supplied separately"):
        bound.begin(RoutedBatch(context, targets, context, targets))
    with pytest.raises(TypeError, match="RoutedBatch"):
        bound.begin((context, targets))


def test_evidence_only_refresh_probes_rows_without_changing_the_bank_or_rng():
    bound, table, optimizer, _ = control()
    assert bound.refresh_evidence() is None  # not enough contexts yet
    bound.begin(routed_batch(0))
    loss = bound.generate(batch(0)[0]).square().sum()
    loss.backward()
    bound.observe_backward()
    assert torch.equal(bound.latest_gradient, table.grad)
    before, rng = table.detach().clone(), torch.get_rng_state()
    result = bound.refresh_evidence()
    assert result["evidence_only"] is True and result["moves"] == 0
    assert bound.counters["evals"] == 1 and bound.counters["probes"] == N
    assert torch.equal(table.detach(), before) and torch.equal(rng, torch.get_rng_state())
    assert bool((bound.evidence.last_probe == 1).all())


def test_guarded_split_moves_exactly_the_reported_rows_and_resets_evidence():
    bound, table, optimizer, _ = control(improvement_margin=0., max_context_harm=1e9)
    before = table.detach().clone()
    result = None
    for step in range(6):
        bound.begin(routed_batch(step))
        optimizer.zero_grad()
        loss = (bound.generate(batch(step)[0]) - batch(step)[1]).square().sum()
        loss.backward()
        bound.observe_backward()
        optimizer.step()
        before = table.detach().clone()
        result = bound.maybe_apply()
        if result and result.get("accepted"):
            break
    assert result is not None and result.get("accepted") is True, result
    moved = bound.moved_rows.tolist()
    assert len(moved) == 2 and moved[0] == result["parent"] and moved[1] == result["child"]
    unchanged = [row for row in range(N) if row not in moved]
    assert torch.equal(table.detach()[unchanged], before[unchanged])
    assert bound.counters["splits"] == 1 and bound.counters["moves"] == 2
    assert not bound.evidence.W.any() and bound.latest_gradient is None and bound.restart_router


def test_control_checkpoint_roundtrip_and_validation():
    bound, table, optimizer, router = control()
    bound.begin(routed_batch(0))
    bound.generate(batch(0)[0]).square().sum().backward()
    bound.observe_backward()
    bound.refresh_evidence()
    saved = bound.state_dict()
    fresh, *_ = control()
    fresh.load_state_dict(saved)
    assert fresh.counters == bound.counters and fresh.fit_fill == bound.fit_fill
    assert torch.equal(fresh.fit_context, bound.fit_context)
    assert torch.equal(fresh.stream.get_state(), bound.stream.get_state())
    for corrupt, match in (
            (lambda s: s.update(schema=2), "checkpoint schema"),
            (lambda s: s["config"].update(routed_geometry="old"), "DV12 law changed"),
            (lambda s: s["config"].update(min_observations=5), "configuration does not match"),
            (lambda s: s.update(probe_cursor=N), "probe cursor"),
            (lambda s: s.update(fit_fill=99), "reservoir counters"),
            (lambda s: s["pools"].update(guard_context=s["pools"]["fit_context"].clone()), "must not overlap"),
            (lambda s: s.update(latest_gradient=torch.zeros(N + 1, D)), "stored gradient"),
            (lambda s: s.update(stream=torch.zeros(3, dtype=torch.uint8)), "private RNG state")):
        state = deepcopy(saved)
        corrupt(state)
        with pytest.raises(ValueError, match=match):
            fresh.check_state(state)

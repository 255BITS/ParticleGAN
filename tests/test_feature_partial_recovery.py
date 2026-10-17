"""Synthetic CPU contracts for partial feature recovery after an R1 fire."""
from copy import deepcopy
from dataclasses import fields
import math
from types import SimpleNamespace

import pytest
import torch

from particlegan.feature_policy import FeatureFacade
from particlegan.output_moments import fit_projection, freeze_moment, OutputObservation, odd_witness
import particlegan.mean_transport as mt



@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads, device, dtype = torch.get_num_threads(), torch.get_default_device(), torch.get_default_dtype()
    rng = torch.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU contract initialized CUDA'))
    torch.set_num_threads(1)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    try:
        yield
        # Other tests may have initialized CUDA before this CPU fixture.
        assert torch.cuda.is_initialized() == cuda_initialized
    finally:
        torch.set_num_threads(threads)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        torch.set_rng_state(rng)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif hasattr(a, '__dataclass_fields__'):
        for field in fields(a): equal(getattr(a, field.name), getattr(b, field.name))
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    else:
        assert a == b


def case(*, full=False):
    # Two occupied modes and one missing EMA mode. The frame uses both raw
    # axes, and the odd observations include the missing mode equally.
    target = torch.tensor([[-1., 0.], [1., 0.], [0., 3.]], dtype=torch.float64)
    offsets = torch.tensor([[-.03, -.03], [-.03, .03], [.03, -.03], [.03, .03]], dtype=torch.float64)
    even = (target[:, None] + offsets[None]).reshape(-1, 2).repeat(64, 1)
    real = torch.stack((even, even.clone()), dim=1).reshape(-1, 2)
    occupied = target if full else target[:2]
    ema = (occupied[:, None] + offsets[None] + torch.tensor([.06, 0.])).reshape(-1, 2).repeat(64, 1)

    class Snapshot:
        valid_metric = True
        rank = 2
        duplicate_fraction = 0.
        mass_groups = 3
        cells = 4
        calibration_rows = len(even)

        def transform(self, x):return x.double()
        def _assign_metric(self, x):
            values, ids = torch.cdist(x, target).square().min(1)
            return ids, values
        def _mass_topology(self):return torch.arange(3)

    projection, reason = fit_projection(real[0::2])
    assert reason is None
    observed = OutputObservation(ema, projection.transform(ema))
    return Snapshot(), real, observed, projection


def partial_case():
    snapshot, real, observed, projection = case()
    fixed, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection,
                                 allow_partial_groups=True)
    assert fixed is not None and reason is None
    return snapshot, real, observed, projection, fixed


def test_default_missing_group_veto_is_preserved():
    snapshot, real, observed, projection = case()
    fixed, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection)
    assert fixed is None and reason == 'missing_EMA_group'
    with pytest.raises(TypeError, match='boolean'):
        freeze_moment(snapshot, real[0::2], real[0::2], observed, projection, allow_partial_groups=1)


def test_partial_witness_retains_original_mass_full_sample_and_bound():
    snapshot, real, observed, projection, fixed = partial_case()
    assert fixed.ema_counts[2] == 0 and fixed.weights[2] == 0
    assert torch.equal(fixed.directions[2], torch.zeros(2, dtype=torch.float64))
    groups = snapshot._assign_metric(real[0::2])[0]
    original_weights = torch.bincount(groups, minlength=3).double() / len(real[0::2])
    torch.testing.assert_close(fixed.weights[:2], original_weights[:2], rtol=0, atol=0)
    assert float(fixed.weights.sum()) == pytest.approx(2. / 3.)
    state = deepcopy(fixed)
    witness = odd_witness(snapshot, fixed, real[1::2], real[1::2])
    assert witness['authoritative'] is False
    assert witness['observations'] == len(real[1::2])
    assert witness['zero_direction_observations'] == len(real[1::2]) // 3
    assert witness['valid'] and witness['fires'] and witness['lower_bound'] > 0
    assert witness['alpha'] == .05 / (3 * snapshot.cells + 3)
    assert witness['radius'] == math.sqrt(projection.rank / .05)
    assert witness['known_range'] == 4 * math.sqrt(projection.rank / .05)
    assert abs(witness['scalar_min']) <= 2 * fixed.radius and abs(witness['scalar_max']) <= 2 * fixed.radius
    equal(state, fixed)


def test_fully_supported_recovery_matches_default_exactly():
    snapshot, real, observed, projection = case(full=True)
    default, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection)
    assert default is not None and reason is None
    recovery, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection,
                                     allow_partial_groups=True)
    assert reason is None
    equal(default, recovery)
    equal(odd_witness(snapshot, default, real[1::2], real[1::2]),
          odd_witness(snapshot, recovery, real[1::2], real[1::2]))


@pytest.mark.parametrize('fires,expected', [(0, False), (1, True), (2, True),
    (-1, False), (True, False), (1., False), (None, False)])
def test_facade_requires_typed_actual_fire_history(fires, expected):
    policy = SimpleNamespace(table=torch.zeros(2, 2), averaged_table=torch.zeros(2, 2),
                             table_optimizer=SimpleNamespace(), surprise=SimpleNamespace(fires=fires))
    facade = FeatureFacade(policy)
    before = dict(facade.__dict__)
    assert facade.mean_partial_recovery is expected
    assert facade.__dict__ == before
    policy.surprise = None
    assert facade.mean_partial_recovery is False


def view(n, groups):
    metric = torch.zeros(n, 2, dtype=torch.float64)
    return mt.View(metric, metric, torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long),
                   groups, torch.ones(n, dtype=torch.bool), torch.ones(n), metric)


def test_inactive_prefix_births_are_filtered_before_proposal_count_division():
    snapshot, _, _, _, fixed = partial_case()
    # Simulates prefix births in a group absent at the preaction freeze.
    inactive = view(16, torch.full((16,), 2, dtype=torch.long))
    pairs = mt.propose_pairs(snapshot, fixed, inactive, inactive, earlier_ordinary=0,
                            reserved_rows=torch.empty(0, dtype=torch.long))
    assert len(pairs.children) == len(pairs.parents) == len(pairs.pre_gain) == 0
    assert pairs.eligible_rows == 0 and not torch.isnan(pairs.pre_gain).any()


def test_inactive_preview_never_divides_or_commits_a_packet(monkeypatch):
    snapshot, _, _, _, fixed = partial_case()
    inactive = view(20, torch.full((20,), 2, dtype=torch.long))
    pairs = mt.CandidatePairs(torch.tensor([0]), torch.tensor([1]), torch.tensor([1.]), 20, 1,
                              torch.empty(0, dtype=torch.long), 1, 0)
    prior = SimpleNamespace(z=torch.zeros(20, 2, dtype=torch.float64))
    average = SimpleNamespace(z=prior.z.clone())
    stream = torch.Generator(device='cpu').manual_seed(1234)
    state = SimpleNamespace(prior=prior, ema_prior=average, bandwidth=torch.ones(2), stream=stream)
    geometry = SimpleNamespace(displacement=lambda latent, *a, **k: torch.zeros_like(latent), work={})
    monkeypatch.setattr(mt, 'epoch', lambda *a, **k: ('same',))
    monkeypatch.setattr(mt, 'observe_view', lambda *a, **k: view(1, torch.tensor([2])))
    measure = lambda x: OutputObservation(x, x.double())
    before = prior.z.clone(), average.z.clone()
    packet, detail = mt.preview_pairs(snapshot, fixed, inactive, inactive, pairs, state, stream=stream,
        measure_fast=measure, measure_ema=measure, geometry=geometry)
    assert packet is None and detail['accepted'] == detail['category_retained'] == 0
    assert math.isfinite(detail['objective_before']) and math.isfinite(detail['objective_after_virtual'])
    equal(before, (prior.z, average.z))


def phase_fixture(monkeypatch, *, erase_active=False, birth_in_inactive=False):
    snapshot, real, observed, _, fixed = partial_case()
    groups = snapshot._assign_metric(observed.features)[0]
    if erase_active:groups[groups == 0] = 1
    if birth_in_inactive:groups[0] = 2
    v = view(len(groups), groups)
    v = mt.View(v.features, v.metric, v.cells, v.categories, v.groups, v.eligible,
                v.pvalues, observed.projected_outputs)
    monkeypatch.setattr(mt, 'capture_outputs_features', lambda *a, **k: observed)
    monkeypatch.setattr(mt, 'observe_view', lambda *a, **k: v)
    trainer = SimpleNamespace(prior=SimpleNamespace(z=observed.features),
                              ema_prior=SimpleNamespace(z=observed.features), ema_G=None)
    backend = SimpleNamespace(N=len(groups), dry_run=False, counters={'mean_forward_rows': 0})
    stamp = {'status': 'firing'}
    return snapshot, fixed, trainer, backend, stamp


def test_current_active_group_loss_vetoes_whole_phase_before_proposals(monkeypatch):
    snapshot, fixed, trainer, backend, stamp = phase_fixture(monkeypatch, erase_active=True)
    monkeypatch.setattr(mt, 'propose_pairs', lambda *a, **k: pytest.fail('active empty group reached proposals'))
    result = mt.run_mean_phase(backend, trainer, snapshot, fixed, stamp, earlier_ordinary=0,
        reserved_rows=torch.empty(0, dtype=torch.long), allow_partial_groups=True)
    assert result['diagnostics']['reason'] == 'missing_current_EMA_group'
    assert len(result['children']) == 0 and result['packet'] is None


@pytest.mark.parametrize('birth_in_inactive', [False, True])
def test_current_inactive_group_stays_inactive_without_global_veto(monkeypatch, birth_in_inactive):
    snapshot, fixed, trainer, backend, stamp = phase_fixture(monkeypatch, birth_in_inactive=birth_in_inactive)
    called = []
    def no_pairs(snapshot, refreshed, fast, ema, **kwargs):
        called.append(True)
        equal(fixed.weights, refreshed.weights)
        assert refreshed.weights[2] == 0 and torch.equal(refreshed.directions[2], torch.zeros(2, dtype=torch.float64))
        empty = torch.empty(0, dtype=torch.long)
        budget = math.floor(.05 * len(ema.features))
        return mt.CandidatePairs(empty, empty, torch.empty(0, dtype=torch.float64), 0, 0, empty, budget, 0)
    monkeypatch.setattr(mt, 'propose_pairs', no_pairs)
    result = mt.run_mean_phase(backend, trainer, snapshot, fixed, stamp, earlier_ordinary=0,
        reserved_rows=torch.empty(0, dtype=torch.long), allow_partial_groups=True)
    assert called and result['diagnostics']['reason'] == 'no_legal_positive_pairs'
    assert result['packet'] is None and len(result['children']) == 0


def test_default_current_missing_group_still_vetoes(monkeypatch):
    snapshot, fixed, trainer, backend, stamp = phase_fixture(monkeypatch)
    result = mt.run_mean_phase(backend, trainer, snapshot, fixed, stamp, earlier_ordinary=0,
                              reserved_rows=torch.empty(0, dtype=torch.long))
    assert result['diagnostics']['reason'] == 'missing_current_EMA_group'


def test_no_fire_facade_preparation_preserves_default_veto(monkeypatch):
    snapshot, real, observed, projection = case()
    monkeypatch.setattr(mt, 'capture_outputs_features', lambda *a, **k: observed)
    policy = SimpleNamespace(table=observed.features, averaged_table=observed.features,
        table_optimizer=SimpleNamespace(), surprise=SimpleNamespace(fires=0), ema_G=None,
        completed_steps=0)
    facade = FeatureFacade(policy)
    def backend():
        return SimpleNamespace(N=len(observed.features), reservoir=real, sample_shape=(2,), snapshot_serial=1,
            counters={'mean_evals': 0, 'mean_forward_rows': 0, 'mean_witness_fires': 0})
    default, zero_fire = backend(), backend()
    old_fixed, old_stamp = mt.prepare_mean_witness(default, facade, snapshot, real)
    fixed, stamp = mt.prepare_mean_witness(zero_fire, facade, snapshot, real,
        allow_partial_groups=facade.mean_partial_recovery)
    assert old_fixed is fixed is None
    equal(old_stamp, stamp)
    equal(default.counters, zero_fire.counters)
    assert stamp['status'] == 'invalid' and stamp['reason'] == 'missing_EMA_group'
    policy.surprise.fires = 1
    recovered, recovery_stamp = mt.prepare_mean_witness(backend(), facade, snapshot, real,
        allow_partial_groups=facade.mean_partial_recovery)
    assert recovered is not None and recovery_stamp['status'] == 'firing'

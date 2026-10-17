"""CPU contracts for recovery on a frozen occupied subset after an R1 fire."""
from copy import deepcopy
from dataclasses import fields
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from particlegan.feature_policy import FeatureFacade
from particlegan.output_moments import fit_projection, freeze_moment, OutputObservation, odd_witness
import particlegan.mean_transport as mt

STUDY = Path(__file__).resolve().parents[2]
ARRAYS = STUDY / 'diagnostics/moving-rotated-recovery/isolation-arrays.npz'
REFERENCE = STUDY / 'pkg-RA14-replay/particlegan'
spec = importlib.util.spec_from_file_location('_ra14_partial_reference', REFERENCE / '__init__.py',
                                           submodule_search_locations=[str(REFERENCE)])
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)
old_moments = importlib.import_module(spec.name + '.output_moments')
old_transport = importlib.import_module(spec.name + '.mean_transport')


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU contract initialized CUDA'))
    torch.set_num_threads(1)
    try:
        yield
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(threads)
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
    with np.load(ARRAYS) as arrays:
        target = torch.from_numpy(arrays['target_centers'].copy())
        real = torch.from_numpy(arrays['real_fifo'].copy())
        ema = torch.from_numpy(arrays['ema_outputs'].copy())

    class Snapshot:
        valid_metric = True
        rank = 2
        duplicate_fraction = 0.
        mass_groups = 100
        cells = 128
        calibration_rows = 10000

        def transform(self, x):return x.double()
        def _assign_metric(self, x):
            values, ids = torch.cdist(x, target).square().min(1)
            return ids, values
        def _mass_topology(self):return torch.arange(100)

    snapshot = Snapshot()
    if full:
        ema = ema.clone(); ema[0] = target[99]
    projection, reason = fit_projection(real[0::2])
    assert reason is None
    observed = OutputObservation(ema, projection.transform(ema))
    return snapshot, real, observed, projection


def partial_case():
    snapshot, real, observed, projection = case()
    fixed, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection,
                                 allow_partial_groups=True)
    assert fixed is not None and reason is None
    return snapshot, real, observed, projection, fixed


def test_actual_checkpoint_default_keeps_original_missing_group_veto():
    snapshot, real, observed, projection = case()
    for function in (old_moments.freeze_moment, freeze_moment):
        fixed, reason = function(snapshot, real[0::2], real[0::2], observed, projection)
        assert fixed is None and reason == 'missing_EMA_group'
    with pytest.raises(TypeError, match='boolean'):
        freeze_moment(snapshot, real[0::2], real[0::2], observed, projection, allow_partial_groups=1)


def test_actual_checkpoint_partial_witness_retains_all_observations_and_bound():
    snapshot, real, observed, projection, fixed = partial_case()
    assert fixed.ema_counts[99] == 0 and fixed.weights[99] == 0
    assert torch.equal(fixed.directions[99], torch.zeros(2, dtype=torch.float64))
    groups = snapshot._assign_metric(real[0::2])[0]
    original_weights = torch.bincount(groups, minlength=100).double() / 10000
    torch.testing.assert_close(fixed.weights[:99], original_weights[:99], rtol=0, atol=0)
    assert float(fixed.weights.sum()) == pytest.approx(.9891)
    state = deepcopy(fixed)
    witness = odd_witness(snapshot, fixed, real[1::2], real[1::2])
    assert witness['authoritative'] is False
    assert witness['observations'] == 10000 and witness['zero_direction_observations'] == 112
    assert witness['valid'] and witness['fires']
    assert witness['lower_bound'] == pytest.approx(.5195955732330525, abs=1e-12)
    assert witness['alpha'] == .05 / (3 * 128 + 3)
    assert witness['radius'] == math.sqrt(2 / .05)
    assert witness['known_range'] == 4 * math.sqrt(2 / .05)
    assert abs(witness['scalar_min']) <= 2 * fixed.radius and abs(witness['scalar_max']) <= 2 * fixed.radius
    equal(state, fixed)


def test_fully_supported_default_and_recovery_match_original_exactly():
    snapshot, real, observed, projection = case(full=True)
    old, reason = old_moments.freeze_moment(snapshot, real[0::2], real[0::2], observed, projection)
    assert old is not None and reason is None
    for flag in (False, True):
        new, reason = freeze_moment(snapshot, real[0::2], real[0::2], observed, projection,
                                   allow_partial_groups=flag)
        assert reason is None
        equal(old, new)
        equal(old_moments.odd_witness(snapshot, old, real[1::2], real[1::2]),
              odd_witness(snapshot, new, real[1::2], real[1::2]))


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
    inactive = view(16, torch.full((16,), 99, dtype=torch.long))
    pairs = mt.propose_pairs(snapshot, fixed, inactive, inactive, earlier_ordinary=0,
                            reserved_rows=torch.empty(0, dtype=torch.long))
    assert len(pairs.children) == len(pairs.parents) == len(pairs.pre_gain) == 0
    assert pairs.eligible_rows == 0 and not torch.isnan(pairs.pre_gain).any()


def test_inactive_preview_never_divides_or_commits_a_packet(monkeypatch):
    snapshot, _, _, _, fixed = partial_case()
    inactive = view(4, torch.full((4,), 99, dtype=torch.long))
    pairs = mt.CandidatePairs(torch.tensor([0]), torch.tensor([1]), torch.tensor([1.]), 4, 1,
                              torch.empty(0, dtype=torch.long), 1, 0)
    prior = SimpleNamespace(z=torch.zeros(4, 2, dtype=torch.float64))
    average = SimpleNamespace(z=prior.z.clone())
    stream = torch.Generator(device='cpu').manual_seed(1234)
    state = SimpleNamespace(prior=prior, ema_prior=average, bandwidth=torch.ones(2), stream=stream)
    geometry = SimpleNamespace(displacement=lambda latent, *a, **k: torch.zeros_like(latent), work={})
    monkeypatch.setattr(mt, 'epoch', lambda *a, **k: ('same',))
    monkeypatch.setattr(mt, 'observe_view', lambda *a, **k: view(1, torch.tensor([99])))
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
    if birth_in_inactive:groups[0] = 99
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
        assert refreshed.weights[99] == 0 and torch.equal(refreshed.directions[99], torch.zeros(2, dtype=torch.float64))
        empty = torch.empty(0, dtype=torch.long)
        return mt.CandidatePairs(empty, empty, torch.empty(0, dtype=torch.float64), 0, 0, empty, 1000, 0)
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


def test_no_fire_feature_training_and_checkpoint_match_ra14_exactly(monkeypatch):
    from particlegan import get_recipe, GANTrainer
    from particlegan.particle_prior import ParticlePrior
    old_training = importlib.import_module(spec.name + '.training')
    old_recipes = importlib.import_module(spec.name + '.recipes')
    old_prior = importlib.import_module(spec.name + '.particle_prior')
    old_cells = importlib.import_module(spec.name + '.feature_cells')
    new_cells = importlib.import_module('particlegan.feature_cells')
    # Measurement duration is not numerical state. A shared diagnostic clock
    # makes even that metadata comparable without changing training code.
    clock = SimpleNamespace(perf_counter=lambda: 0.)
    monkeypatch.setattr(old_cells, 'time', clock)
    monkeypatch.setattr(new_cells, 'time', clock)
    import json
    config = json.loads((STUDY / 'configs/RA15-partial-recovery.json').read_text())
    config.update(num_particles=1024, z_dim=2, batch_size=128)
    def make(old):
        torch.manual_seed(1234)
        G = torch.nn.Linear(2, 2)
        D = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        prior_type = old_prior.ParticlePrior if old else ParticlePrior
        prior = prior_type(1024, 2, generator=torch.Generator().manual_seed(1234))
        recipe = old_recipes.Recipe(**config) if old else get_recipe(**config)
        trainer_type = old_training.GANTrainer if old else GANTrainer
        return trainer_type(recipe, G, D, prior=prior, seed=1234, serial_backward=True,
                            optimizer_options={'foreach': False, 'fused': False})
    old, new = make(True), make(False)
    for index in range(18):
        batch = torch.arange(256, dtype=torch.float32).reshape(128, 2).sin() + index * .001
        global_rng = torch.get_rng_state()
        old.step(batch, generator_real=batch)
        old_rng = torch.get_rng_state()
        torch.set_rng_state(global_rng)
        new.step(batch, generator_real=batch)
        equal(old_rng, torch.get_rng_state())
        assert old.policy.surprise.fires == new.policy.surprise.fires == 0
    old_state, new_state = old.state_dict(), new.state_dict()
    assert new_state['backend_selection']['actual_backend'] == 'feature_cells'
    assert new.birth_death.counters['mean_evals'] > 0
    equal(old_state, new_state)
    assert old_state['schema'] == new_state['schema'] == 4

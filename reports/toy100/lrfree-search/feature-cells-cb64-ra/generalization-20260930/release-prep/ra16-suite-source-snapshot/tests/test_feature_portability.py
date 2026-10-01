"""Device-independent CPU planning and atomic feature shape restore contracts."""
from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from particlegan.feature_policy import FeatureFacade
from particlegan.feature_reference import ParticleBirthDeath
from particlegan.output_moments import (
    FixedOutputMoment, OutputObservation, OutputProjection, fit_projection,
    freeze_moment, odd_witness,
)
from particlegan.particle_prior import ParticlePrior
from particlegan.population_continuity import PopulationSequentialSettleTest
from particlegan.recipes import Recipe
from particlegan.training import GANTrainer
import particlegan.mean_transport as mt


BASE = json.loads((Path(__file__).parent / 'fixtures' / 'feature-auto-base.json').read_text())
SEED = 1234
CUDA_TEST = 'test_cuda_default_preserves_feature_reactions_and_checkpoint_replay'


@pytest.fixture(autouse=True)
def owned_defaults(request, monkeypatch):
    threads, device, dtype = torch.get_num_threads(), torch.get_default_device(), torch.get_default_dtype()
    rng = torch.get_rng_state().clone()
    initialized = torch.cuda.is_initialized()
    if request.node.name != CUDA_TEST:
        monkeypatch.setattr(torch.cuda, '_lazy_init',
                            lambda *a, **k: pytest.fail('CPU contract initialized CUDA'))
    torch.set_num_threads(1)
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.float32)
    try:
        yield
        if request.node.name != CUDA_TEST:
            assert torch.cuda.is_initialized() == initialized
    finally:
        torch.set_num_threads(threads)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        torch.set_rng_state(rng)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.device == b.device and a.dtype == b.dtype
        assert torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)
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


def make(*, width=2, backend='auto', device='cpu'):
    torch.manual_seed(SEED)
    G = torch.nn.Sequential(torch.nn.Linear(2, 8, device=device), torch.nn.Tanh(),
                            torch.nn.Linear(8, width, device=device))
    D = torch.nn.Sequential(torch.nn.Linear(width, 8, device=device), torch.nn.Tanh(),
                            torch.nn.Linear(8, 1, device=device))
    prior = ParticlePrior(1024, 2, device=device,
                          generator=torch.Generator(device=device).manual_seed(SEED + 1))
    recipe = Recipe(**dict(BASE, z_dim=2, num_particles=1024, batch_size=128,
                           birth_death_backend=backend))
    return GANTrainer(recipe, G, D, prior=prior, seed=SEED, serial_backward=True)


def real(width=2, *, device='cpu'):
    return torch.arange(128 * width, dtype=torch.float32, device=device).reshape(128, width).sin()


class Snapshot:
    valid_metric, rank, width, duplicate_fraction = True, 2, 2, 0.
    mass_groups, cells, cache_version = 3, 3, 0
    device = torch.device('cpu')

    def __init__(self):
        self.centers = torch.tensor([[-1., 0.], [1., 0.], [0., 3.]], dtype=torch.float64, device='cpu')
        self.mean = torch.zeros(2, dtype=torch.float64, device='cpu')
        self.scale = torch.ones(2, dtype=torch.float64, device='cpu')
        self.basis = torch.eye(2, dtype=torch.float64, device='cpu')
        self.cell_scale = torch.ones(3, dtype=torch.float64, device='cpu')
        self.count_boundary = torch.ones(3, dtype=torch.float64, device='cpu')
        self.null_scores = torch.zeros(20, dtype=torch.float64, device='cpu')
        self.topology = torch.arange(3, device='cpu')

    def transform(self, x):return x.double()
    def _assign_metric(self, x):
        distance, ids = torch.cdist(x, self.centers).square().min(1)
        return ids, distance
    def _mass_topology(self):return self.topology
    def _count_categories_metric(self, x, cells):
        return 2 * cells, torch.zeros_like(cells, dtype=torch.float64)


def moment_case(*, partial=False):
    snapshot = Snapshot()
    offsets = torch.tensor([[-.03, -.03], [-.03, .03], [.03, -.03], [.03, .03]],
                           dtype=torch.float64, device='cpu')
    even = (snapshot.centers[:, None] + offsets[None]).reshape(-1, 2).repeat(64, 1)
    occupied = snapshot.centers[:2] if partial else snapshot.centers
    ema = (occupied[:, None] + offsets[None] + torch.tensor([.06, 0.], device='cpu')).reshape(-1, 2)
    return snapshot, even, ema.repeat(64, 1)


def test_streamed_projection_uses_cpu_under_meta_default():
    values = torch.arange(600 * 10, dtype=torch.float64, device='cpu').reshape(600, 10).sin()
    reference, reason = fit_projection(values)
    assert reason is None and reference.rank == 8
    rng = torch.get_rng_state().clone()
    with torch.device('meta'):
        projection, reason = fit_projection(values)
    assert reason is None
    equal(reference, projection)
    assert projection.axes.device.type == projection.even_mean.device.type == 'cpu'
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('partial', [False, True])
def test_frozen_moment_and_odd_query_ignore_meta_default(partial):
    snapshot, even, ema = moment_case(partial=partial)
    projection, _ = fit_projection(even)
    observation = OutputObservation(ema, projection.transform(ema))
    expected, reason = freeze_moment(snapshot, even, even, observation, projection,
                                     allow_partial_groups=partial)
    assert reason is None
    expected_witness = odd_witness(snapshot, expected, even, even)
    with torch.device('meta'):
        actual, reason = freeze_moment(snapshot, even, even, observation, projection,
                                      allow_partial_groups=partial)
        witness = odd_witness(snapshot, actual, even, even)
    assert reason is None and witness['valid']
    equal(expected, actual)
    equal(expected_witness, witness)
    assert all(getattr(actual, name).device.type == 'cpu' for name in
               ('centers', 'scales', 'even_means', 'ema_means', 'ema_counts', 'weights', 'directions'))


def test_missing_fixed_moment_proposes_cpu_empty_result_under_meta():
    snapshot, _, ema = moment_case()
    view = mt.observe_view(snapshot, ema, coordinates=ema, projected_outputs=ema)
    reserved = torch.empty(0, dtype=torch.long, device='cpu')
    with torch.device('meta'):
        pairs = mt.propose_pairs(snapshot, None, view, view, earlier_ordinary=0, reserved_rows=reserved)
    assert len(pairs.children) == len(pairs.parents) == len(pairs.pre_gain) == 0
    assert pairs.pre_gain.device.type == pairs.children.device.type == 'cpu'


def test_accepted_packet_indices_remain_cpu_under_meta_default():
    # Real preview, chart queries, epoch guards, objective and packet hashing.
    snapshot = Snapshot()
    table = snapshot.centers[0].repeat(20, 1)
    table[1, 0] += .2
    snapshot.row_features = table.clone()
    snapshot.query_cell_ids = snapshot._assign_metric(table)[0]
    projection = OutputProjection(2, torch.arange(2, device='cpu'), 20,
        torch.zeros(2, dtype=torch.float64, device='cpu'), torch.ones(2, dtype=torch.float64, device='cpu'))
    fixed = FixedOutputMoment(projection, snapshot.centers.clone(),
        torch.ones(3, dtype=torch.float64, device='cpu'),
        torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64, device='cpu'),
        torch.zeros(3, 2, dtype=torch.float64, device='cpu'),
        torch.tensor([20, 1, 1], device='cpu'), torch.tensor([1., 0., 0.], dtype=torch.float64, device='cpu'),
        torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64, device='cpu'),
        math.sqrt(2 / .05), 2, 2, 3)
    fast = mt.observe_view(snapshot, table, coordinates=table, projected_outputs=table)
    pairs = mt.CandidatePairs(torch.tensor([0], device='cpu'), torch.tensor([1], device='cpu'),
        torch.tensor([1.], device='cpu'), 20, 1, torch.empty(0, dtype=torch.long, device='cpu'), 1, 0)
    def preview():
        stream = torch.Generator(device='cpu').manual_seed(SEED)
        state = SimpleNamespace(prior=SimpleNamespace(z=table.clone()),
            ema_prior=SimpleNamespace(z=table.clone()), stream=stream,
            lineage=SimpleNamespace(neighbors=torch.full((20, 1), -1, dtype=torch.long, device='cpu')),
            model_tensors=(), row_state={}, history=None, buffer_epoch=lambda: (),
            bandwidth=torch.ones(2, device='cpu'))
        geometry = SimpleNamespace(displacement=lambda latent, *a, **k: torch.zeros_like(latent), work={})
        measure = lambda x: OutputObservation(x, projection.transform(x))
        before = state.prior.z.clone(), state.ema_prior.z.clone()
        packet, detail = mt.preview_pairs(snapshot, fixed, fast, fast, pairs, state, stream=stream,
            measure_fast=measure, measure_ema=measure, geometry=geometry)
        equal(before, (state.prior.z, state.ema_prior.z))
        return packet, detail
    reference, reference_detail = preview()
    with torch.device('meta'):
        packet, detail = preview()
    assert packet is not None and detail['accepted'] == 1
    equal(reference_detail, detail)
    assert packet.content_digest() == reference.content_digest()
    equal(packet.fast_coordinates, table[1:2])
    assert packet.children.device.type == packet.fast_output_metric.device.type == 'cpu'


def test_feature_control_install_uses_cpu_calibration_under_meta():
    trainer = make()
    facade = FeatureFacade(trainer.policy)
    expected = ParticleBirthDeath(facade, SEED)
    batch = real()
    with torch.device('meta'):
        actual = ParticleBirthDeath(facade, SEED)
        trainer.policy._feature_selection.observe_shape(batch)
    assert actual.s_k == expected.s_k
    equal(expected.state_dict(), actual.state_dict())
    assert trainer.birth_death.S.device.type == 'cpu'
    assert trainer.policy._feature_selection.state['actual_backend'] == 'feature_cells'


def test_population_empty_and_nonempty_scale_queries_use_cpu_under_meta():
    tester = PopulationSequentialSettleTest()
    with torch.device('meta'):
        empty = tester._scale_values()
        stats = [tester._early_stat(x) for x in empty]
    assert all(x.device.type == 'cpu' and len(x) == 0 for x in empty)
    assert all(x['n'] == 0 and x['verdict'] == 0 for x in stats)
    tester.rows = 2
    tester.r_b = [torch.tensor([.2, .4], device='cpu')]
    tester.r_2b = [torch.tensor([-.1, -.3], device='cpu')]
    expected = tester._scale_values()
    with torch.device('meta'):
        actual = tester._scale_values()
    equal(expected, actual)


def reject_atomically(target, saved, owner, message):
    before = target.state_dict()
    parameters = list(target.policy._training_modules().values())
    parameters = [p for module in parameters for p in module.parameters()] + [target.prior.z]
    versions = [p._version for p in parameters]
    controls = target.birth_death, target.policy.lr_settle, target.policy._feature_selection
    invalid = deepcopy(saved)
    with pytest.raises(ValueError, match=message):
        (target.load_state_dict if owner == 'trainer' else target.policy.load_state_dict)(invalid)
    assert versions == [p._version for p in parameters]
    assert controls == (target.birth_death, target.policy.lr_settle, target.policy._feature_selection)
    equal(before, target.state_dict())
    equal(saved, invalid)


@pytest.mark.parametrize('owner', ['trainer', 'policy'])
@pytest.mark.parametrize('width', [2, 9])
def test_equal_width_different_fifo_shape_is_rejected_atomically(owner, width):
    source = make(width=width)
    source.step(real(width))
    saved = (source.state_dict() if owner == 'trainer' else source.policy.state_dict())
    assert saved['backend_selection']['output_shape'] == [width]
    saved['birth_death']['sample_shape'] = (1, width)
    target = make(width=width)
    reject_atomically(target, saved, owner, 'sample shape differs')


@pytest.mark.parametrize('owner', ['trainer', 'policy'])
@pytest.mark.parametrize('backend', ['auto', 'feature_cells'])
def test_unresolved_selection_cannot_restore_initialized_fifo(owner, backend):
    source = make(backend=backend)
    source.birth_death.observe_real(real())
    saved = (source.state_dict() if owner == 'trainer' else source.policy.state_dict())
    assert saved['completed_steps'] == 0 and saved['backend_selection']['output_shape'] is None
    assert saved['birth_death']['reservoir'] is not None
    target = make(backend=backend)
    reject_atomically(target, saved, owner, 'unresolved backend output shape')


@pytest.mark.parametrize('owner', ['trainer', 'policy'])
@pytest.mark.parametrize('resolved', [False, True])
def test_uninitialized_fifo_valid_pending_or_resolved_checkpoint_restores(owner, resolved):
    source = make()
    if resolved:source.policy._feature_selection.observe_shape(real())
    saved = (source.state_dict() if owner == 'trainer' else source.policy.state_dict())
    assert saved['birth_death']['reservoir'] is saved['birth_death']['sample_shape'] is None
    target = make()
    (target.load_state_dict if owner == 'trainer' else target.policy.load_state_dict)(saved)
    equal(saved, target.state_dict() if owner == 'trainer' else target.policy.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_cuda_default_preserves_feature_reactions_and_checkpoint_replay(monkeypatch):
    # Same explicit GPU model/data law and fixed seed, only the ambient default
    # device differs. This exercises both raw-moment evaluations, not a scorer.
    device = torch.device('cuda', torch.cuda.current_device())
    import particlegan.feature_cells as feature_cells
    monkeypatch.setattr(feature_cells, 'time', SimpleNamespace(perf_counter=lambda: 0.))
    def run(default):
        with torch.device(default):
            trainer = make(device=device)
            batch = real(device=device)
            losses = [trainer.step(batch + step * .001) for step in range(18)]
            assert trainer.birth_death.snapshot_serial >= 2
            assert trainer.birth_death.counters['mean_forward_rows'] > 0
            saved = trainer.state_dict()
            restored = make(device=device)
            restored.load_state_dict(saved)
            equal(saved, restored.state_dict())
            next_a, next_b = trainer.step(batch + .018), restored.step(batch + .018)
            equal(next_a, next_b)
            equal(trainer.state_dict(), restored.state_dict())
            samples = trainer.sample(64, output_noise=True,
                generator=torch.Generator(device=device).manual_seed(SEED + 100))
            return losses, trainer.state_dict(), samples
    with torch.random.fork_rng(devices=[device.index]):
        reference = run('cpu')
        actual = run(device)
    equal(reference, actual)

"""Analytical and bounded public-API software checks, not quality experiments."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from experiments.forge.api import TRAINER_STREAM_BINDINGS
from experiments.forge.boundaries import recipe_field_owner, validate_registry
from experiments.forge.rng import NamedStreams
from particlegan import GANTrainer, get_recipe
from particlegan.conditional_transport import OutputMarginalTransport
from particlegan.init import deterministic_orthogonal_
from particlegan.kinetic_transport import kinetic_transport_loss
from particlegan.transport_mobility import block_mmd_mobility


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def test_exact_block_reference_all_rows_uncertainty_and_detachment():
    x = torch.linspace(-1, 1, 11, dtype=torch.float64).reshape(-1, 1).requires_grad_()
    y = (x.detach().square() + .3).requires_grad_()
    before = torch.get_rng_state().clone()
    factor, receipt = block_mmd_mobility(x, y)
    width = float(((y.detach() - y.detach().mean(0)) ** 2).sum(1).mean())
    values = []
    for a, b in zip(x.detach().tensor_split(2), y.detach().tensor_split(2)):
        h = 0.
        for i in range(len(a)):
            for j in range(len(a)):
                if i != j:
                    kernel = lambda p, q: math.exp(-float((p-q).square().sum()) / (2 * width))
                    h += kernel(a[i], a[j]) + kernel(b[i], b[j]) - kernel(a[i], b[j]) - kernel(b[i], a[j])
        values.append(h / (len(a) * (len(a)-1)))
    values = torch.tensor(values, dtype=x.dtype)
    mean, error = float(values.mean()), float(values.std() / math.sqrt(2))
    expected = max(0., mean-error) / (max(0., mean) + error + torch.finfo(x.dtype).eps)
    assert receipt['rows'] == 11 and receipt['blocks'] == 2
    assert receipt['signal'] == pytest.approx(mean)
    assert receipt['standard_error'] == pytest.approx(error)
    assert float(factor) == pytest.approx(expected)
    assert not factor.requires_grad and torch.equal(before, torch.get_rng_state())


def test_null_suppression_and_repeated_systematic_signal():
    real = torch.linspace(-1., 1., 32, dtype=torch.float64).reshape(-1, 1).repeat(4, 1)
    null, evidence = block_mmd_mobility(real, real)
    assert null == 0 and evidence['signal'] == 0 and evidence['standard_error'] == 0
    shifted, evidence = block_mmd_mobility(real + 2, real)
    assert shifted > .999 and evidence['signal'] > 0
    # Equal-marginal batches need not share row pairing: unbiased estimates can be negative.
    equal_law, _ = block_mmd_mobility(real.flip(0), real)
    assert equal_law == 0
    scaled, _ = block_mmd_mobility(7*(real+2) + 11, 7*real + 11)
    assert float(scaled) == pytest.approx(float(shifted))
    small, e = block_mmd_mobility(real[:3], real[:3]+2)
    assert small == 0 and e['blocks'] == 1


def test_auxiliary_gradient_scalar_and_original_conditional_objective():
    recipe = get_recipe('bcap', kinetic_transport_weight=1., transport_mobility_mode='block_mmd_v1')
    real = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(-1, 1).repeat(2, 1)
    fake = (real + .2).requires_grad_()
    factor, _ = block_mmd_mobility(fake, real)
    raw = kinetic_transport_loss(fake, real)
    weighted = recipe.kinetic_transport_loss(fake, real)
    torch.testing.assert_close(torch.autograd.grad(weighted, fake, retain_graph=True)[0],
                               factor * torch.autograd.grad(raw, fake)[0])
    original = (fake-real).square().mean()
    consumer = OutputMarginalTransport(recipe)
    total = consumer.add(original, fake, real, conditioning=real)
    torch.testing.assert_close(total-original, weighted)
    assert consumer.mobility.stats['calls'] == 1
    permuted = real.flip(0).clone().requires_grad_()
    identity = (permuted-real).square().mean()
    marginal = consumer.add(identity, permuted, real, conditioning=real)
    assert float(identity.detach()) > .1 and float(marginal.detach()) == float(identity.detach())


def trainer(active, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        r = get_recipe('bcap', z_dim=2, num_particles=8, batch_size=8, total_steps=8,
                       prior_kind='mog', sigma_rel=.025, standardize=False,
                       kinetic_transport_weight=1. if active else 0.,
                       kinetic_transport_local_weight=1. if active else 0.,
                       transport_mobility_mode='block_mmd_v1' if active else 'none')
        g = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))
        d = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
        prior = r.make_prior()
        for model in (g, d, prior):
            deterministic_orthogonal_(model, seed=0)
            model.to(device)
        streams = NamedStreams(0, device=device)
        return GANTrainer(r, g, d, prior=prior, seed=0,
            **{name: streams.generator(f, component=c, purpose=p)
               for name, (f, c, p) in TRAINER_STREAM_BINDINGS.items()})


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_active_public_trainer_resume_exact_clock_rng_and_invalid_atomicity(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    live = trainer(True, device)
    real = torch.arange(16, device=device, dtype=torch.float32).reshape(8, 2) / 8 - 1
    for _ in range(3):
        live.step(real)
    saved = live.state_dict()
    assert saved['transport_mobility']['stats']['calls'] == 3
    resumed = trainer(True, device)
    resumed.load_state_dict(saved)
    for _ in range(3):
        output = live.step(real)
        uninterrupted = live.state_dict()
        resume_output = resumed.step(real)
        same(output, resume_output)
        same(uninterrupted, resumed.state_dict())
    bad = deepcopy(resumed.state_dict())
    bad['transport_mobility']['stats']['calls'] += 1
    before = resumed.state_dict()
    with pytest.raises(ValueError, match='clock'):
        resumed.load_state_dict(bad)
    same(before, resumed.state_dict())


def test_disabled_omits_state_and_never_estimates(monkeypatch):
    import particlegan.transport_mobility as module
    monkeypatch.setattr(module, 'block_mmd_mobility', lambda *_: pytest.fail('inactive estimator'))
    live = trainer(False, 'cpu')
    state = live.state_dict()
    assert 'transport_mobility' not in state and 'transport_mobility_mode' not in state['recipe']
    live.step(torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8 - 1)
    restored = trainer(False, 'cpu')
    restored.load_state_dict(state)
    validate_registry()
    assert recipe_field_owner('transport_mobility_mode') == 'technique'


@pytest.mark.parametrize('override', [dict(transport_mobility_mode='bad'),
                                     dict(transport_mobility_mode='block_mmd_v1')])
def test_invalid_recipe(override):
    with pytest.raises(ValueError, match='mobility'):
        get_recipe('bcap', **override)


def test_consumer_rejects_bad_counters_before_mutation():
    r = get_recipe('bcap', kinetic_transport_weight=1., transport_mobility_mode='block_mmd_v1')
    consumer = OutputMarginalTransport(r)
    panel = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    consumer.add(panel.sum(), panel, panel+1)
    before = consumer.state_dict()
    bad = deepcopy(before)
    bad['transport_mobility']['stats']['standard_error_sum'] = float('nan')
    with pytest.raises(ValueError, match='counters'):
        consumer.load_state_dict(bad)
    same(before, consumer.state_dict())


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_active_word_joint_resume_with_actual_data_streams(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from experiments.forge.api import FormulationContext
    from experiments.forge.state import state_digest
    from benchmarks.toy_audit.api_images import WordFixture
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    overrides = json.loads((root / 'reports/forge/bcap-three-phase/baseline.json').read_text())['global_recipe_overrides']
    overrides.update(kinetic_transport_weight=1., kinetic_transport_local_weight=1.,
        transport_mobility_mode='block_mmd_v1', optimizer_svd_backend='cpu',
        num_particles=5, z_dim=2, batch_size=256, total_steps=20000)
    def build():
        context = FormulationContext(recipe_preset='bcap', recipe_overrides=overrides,
            prior=dict(kind='particle_cloud', sigma=0., standardize=False, learnable=True,
                       exception_reason='Bounded original finite-vocabulary software replay'),
            seed=0, device=device, initializer='deterministic_orthogonal',
            execution_path='public_components', component_transport='output_marginal_v1')
        return WordFixture(device=device, seed=0, recipe_name=None, max_steps=4, components=context), context
    before_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        live, context = build()
        live.step()
        saved, streams = deepcopy(live.state_dict()), deepcopy(context.streams.state_dict())
        expected = state_digest([live.step(), live.step()])
        final = live.state_dict()
        restored, restored_context = build()
        restored_context.streams.load_state_dict(streams)
        restored.policy.load_state_dict(saved['api_state'])
        restored.data_generator.set_state(saved['data_generator'])
        restored.restore_component_transport(saved['component_transport'])
        assert state_digest([restored.step(), restored.step()]) == expected
        same(final, restored.state_dict())
        same(context.streams.state_dict(), restored_context.streams.state_dict())
        assert restored.transport.mobility.stats['calls'] == 3
    finally:
        torch.set_num_threads(before_threads)

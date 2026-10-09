from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from experiments.forge.api import task_policy_blockers
from experiments.forge.state import state_digest


def trainer(enabled=True):
    torch.manual_seed(0)
    recipe = get_recipe('bcap', num_particles=24, z_dim=2, batch_size=8, total_steps=8,
                        prior_kind='mog', standardize=False, sigma_rel=.025,
                        same_batch_extragradient=enabled)
    return GANTrainer(recipe, nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)),
                      nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)),
                      model_generator=torch.Generator().manual_seed(8))


def test_exact_alternating_map_and_single_clock():
    candidate, reference = trainer(), trainer()
    real = torch.linspace(-1, 1, 16).reshape(8, 2)
    original = reference.state_dict()
    # Independent composition of the existing public step, as a software oracle.
    reference._step(real)
    preview = {name: deepcopy(getattr(reference, name).state_dict()) for name in ('G', 'D', 'prior')}
    reference.load_state_dict(original)
    for name in preview:
        getattr(reference, name).load_state_dict(preview[name])
    reference._step(real)
    expected = reference.state_dict()
    candidate.step(real, collect_stats=True)
    actual = candidate.state_dict()
    for name in preview:
        for key, p in getattr(candidate, name).named_parameters():
            base = original['models'][name][key]
            delta = expected['models'][name][key] - preview[name][key]
            assert torch.equal(p, torch.where(delta == 0, base, base + delta))
    assert actual['completed_steps'] == 1
    for left, right in zip(actual['optimizers'], expected['optimizers']):
        assert state_digest(left) == state_digest(right)
    for name in ('latent_generator', 'noise_generator', 'prior_noise_generator', 'penalty_generator'):
        assert torch.equal(actual['streams'][name], expected['streams'][name])
    assert all(v['step'] == 1 for opt in actual['optimizers'] for v in opt['state'].values())
    assert state_digest(actual['models']['ema_G']) == state_digest(actual['models']['G'])
    assert state_digest(actual['models']['ema_prior']) == state_digest(actual['models']['prior'])


def test_checkpoint_replay_and_matched_consumed_streams():
    candidate, control = trainer(), trainer(False)
    real = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    candidate.step(real)
    control.step(real)
    saved = candidate.state_dict()
    for key, value in saved['streams'].items():
        assert torch.equal(value, control.state_dict()['streams'][key]), key
    candidate.step(real)
    expected = candidate.state_dict()
    restored = trainer()
    restored.load_state_dict(saved)
    restored.step(real)
    assert state_digest(restored.state_dict()) == state_digest(expected)


def test_data_callable_once_and_failure_rolls_back():
    model = trainer()
    real = torch.zeros(8, 2)
    called = []
    model.step(real, generator_real=lambda: (called.append(1) or real))
    assert called == [1]
    before = model.state_dict()
    original = model.G.forward
    calls = []
    def broken(x):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('corrector failure')
        return original(x)
    model.G.forward = broken
    with pytest.raises(RuntimeError, match='corrector failure'):
        model.step(real)
    assert state_digest(model.state_dict()) == state_digest(before)


def test_zero_displacement_and_unsampled_rows_are_bitwise_preserved():
    model = trainer()
    real = torch.zeros(8, 2)
    for p in model.D.parameters():
        p.data.zero_()
    before = model.prior.z.detach().clone()
    model.step(real)
    assert torch.equal(before, model.prior.z)


@pytest.mark.parametrize('change', [dict(optimizer_family='adam'), dict(optimizer_momentum=.5),
                                    dict(ema_decay=.9), dict(same_batch_extragradient=1)])
def test_unsupported_recipe_rejected(change):
    with pytest.raises(ValueError, match='same_batch_extragradient'):
        trainer().recipe.replace(**change)


def test_components_blocked_and_default_packet_preserved():
    recipe = trainer().recipe
    assert task_policy_blockers({'id':'two_pole','adapter':'transfer_behavior',
                                'execution':{'host':'two_pole'}},
                               {'recipe_overrides':recipe.to_dict()})
    assert 'same_batch_extragradient' not in trainer(False).recipe.to_dict()


def test_model_noise_and_buffers_advance_once():
    def build(enabled):
        torch.manual_seed(2)
        recipe = trainer(enabled).recipe.replace(input_noise_std=.03, output_noise_std=.04)
        g = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.Dropout(.25), nn.Linear(8, 2))
        d = nn.Sequential(nn.Linear(2, 8), nn.Dropout(.2), nn.Linear(8, 1))
        return GANTrainer(recipe, g, d, model_generator=torch.Generator().manual_seed(8))
    candidate, control = build(True), build(False)
    real = torch.zeros(8, 2)
    candidate.step(real)
    control.step(real)
    for name, stream in candidate.state_dict()['streams'].items():
        assert torch.equal(stream, control.state_dict()['streams'][name]), name
    assert candidate.G[1].num_batches_tracked == control.G[1].num_batches_tracked == 1

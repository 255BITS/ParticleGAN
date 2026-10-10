"""Hydraulic travel software checks; tiny public API updates confer no scientific qualification."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from experiments.forge.api import task_formulation_context, task_policy_blockers
from experiments.forge.state import state_digest
from experiments.forge.vectorprofiles import build_vector_models, resolve_vector_spec
from particlegan import get_recipe
from particlegan.hydraulic import HydraulicTravel

ROOT = Path(__file__).resolve().parents[1]


def card(name):
    return json.loads((ROOT / f'configs/forge/ideas/{name}.json').read_text())


def task(name):
    return json.loads((ROOT / f'configs/forge/tasks/{name}.json').read_text())


def test_nonlinear_accepted_output_is_bounded_and_direction_preserved():
    weight = torch.nn.Parameter(torch.tensor([[1.0]]))
    optimizer = torch.optim.SGD([weight], lr=10.)
    weight.grad = torch.tensor([[-1.]])
    limiter = HydraulicTravel(1.)
    real = torch.tensor([[0.], [.1], [.2]])
    old = weight.detach().clone() ** 3
    def probe():
        result = weight ** 3
        return result, result
    limiter.step(optimizer, real, probe)
    assert 0 < float(weight.detach() - 1) < 10
    assert float((weight.detach() ** 3 - old).abs()) <= .1
    assert limiter.summary['max_accepted_radius_ratio'] <= 1
    assert limiter.summary['limited'] == 1


def test_gap_radius_relaxes_only_when_generated_is_far():
    def run(start, radius):
        weight = torch.nn.Parameter(torch.tensor([[start]]))
        optimizer = torch.optim.SGD([weight], lr=1.)
        weight.grad = torch.tensor([[-1.]])
        limiter = HydraulicTravel(1., radius=radius)
        real = torch.tensor([[0.], [.1], [.2]])
        limiter.step(optimizer, real, lambda: (weight * 1., weight * 1.))
        return float(weight.detach() - start), limiter.summary
    near_v1, _ = run(.1, 'real_spacing')
    near_gap, near = run(.1, 'gap_adaptive')
    far_v1, _ = run(-5., 'real_spacing')
    far_gap, far = run(-5., 'gap_adaptive')
    assert near_gap == pytest.approx(near_v1) == pytest.approx(.1)
    assert near['gap_wider'] == 0 and far['gap_wider'] == 1
    assert far_v1 == pytest.approx(.1) and far_gap == pytest.approx(1.)


def test_recipe_flag_is_opt_in_and_preserves_archived_identity():
    winner = get_recipe('bcap')
    assert winner.hydraulic_travel_fraction == 0 and winner.make_hydraulic_travel() is None
    assert 'hydraulic_travel_fraction' not in winner.to_dict()
    assert 'hydraulic_travel_radius' not in winner.to_dict()
    active = replace(winner, hydraulic_travel_fraction=1.)
    assert isinstance(active.make_hydraulic_travel(), HydraulicTravel)
    with pytest.raises(ValueError, match='hydraulic'):
        replace(winner, hydraulic_travel_fraction=-1.)
    with pytest.raises(ValueError, match='hydraulic'):
        replace(winner, hydraulic_travel_fraction=1., hydraulic_travel_radius='other')
    with pytest.raises(ValueError, match='hydraulic'):
        replace(winner, hydraulic_travel_fraction=1., output_noise_std=.1)


def build(name, steps=4):
    context = task_formulation_context(card(name), task('gaussian1d_smoke'), device='cpu', root=ROOT)
    g, d = build_vector_models(context, resolve_vector_spec(task('gaussian1d_smoke')))
    return context, context.build_trainer(g, d, max_steps=steps)


@pytest.mark.parametrize('name', ['bcap-hydraulic-winner-travel-v1', 'bcap-hydraulic-winner-gap-v1'])
def test_public_checkpoint_replay_and_no_probe_rng_consumption(name):
    torch.set_num_threads(1)
    context, trainer = build(name)
    assert trainer.recipe.constraint_geometry_mode == 'direction_blend'
    real = torch.linspace(1., 3., 128).reshape(-1, 1)
    trainer.step(real, generator_real=real)
    saved = context.state_dict()
    trainer.step(real, generator_real=real)
    expected = state_digest(context.state_dict())
    context.load_state_dict(saved)
    trainer.step(real, generator_real=real)
    assert state_digest(context.state_dict()) == expected
    assert trainer.hydraulic.summary['updates'] == 2
    assert trainer.hydraulic.summary['max_accepted_radius_ratio'] <= 1
    # Probes consume no named stream: compare with the disabled mechanism.
    control_context, control = build(name)
    control.hydraulic = None
    control.step(real, generator_real=real)
    control.step(real, generator_real=real)
    for stream in trainer._STREAMS:
        assert torch.equal(getattr(trainer, stream).get_state(), getattr(control, stream).get_state())
    packet = trainer.state_dict()
    packet['hydraulic']['fraction'] = .5
    before = state_digest(trainer.state_dict())
    with pytest.raises(ValueError, match='hydraulic'):
        trainer.load_state_dict(packet)
    assert state_digest(trainer.state_dict()) == before


def test_disabled_flag_keeps_winner_update_bytes():
    torch.set_num_threads(1)
    winner = card('bcap-default-baseline-direction-v1')
    contexts = []
    for overrides in ({}, {'hydraulic_travel_fraction': 0.0}):
        candidate = {**winner, 'recipe_overrides': {**winner['recipe_overrides'], **overrides}}
        context = task_formulation_context(candidate, task('gaussian1d_smoke'), device='cpu', root=ROOT)
        g, d = build_vector_models(context, resolve_vector_spec(task('gaussian1d_smoke')))
        trainer = context.build_trainer(g, d, max_steps=3)
        assert trainer.hydraulic is None
        real = torch.linspace(1., 3., 128).reshape(-1, 1)
        for _ in range(3):
            trainer.step(real, generator_real=real)
        contexts.append(state_digest(context.state_dict()))
    assert contexts[0] == contexts[1]


@pytest.mark.parametrize('host', ['two_pole', 'trajectory', 'unused_token_hold', 'five_word_joint_smoke'])
def test_caller_owned_hosts_block_before_training(host):
    blockers = task_policy_blockers(task(host), card('bcap-hydraulic-winner-travel-v1'))
    assert blockers and 'hydraulic' in blockers[0]


def test_public_trainer_hosts_are_supported():
    for name in ('mode_hold', 'grid100', 'img_bars4', 'gaussian1d_stability'):
        assert task_policy_blockers(task(name), card('bcap-hydraulic-winner-gap-v1')) == []

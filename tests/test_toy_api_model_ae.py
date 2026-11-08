"""AE toy scorers and public-policy replay, not full-budget quality claims."""
from copy import deepcopy
import json

import pytest
import torch

from benchmarks.toy_audit import api_contract, api_model_ae
from benchmarks.toy_audit.api_model_ae import (
    CASES, DEFAULT_RECIPE, build_case, list_cases, sample_target, score_samples,
)
from experiments.forge.state import state_digest
from particlegan import get_recipe


@pytest.fixture(autouse=True)
def serial_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_declarations_are_frozen_and_public():
    for case in list_cases():
        api_contract.validate_case(case)
        assert case['default_recipe'] == DEFAULT_RECIPE
        assert case['default_steps'] == 1200
        assert case['evaluation_observations'] == 24
        assert case['terminal_observations'] == 5
        assert case['thresholds'] == case['forge_thresholds']


def test_oracle_and_destructive_controls_validate_meaningful_gates():
    target = sample_target(4096, torch.Generator().manual_seed(10000))
    independent = sample_target(4096, torch.Generator().manual_seed(10001))
    assert score_samples(target, target, independent)['passed']
    centers = independent.sign()
    collapsed = score_samples(target, target, centers)
    assert not collapsed['passed']
    assert any('component_min_eigen_ratio' in bound for bound in collapsed['failed_bounds'])
    skew = independent.clone()
    skew[:3072] = skew[:3072].abs()
    assert any('mass_tv' in bound for bound in score_samples(target, target, skew)['failed_bounds'])
    # This distribution is perfect, but the paired mapping is wrong.
    mismatched = score_samples(target, -target, independent)
    assert any('reconstruction_mode_agreement' in bound for bound in mismatched['failed_bounds'])
    assert not score_samples(target, target, torch.full_like(independent, float('nan')))['passed']
    assert score_samples(target, target, torch.zeros_like(independent), adversarial=False)['passed']


@pytest.mark.parametrize('case_id', CASES)
def test_actual_public_updates_evaluation_isolation_and_exact_resume(case_id):
    fixture = build_case(case_id, max_steps=4)
    fixture.step()
    saved = fixture.state_dict()
    initial = state_digest(saved)
    observation = api_contract.validate_observation(fixture.observe(n=256))
    assert observation['views'] and all(torch.isfinite(torch.tensor(value)) for value in observation['metrics'].values())
    assert state_digest(fixture.state_dict()) == initial
    expected_step = fixture.step()
    expected = state_digest(fixture.state_dict())
    fixture.load_state_dict(saved)
    assert fixture.step() == expected_step
    assert state_digest(fixture.state_dict()) == expected
    fresh = build_case(case_id, max_steps=4)
    fresh.load_state_dict(saved)
    assert fresh.step() == expected_step
    assert state_digest(fresh.state_dict()) == expected


def test_recipe_injection_preserves_recipe_prior_policy_and_matches_data():
    baseline = build_case('api-model-ae-gan-multimodal', max_steps=2)
    frozen_recipe = baseline.recipe.replace(prior_update='frozen', prior_reg=.123)
    frozen = build_case('api-model-ae-gan-multimodal', max_steps=2, recipe=frozen_recipe)
    assert frozen.recipe.prior_update == 'frozen'
    assert frozen.recipe.prior_reg == .123
    assert not frozen.prior.z.requires_grad
    assert not any(p is frozen.prior.z for group in frozen.opt_g.param_groups for p in group["params"])
    assert float(frozen.prior.sigma) == pytest.approx(.025)
    for role in ('G', 'D', 'E'):
        assert state_digest(getattr(baseline, role).state_dict()) == state_digest(getattr(frozen, role).state_dict())
    torch.testing.assert_close(baseline.prior.z, frozen.prior.z, rtol=0, atol=0)
    baseline.step(); frozen.step()
    assert baseline.batch_sequence_sha256 == frozen.batch_sequence_sha256
    assert not torch.equal(baseline.prior.z, frozen.prior.z)
    assert frozen.prior.z.grad is None


def test_seed_budget_and_checkpoint_identity_reject_contract_changes():
    with pytest.raises(ValueError, match='seed 0'):
        build_case('api-model-ae-reconstruction', seed=1)
    with pytest.raises(ValueError, match='prefix'):
        build_case('api-model-ae-reconstruction', max_steps=1201)
    fixture = build_case('api-model-ae-reconstruction', max_steps=2)
    bad = deepcopy(fixture.state_dict()); bad['case_id'] = 'wrong'
    original = state_digest(fixture.state_dict())
    with pytest.raises(ValueError, match='case_id'):
        fixture.load_state_dict(bad)
    assert state_digest(fixture.state_dict()) == original


@pytest.mark.parametrize('change', ('bounds', 'steps', 'prior_width', 'resources', 'architecture'))
def test_task_json_cannot_redefine_its_executable_contract(tmp_path, monkeypatch, change):
    directory = tmp_path / 'configs/forge/tasks'
    directory.mkdir(parents=True)
    for task_id in CASES.values():
        source = api_model_ae.ROOT / f'configs/forge/tasks/{task_id}.json'
        task = json.loads(source.read_text())
        if change == 'bounds':
            task['evaluation']['thresholds'][0][2] = 10.
        elif change == 'steps':
            task['execution']['steps'] = 12
        elif change == 'prior_width':
            task['execution']['prior']['sigma'] = .5
        elif change == 'resources':
            task['execution']['host_definition']['resources']['num_particles'] = 8
        else:
            task['execution']['host_definition']['hidden'] = 32
        (directory / f'{task_id}.json').write_text(json.dumps(task))
    monkeypatch.setattr(api_model_ae, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='static AE scorer|frozen AE protocol'):
        list_cases()


@pytest.mark.parametrize('changes', ({'encoder_mode': 'none'}, {'num_particles': 32},
                                     {'standardize': True}, {'total_steps': 7000},
                                     {'reconstruction_weight': 2.}))
def test_injected_resolved_recipe_cannot_silently_change_task_fields(changes):
    baseline = build_case('api-model-ae-reconstruction', max_steps=2)
    assert baseline.recipe.total_steps == 1200
    with pytest.raises(ValueError, match='task-owned AE fields'):
        build_case('api-model-ae-reconstruction', max_steps=2, recipe=baseline.recipe.replace(**changes))


@pytest.mark.parametrize('case_id, expected_calls', (('api-model-ae-reconstruction', 1),
                                                    ('api-model-ae-gan-multimodal', 2)))
def test_only_objective_consumed_prior_rows_gain_optimizer_ownership(case_id, expected_calls):
    fixture = build_case(case_id, max_steps=2)
    calls = []
    observe = fixture.policy.observe_sampled_rows
    def record(rows):
        calls.append(rows.clone())
        return observe(rows)
    fixture.policy.observe_sampled_rows = record
    fixture.step()
    assert len(calls) == expected_calls
    assert all(rows.ndim == 1 and rows.dtype == torch.long for rows in calls)
    fixture.observe(n=128)
    assert len(calls) == expected_calls


def test_schedule_free_named_recipe_keeps_its_absent_horizon():
    schedule_free = get_recipe('ka2', continuous_policy='dv1', total_steps=None,
                               **api_model_ae.FROZEN_RECIPE_FIELDS)
    assert api_model_ae.task_recipe(list_cases()[0], recipe=schedule_free).total_steps is None

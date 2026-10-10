"""Opt-in host admission and unchanged full-suite scientific conditions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.forge.api import task_policy_blockers

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def test_original_full_suite_conditions_preserved(tmp_path):
    import shutil
    from tests.archived_forge_contracts import restore_archived_contracts
    path = ROOT/'reports/forge/bcap-develop-integration/original-task-contracts.json'
    if not path.exists():
        pytest.skip('prepare study before verifying registered contracts')
    originals = json.loads(path.read_text())['tasks']
    shutil.copytree(ROOT / 'configs', tmp_path / 'configs')
    restore_archived_contracts(tmp_path, 'bcap_legacy_questions')
    for name, old in originals.items():
        new = json.loads((tmp_path/f'configs/forge/tasks/{name}.json').read_text())
        current_execution = deepcopy(new['execution'])
        for field in ('transport_consumer', 'transport_contract'):
            current_execution.pop(field, None)
        assert current_execution == old['execution'], name
        assert new['adapter'] == old['adapter'], name
        assert new['resources'] == old['resources'], name
        assert new['requires_capabilities'] == old['requires_capabilities'], name
        assert new['dependencies'] == old['dependencies'], name
        current_eval = deepcopy(new['evaluation'])
        original_eval = deepcopy(old['evaluation'])
        for field in ('sources', 'evaluator_revision'):
            current_eval.pop(field, None)
            original_eval.pop(field, None)
        assert current_eval == original_eval, name


@pytest.mark.parametrize('preset', ['bcap', 'k3p', 'ka2', 'r1r2', 'release07_gan_v3'])
def test_disabled_techniques_do_not_require_optional_consumers(preset):
    task = read('configs/forge/tasks/trajectory.json')
    task['execution'].pop('transport_consumer', None)
    task['execution'].pop('transport_contract', None)
    blockers = task_policy_blockers(task, dict(recipe_preset=preset, recipe_overrides={}))
    assert not any('transport' in reason or 'constraint_geometry' in reason for reason in blockers)


def test_active_transport_rejects_missing_hook():
    task = read('configs/forge/tasks/trajectory.json')
    task['execution'].pop('transport_consumer', None)
    blockers = task_policy_blockers(task, dict(recipe_preset='bcap', recipe_overrides={'kinetic_transport_weight':1.}))
    assert any('does not consume' in reason for reason in blockers)


def test_unrecognized_consumer_is_explicit_blocker():
    task = read('configs/forge/tasks/trajectory.json')
    task['execution']['transport_consumer'] = 'unknown'
    assert task_policy_blockers(task, dict(recipe_preset='bcap', recipe_overrides={}))

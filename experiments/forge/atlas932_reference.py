"""Verified historical927 metadata for the exact932 objective diagnostic.

The completed927 reference uses original two_pole/objective .02.  Current932
uses a distinct Task/objective0.  This resolver returns declaration bindings
only; it never invokes the old factory or transfers results, grades or costs.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .atlas932_contract import (BASE_RECIPE_SHA256, CANDIDATE_ID, VIEW_ID, STUDY_ID,
    TASK_ID, TASK_DIGEST, OPTIMIZER_VARIANT, OPTIMIZER_VARIANT_SHA256,
    OBJECTIVE_VARIANT, OBJECTIVE_VARIANT_SHA256, UNCHANGED_SOURCE_PINS,
    canonical, digest, task_digest)

CONTROL_ID = 'atlas-two-pole-particle-amsgrad-off927-v1'
PARENT_TASK_ID = 'two_pole'
TASK_PATH = 'configs/forge/tasks/two_pole_l2_off932_v1.json'
PARENT_TASK_PATH = 'configs/forge/tasks/two_pole.json'
PARENT_TASK_DIGEST = '2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b'
BASELINE_OWNER_PATH = 'experiments/forge/atlas927_two_pole_owner.py'
BASELINE_OWNER_PIN = {
    'sha256': '0fdd73021ab38f660b8c1ee4b292e32ef9fa39077751ba85d7679a64d283a587',
    'bytes': 68998}
CHANGED_REGISTRATION_PATHS = frozenset({
    'experiments/forge/api.py', 'experiments/forge/atlas_existing_mog.py',
    'experiments/forge/adapters.py', 'experiments/forge/studies.py'})


def _source(root, relative, pin=None):
    base = Path(root).resolve()
    target = base / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or target.resolve() != target or not target.is_file()):
        raise ValueError('unsafe or missing historical Source: ' + relative)
    raw = target.read_bytes()
    if pin is not None and (len(raw) != pin['bytes']
            or hashlib.sha256(raw).hexdigest() != pin['sha256']):
        raise ValueError('historical Source byte identity changed: ' + relative)
    return raw


def _assignment(source, name):
    matches = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(matches) != 1:
        raise ValueError('historical Source must declare one ' + name)
    return ast.literal_eval(matches[0])


def _guard(study, candidate, current_candidate, tasks):
    from .atlas932_two_pole_owner import supports_candidate
    if (study.get('id') != STUDY_ID or study.get('candidate') != CANDIDATE_ID
            or study.get('status') not in ('draft', 'ready')
            or study.get('scope') != dict(view=VIEW_ID, through_tier=1, execution_backend='cpu')
            or study.get('control') != dict(candidate_id=CONTROL_ID, task_map={TASK_ID: PARENT_TASK_ID})
            or study.get('campaign') != dict(id=STUDY_ID, budget_seconds=300, candidate_budget_seconds=300)
            or study.get('max_rounds') != 1 or candidate.get('id') != CONTROL_ID
            or not supports_candidate(current_candidate)
            or not isinstance(tasks, dict) or not set(tasks) <= {TASK_ID}):
        raise ValueError('historical metadata is only the exact932 Study/completed927 reference')


def historical_reference_bindings(candidate, tasks, protocol, root, *, study, current_candidate):
    """Expose only pinned927 metadata, with its original Task and objective .02."""
    _guard(study, candidate, current_candidate, tasks)
    original_owner = _source(root, BASELINE_OWNER_PATH, BASELINE_OWNER_PIN)
    original_pins = _assignment(original_owner, 'SOURCE_PINS')
    if not CHANGED_REGISTRATION_PATHS <= original_pins.keys():
        raise ValueError('historical registration boundary differs from its pinned927 owner')
    checked = {p: pin for p, pin in original_pins.items() if p not in CHANGED_REGISTRATION_PATHS}
    source = {p: _source(root, p, pin) for p, pin in checked.items()}
    original_candidate = json.loads(source['configs/forge/ideas/' + CONTROL_ID + '.json'])
    if any(canonical(candidate.get(k)) != canonical(v) for k, v in original_candidate.items()):
        raise ValueError('historical control declaration differs from pinned927 Idea')
    original_task = json.loads(source[PARENT_TASK_PATH])
    new_task = json.loads(_source(root, TASK_PATH, UNCHANGED_SOURCE_PINS[TASK_PATH]))
    expected_task = deepcopy(original_task)
    expected_task.update(id=TASK_ID, task_cohort='atlas_two_pole_l2_off932_v1',
        objective_substitution_parent={
            'task_id': PARENT_TASK_ID,
            **UNCHANGED_SOURCE_PINS[PARENT_TASK_PATH],
            'objective_variant': deepcopy(OBJECTIVE_VARIANT)})
    expected_task['execution']['particle_l2'] = 0.0
    if (task_digest(original_task) != PARENT_TASK_DIGEST
            or task_digest(new_task) != TASK_DIGEST
            or canonical(new_task) != canonical(expected_task)
            or any(task_digest(task) != PARENT_TASK_DIGEST for task in tasks.values())
            or canonical(protocol) != canonical(json.loads(source['configs/forge/protocols/screening.json']))):
        raise ValueError('reference requires the exact Task delta and original seed0 protocol')
    recipe_ast = ast.parse(source['particlegan/recipes.py'])
    recipe_class = next(n for n in recipe_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Recipe')
    fields = {n.target.id: ast.literal_eval(n.value) for n in recipe_class.body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    fields.update(name='atlas', **json.loads(source['configs/100gaussians/atlas.json']))
    fields.update(num_particles=12, z_dim=1, batch_size=12, sigma_rel=0., standardize=False)
    if fields['reg_arm'] is not None:
        fields['critic_formulation'] = 'k3p'
    fields = json.loads(canonical(fields))
    if len(fields) != 79 or digest(fields) != BASE_RECIPE_SHA256 or fields['total_steps'] is not None:
        raise ValueError('historical Source no longer reconstructs original full79 baseD5')
    from . import decision_contracts as decisions
    from .atlas932_two_pole_owner import resolve_binding
    from .views import task_execution_fingerprint, task_evaluation_fingerprint
    current_tasks = {name: deepcopy(new_task) for name in tasks}
    result = decisions._bindings(current_candidate, current_tasks, protocol, root)
    for name, task in current_tasks.items():
        binding = resolve_binding(root, current_candidate, task, protocol)
        if (canonical(binding['recipe']) != canonical(fields)
                or binding.get('base_recipe_sha256') != BASE_RECIPE_SHA256
                or canonical(binding.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
                or binding.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256
                or canonical(binding.get('effective_objective_variant')) != canonical(OBJECTIVE_VARIANT)
                or binding.get('effective_objective_variant_sha256') != OBJECTIVE_VARIANT_SHA256):
            raise ValueError('current932 must preserve baseD5/optimizer950ed and declare objective15145')
        host = result['host'][name]['bound']
        ablation = host.get('objective_ablation')
        if (not isinstance(ablation, dict)
                or canonical(ablation.get('effective_objective_variant')) != canonical(OBJECTIVE_VARIANT)
                or ablation.get('effective_objective_variant_sha256') != OBJECTIVE_VARIANT_SHA256):
            raise ValueError('current932 host must expose its explicit objective law')
        result['host'][name]['adapter_parameters'].pop('particle_l2', None)
        result['task_identity'][name] = {
            'execution': task_execution_fingerprint(original_task),
            'evaluation': task_evaluation_fingerprint(original_task)}
        base_host = {k: deepcopy(v) for k, v in host.items() if k != 'objective_ablation'}
        result['host'][name]['bound'] = {**base_host, 'historical_objective_reference': {
            'schema': 'forge_atlas932_completed927_metadata_reference_v1',
            'candidate_id': CONTROL_ID, 'task_id': PARENT_TASK_ID,
            'task_pin': deepcopy(UNCHANGED_SOURCE_PINS[PARENT_TASK_PATH]),
            'task_digest': PARENT_TASK_DIGEST, 'particle_l2': 0.02,
            'base_recipe_sha256': BASE_RECIPE_SHA256,
            'effective_optimizer_variant': deepcopy(OPTIMIZER_VARIANT),
            'effective_optimizer_variant_sha256': OPTIMIZER_VARIANT_SHA256,
            'owner_pin': deepcopy(BASELINE_OWNER_PIN), 'verified_source_pins': deepcopy(checked),
            'changed_registration_paths': sorted(CHANGED_REGISTRATION_PATHS),
            'factory_runnable': False, 'historical_reference_only': True,
            'base_bindings': 'exact parent/new Task delta proves data/geometry/initialization/schedule/gates equal; objective and Task identity differ',
            'result_or_grade_transfer': False}}
    return result

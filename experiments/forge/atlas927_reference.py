"""Verified historical metadata for the exact927 Study's completed921 control.

No old factory is invoked or made runnable on changed registration Source.
This helper returns declaration bindings only, never old results, grades or cost.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .atlas927_contract import (BASE_RECIPE_SHA256, CANDIDATE_ID, VIEW_ID, STUDY_ID,
    TASK_ID, TASK_DIGEST, OPTIMIZER_VARIANT, OPTIMIZER_VARIANT_SHA256,
    canonical, digest, task_digest)

CONTROL_ID = 'atlas-original-two-pole-passive889-v1'
BASELINE_OWNER_PATH = 'experiments/forge/atlas889_two_pole_owner.py'
BASELINE_OWNER_PIN = {
    'sha256': 'c0eb7e20fb6c2caca5c759b54c00bd742a4f2f3b1e7e4a42545ac8233af3aa76',
    'bytes': 64151}
CHANGED_REGISTRATION_PATHS = frozenset({
    'experiments/forge/api.py', 'experiments/forge/atlas_existing_mog.py',
    'experiments/forge/adapters.py'})
BASELINE_REFERENCE = {
    'request_id': 'd9fbd5fa5a55b4af405035a3',
    'attempt': '6a64afbc99b742889fedefeaf3d2b7e1',
    'source_digest': '2868968d572545047911bd7445a51a6bbcf6b723312606bae5db1af45512a43b',
    'candidate_revision': '19344b170308e8db3818acb5fbbd20be103a0e0d36086fec56ae8624c14eedc6',
    'request_sha256': 'b5967e908cb0f8a950cc3a8e90e704e709c1dbf26422b77bf42645c53b426eba',
    'raw_pin': {'sha256': '17ff337a808fe7f89042ffe3d6c27ef3536f56a634e20e2db5223c4cbe477e18',
                'bytes': 12967018},
    'provenance': 'ROOT-retained completed921 identity; references only; no result read or replay'}


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
    from .atlas927_two_pole_owner import supports_candidate
    if (study.get('id') != STUDY_ID or study.get('candidate') != CANDIDATE_ID
            or study.get('status') not in ('draft', 'ready')
            or study.get('scope') != dict(view=VIEW_ID, through_tier=1, execution_backend='cpu')
            or study.get('control') != dict(candidate_id=CONTROL_ID, task_map={TASK_ID: TASK_ID})
            or study.get('campaign') != dict(id=STUDY_ID, budget_seconds=300, candidate_budget_seconds=300)
            or study.get('max_rounds') != 1 or candidate.get('id') != CONTROL_ID
            or not supports_candidate(current_candidate)
            or not isinstance(tasks, dict) or not set(tasks) <= {TASK_ID}):
        raise ValueError('historical metadata is only the exact927 Study/control reference')


def historical_reference_bindings(candidate, tasks, protocol, root, *, study, current_candidate):
    """Read preserved baseline Source and expose a non-runnable control descriptor."""
    _guard(study, candidate, current_candidate, tasks)
    original_owner = _source(root, BASELINE_OWNER_PATH, BASELINE_OWNER_PIN)
    original_pins = _assignment(original_owner, 'SOURCE_PINS')
    if not CHANGED_REGISTRATION_PATHS <= original_pins.keys():
        raise ValueError('historical registration boundary no longer matches its pinned owner')
    checked = {p: pin for p, pin in original_pins.items() if p not in CHANGED_REGISTRATION_PATHS}
    source = {p: _source(root, p, pin) for p, pin in checked.items()}
    original_candidate = json.loads(source['configs/forge/ideas/' + CONTROL_ID + '.json'])
    # load_idea adds inherited/resolved metadata; only the raw declaration fields
    # are compared here. Unknown task changes still fail the exact task digest.
    if any(canonical(candidate.get(k)) != canonical(v) for k, v in original_candidate.items()):
        raise ValueError('historical control declaration differs from pinned completed921 Idea')
    original_task = json.loads(source['configs/forge/tasks/two_pole.json'])
    if (task_digest(original_task) != TASK_DIGEST
            or any(name != TASK_ID or task_digest(task) != TASK_DIGEST for name, task in tasks.items())
            or canonical(protocol) != canonical(json.loads(source['configs/forge/protocols/screening.json']))):
        raise ValueError('historical reference requires the complete original two_pole and seed0 protocol')
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
        raise ValueError('historical Source no longer reconstructs original full79 D5')
    from . import decision_contracts as decisions
    from .atlas927_two_pole_owner import resolve_binding
    result = decisions._bindings(current_candidate, tasks, protocol, root)
    baseline_variant = deepcopy(OPTIMIZER_VARIANT)
    baseline_variant['groups'][0]['amsgrad'] = True
    for name, task in tasks.items():
        current_binding = resolve_binding(root, current_candidate, task, protocol)
        if (canonical(current_binding['recipe']) != canonical(fields)
                or current_binding.get('base_recipe_sha256') != BASE_RECIPE_SHA256
                or canonical(current_binding.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
                or current_binding.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256):
            raise ValueError('current927 does not preserve the historical base task Recipe')
        host = result['host'][name]['bound']
        ablation = host.get('optimizer_ablation')
        if (not isinstance(ablation, dict)
                or canonical(ablation.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
                or ablation.get('base_recipe_sha256') != BASE_RECIPE_SHA256
                or ablation.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256):
            raise ValueError('current927 host must expose its actual explicit optimizer law')
        base_host = {k: deepcopy(v) for k, v in host.items() if k != 'optimizer_ablation'}
        result['host'][name]['bound'] = {**base_host, 'historical_optimizer_reference': {
            'schema': 'forge_atlas927_completed921_metadata_reference_v1',
            'candidate_id': CONTROL_ID, 'base_recipe_sha256': BASE_RECIPE_SHA256,
            'effective_optimizer_variant': baseline_variant,
            'effective_optimizer_variant_sha256': digest(baseline_variant),
            'owner_pin': deepcopy(BASELINE_OWNER_PIN), 'verified_source_pins': deepcopy(checked),
            'changed_registration_paths': sorted(CHANGED_REGISTRATION_PATHS),
            'completed_reference': deepcopy(BASELINE_REFERENCE),
            'factory_runnable': False, 'historical_reference_only': True,
            'source_law': 'pinned c0eb constructor: original table/noise/critic AMSGradTrue; completed LR clock',
            'base_bindings': 'original pinned task/Recipe/protocol laws also verified against current927 metadata',
            'result_or_grade_transfer': False}}
    return result
